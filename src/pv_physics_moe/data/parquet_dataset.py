from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from pv_physics_moe.parquet_config import ParquetForecastConfig
from pv_physics_moe.physics.rest2 import PHYSICS_FEATURES, Rest2FeatureBuilder
from pv_physics_moe.physics.solar_geometry import solar_geometry


@dataclass
class ParquetNormalization:
    serial_mean: np.ndarray
    serial_std: np.ndarray
    serial_missing_fraction: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, observed: np.ndarray | None = None) -> "ParquetNormalization":
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        if not np.isfinite(mean).all():
            bad = np.flatnonzero(~np.isfinite(mean)).tolist()
            raise ValueError(f"training rows contain no finite values for serial column indices {bad}")
        missing = 1.0 - observed.mean(axis=0) if observed is not None else np.isnan(values).mean(axis=0)
        return cls(
            mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32), missing.astype(np.float32)
        )

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "serial_mean": self.serial_mean.tolist(),
            "serial_std": self.serial_std.tolist(),
            "serial_missing_fraction": self.serial_missing_fraction.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ParquetNormalization":
        mean = np.asarray(state["serial_mean"], np.float32)
        return cls(
            mean,
            np.asarray(state["serial_std"], np.float32),
            np.asarray(state.get("serial_missing_fraction", np.zeros_like(mean)), np.float32),
        )


def _parse_list(value: Any, field_name: str) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                result = parser(stripped)
                if isinstance(result, (list, tuple)):
                    return list(result)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    raise ValueError(f"{field_name} must contain a list or a serialized list")


def _timestamp(value: Any, timezone: str | None) -> pd.Timestamp:
    result = pd.to_datetime(value, errors="coerce")
    if pd.isna(result):
        raise ValueError(f"unparseable timestamp: {value!r}")
    result = pd.Timestamp(result)
    if timezone:
        result = result.tz_localize(timezone) if result.tzinfo is None else result.tz_convert(timezone)
    return result


def _range_mask(times: pd.Series, bounds: Iterable[Any] | None, timezone: str | None) -> np.ndarray:
    if not bounds:
        return np.ones(len(times), dtype=bool)
    start, end = list(bounds)
    start_ts, end_ts = _timestamp(start, timezone), _timestamp(end, timezone)
    return ((times >= start_ts) & (times < end_ts)).to_numpy()


class ConfigurableParquetDataset(Dataset):
    """One strict, configuration-driven adapter for YLJ and Luoyang Parquet data."""

    def __init__(
        self,
        config: ParquetForecastConfig,
        split: str,
        normalization: ParquetNormalization | None = None,
        frame: pd.DataFrame | None = None,
    ) -> None:
        self.config = config
        self.split = split
        cfg = config.dataset
        if frame is None:
            if not cfg.parquet_file:
                raise ValueError("dataset.parquet_file is null/TODO; supply the real Parquet path")
            frame = pd.read_parquet(cfg.parquet_file)
        self.frame = frame.copy()
        required = {cfg.timestamp_column, cfg.target_column, *cfg.serial_columns}
        for value in cfg.irradiance_columns.values():
            if value:
                required.add(value)
        if config.images.enabled:
            required.update((config.images.paths_column, config.images.timestamps_column))
        missing = sorted(str(name) for name in required if name not in self.frame.columns)
        if missing:
            raise ValueError(f"Parquet is missing configured columns: {missing}")

        parsed = pd.to_datetime(self.frame[cfg.timestamp_column], errors="coerce")
        if parsed.isna().any():
            rows = np.flatnonzero(parsed.isna().to_numpy())[:5].tolist()
            raise ValueError(f"unparseable timestamps in rows {rows}")
        if parsed.duplicated().any():
            duplicate = parsed[parsed.duplicated(keep=False)].iloc[0]
            raise ValueError(f"duplicate timestamp is not allowed: {duplicate}")
        self.frame[cfg.timestamp_column] = parsed
        self.frame = self.frame.sort_values(cfg.timestamp_column).reset_index(drop=True)
        self.times = self.frame[cfg.timestamp_column]
        if config.site.timezone:
            if self.times.dt.tz is None:
                self.times = self.times.dt.tz_localize(config.site.timezone, ambiguous="raise", nonexistent="raise")
            else:
                self.times = self.times.dt.tz_convert(config.site.timezone)
            self.frame[cfg.timestamp_column] = self.times
        diffs = self.times.diff().dropna().dt.total_seconds().to_numpy() / 60.0
        expected = float(cfg.sampling_interval_minutes)
        bad = diffs[~np.isclose(diffs, expected, rtol=0.0, atol=1e-6)]
        if bad.size:
            actual = float(bad[0])
            raise ValueError(
                f"Parquet time granularity mismatch: expected {expected:g} minutes, actual {actual:g} minutes"
            )
        if cfg.forecast_step_minutes % cfg.sampling_interval_minutes:
            raise ValueError(
                "forecast_step_minutes must be an integer multiple of the actual sampling interval"
            )
        self.step_rows = cfg.forecast_step_minutes // cfg.sampling_interval_minutes
        self.horizon = int(cfg.forecast_steps)

        numeric = self.frame[cfg.serial_columns].apply(pd.to_numeric, errors="coerce")
        observed = numeric.notna().to_numpy()
        if cfg.causal_fill == "forward_fill":
            numeric = numeric.ffill()
        self.serial_raw = numeric.to_numpy(np.float32)
        self.serial_observed = observed

        development_bounds = config.splits.get("development")
        split_bounds = config.splits.get(split, development_bounds if split in {"train", "val"} else None)
        allowed_issue = _range_mask(self.times, split_bounds, config.site.timezone)
        targets = pd.to_numeric(self.frame[cfg.target_column], errors="coerce").to_numpy(np.float32)
        candidates: list[int] = []
        final_offset = self.horizon * self.step_rows
        for issue in np.flatnonzero(allowed_issue):
            history_start = issue - cfg.history_points + 1
            target_indices = issue + np.arange(1, self.horizon + 1) * self.step_rows
            if history_start < 0 or target_indices[-1] >= len(self.frame):
                continue
            if split_bounds and not allowed_issue[target_indices[-1]]:
                continue
            if not np.isfinite(targets[target_indices]).all():
                continue
            candidates.append(int(issue))
        if development_bounds and split in {"train", "val"}:
            validation_fraction = float(config.splits.get("validation_fraction", 0.15))
            if not 0.0 < validation_fraction < 1.0:
                raise ValueError("splits.validation_fraction must be in (0,1)")
            cut = int(np.floor(len(candidates) * (1.0 - validation_fraction)))
            candidates = candidates[:cut] if split == "train" else candidates[cut:]
        self.issue_indices = candidates
        self.targets = targets
        if normalization is None:
            if split != "train":
                raise ValueError("validation/test datasets require normalization fitted on training rows")
            train_rows = np.zeros(len(self.frame), dtype=bool)
            for issue in self.issue_indices:
                train_rows[issue - cfg.history_points + 1 : issue + 1] = True
            if not train_rows.any():
                raise ValueError("training split contains no valid issue_time windows")
            normalization = ParquetNormalization.fit(
                self.serial_raw[train_rows], self.serial_observed[train_rows]
            )
        self.normalization = normalization
        self.metadata = {
            "dataset": cfg.name,
            "serial_columns": list(cfg.serial_columns),
            "serial_input_dim": len(cfg.serial_columns),
            "physics_features": list(PHYSICS_FEATURES),
            "forecast_step_minutes": cfg.forecast_step_minutes,
            "forecast_horizon_minutes": cfg.forecast_horizon_minutes,
            "forecast_steps": self.horizon,
            "normalization": normalization.state_dict(),
            "physics_sources": self._physics_sources(),
        }

    def _physics_sources(self) -> dict[str, str]:
        cfg = self.config.dataset
        return {
            "solar_geometry": "derived",
            "pressure_pa": "derived_from_msl" if cfg.pressure_column and self.config.site.altitude_m is not None else "default",
            "pwv_cm": "observed_converted" if cfg.pwv_column else "default",
            "pwv_source_unit": cfg.pwv_unit or "not_configured",
            "aod700": "default",
            "precip": "observed" if cfg.precip_column else "default",
        }

    def __len__(self) -> int:
        return len(self.issue_indices)

    def _future_indices(self, issue: int) -> np.ndarray:
        return issue + np.arange(1, self.horizon + 1) * self.step_rows

    def _physics(self, issue: int, future: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.config.site.require_solar_geometry()
        cfg, defaults = self.config.dataset, self.config.physics_defaults
        geometry = solar_geometry(
            self.times.iloc[future],
            float(self.config.site.latitude),
            float(self.config.site.longitude),
            str(self.config.site.timezone),
        )
        mu0 = torch.from_numpy(geometry["mu0"].to_numpy(np.float32)).view(1, -1)
        dni_extra = torch.from_numpy(geometry["dni_extra"].to_numpy(np.float32)).view(1, -1)
        ghi_column = cfg.irradiance_columns.get("ghi")
        input_ghi = float(pd.to_numeric(self.frame.loc[issue, ghi_column], errors="coerce")) if ghi_column else 0.0
        if not np.isfinite(input_ghi):
            input_ghi = 0.0
        pressure = float(defaults.pressure_pa)
        if cfg.pressure_column and self.config.site.altitude_m is not None:
            raw_pressure = float(pd.to_numeric(self.frame.loc[issue, cfg.pressure_column], errors="coerce"))
            if np.isfinite(raw_pressure):
                if cfg.pressure_unit == "hpa":
                    raw_pressure *= 100.0
                elif cfg.pressure_unit != "pa":
                    raise ValueError(f"unsupported pressure_unit: {cfg.pressure_unit}")
                pressure = raw_pressure * np.exp(-float(self.config.site.altitude_m) / 8434.5)
        pwv = float(defaults.pwv_cm)
        if cfg.pwv_column:
            raw_pwv = float(pd.to_numeric(self.frame.loc[issue, cfg.pwv_column], errors="coerce"))
            if np.isfinite(raw_pwv):
                pwv = raw_pwv * float(cfg.pwv_to_cm)
        precip = float(defaults.precip)
        if cfg.precip_column:
            raw_precip = float(pd.to_numeric(self.frame.loc[issue, cfg.precip_column], errors="coerce"))
            if np.isfinite(raw_precip):
                precip = raw_precip
        wind = 0.0
        if cfg.wind_columns:
            components = pd.to_numeric(self.frame.loc[issue, cfg.wind_columns], errors="coerce").to_numpy(float)
            if np.isfinite(components).all():
                wind = float(np.linalg.norm(components)) if len(components) > 1 else float(components[0])
        inputs = {
            "mu0": mu0,
            "input_ghi": torch.full_like(mu0, input_ghi),
            "previous_ghi": torch.full_like(mu0, input_ghi),
            "pressure_pa": torch.full_like(mu0, pressure),
            "pwv_cm": torch.full_like(mu0, pwv),
            "aod700": torch.full_like(mu0, float(defaults.aod700)),
            "precip": torch.full_like(mu0, precip),
            "wind_speed": torch.full_like(mu0, wind),
            "dni_extra": dni_extra,
        }
        raw = Rest2FeatureBuilder()(inputs).squeeze(0)
        zenith = torch.from_numpy(geometry["apparent_zenith"].to_numpy(np.float32))
        return raw.clone(), raw, zenith

    def _images(self, issue: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config.images
        paths = _parse_list(self.frame.loc[issue, cfg.paths_column], str(cfg.paths_column))
        stamps = _parse_list(self.frame.loc[issue, cfg.timestamps_column], str(cfg.timestamps_column))
        if len(paths) != len(stamps):
            raise ValueError(f"image path/timestamp list length mismatch: {len(paths)} != {len(stamps)}")
        issue_time = _timestamp(self.times.iloc[issue], self.config.site.timezone)
        pairs: list[tuple[pd.Timestamp, str]] = []
        seen: set[pd.Timestamp] = set()
        for path, stamp in zip(paths, stamps):
            image_time = _timestamp(stamp, self.config.site.timezone)
            if image_time > issue_time and cfg.reject_future:
                raise ValueError(f"future image {image_time} is later than issue_time {issue_time}")
            if image_time < issue_time - pd.Timedelta(minutes=cfg.history_minutes):
                continue
            if image_time in seen:
                raise ValueError(f"duplicate image timestamp is not allowed: {image_time}")
            seen.add(image_time)
            if image_time <= issue_time:
                pairs.append((image_time, str(path)))
        pairs.sort(key=lambda item: item[0])
        if len(pairs) > cfg.max_frames:
            selection = np.linspace(0, len(pairs) - 1, cfg.max_frames).round().astype(int)
            pairs = [pairs[index] for index in selection]
        images = torch.zeros(cfg.max_frames, cfg.channels, cfg.size, cfg.size, dtype=torch.float32)
        mask = torch.zeros(cfg.max_frames, dtype=torch.long)
        offsets = torch.zeros(cfg.max_frames, dtype=torch.float32)
        root = Path(str(cfg.root))
        for slot, (stamp, relative) in enumerate(pairs):
            path = Path(relative)
            if path.is_absolute():
                raise ValueError(f"image path must be relative to images.root: {relative}")
            resolved = root / path
            if not resolved.is_file():
                continue
            mode = "RGB" if cfg.channels == 3 else "L"
            with Image.open(resolved) as handle:
                array = np.asarray(handle.convert(mode).resize((cfg.size, cfg.size)), dtype=np.float32) / 255.0
            if cfg.channels == 1:
                array = array[..., None]
            images[slot] = torch.from_numpy(array).permute(2, 0, 1)
            mask[slot] = 1
            offsets[slot] = float((stamp - issue_time).total_seconds() / 60.0)
        return images, mask, offsets

    def __getitem__(self, index: int) -> dict[str, Any]:
        issue = self.issue_indices[index]
        cfg = self.config.dataset
        history = np.arange(issue - cfg.history_points + 1, issue + 1)
        future = self._future_indices(issue)
        serial = (self.serial_raw[history] - self.normalization.serial_mean) / self.normalization.serial_std
        serial_mask = np.isfinite(self.serial_raw[history]).any(axis=1).astype(np.int64)
        serial = np.nan_to_num(serial, nan=0.0, posinf=0.0, neginf=0.0)
        physics, physics_raw, zenith = self._physics(issue, future)
        irr = np.zeros((self.horizon, 3), np.float32)
        irr_mask = np.zeros((self.horizon, 3), np.int64)
        for component, slot in (("ghi", 0), ("dni", 1), ("dhi", 2)):
            column = cfg.irradiance_columns.get(component)
            if column:
                values = pd.to_numeric(self.frame.loc[future, column], errors="coerce").to_numpy(np.float32)
                valid = np.isfinite(values)
                irr[valid, slot] = values[valid]
                irr_mask[valid, slot] = 1
        target = self.targets[future, None]
        issue_time = self.times.iloc[issue]
        target_times = [self.times.iloc[position] for position in future]
        sample: dict[str, Any] = {
            "serial": torch.from_numpy(serial).float(),
            "serial_valid_mask": torch.from_numpy(serial_mask),
            "physics": physics.float(),
            "physics_raw": physics_raw.float(),
            "physics_valid_mask": torch.ones(self.horizon, dtype=torch.long),
            "future_zenith": zenith.float(),
            "target": torch.from_numpy(target).float(),
            "target_valid_mask": torch.ones(self.horizon, dtype=torch.long),
            "irradiance_target": torch.from_numpy(irr),
            "irradiance_target_mask": torch.from_numpy(irr_mask),
            "issue_time": issue_time.isoformat(),
            "target_times": [stamp.isoformat() for stamp in target_times],
            "current_power": torch.tensor([self.targets[issue]], dtype=torch.float32),
        }
        if self.config.images.enabled:
            sample["images"], sample["image_valid_mask"], sample["image_time_offsets"] = self._images(issue)
        return sample


def collate_parquet_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    if any(sample.keys() != samples[0].keys() for sample in samples):
        raise ValueError("all Parquet samples must expose the same batch contract")
    result: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        result[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return result


def build_parquet_datasets(
    config: ParquetForecastConfig, frame: pd.DataFrame | None = None
) -> dict[str, ConfigurableParquetDataset]:
    train = ConfigurableParquetDataset(config, "train", frame=frame)
    datasets = {"train": train}
    for split in ("val", "test"):
        if split in config.splits:
            datasets[split] = ConfigurableParquetDataset(
                config, split, normalization=train.normalization, frame=frame
            )
    return datasets
