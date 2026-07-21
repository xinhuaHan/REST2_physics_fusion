from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe import load_config
from pv_physics_moe.physics.rest2 import PHYSICS_FEATURES, rest2_torch_clear_sky
from pv_physics_moe.physics.solar_geometry import solar_geometry


SERIAL_FEATURES = (
    "power_file_ghi", "observed_power", "irradiance_file_ghi", "observed_dni",
    "observed_dhi", "weather_ghi", "temperature", "wind_speed",
    "wind_direction_sin", "wind_direction_cos", "precipitation", "pwat", "sdwe",
)
STATION_PHYSICS_FEATURES = list(PHYSICS_FEATURES)
STATION_PHYSICS_FEATURES[17] = "clear_ghi_target_lead"
STATION_PHYSICS_FEATURES[18] = "forecast_4h_ghi_target"
STATION_PHYSICS_FEATURES[19] = "forecast_1d_ghi_target"
STATION_PHYSICS_FEATURES[23] = "observed_weather_adjusted_clear_ghi"
STATION_PHYSICS_FEATURES[24] = "forecast_4h_weather_adjusted_clear_ghi"
STATION_PHYSICS_FEATURES[25] = "forecast_1d_weather_adjusted_clear_ghi"


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"station input does not exist: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError(f"unsupported station table type: {path.suffix}")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"cannot decode CSV {path}: {last_error}")


def require_columns(frame: pd.DataFrame, required: list[str], role: str) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{role} is missing columns {missing}; actual={list(frame.columns)}")


def prepare_regular_table(frame: pd.DataFrame, timestamp: str, required: list[str], role: str) -> pd.DataFrame:
    require_columns(frame, [timestamp, *required], role)
    output = frame[[timestamp, *required]].copy()
    output[timestamp] = pd.to_datetime(output[timestamp], errors="raise")
    if output[timestamp].duplicated().any():
        examples = output.loc[output[timestamp].duplicated(False), timestamp].head(5).astype(str).tolist()
        raise ValueError(f"{role} contains duplicate timestamps, examples={examples}")
    for column in required:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.sort_values(timestamp).reset_index(drop=True)


def forecast_issue_time(
    frame: pd.DataFrame,
    timestamp: str,
    date_column: str,
    time_column: str,
    interval_column: str,
) -> pd.Series:
    valid = pd.to_datetime(frame[timestamp], errors="raise")
    fallback = valid - pd.to_timedelta(pd.to_numeric(frame[interval_column], errors="coerce"), unit="m")
    if date_column not in frame or time_column not in frame:
        return fallback
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    parts = frame[time_column].astype(str).str.extract(r"(?P<hour>\d{1,2})[_:](?P<minute>\d{2})")
    minutes = pd.to_numeric(parts["hour"], errors="coerce") * 60 + pd.to_numeric(parts["minute"], errors="coerce")
    parsed = dates + pd.to_timedelta(minutes, unit="m")
    return parsed.fillna(fallback)


def align_forecast(
    frame: pd.DataFrame,
    base_times: pd.DatetimeIndex,
    *,
    timestamp: str,
    feature_columns: list[str],
    date_column: str,
    time_column: str,
    interval_column: str,
    lead_minutes: int,
    prefix: str,
) -> pd.DataFrame:
    """Align forecasts available at the forecast origin without future-row interpolation."""
    require_columns(frame, [timestamp, *feature_columns, interval_column], prefix)
    data = frame.copy()
    data[timestamp] = pd.to_datetime(data[timestamp], errors="raise")
    for column in feature_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["_issue_time"] = forecast_issue_time(data, timestamp, date_column, time_column, interval_column)
    origin_time = data[timestamp] - pd.to_timedelta(lead_minutes, unit="m")
    data = data.loc[data["_issue_time"] <= origin_time].copy()
    names = [f"{prefix}_{name}" for name in feature_columns]
    if data.empty:
        empty = pd.DataFrame({timestamp: base_times})
        for name in names:
            empty[name] = np.nan
        return empty
    data = data.sort_values([timestamp, "_issue_time"]).drop_duplicates(timestamp, keep="last")
    values = data.set_index(timestamp)[feature_columns].sort_index()
    # Forward fill can only carry an older forecast forward, so it cannot import
    # information issued after the current forecast origin.
    values = values.reindex(values.index.union(base_times).sort_values()).ffill().reindex(base_times)
    values.columns = names
    values.index.name = timestamp
    return values.reset_index()


def standard_pressure_pa(altitude_m: float) -> float:
    """ICAO tropospheric standard-atmosphere pressure for the configured altitude."""
    base = 1.0 - 2.25577e-5 * float(altitude_m)
    if base <= 0.0:
        raise ValueError("altitude is outside the standard-atmosphere formula range")
    return 101325.0 * base ** 5.25588


def clear_sky(
    mu0: np.ndarray,
    pressure: np.ndarray,
    pwv: np.ndarray,
    aod: float,
    dni_extra: np.ndarray,
) -> tuple[np.ndarray, ...]:
    outputs = [np.empty(len(mu0), dtype=np.float32) for _ in range(4)]
    for start in range(0, len(mu0), 100_000):
        end = min(start + 100_000, len(mu0))
        with torch.no_grad():
            result = rest2_torch_clear_sky(
                mu0=torch.from_numpy(mu0[start:end].astype(np.float32)),
                pressure_pa=torch.from_numpy(pressure[start:end].astype(np.float32)),
                pwv_cm=torch.from_numpy(pwv[start:end].astype(np.float32)),
                aod700=torch.full((end - start,), float(aod)),
                dni_extra=torch.from_numpy(dni_extra[start:end].astype(np.float32)),
            )
        for target, name in zip(outputs, ("ghi_clear_target", "dni_clear_target", "dhi_clear_target", "t_direct")):
            target[start:end] = result[name].numpy()
    return tuple(outputs)


def shift_future(values: np.ndarray, offset: int, fill: float = np.nan) -> np.ndarray:
    output = np.full(values.shape, fill, dtype=np.float32)
    if 0 < offset < len(values):
        output[:-offset] = values[offset:]
    return output


def rolling_all(mask: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(mask.astype(np.int64))))
    output = np.zeros(len(mask), dtype=bool)
    output[window - 1:] = cumulative[window:] - cumulative[:-window] == window
    return output


def normalize_serial(values: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = values[train_rows].astype(np.float64)
    mean, std = selected.mean(0), selected.std(0)
    std[std < 1e-6] = 1.0
    return ((values - mean) / std).astype(np.float32), mean, std


def normalize_physics(values: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = values[train_rows].astype(np.float64)
    mean, std = selected.mean((0, 1)), selected.std((0, 1))
    std[std < 1e-6] = 1.0
    return ((values - mean[None, None, :]) / std[None, None, :]).astype(np.float32), mean, std


def attenuation(pwv: np.ndarray, precip: np.ndarray, wind: np.ndarray, sky_last: np.ndarray) -> np.ndarray:
    value = 1.0 - 0.025 * np.maximum(pwv - 1.5, 0.0) - 0.18 * (1.0 - np.exp(-np.maximum(precip, 0.0)))
    value = value - 0.35 * np.clip(1.0 - sky_last, 0.0, 1.0) + 0.03 * np.tanh(np.maximum(wind, 0.0) / 6.0)
    return np.clip(value, 0.05, 1.2).astype(np.float32)


def coalesce(*arrays: np.ndarray) -> np.ndarray:
    result = arrays[-1].astype(np.float32).copy()
    for values in reversed(arrays[:-1]):
        result = np.where(np.isfinite(values), values, result)
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe five-horizon station data from five tables.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "server_station.yaml"))
    parser.add_argument("--data-root", help="Override data.root_dir")
    parser.add_argument("--output", help="Override data.processed_dir")
    args = parser.parse_args()
    config = load_config(args.config)
    data_cfg = config.data
    data_root = resolve_path(ROOT, args.data_root or data_cfg.root_dir)
    output = resolve_path(ROOT, args.output or data_cfg.processed_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "timeline").mkdir(exist_ok=True)
    (output / "splits").mkdir(exist_ok=True)

    paths = {
        "power_ghi": data_root / data_cfg.power_ghi_file,
        "irradiance": data_root / data_cfg.irradiance_file,
        "forecast_4h": data_root / data_cfg.forecast_4h_file,
        "forecast_1d": data_root / data_cfg.forecast_1d_file,
        "weather": data_root / data_cfg.weather_file,
    }
    print("reading five station tables", flush=True)
    tables = {name: read_table(path) for name, path in paths.items()}
    timestamp = data_cfg.timestamp_column
    power = prepare_regular_table(tables["power_ghi"], timestamp, [data_cfg.power_column, data_cfg.power_ghi_column], "power_ghi")
    irradiance = prepare_regular_table(tables["irradiance"], timestamp, [data_cfg.irradiance_ghi_column, data_cfg.irradiance_dni_column, data_cfg.irradiance_dhi_column], "irradiance")
    weather = prepare_regular_table(tables["weather"], timestamp, data_cfg.weather_columns, "weather")
    base_times = pd.DatetimeIndex(power[timestamp])

    master = power.rename(columns={data_cfg.power_column: "power", data_cfg.power_ghi_column: "power_ghi"})
    master = master.merge(
        irradiance.rename(columns={data_cfg.irradiance_ghi_column: "irr_ghi", data_cfg.irradiance_dni_column: "irr_dni", data_cfg.irradiance_dhi_column: "irr_dhi"}),
        on=timestamp, how="left", validate="one_to_one",
    )
    master = master.merge(
        weather.rename(columns={name: f"weather_{name}" for name in data_cfg.weather_columns}),
        on=timestamp, how="left", validate="one_to_one",
    )
    n = len(master)
    offsets = [lead // data_cfg.sample_minutes for lead in data_cfg.forecast_lead_minutes]
    max_offset = max(offsets)
    if n <= data_cfg.history_length + max_offset:
        raise ValueError("station timeline is too short for history_length and the 1-day horizon")

    wd = np.deg2rad(master["weather_WD"].to_numpy(dtype=np.float64))
    serial_raw = np.column_stack((
        master["power_ghi"], master["power"], master["irr_ghi"], master["irr_dni"], master["irr_dhi"],
        master["weather_GHI"], master["weather_TEMP"], master["weather_WS"], np.sin(wd), np.cos(wd),
        master["weather_PREC"], master["weather_PWAT"], master["weather_SDWE"],
    )).astype(np.float32)

    print("computing five target-time REST2 physics vectors", flush=True)
    geometry = solar_geometry(master[timestamp], data_cfg.latitude, data_cfg.longitude, data_cfg.timezone)
    mu0 = geometry["mu0"].to_numpy(dtype=np.float32)
    zenith = geometry["apparent_zenith"].to_numpy(dtype=np.float32)
    dni_extra = geometry["dni_extra"].to_numpy(dtype=np.float32)
    pressure_pa = float(data_cfg.default_pressure_pa) if data_cfg.default_pressure_pa is not None else standard_pressure_pa(data_cfg.altitude_m)
    pressure = np.full(n, pressure_pa, dtype=np.float32)
    observed_pwv = np.clip(master["weather_PWAT"].to_numpy(dtype=np.float32), 0.05, 10.0)
    current_clear_ghi, _, _, _ = clear_sky(mu0, pressure, observed_pwv, data_cfg.default_aod700, dni_extra)
    current_ghi = master["power_ghi"].to_numpy(dtype=np.float32)
    previous_ghi = np.roll(current_ghi, 1)
    previous_ghi[0] = current_ghi[0]
    previous_clear = np.roll(current_clear_ghi, 1)
    previous_clear[0] = current_clear_ghi[0]
    sky_last = np.where(previous_clear > 20.0, previous_ghi / np.maximum(previous_clear, 1.0), 0.0).clip(0.0, 2.0).astype(np.float32)
    sky_current = np.where(current_clear_ghi > 20.0, current_ghi / np.maximum(current_clear_ghi, 1.0), 0.0).clip(0.0, 2.0).astype(np.float32)
    current_precip = master["weather_PREC"].to_numpy(dtype=np.float32)
    current_wind = master["weather_WS"].to_numpy(dtype=np.float32)
    observed_att = attenuation(observed_pwv, current_precip, current_wind, sky_last)

    physics_by_horizon: list[np.ndarray] = []
    power_by_horizon: list[np.ndarray] = []
    irradiance_by_horizon: list[np.ndarray] = []
    zenith_by_horizon: list[np.ndarray] = []
    power_raw_all = master["power"].to_numpy(dtype=np.float32)
    irr_ghi_all = master["irr_ghi"].to_numpy(dtype=np.float32)
    irr_dni_all = master["irr_dni"].to_numpy(dtype=np.float32)
    irr_dhi_all = master["irr_dhi"].to_numpy(dtype=np.float32)

    for lead, offset in zip(data_cfg.forecast_lead_minutes, offsets):
        fc4 = align_forecast(
            tables["forecast_4h"], base_times, timestamp=timestamp,
            feature_columns=data_cfg.forecast_columns, date_column=data_cfg.forecast_issue_date_column,
            time_column=data_cfg.forecast_issue_time_column, interval_column=data_cfg.forecast_interval_column,
            lead_minutes=lead, prefix="fc4",
        )
        fc1 = align_forecast(
            tables["forecast_1d"], base_times, timestamp=timestamp,
            feature_columns=data_cfg.forecast_columns, date_column=data_cfg.forecast_issue_date_column,
            time_column=data_cfg.forecast_issue_time_column, interval_column=data_cfg.forecast_interval_column,
            lead_minutes=lead, prefix="fc1d",
        )

        def forecast_target(frame: pd.DataFrame, prefix: str, name: str) -> np.ndarray:
            return shift_future(frame[f"{prefix}_{name}"].to_numpy(dtype=np.float32), offset)

        fc4_pwv, fc1_pwv = forecast_target(fc4, "fc4", "PWAT"), forecast_target(fc1, "fc1d", "PWAT")
        fc4_prec, fc1_prec = forecast_target(fc4, "fc4", "PREC"), forecast_target(fc1, "fc1d", "PREC")
        fc4_ws, fc1_ws = forecast_target(fc4, "fc4", "WS"), forecast_target(fc1, "fc1d", "WS")
        fc4_ghi_raw, fc1_ghi_raw = forecast_target(fc4, "fc4", "GHI"), forecast_target(fc1, "fc1d", "GHI")
        if lead <= 240:
            primary_pwv, secondary_pwv = fc4_pwv, fc1_pwv
            primary_prec, secondary_prec = fc4_prec, fc1_prec
            primary_ws, secondary_ws = fc4_ws, fc1_ws
            primary_ghi, secondary_ghi = fc4_ghi_raw, fc1_ghi_raw
        else:
            primary_pwv, secondary_pwv = fc1_pwv, fc4_pwv
            primary_prec, secondary_prec = fc1_prec, fc4_prec
            primary_ws, secondary_ws = fc1_ws, fc4_ws
            primary_ghi, secondary_ghi = fc1_ghi_raw, fc4_ghi_raw

        target_pwv = np.clip(coalesce(primary_pwv, secondary_pwv, observed_pwv), 0.05, 10.0)
        target_prec = coalesce(primary_prec, secondary_prec, current_precip)
        target_ws = coalesce(primary_ws, secondary_ws, current_wind)
        target_mu0 = shift_future(mu0, offset)
        target_zenith = shift_future(zenith, offset, 180.0)
        target_extra = shift_future(dni_extra, offset)
        clear_ghi, clear_dni, clear_dhi, direct_t = clear_sky(
            target_mu0, pressure, target_pwv, data_cfg.default_aod700, target_extra,
        )
        primary_att = attenuation(target_pwv, target_prec, target_ws, sky_last)
        fc4_att = attenuation(np.clip(coalesce(fc4_pwv, target_pwv), 0.05, 10.0), coalesce(fc4_prec, target_prec), coalesce(fc4_ws, target_ws), sky_last)
        fc1_att = attenuation(np.clip(coalesce(fc1_pwv, target_pwv), 0.05, 10.0), coalesce(fc1_prec, target_prec), coalesce(fc1_ws, target_ws), sky_last)
        adjusted_observed = clear_ghi * observed_att
        adjusted_fc4, adjusted_fc1 = clear_ghi * fc4_att, clear_ghi * fc1_att
        fc4_ghi = coalesce(fc4_ghi_raw, primary_ghi, secondary_ghi, current_ghi)
        fc1_ghi = coalesce(fc1_ghi_raw, primary_ghi, secondary_ghi, current_ghi)

        physics_h = np.column_stack((
            clear_ghi, clear_dni, clear_dhi, target_mu0, target_zenith, target_extra,
            pressure, target_pwv, np.full(n, data_cfg.default_aod700, np.float32), direct_t,
            sky_last, sky_last * clear_ghi, sky_current, primary_att, clear_ghi * primary_att,
            previous_ghi - previous_clear, np.maximum(clear_ghi - current_ghi, 0.0),
            clear_ghi, fc4_ghi, fc1_ghi, target_mu0, target_mu0, target_mu0,
            adjusted_observed, adjusted_fc4, adjusted_fc1,
        )).astype(np.float32)
        physics_by_horizon.append(physics_h)
        power_by_horizon.append(shift_future(power_raw_all, offset))
        irradiance_by_horizon.append(np.column_stack((
            shift_future(irr_ghi_all, offset),
            shift_future(irr_dni_all, offset),
            shift_future(irr_dhi_all, offset),
        )).astype(np.float32))
        zenith_by_horizon.append(target_zenith)

    physics_raw = np.stack(physics_by_horizon, axis=1)
    target_power_raw = np.stack(power_by_horizon, axis=1)
    clipped_count = int(np.sum(np.isfinite(target_power_raw) & (target_power_raw < data_cfg.power_target_floor)))
    target_power = np.maximum(target_power_raw, data_cfg.power_target_floor)[..., None].astype(np.float32)
    target_irradiance = np.stack(irradiance_by_horizon, axis=1).astype(np.float32)
    future_zenith = np.stack(zenith_by_horizon, axis=1).astype(np.float32)

    seconds = master[timestamp].to_numpy(dtype="datetime64[s]").astype(np.int64)
    contiguous = np.ones(n, dtype=bool)
    contiguous[1:] = np.diff(seconds) == data_cfg.sample_minutes * 60
    segment = np.cumsum(~contiguous)
    serial_finite = np.isfinite(serial_raw).all(1)
    history_finite = rolling_all(serial_finite, data_cfg.history_length)
    endpoints = np.arange(n, dtype=np.int64)
    target_indices = np.minimum(endpoints + max_offset, n - 1)
    valid = endpoints >= data_cfg.history_length - 1
    valid &= endpoints + max_offset < n
    valid &= history_finite
    valid &= np.isfinite(physics_raw).all((1, 2))
    valid &= np.isfinite(target_power).all((1, 2)) & np.isfinite(target_irradiance).all((1, 2))
    valid &= segment[np.maximum(endpoints - data_cfg.history_length + 1, 0)] == segment[target_indices]
    sample_indices = endpoints[valid].astype(np.int32)
    if len(sample_indices) < 3:
        raise RuntimeError(f"only {len(sample_indices)} valid samples after five-table/five-horizon alignment")
    train_end = max(1, int(len(sample_indices) * data_cfg.train_fraction))
    val_end = max(train_end + 1, int(len(sample_indices) * (data_cfg.train_fraction + data_cfg.val_fraction)))
    val_end = min(val_end, len(sample_indices) - 1)
    splits = {"train": sample_indices[:train_end], "val": sample_indices[train_end:val_end], "test": sample_indices[val_end:]}

    train_cutoff = int(splits["train"][-1])
    train_serial_rows = np.flatnonzero(serial_finite & (endpoints <= train_cutoff))
    serial, serial_mean, serial_std = normalize_serial(serial_raw, train_serial_rows)
    physics, physics_mean, physics_std = normalize_physics(physics_raw, splits["train"])
    power_scale = float(max(1.0, np.std(target_power[splits["train"]], dtype=np.float64)))
    irradiance_scale = np.std(target_irradiance[splits["train"]].astype(np.float64), axis=(0, 1))
    irradiance_scale[irradiance_scale < 1.0] = 1.0

    timeline = output / "timeline"
    np.save(timeline / "serial.npy", serial)
    np.save(timeline / "physics.npy", physics)
    np.save(timeline / "physics_raw.npy", physics_raw)
    np.save(timeline / "target_power.npy", target_power)
    np.save(timeline / "target_irradiance.npy", target_irradiance)
    np.save(timeline / "future_zenith.npy", future_zenith)
    np.save(timeline / "timestamps.npy", master[timestamp].to_numpy(dtype="datetime64[s]"))
    for name, indices in splits.items():
        np.save(output / "splits" / f"{name}_end_indices.npy", indices)

    normalizers = {
        "serial": {"features": list(SERIAL_FEATURES), "mean": serial_mean.tolist(), "std": serial_std.tolist()},
        "physics": {"features": STATION_PHYSICS_FEATURES, "mean": physics_mean.tolist(), "std": physics_std.tolist()},
        "target_power": {"scale": power_scale},
        "target_irradiance": {"features": ["GHI", "DNI", "DHI"], "scale": irradiance_scale.tolist(), "units": "W/m2"},
    }
    with (output / "normalization.json").open("w", encoding="utf-8") as handle:
        json.dump(normalizers, handle, ensure_ascii=False, indent=2)
    metadata = {
        "source_files": {name: str(path) for name, path in paths.items()},
        "rows": n,
        "history_length": data_cfg.history_length,
        "forecast_lead_minutes": data_cfg.forecast_lead_minutes,
        "forecast_horizon": len(data_cfg.forecast_lead_minutes),
        "sample_minutes": data_cfg.sample_minutes,
        "station": {
            "longitude": data_cfg.longitude,
            "latitude": data_cfg.latitude,
            "altitude_m": data_cfg.altitude_m,
            "timezone": data_cfg.timezone,
            "pressure_pa": pressure_pa,
            "pressure_source": "configured" if data_cfg.default_pressure_pa is not None else "ICAO standard atmosphere from altitude",
        },
        "serial_features": list(SERIAL_FEATURES),
        "physics_features": STATION_PHYSICS_FEATURES,
        "modalities": ["serial", "physics"],
        "target_mode": "power",
        "auxiliary_target": ["GHI", "DNI", "DHI"],
        "pwat_unit": "cm (same REST2/Folsom convention)",
        "negative_power_targets_clipped": clipped_count,
        "negative_power_floor": data_cfg.power_target_floor,
        "split_samples": {name: int(len(indices)) for name, indices in splits.items()},
        "forecast_leakage_rule": "for each horizon: issue_time <= target_valid_time - horizon (= forecast origin)",
        "forecast_source_rule": "4h source is primary through 240 min; 1d source is primary at 1440 min",
        "physics_closure": "GHI = cos(zenith) * DNI + DHI",
        "tensor_shapes": {
            "serial": ["timeline", data_cfg.history_length, len(SERIAL_FEATURES)],
            "physics": ["timeline", len(data_cfg.forecast_lead_minutes), len(STATION_PHYSICS_FEATURES)],
            "target_power": ["timeline", len(data_cfg.forecast_lead_minutes), 1],
            "target_irradiance": ["timeline", len(data_cfg.forecast_lead_minutes), 3],
        },
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
