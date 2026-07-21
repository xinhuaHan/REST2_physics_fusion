from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe import load_config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_experiment(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    experiment = raw.get("experiment", {})
    return {
        "enabled": bool(experiment.get("fixed_year_split", True)),
        "training_years": [int(year) for year in experiment.get("training_years", [2024])],
        "holdout_years": [int(year) for year in experiment.get("holdout_years", [2025])],
        "train_fraction": float(experiment.get("train_fraction_within_training_years", 0.85)),
    }


def time_range(indices: np.ndarray, timestamps: pd.DatetimeIndex, history: int, max_offset: int) -> dict | None:
    if len(indices) == 0:
        return None
    first, last = int(indices[0]), int(indices[-1])
    return {
        "samples": int(len(indices)),
        "history_start": str(timestamps[first - history + 1]),
        "first_origin": str(timestamps[first]),
        "last_origin": str(timestamps[last]),
        "last_target": str(timestamps[last + max_offset]),
    }


def normalize_serial(serial_raw: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = serial_raw[rows].astype(np.float64)
    mean, std = selected.mean(0), selected.std(0)
    std[std < 1e-6] = 1.0
    return ((serial_raw - mean) / std).astype(np.float32), mean, std


def normalize_physics(physics_raw: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = physics_raw[indices].astype(np.float64)
    mean, std = selected.mean((0, 1)), selected.std((0, 1))
    std[std < 1e-6] = 1.0
    normalized = (physics_raw - mean[None, None, :]) / std[None, None, :]
    return normalized.astype(np.float32), mean, std


def apply_fixed_year_split(output: Path, experiment: dict) -> None:
    if not experiment["enabled"]:
        print("fixed-year split disabled; keeping the generic chronological split", flush=True)
        return
    training_years = experiment["training_years"]
    holdout_years = experiment["holdout_years"]
    train_fraction = experiment["train_fraction"]
    if not training_years:
        raise ValueError("experiment.training_years must not be empty")
    if set(training_years) & set(holdout_years):
        raise ValueError("experiment.training_years and holdout_years must not overlap")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("experiment.train_fraction_within_training_years must be in (0,1)")

    with (output / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (output / "normalization.json").open("r", encoding="utf-8") as handle:
        normalizers = json.load(handle)
    timeline = output / "timeline"
    split_dir = output / "splits"
    timestamps = pd.DatetimeIndex(np.load(timeline / "timestamps.npy"))
    history = int(metadata["history_length"])
    max_offset = max(int(lead) // int(metadata["sample_minutes"]) for lead in metadata["forecast_lead_minutes"])

    previous_splits = [np.load(split_dir / f"{name}_end_indices.npy") for name in ("train", "val", "test")]
    all_valid = np.unique(np.concatenate(previous_splits)).astype(np.int32)
    years = timestamps.year.to_numpy()
    history_start = all_valid - history + 1
    max_target = all_valid + max_offset
    training_year_mask = (
        np.isin(years[history_start], training_years)
        & np.isin(years[all_valid], training_years)
        & np.isin(years[max_target], training_years)
    )
    development = all_valid[training_year_mask]
    if len(development) < 3:
        raise RuntimeError(
            f"only {len(development)} valid samples are fully contained in training_years={training_years}"
        )

    boundary = min(len(development) - 1, max(1, int(len(development) * train_fraction)))
    validation = development[boundary:]
    first_validation_origin = int(validation[0])
    # Purge training origins whose longest-horizon label reaches the validation period.
    training = development[:boundary]
    training = training[training + max_offset < first_validation_origin]
    if len(training) == 0 or len(validation) == 0:
        raise RuntimeError("fixed-year split did not leave non-empty train and validation sets")

    if holdout_years:
        holdout_mask = (
            np.isin(years[history_start], holdout_years)
            & np.isin(years[all_valid], holdout_years)
            & np.isin(years[max_target], holdout_years)
        )
        holdout = all_valid[holdout_mask]
    else:
        holdout = np.empty(0, dtype=np.int32)
    splits = {
        "train": training.astype(np.int32),
        "val": validation.astype(np.int32),
        "test": holdout.astype(np.int32),
    }

    # Recover the raw serial values from the generic preprocessing normalization,
    # then fit every normalization statistic again using the fixed 2024 train split only.
    old_serial = np.load(timeline / "serial.npy").astype(np.float64)
    old_mean = np.asarray(normalizers["serial"]["mean"], dtype=np.float64)
    old_std = np.asarray(normalizers["serial"]["std"], dtype=np.float64)
    serial_raw = old_serial * old_std[None, :] + old_mean[None, :]
    physics_raw = np.load(timeline / "physics_raw.npy")
    target_power = np.load(timeline / "target_power.npy")
    target_irradiance = np.load(timeline / "target_irradiance.npy")

    coverage = np.zeros(len(timestamps) + 1, dtype=np.int64)
    starts = training.astype(np.int64) - history + 1
    ends = training.astype(np.int64) + 1
    np.add.at(coverage, starts, 1)
    np.add.at(coverage, ends, -1)
    train_serial_rows = np.flatnonzero(np.cumsum(coverage[:-1]) > 0)
    serial, serial_mean, serial_std = normalize_serial(serial_raw, train_serial_rows)
    physics, physics_mean, physics_std = normalize_physics(physics_raw, training)
    power_scale = float(max(1.0, np.std(target_power[training], dtype=np.float64)))
    irradiance_scale = np.std(target_irradiance[training].astype(np.float64), axis=(0, 1))
    irradiance_scale[irradiance_scale < 1.0] = 1.0

    np.save(timeline / "serial.npy", serial)
    np.save(timeline / "physics.npy", physics)
    for name, indices in splits.items():
        np.save(split_dir / f"{name}_end_indices.npy", indices)

    normalizers["serial"].update({
        "mean": serial_mean.tolist(), "std": serial_std.tolist(),
        "fit_split": "train", "fit_years": training_years,
    })
    normalizers["physics"].update({
        "mean": physics_mean.tolist(), "std": physics_std.tolist(),
        "fit_split": "train", "fit_years": training_years,
    })
    normalizers["target_power"].update({
        "scale": power_scale, "fit_split": "train", "fit_years": training_years,
    })
    normalizers["target_irradiance"].update({
        "scale": irradiance_scale.tolist(), "fit_split": "train", "fit_years": training_years,
    })
    with (output / "normalization.json").open("w", encoding="utf-8") as handle:
        json.dump(normalizers, handle, ensure_ascii=False, indent=2)

    clipped_mask = np.isfinite(target_power) & (target_power <= float(metadata["negative_power_floor"]))
    metadata.update({
        "split_policy": "fixed calendar years; 2024 train/validation, 2025 locked client holdout",
        "training_years": training_years,
        "holdout_years": holdout_years,
        "train_fraction_within_training_years": train_fraction,
        "validation_fraction_within_training_years": 1.0 - train_fraction,
        "train_validation_embargo_minutes": int(metadata["forecast_lead_minutes"][-1]),
        "normalization_fit_split": "train",
        "normalization_fit_years": training_years,
        "client_holdout_locked": True,
        "split_samples": {name: int(len(indices)) for name, indices in splits.items()},
        "split_time_ranges": {
            name: time_range(indices, timestamps, history, max_offset) for name, indices in splits.items()
        },
        "zero_or_clipped_power_values_by_split": {
            name: int(clipped_mask[indices].sum()) for name, indices in splits.items()
        },
    })
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps({
        "fixed_year_split": True,
        "training_years": training_years,
        "holdout_years": holdout_years,
        "split_samples": metadata["split_samples"],
        "split_time_ranges": metadata["split_time_ranges"],
        "normalization_fit_years": training_years,
        "embargo_minutes": metadata["train_validation_embargo_minutes"],
    }, ensure_ascii=False, indent=2), flush=True)
    if len(holdout) == 0:
        print(
            f"WARNING: no complete holdout samples were found for years={holdout_years}; "
            "training is still valid, but test evaluation requires the held-out files.",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare five-horizon station data and enforce the 2024-train/2025-holdout experiment."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "server_station.yaml"))
    parser.add_argument("--data-root", help="Override data.root_dir")
    parser.add_argument("--output", help="Override data.processed_dir")
    parser.add_argument(
        "--resplit-only", action="store_true",
        help="Skip base preprocessing and only rebuild year splits/normalization in an existing processed directory",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output = resolve_path(args.output or config.data.processed_dir)
    if not args.resplit_only:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "prepare_station_dataset_all_years.py"),
            "--config", str(config_path),
        ]
        if args.data_root:
            command.extend(["--data-root", args.data_root])
        if args.output:
            command.extend(["--output", args.output])
        subprocess.run(command, check=True)
    apply_fixed_year_split(output, load_experiment(config_path))


if __name__ == "__main__":
    main()
