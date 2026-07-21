from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe.physics.rest2 import PHYSICS_FEATURES, rest2_torch_clear_sky
from pv_physics_moe.physics.solar_geometry import solar_geometry


SERIAL_FEATURES = (
    "ghi", "dni", "dhi", "air_temp", "relhum", "pressure_pa", "windsp",
    "winddir_sin", "winddir_cos", "max_windsp", "precipitation",
)


def parse_image_time(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.stem, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def scan_images(dataset_root: Path) -> tuple[np.ndarray, list[str]]:
    records: list[tuple[int, str]] = []
    for folder, _, filenames in os.walk(dataset_root):
        base = Path(folder)
        for filename in filenames:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = base / filename
            timestamp = parse_image_time(path)
            if timestamp is not None:
                seconds = np.datetime64(timestamp, "s").astype(np.int64)
                records.append((int(seconds), path.relative_to(dataset_root).as_posix()))
    if not records:
        raise RuntimeError(f"no timestamped images found below {dataset_root}")
    records.sort(key=lambda item: item[0])
    return np.asarray([item[0] for item in records], dtype=np.int64), [item[1] for item in records]


def nearest_image_ids(row_seconds: np.ndarray, image_seconds: np.ndarray, tolerance: int) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(image_seconds, row_seconds, side="left")
    right_clip = np.clip(right, 0, len(image_seconds) - 1)
    left_clip = np.clip(right - 1, 0, len(image_seconds) - 1)
    right_delta = np.abs(image_seconds[right_clip] - row_seconds)
    left_delta = np.abs(image_seconds[left_clip] - row_seconds)
    choose_left = left_delta <= right_delta
    ids = np.where(choose_left, left_clip, right_clip).astype(np.int32)
    deltas = np.where(choose_left, left_delta, right_delta).astype(np.int16)
    ids[deltas > tolerance] = -1
    return ids, deltas


def estimate_pwv_cm(temp_c: np.ndarray, relative_humidity: np.ndarray) -> np.ndarray:
    saturation_hpa = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    vapor_pressure_hpa = saturation_hpa * np.clip(relative_humidity, 0.0, 100.0) / 100.0
    return np.clip(0.12 * vapor_pressure_hpa + 0.10, 0.05, 8.0)


def clear_sky_arrays(mu0: np.ndarray, pressure_pa: np.ndarray, pwv_cm: np.ndarray, dni_extra: np.ndarray, chunk: int = 100_000) -> tuple[np.ndarray, ...]:
    outputs = [np.empty(len(mu0), dtype=np.float32) for _ in range(4)]
    for start in range(0, len(mu0), chunk):
        end = min(start + chunk, len(mu0))
        with torch.no_grad():
            clear = rest2_torch_clear_sky(
                mu0=torch.from_numpy(mu0[start:end].astype(np.float32)),
                pressure_pa=torch.from_numpy(pressure_pa[start:end].astype(np.float32)),
                pwv_cm=torch.from_numpy(pwv_cm[start:end].astype(np.float32)),
                aod700=torch.full((end - start,), 0.08),
                dni_extra=torch.from_numpy(dni_extra[start:end].astype(np.float32)),
            )
        for output, name in zip(outputs, ("ghi_clear_target", "dni_clear_target", "dhi_clear_target", "t_direct")):
            output[start:end] = clear[name].numpy()
    return tuple(outputs)


def shifted(values: np.ndarray, offset: int, fill: float = 0.0) -> np.ndarray:
    result = np.full_like(values, fill)
    if offset:
        result[:-offset] = values[offset:]
    else:
        result[:] = values
    return result


def rolling_all(mask: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(mask.astype(np.int64))))
    result = np.zeros(len(mask), dtype=bool)
    result[window - 1 :] = cumulative[window:] - cumulative[:-window] == window
    return result


def normalize(values: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = values[train_rows].astype(np.float64)
    mean = selected.mean(axis=0)
    std = selected.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = ((values.astype(np.float64) - mean) / std).astype(np.float32)
    return normalized, mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Align the real Folsom CSV/image data without changing raw files.")
    parser.add_argument("--dataset", default=r"C:\Users\ADMIN\Desktop\dataset")
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "folsom_15min"))
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--lead-minutes", type=int, default=15)
    parser.add_argument("--image-tolerance-seconds", type=int, default=45)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, help="Development-only prefix; omit for full data")
    args = parser.parse_args()

    dataset_root, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "timeline").mkdir(exist_ok=True)
    (output / "splits").mkdir(exist_ok=True)
    print("reading CSV files", flush=True)
    irradiance = pd.read_csv(dataset_root / "Folsom_irradiance.csv")
    weather = pd.read_csv(dataset_root / "Folsom_weather.csv")
    if args.max_rows:
        irradiance, weather = irradiance.iloc[: args.max_rows].copy(), weather.iloc[: args.max_rows].copy()
    times_irr = pd.to_datetime(irradiance["timeStamp"], errors="raise")
    times_weather = pd.to_datetime(weather["timeStamp"], errors="raise")
    if len(irradiance) != len(weather) or not np.array_equal(times_irr.to_numpy(), times_weather.to_numpy()):
        raise ValueError("irradiance and weather CSV timestamps are not exactly aligned")
    n_rows = len(irradiance)
    if n_rows <= args.history + args.lead_minutes:
        raise ValueError("not enough rows for the requested history and lead time")

    raw = pd.DataFrame({
        "ghi": pd.to_numeric(irradiance["ghi"], errors="coerce"),
        "dni": pd.to_numeric(irradiance["dni"], errors="coerce"),
        "dhi": pd.to_numeric(irradiance["dhi"], errors="coerce"),
        "air_temp": pd.to_numeric(weather["air_temp"], errors="coerce"),
        "relhum": pd.to_numeric(weather["relhum"], errors="coerce"),
        "pressure_pa": pd.to_numeric(weather["press"], errors="coerce") * 100.0,
        "windsp": pd.to_numeric(weather["windsp"], errors="coerce"),
        "winddir": pd.to_numeric(weather["winddir"], errors="coerce"),
        "max_windsp": pd.to_numeric(weather["max_windsp"], errors="coerce"),
        "precipitation": pd.to_numeric(weather["precipitation"], errors="coerce"),
    })
    finite_rows = np.isfinite(raw.to_numpy(dtype=np.float64)).all(axis=1)
    safe = raw.fillna(0.0)
    direction = np.deg2rad(safe["winddir"].to_numpy(dtype=np.float64))
    serial_raw = np.column_stack((
        safe["ghi"], safe["dni"], safe["dhi"], safe["air_temp"], safe["relhum"],
        safe["pressure_pa"], safe["windsp"], np.sin(direction), np.cos(direction),
        safe["max_windsp"], safe["precipitation"],
    )).astype(np.float32)

    print("computing solar geometry and REST2 priors", flush=True)
    geometry = solar_geometry(times_irr, 38.677, -121.148, "UTC")
    mu0 = geometry["mu0"].to_numpy(dtype=np.float32)
    zenith = geometry["apparent_zenith"].to_numpy(dtype=np.float32)
    dni_extra = geometry["dni_extra"].to_numpy(dtype=np.float32)
    pressure = safe["pressure_pa"].to_numpy(dtype=np.float32)
    pwv = estimate_pwv_cm(safe["air_temp"].to_numpy(), safe["relhum"].to_numpy()).astype(np.float32)
    clear_ghi, clear_dni, clear_dhi, direct_t = clear_sky_arrays(mu0, pressure, pwv, dni_extra)

    lead = args.lead_minutes
    target_mu0, target_zenith = shifted(mu0, lead), shifted(zenith, lead, 180.0)
    target_extra = shifted(dni_extra, lead)
    target_clear_ghi, target_clear_dni = shifted(clear_ghi, lead), shifted(clear_dni, lead)
    target_clear_dhi, target_direct = shifted(clear_dhi, lead), shifted(direct_t, lead)
    current_ghi = safe["ghi"].to_numpy(dtype=np.float32)
    previous_ghi = np.roll(current_ghi, 1); previous_ghi[0] = current_ghi[0]
    previous_clear = np.roll(clear_ghi, 1); previous_clear[0] = clear_ghi[0]
    sky_last = np.where(previous_clear > 20.0, previous_ghi / np.maximum(previous_clear, 1.0), 0.0).clip(0.0, 2.0)
    sky_current = np.where(clear_ghi > 20.0, current_ghi / np.maximum(clear_ghi, 1.0), 0.0).clip(0.0, 2.0)
    precipitation = safe["precipitation"].to_numpy(dtype=np.float32).clip(0.0)
    wind = safe["windsp"].to_numpy(dtype=np.float32).clip(0.0)
    attenuation = 1.0 - 0.025 * np.maximum(pwv - 1.5, 0.0) - 0.18 * (1.0 - np.exp(-precipitation))
    attenuation = attenuation - 0.35 * np.clip(1.0 - sky_last, 0.0, 1.0) + 0.03 * np.tanh(wind / 6.0)
    attenuation = np.clip(attenuation, 0.05, 1.2).astype(np.float32)
    adjusted = target_clear_ghi * attenuation
    physics_raw = np.column_stack((
        target_clear_ghi, target_clear_dni, target_clear_dhi, target_mu0, target_zenith,
        target_extra, pressure, pwv, np.full(n_rows, 0.08, np.float32), target_direct,
        sky_last, sky_last * target_clear_ghi, sky_current, attenuation, adjusted,
        previous_ghi - previous_clear, np.maximum(target_clear_ghi - current_ghi, 0.0),
        target_clear_ghi, target_clear_ghi, target_clear_ghi,
        target_mu0, target_mu0, target_mu0, adjusted, adjusted, adjusted,
    )).astype(np.float32)
    targets = np.column_stack((
        shifted(safe["ghi"].to_numpy(dtype=np.float32), lead),
        shifted(safe["dni"].to_numpy(dtype=np.float32), lead),
        shifted(safe["dhi"].to_numpy(dtype=np.float32), lead),
    )).astype(np.float32)

    print("scanning and matching sky images", flush=True)
    image_seconds, image_paths = scan_images(dataset_root)
    row_seconds = times_irr.to_numpy(dtype="datetime64[s]").astype(np.int64)
    image_ids, image_deltas = nearest_image_ids(row_seconds, image_seconds, args.image_tolerance_seconds)
    with (output / "image_paths.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(image_paths))
        handle.write("\n")

    minute_contiguous = np.ones(n_rows, dtype=bool)
    minute_contiguous[1:] = np.diff(row_seconds) == 60
    segment = np.cumsum(~minute_contiguous)
    full_history_data = rolling_all(finite_rows, args.history)
    full_history_images = rolling_all(image_ids >= 0, args.history)
    end_indices = np.arange(n_rows, dtype=np.int64)
    target_indices = np.minimum(end_indices + lead, n_rows - 1)
    valid = end_indices >= args.history - 1
    valid &= end_indices + lead < n_rows
    valid &= full_history_data & full_history_images
    valid &= finite_rows[target_indices]
    valid &= segment[np.maximum(end_indices - args.history + 1, 0)] == segment[target_indices]
    target_year = pd.DatetimeIndex(times_irr.to_numpy()[target_indices]).year.to_numpy()
    split_masks = {"train": target_year == 2014, "val": target_year == 2015, "test": target_year == 2016}
    split_indices = {name: end_indices[valid & mask].astype(np.int32) for name, mask in split_masks.items()}
    if any(len(rows) == 0 for rows in split_indices.values()) and not args.max_rows:
        raise RuntimeError(f"an expected chronological split is empty: { {k: len(v) for k, v in split_indices.items()} }")
    train_rows = split_indices["train"]
    if not len(train_rows):
        train_rows = end_indices[valid]
    serial, serial_mean, serial_std = normalize(serial_raw, train_rows)
    physics, physics_mean, physics_std = normalize(physics_raw, train_rows)
    target_scale = targets[train_rows].astype(np.float64).std(axis=0)
    target_scale[target_scale < 1.0] = 1.0

    timeline = output / "timeline"
    np.save(timeline / "serial.npy", serial)
    np.save(timeline / "physics.npy", physics)
    np.save(timeline / "physics_raw.npy", physics_raw)
    np.save(timeline / "target.npy", targets)
    np.save(timeline / "future_zenith.npy", target_zenith.astype(np.float32))
    np.save(timeline / "image_ids.npy", image_ids)
    np.save(timeline / "timestamps_utc.npy", times_irr.to_numpy(dtype="datetime64[s]"))
    for name, rows in split_indices.items():
        np.save(output / "splits" / f"{name}_end_indices.npy", rows)

    normalizers = {
        "serial": {"features": list(SERIAL_FEATURES), "mean": serial_mean.tolist(), "std": serial_std.tolist()},
        "physics": {"features": list(PHYSICS_FEATURES), "mean": physics_mean.tolist(), "std": physics_std.tolist()},
        "target": {"features": ["ghi", "dni", "dhi"], "scale": target_scale.tolist(), "units": "W/m2"},
    }
    with (output / "normalization.json").open("w", encoding="utf-8") as handle:
        json.dump(normalizers, handle, ensure_ascii=False, indent=2)
    metadata = {
        "source_dataset": str(dataset_root), "timestamps_are_utc": True,
        "latitude": 38.677, "longitude": -121.148, "altitude_m": 70.0,
        "history_length": args.history, "lead_minutes": lead, "forecast_horizon": 1,
        "image_size": args.image_size, "image_tolerance_seconds": args.image_tolerance_seconds,
        "rows": n_rows, "images": len(image_paths), "matched_rows": int((image_ids >= 0).sum()),
        "max_matched_delta_seconds": int(image_deltas[image_ids >= 0].max(initial=0)),
        "split_samples": {name: int(len(rows)) for name, rows in split_indices.items()},
        "target_mode": "irradiance", "target_order": ["GHI", "DNI", "DHI"],
        "physics_closure": "GHI = cos(zenith) * DNI + DHI",
        "raw_data_modified": False,
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

