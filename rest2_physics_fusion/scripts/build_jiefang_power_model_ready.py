from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_mock_training_csv import MODEL_READY_COLUMNS
from rest2_physics_fusion.data.merge_sources import WEATHER_COLUMNS, merge_station_with_weather
from rest2_physics_fusion.physics.physics_features import PhysicsConfig, build_physics_features
from rest2_physics_fusion.physics.solar_geometry import SiteConfig


TARGET_HORIZONS = {
    "5min": pd.Timedelta(minutes=5),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

WEATHER_RENAME = {
    "temp": "TEMP",
    "temperature": "TEMP",
    "air_temp": "TEMP",
    "TEMP": "TEMP",
    "ws": "WS",
    "wind_speed": "WS",
    "windsp": "WS",
    "WS": "WS",
    "wd": "WD",
    "wind_dir": "WD",
    "winddir": "WD",
    "WD": "WD",
    "prec": "PREC",
    "precip": "PREC",
    "precipitation": "PREC",
    "PREC": "PREC",
    "pwat": "PWAT",
    "pwv": "PWAT",
    "PWAT": "PWAT",
    "sdwe": "SDWE",
    "SDWE": "SDWE",
    "ghi": "GHI",
    "GHI": "GHI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Jiefang-only model_ready CSV from station power, a timestamp/ghi/dhi/dni irradiance table, "
            "weather history, and optional 4h/1d forecast tables."
        )
    )
    parser.add_argument("--power-file", required=True, help="Station table with timestamp and power.")
    parser.add_argument("--irradiance-file", required=True, help="Table with timestamp, ghi, dhi, dni.")
    parser.add_argument("--weather-history-file", required=True, help="Weather history table, e.g. solar_history.")
    parser.add_argument("--forecast-4h-file", default=None)
    parser.add_argument("--forecast-1d-file", default=None)
    parser.add_argument("--model-ready-output", default=str(ROOT / "data" / "jiefang_power_model_ready" / "train_jiefang_power.csv"))
    parser.add_argument("--enriched-output", default=str(ROOT / "data" / "jiefang_power_enriched" / "train_jiefang_power.csv"))
    parser.add_argument("--physics-output", default=str(ROOT / "outputs" / "jiefang_power_physics" / "physics_jiefang_power.csv"))
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--power-column", default=None)
    parser.add_argument("--latitude", type=float, default=29.919)
    parser.add_argument("--longitude", type=float, default=100.641)
    parser.add_argument("--altitude-m", type=float, default=0.0)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--join-tolerance-minutes", type=int, default=60)
    parser.add_argument("--target-tolerance-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--clear-sky-backend",
        choices=["auto", "pvlib", "pvlib_simplified_solis", "rest2", "rest2_numpy", "rest2_like", "fallback", "fallback_rest2_like"],
        default="auto",
    )
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_timestamp(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    out = frame.copy()
    if timestamp_column not in out.columns:
        if "dtime" in out.columns:
            out[timestamp_column] = out["dtime"]
        elif "timeStamp" in out.columns:
            out[timestamp_column] = out["timeStamp"]
        else:
            raise ValueError(f"Missing timestamp column. Expected {timestamp_column}, dtime, or timeStamp.")
    out[timestamp_column] = pd.to_datetime(out[timestamp_column])
    return out.sort_values(timestamp_column).reset_index(drop=True)


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def normalize_irradiance(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    out = normalize_timestamp(frame, timestamp_column)
    rename = {}
    for canonical in ("ghi", "dhi", "dni"):
        source = first_existing(list(out.columns), [canonical, canonical.upper()])
        if source is None:
            raise ValueError(f"Irradiance table is missing required column: {canonical}")
        rename[source] = canonical
    out = out.rename(columns=rename)
    keep = [timestamp_column, "ghi", "dhi", "dni"]
    for column in ("ghi", "dhi", "dni"):
        out[column] = pd.to_numeric(out[column], errors="coerce").clip(lower=0.0)
    return out[keep]


def normalize_power(frame: pd.DataFrame, timestamp_column: str, power_column: str | None) -> pd.DataFrame:
    out = normalize_timestamp(frame, timestamp_column)
    source = power_column or first_existing(
        list(out.columns),
        ["observe_power", "power", "active_power", "p", "POWER", "Power"],
    )
    if source is None or source not in out.columns:
        raise ValueError("Power table must contain a power column. Pass --power-column if it is not obvious.")
    out["observe_power"] = pd.to_numeric(out[source], errors="coerce").clip(lower=0.0)
    return out[[timestamp_column, "observe_power"]]


def normalize_weather(frame: pd.DataFrame, timestamp_column: str, prefix: str | None = None) -> pd.DataFrame:
    out = normalize_timestamp(frame, timestamp_column)
    rename = {}
    for column in out.columns:
        key = WEATHER_RENAME.get(str(column))
        if key is None:
            key = WEATHER_RENAME.get(str(column).lower())
        if key is not None and key not in rename.values():
            rename[column] = key
    out = out.rename(columns=rename)
    keep = [timestamp_column]
    for column in WEATHER_COLUMNS + ["GHI"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            keep.append(column)
    out = out[keep].copy()
    if prefix:
        out = out.rename(columns={column: f"{prefix}_{column}" for column in keep if column != timestamp_column})
    return out


def merge_nearest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    timestamp_column: str,
    tolerance_minutes: int,
) -> pd.DataFrame:
    left_work = left.sort_values(timestamp_column)
    right_work = right.sort_values(timestamp_column)
    return pd.merge_asof(
        left_work,
        right_work,
        on=timestamp_column,
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )


def add_forecast_features(
    frame: pd.DataFrame,
    forecast: pd.DataFrame | None,
    *,
    timestamp_column: str,
    prefix: str,
    tolerance_minutes: int,
    pwat_to_cm: float,
) -> pd.DataFrame:
    out = frame.copy()
    prefixed_weather = [f"{prefix}_{column}" for column in WEATHER_COLUMNS]
    if forecast is not None:
        out = merge_nearest(
            out,
            forecast,
            timestamp_column=timestamp_column,
            tolerance_minutes=tolerance_minutes,
        )
    joined = pd.Series(False, index=out.index)
    for source, target in [
        (f"{prefix}_TEMP", f"{prefix}_temp_c"),
        (f"{prefix}_WS", f"{prefix}_wind_speed"),
        (f"{prefix}_WD", f"{prefix}_wind_dir"),
        (f"{prefix}_PREC", f"{prefix}_precip"),
    ]:
        if source in out.columns:
            out[target] = pd.to_numeric(out[source], errors="coerce")
            joined = joined | out[target].notna()
        else:
            out[target] = np.nan
    pwat_column = f"{prefix}_PWAT"
    if pwat_column in out.columns:
        out[f"{prefix}_pwv_cm"] = (pd.to_numeric(out[pwat_column], errors="coerce") * pwat_to_cm).clip(
            lower=0.05,
            upper=10.0,
        )
        joined = joined | out[f"{prefix}_pwv_cm"].notna()
    else:
        out[f"{prefix}_pwv_cm"] = np.nan
    ghi_column = f"{prefix}_GHI"
    out[f"{prefix}_ghi"] = pd.to_numeric(out[ghi_column], errors="coerce").clip(lower=0.0) if ghi_column in out.columns else np.nan
    out[f"{prefix}_is_joined"] = joined.astype(float)
    for column in [
        f"{prefix}_temp_c",
        f"{prefix}_wind_speed",
        f"{prefix}_wind_dir",
        f"{prefix}_precip",
        f"{prefix}_pwv_cm",
        f"{prefix}_ghi",
    ]:
        out[column] = out[column].fillna(0.0)
    out = out.drop(columns=[column for column in prefixed_weather + [f"{prefix}_GHI"] if column in out.columns])
    return out


def infer_step(series: pd.Series) -> pd.Timedelta:
    diffs = pd.to_datetime(series).sort_values().diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=15)
    return diffs.median()


def add_future_targets(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    value_column: str,
    output_prefix: str,
    tolerance_multiplier: float,
) -> pd.DataFrame:
    out = frame.copy()
    step = infer_step(out[timestamp_column])
    tolerance = max(step * tolerance_multiplier, pd.Timedelta(minutes=1))
    lookup = out[[timestamp_column, value_column]].dropna().sort_values(timestamp_column)
    lookup = lookup.rename(columns={timestamp_column: "_target_time", value_column: "_target_value"})
    for name, horizon in TARGET_HORIZONS.items():
        query = pd.DataFrame(
            {
                "_row_id": out.index,
                "_target_time": out[timestamp_column] + horizon,
            }
        ).sort_values("_target_time")
        matched = pd.merge_asof(
            query,
            lookup,
            on="_target_time",
            direction="nearest",
            tolerance=tolerance,
        ).sort_values("_row_id")
        out[f"{output_prefix}_{name}"] = pd.to_numeric(matched["_target_value"], errors="coerce").reset_index(drop=True)
    return out


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote={path} rows={len(frame)} columns={len(frame.columns)}")


def model_ready(frame: pd.DataFrame) -> pd.DataFrame:
    extra_columns = [
        "target_ghi_5min",
        "target_ghi_4h",
        "target_ghi_1d",
        "target_power_5min",
        "target_power_4h",
        "target_power_1d",
        "input_dhi",
        "input_dni",
        "observe_power",
        "forecast_4h_temp_c",
        "forecast_4h_wind_speed",
        "forecast_4h_wind_dir",
        "forecast_4h_precip",
        "forecast_4h_pwv_cm",
        "forecast_4h_ghi",
        "forecast_4h_is_joined",
        "forecast_1d_temp_c",
        "forecast_1d_wind_speed",
        "forecast_1d_wind_dir",
        "forecast_1d_precip",
        "forecast_1d_pwv_cm",
        "forecast_1d_ghi",
        "forecast_1d_is_joined",
    ]
    columns = list(dict.fromkeys([*MODEL_READY_COLUMNS, *extra_columns]))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot write model_ready CSV; missing columns: {missing}")
    return frame.loc[:, columns].copy()


def main() -> None:
    args = parse_args()
    timestamp_column = args.timestamp_column
    config = PhysicsConfig(
        SiteConfig(args.latitude, args.longitude, args.altitude_m, args.timezone),
        clear_sky_backend=args.clear_sky_backend,
    )

    irradiance = normalize_irradiance(read_table(args.irradiance_file), timestamp_column)
    power = normalize_power(read_table(args.power_file), timestamp_column, args.power_column)
    weather = normalize_weather(read_table(args.weather_history_file), timestamp_column)
    forecast_4h = (
        normalize_weather(read_table(args.forecast_4h_file), timestamp_column, prefix="forecast_4h")
        if args.forecast_4h_file
        else None
    )
    forecast_1d = (
        normalize_weather(read_table(args.forecast_1d_file), timestamp_column, prefix="forecast_1d")
        if args.forecast_1d_file
        else None
    )

    base = irradiance.rename(columns={"ghi": "observe_ghi"}).copy()
    base["input_dhi"] = irradiance["dhi"]
    base["input_dni"] = irradiance["dni"]
    base = merge_nearest(
        base,
        power,
        timestamp_column=timestamp_column,
        tolerance_minutes=args.join_tolerance_minutes,
    )
    base = merge_station_with_weather(
        base,
        weather.rename(columns={timestamp_column: "timestamp"}),
        timestamp_column=timestamp_column,
        tolerance_minutes=args.join_tolerance_minutes,
        weather_label="weather_history",
    )
    base = add_forecast_features(
        base,
        forecast_4h,
        timestamp_column=timestamp_column,
        prefix="forecast_4h",
        tolerance_minutes=args.join_tolerance_minutes,
        pwat_to_cm=config.pwat_to_cm,
    )
    base = add_forecast_features(
        base,
        forecast_1d,
        timestamp_column=timestamp_column,
        prefix="forecast_1d",
        tolerance_minutes=args.join_tolerance_minutes,
        pwat_to_cm=config.pwat_to_cm,
    )

    physics = build_physics_features(
        base,
        timestamp_column=timestamp_column,
        ghi_column="observe_ghi",
        config=config,
        source_type="station_observation",
        station_name="jiefang",
    )
    physics = add_future_targets(
        physics,
        timestamp_column=timestamp_column,
        value_column="input_ghi",
        output_prefix="target_ghi",
        tolerance_multiplier=args.target_tolerance_multiplier,
    )
    physics = add_future_targets(
        physics,
        timestamp_column=timestamp_column,
        value_column="observe_power",
        output_prefix="target_power",
        tolerance_multiplier=args.target_tolerance_multiplier,
    )
    physics["source_file"] = str(args.power_file)
    physics["feature_quality_flags"] = physics["feature_quality_flags"].astype(str) + ";jiefang_power_two_stage"

    write_csv(physics, args.enriched_output)
    write_csv(physics, args.physics_output)
    ready = model_ready(physics)
    write_csv(ready, args.model_ready_output)
    invalid = {
        column: int(pd.to_numeric(ready[column], errors="coerce").isna().sum())
        for column in ["target_ghi_4h", "target_ghi_1d", "target_power_4h", "target_power_1d"]
    }
    print(f"invalid_targets={invalid}")
    print("note=GHI targets come from the timestamp/ghi/dhi/dni irradiance table; power targets come from the station power table.")


if __name__ == "__main__":
    main()
