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

from generate_mock_training_csv import to_model_ready
from rest2_physics_fusion.physics.physics_features import (
    PhysicsConfig,
    add_persistence_features,
    add_weather_attenuation_features,
    normalize_weather_columns,
)
from rest2_physics_fusion.physics.rest2_clear_sky import add_clear_sky
from rest2_physics_fusion.physics.solar_geometry import SiteConfig, add_solar_geometry, standard_pressure_pa


TARGET_HORIZONS = {
    "target_ghi_5min": pd.Timedelta(minutes=5),
    "target_ghi_4h": pd.Timedelta(hours=4),
    "target_ghi_1d": pd.Timedelta(days=1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Folsom irradiance/weather CSV files into the fixed model_ready schema."
    )
    parser.add_argument("--dataset-dir", default=r"C:\Users\ADMIN\Desktop\dataset")
    parser.add_argument("--irradiance-file", default="Folsom_irradiance.csv")
    parser.add_argument("--weather-file", default="Folsom_weather.csv")
    parser.add_argument("--model-ready-output", default=str(ROOT / "data" / "folsom_model_ready" / "train_folsom.csv"))
    parser.add_argument("--enriched-output", default=str(ROOT / "data" / "folsom_enriched" / "train_folsom.csv"))
    parser.add_argument("--latitude", type=float, default=38.677)
    parser.add_argument("--longitude", type=float, default=-121.148)
    parser.add_argument("--altitude-m", type=float, default=70.0)
    parser.add_argument(
        "--timestamp-timezone",
        default="UTC",
        help="Timezone of the source timeStamp column. Folsom CSV timestamps are expected to be UTC.",
    )
    parser.add_argument(
        "--site-timezone",
        default="America/Los_Angeles",
        help="Site timezone used only for enriched local-time diagnostics; model_ready dtime stays in source time.",
    )
    parser.add_argument(
        "--clear-sky-backend",
        choices=["auto", "pvlib", "pvlib_simplified_solis", "rest2", "rest2_numpy", "rest2_like", "fallback", "fallback_rest2_like"],
        default="auto",
    )
    parser.add_argument("--target-tolerance-minutes", type=float, default=2.0)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit after merging.")
    return parser.parse_args()


def canonical_time_from_source(values: pd.Series, source_tz: str) -> pd.Series:
    times = pd.to_datetime(values)
    if times.dt.tz is None:
        times = times.dt.tz_localize(source_tz)
    else:
        times = times.dt.tz_convert(source_tz)
    # Keep canonical source-time timestamps for merging/training. For Folsom this
    # is UTC, which avoids duplicated local timestamps during DST fall-back.
    return times.dt.tz_localize(None)


def local_time_string_from_source(values: pd.Series, source_tz: str, site_tz: str) -> pd.Series:
    times = pd.to_datetime(values)
    if times.dt.tz is None:
        times = times.dt.tz_localize(source_tz)
    else:
        times = times.dt.tz_convert(source_tz)
    return times.dt.tz_convert(site_tz).astype(str)


def estimate_pwv_cm_from_temp_rh(temp_c: pd.Series, relhum: pd.Series) -> pd.Series:
    """Estimate precipitable water from near-surface humidity.

    Folsom has relative humidity but not PWAT. This deterministic approximation
    is a physics prior, not a learned or random value. The coefficients keep the
    resulting PWV in a realistic mid-latitude range for clear-sky calculations.
    """

    temp = pd.to_numeric(temp_c, errors="coerce")
    rh = pd.to_numeric(relhum, errors="coerce").clip(lower=0.0, upper=100.0)
    saturation_hpa = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    vapor_pressure_hpa = saturation_hpa * rh / 100.0
    pwv_cm = 0.12 * vapor_pressure_hpa + 0.10
    return pd.Series(pwv_cm, index=temp.index).clip(lower=0.05, upper=8.0)


def read_and_merge(args: argparse.Namespace) -> pd.DataFrame:
    dataset_dir = Path(args.dataset_dir)
    irradiance = pd.read_csv(dataset_dir / args.irradiance_file)
    weather = pd.read_csv(dataset_dir / args.weather_file)
    irradiance["dtime"] = canonical_time_from_source(irradiance["timeStamp"], args.timestamp_timezone)
    weather["dtime"] = canonical_time_from_source(weather["timeStamp"], args.timestamp_timezone)
    irradiance["site_local_time"] = local_time_string_from_source(
        irradiance["timeStamp"], args.timestamp_timezone, args.site_timezone
    )
    irradiance["site_timezone"] = args.site_timezone

    irradiance = irradiance.sort_values("dtime")
    weather = weather.sort_values("dtime")
    merged = irradiance.merge(weather, on="dtime", how="inner", suffixes=("_irr", "_weather"))
    if args.max_rows is not None:
        merged = merged.head(args.max_rows).copy()
    return merged.reset_index(drop=True)


def to_project_raw(frame: pd.DataFrame, config: PhysicsConfig) -> pd.DataFrame:
    out = pd.DataFrame()
    out["dtime"] = frame["dtime"]
    if "site_local_time" in frame.columns:
        out["site_local_time"] = frame["site_local_time"]
    if "site_timezone" in frame.columns:
        out["site_timezone"] = frame["site_timezone"]
    out["GHI"] = pd.to_numeric(frame["ghi"], errors="coerce").clip(lower=0.0)
    out["DNI_observed"] = pd.to_numeric(frame["dni"], errors="coerce").clip(lower=0.0)
    out["DHI_observed"] = pd.to_numeric(frame["dhi"], errors="coerce").clip(lower=0.0)
    out["TEMP"] = pd.to_numeric(frame["air_temp"], errors="coerce")
    out["WS"] = pd.to_numeric(frame["windsp"], errors="coerce")
    out["WD"] = pd.to_numeric(frame["winddir"], errors="coerce")
    out["PREC"] = pd.to_numeric(frame["precipitation"], errors="coerce")
    out["RELHUM"] = pd.to_numeric(frame["relhum"], errors="coerce")
    out["PRESS_HPA"] = pd.to_numeric(frame["press"], errors="coerce")
    pwv_cm = estimate_pwv_cm_from_temp_rh(out["TEMP"], out["RELHUM"])
    out["PWAT"] = pwv_cm / config.pwat_to_cm
    out["weather_is_joined"] = 1.0
    out["weather_join_status"] = "folsom_weather_exact_timestamp"
    return out


def add_observed_pressure(out: pd.DataFrame, raw: pd.DataFrame, config: PhysicsConfig) -> pd.DataFrame:
    pressure_hpa = pd.to_numeric(raw["PRESS_HPA"], errors="coerce")
    pressure_pa = (pressure_hpa * 100.0).where(pressure_hpa.notna(), standard_pressure_pa(config.site.altitude_m))
    out["pressure_pa"] = pressure_pa
    out["pressure_pa_source"] = np.where(pressure_hpa.notna(), "folsom_weather_press", "altitude_estimated")
    out["pressure_pa_is_observed"] = pressure_hpa.notna().astype(float)
    out["feature_quality_flags"] = out["feature_quality_flags"].astype(str) + ";folsom_weather;pwv_estimated_from_temp_rh"
    return out


def add_time_aligned_future_columns(
    out: pd.DataFrame,
    *,
    timestamp_column: str,
    value_columns: list[str],
    tolerance_minutes: float,
) -> pd.DataFrame:
    result = out.copy()
    base = result[[timestamp_column, *value_columns]].sort_values(timestamp_column)
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    for suffix, horizon in {"5min": TARGET_HORIZONS["target_ghi_5min"], "4h": TARGET_HORIZONS["target_ghi_4h"], "1d": TARGET_HORIZONS["target_ghi_1d"]}.items():
        query = pd.DataFrame(
            {
                "_row_id": result.index,
                timestamp_column: result[timestamp_column] + horizon,
            }
        ).sort_values(timestamp_column)
        matched = pd.merge_asof(query, base, on=timestamp_column, direction="nearest", tolerance=tolerance).sort_values("_row_id")
        for column in value_columns:
            result[f"{column}_horizon_{suffix}"] = matched[column].reset_index(drop=True).fillna(result[column])
    return result


def add_targets(out: pd.DataFrame, tolerance_minutes: float) -> pd.DataFrame:
    result = out.copy()
    lookup = result[["dtime", "input_ghi"]].sort_values("dtime").rename(columns={"input_ghi": "_future_ghi"})
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    for target_column, horizon in TARGET_HORIZONS.items():
        query = pd.DataFrame({"_row_id": result.index, "dtime": result["dtime"] + horizon}).sort_values("dtime")
        matched = pd.merge_asof(query, lookup, on="dtime", direction="nearest", tolerance=tolerance).sort_values("_row_id")
        result[target_column] = matched["_future_ghi"].reset_index(drop=True).fillna(result["input_ghi"]).clip(lower=0.0)
        result[f"{target_column}_source"] = np.where(
            matched["_future_ghi"].notna().to_numpy(),
            "future_observed_ghi",
            "fallback_current_ghi",
        )
    return result


def build_folsom_features(frame: pd.DataFrame, config: PhysicsConfig, target_tolerance_minutes: float) -> pd.DataFrame:
    raw = to_project_raw(frame, config)
    out = normalize_weather_columns(raw, "GHI", config)
    out = add_observed_pressure(out, raw, config)
    out["source_type"] = "folsom_irradiance_weather"
    out["station_name"] = "Folsom"
    out["latitude"] = config.site.latitude
    out["longitude"] = config.site.longitude
    out["altitude_m"] = config.site.altitude_m
    out["timezone"] = config.site.timezone
    out = out.sort_values("dtime").reset_index(drop=True)
    out = add_solar_geometry(out, "dtime", config.site)
    out = add_clear_sky(out, config.clear_sky_backend)
    out = add_persistence_features(out, config.sky_index_max)
    out = add_weather_attenuation_features(out)
    out = add_time_aligned_future_columns(
        out,
        timestamp_column="dtime",
        value_columns=["ghi_clear_target", "mu0_target", "weather_adjusted_clear_sky_ghi"],
        tolerance_minutes=target_tolerance_minutes,
    )
    out = out.rename(
        columns={
            "ghi_clear_target_horizon_5min": "ghi_clear_horizon_5min",
            "ghi_clear_target_horizon_4h": "ghi_clear_horizon_4h",
            "ghi_clear_target_horizon_1d": "ghi_clear_horizon_1d",
            "mu0_target_horizon_5min": "mu0_horizon_5min",
            "mu0_target_horizon_4h": "mu0_horizon_4h",
            "mu0_target_horizon_1d": "mu0_horizon_1d",
            "weather_adjusted_clear_sky_ghi_horizon_5min": "weather_adjusted_clear_sky_ghi_horizon_5min",
            "weather_adjusted_clear_sky_ghi_horizon_4h": "weather_adjusted_clear_sky_ghi_horizon_4h",
            "weather_adjusted_clear_sky_ghi_horizon_1d": "weather_adjusted_clear_sky_ghi_horizon_1d",
        }
    )
    out = add_targets(out, target_tolerance_minutes)
    return out


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote={path} rows={len(frame)} columns={len(frame.columns)}")


def main() -> None:
    args = parse_args()
    config = PhysicsConfig(
        SiteConfig(
            latitude=args.latitude,
            longitude=args.longitude,
            altitude_m=args.altitude_m,
            timezone=args.timestamp_timezone,
        ),
        clear_sky_backend=args.clear_sky_backend,
    )
    merged = read_and_merge(args)
    enriched = build_folsom_features(merged, config, args.target_tolerance_minutes)
    model_ready = to_model_ready(enriched)
    write_csv(enriched, Path(args.enriched_output))
    write_csv(model_ready, Path(args.model_ready_output))
    print("note=Folsom conversion uses irradiance/weather CSV only; image files are ignored.")
    print("note=physical features are deterministic from observed Folsom weather/irradiance plus pvlib/REST2 clear-sky calculations; targets are future observed GHI, not random.")


if __name__ == "__main__":
    main()
