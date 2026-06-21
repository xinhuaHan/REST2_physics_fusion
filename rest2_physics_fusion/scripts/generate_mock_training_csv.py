from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.data.merge_sources import merge_station_with_weather
from rest2_physics_fusion.physics.physics_features import PhysicsConfig, build_physics_features
from rest2_physics_fusion.physics.solar_geometry import SiteConfig


def make_weather_like_frame(start: str, periods: int, freq_minutes: int, seed: int, forecast_noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start=start, periods=periods, freq=f"{freq_minutes}min")
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    day = times.dayofyear.to_numpy()
    daylight_shape = np.clip(np.sin(np.pi * (hour - 7.0) / 11.0), 0.0, None)
    seasonal = 0.78 + 0.18 * np.sin(2.0 * np.pi * (day - 30.0) / 365.0)
    cloud = 0.35 + 0.55 * rng.beta(4.0, 2.0, size=periods)
    cloud = pd.Series(cloud).rolling(5, min_periods=1).mean().to_numpy()
    clear_like = 950.0 * daylight_shape * seasonal
    ghi = np.clip(clear_like * cloud + rng.normal(0.0, 18.0 + forecast_noise, size=periods), 0.0, None)
    temp = -3.0 + 17.0 * daylight_shape + 6.0 * np.sin(2.0 * np.pi * (day - 20.0) / 365.0)
    temp += rng.normal(0.0, 1.2 + forecast_noise * 0.02, size=periods)
    wind_speed = np.clip(1.8 + rng.normal(0.0, 0.5, size=periods), 0.0, None)
    wind_dir = np.mod(210.0 + 35.0 * np.sin(np.arange(periods) / 36.0) + rng.normal(0.0, 8.0, size=periods), 360.0)
    precip = np.where(rng.random(periods) < 0.04, rng.gamma(1.2, 0.2, size=periods), 0.0)
    pwat = np.clip(2.0 + 1.5 * daylight_shape + rng.normal(0.0, 0.25, size=periods), 0.5, 8.0)
    sdwe = np.clip(0.6 + rng.normal(0.0, 0.05, size=periods), 0.0, None)
    return pd.DataFrame(
        {
            "dtime": times,
            "GHI": ghi,
            "TEMP": temp,
            "WS": wind_speed,
            "WD": wind_dir,
            "PREC": precip,
            "PWAT": pwat,
            "SDWE": sdwe,
        }
    )


def make_station_raw(weather_frame: pd.DataFrame, seed: int, station_scale: float, station_bias: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observe_ghi = np.clip(weather_frame["GHI"].to_numpy(dtype=float) * station_scale + rng.normal(0, 16, len(weather_frame)), 0.0, None)
    observe_power = np.clip(observe_ghi * 0.82 + station_bias + rng.normal(0, 5, len(weather_frame)), 0.0, None)
    return pd.DataFrame(
        {
            "dtime": weather_frame["dtime"],
            "observe_power": observe_power,
            "observe_ghi": observe_ghi,
        }
    )


def add_forecast_meta(frame: pd.DataFrame, lead_minutes: int) -> pd.DataFrame:
    out = frame.copy()
    report_times = pd.to_datetime(out["dtime"]) - pd.to_timedelta(lead_minutes, unit="min")
    out["interval"] = lead_minutes
    out["interval_day"] = (report_times.dt.date < pd.to_datetime(out["dtime"]).dt.date).astype(int)
    out["report_date"] = report_times.dt.strftime("%Y-%m-%d")
    out["report_time"] = report_times.dt.strftime("%H_%M")
    out["update_time"] = 0
    out["govern_flag"] = 0
    return out


def add_targets(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000)
    out = df.copy()
    base = out["input_ghi"].to_numpy(dtype=float)
    persistence = out["smart_persistence_ghi"].to_numpy(dtype=float)
    out["target_ghi_5min"] = np.clip(0.72 * np.roll(base, -1) + 0.28 * persistence + rng.normal(0, 12, len(out)), 0.0, None)
    out["target_ghi_4h"] = np.clip(0.55 * np.roll(base, -16) + 0.45 * persistence + rng.normal(0, 25, len(out)), 0.0, None)
    out["target_ghi_1d"] = np.clip(0.35 * np.roll(base, -96) + 0.65 * persistence + rng.normal(0, 35, len(out)), 0.0, None)
    if "observe_power" not in out.columns:
        out["observe_power"] = np.clip(out["target_ghi_5min"] * 0.82 + rng.normal(0, 5, len(out)), 0.0, None)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-output-dir", default=str(ROOT / "data" / "mock_raw"))
    parser.add_argument("--enriched-output-dir", default=str(ROOT / "data" / "mock_enriched"))
    parser.add_argument("--training-output-dir", default=None)
    parser.add_argument("--model-ready-output-dir", default=str(ROOT / "data" / "mock_model_ready"))
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--start", default="2024-01-01 00:00:00")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latitude", type=float, default=29.919)
    parser.add_argument("--longitude", type=float, default=100.641)
    parser.add_argument("--altitude-m", type=float, default=0.0)
    parser.add_argument(
        "--clear-sky-backend",
        choices=["auto", "pvlib", "pvlib_simplified_solis", "rest2", "rest2_numpy", "rest2_like", "fallback", "fallback_rest2_like"],
        default="auto",
    )
    return parser.parse_args()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote={path} rows={len(frame)}")


MODEL_READY_BASE_COLUMNS = [
    "dtime",
    "source_type",
    "station_name",
    "input_ghi",
    "target_ghi_5min",
    "target_ghi_4h",
    "target_ghi_1d",
]

MODEL_READY_SERIAL_COLUMNS = [
    "temp_c",
    "wind_speed",
    "wind_dir",
    "precip",
    "pwv_cm",
    "temp_c_is_observed",
    "wind_speed_is_observed",
    "wind_dir_is_observed",
    "precip_is_observed",
    "pwv_cm_is_observed",
    "pressure_pa_is_observed",
    "aod700_is_observed",
    "weather_is_joined",
]

MODEL_READY_PHYSICS_COLUMNS = [
    "ghi_clear_target",
    "dni_clear_target",
    "dhi_clear_target",
    "mu0_target",
    "apparent_zenith",
    "dni_extra",
    "pressure_pa",
    "aod700",
    "t_clr_dd",
    "clear_sky_index_last",
    "smart_persistence_ghi",
    "clear_sky_index_current",
    "weather_attenuation_prior",
    "weather_adjusted_clear_sky_ghi",
    "clear_sky_residual_last",
    "clear_sky_gap",
    "ghi_clear_horizon_5min",
    "ghi_clear_horizon_4h",
    "ghi_clear_horizon_1d",
    "mu0_horizon_5min",
    "mu0_horizon_4h",
    "mu0_horizon_1d",
    "weather_adjusted_clear_sky_ghi_horizon_5min",
    "weather_adjusted_clear_sky_ghi_horizon_4h",
    "weather_adjusted_clear_sky_ghi_horizon_1d",
]

MODEL_READY_SOURCE_COLUMNS = [
    "pwv_cm_source",
    "pressure_pa_source",
    "aod700_source",
    "weather_join_status",
    "feature_quality_flags",
]

MODEL_READY_COLUMNS = (
    MODEL_READY_BASE_COLUMNS
    + MODEL_READY_SERIAL_COLUMNS
    + MODEL_READY_PHYSICS_COLUMNS
    + MODEL_READY_SOURCE_COLUMNS
)


def to_model_ready(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MODEL_READY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot build model_ready CSV; missing columns: {missing}")
    return frame.loc[:, MODEL_READY_COLUMNS].copy()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_output_dir)
    enriched_dir = Path(args.enriched_output_dir)
    train_dir = Path(args.training_output_dir) if args.training_output_dir else None
    model_ready_dir = Path(args.model_ready_output_dir)
    config = PhysicsConfig(
        SiteConfig(args.latitude, args.longitude, args.altitude_m, "Asia/Shanghai"),
        clear_sky_backend=args.clear_sky_backend,
    )

    periods_15m = args.days * 24 * 4
    periods_1h = args.days * 24
    solar_history = make_weather_like_frame(args.start, periods_15m, 15, args.seed)
    jiefang = make_station_raw(solar_history, args.seed + 1, station_scale=0.98, station_bias=-1.0)
    qingda = make_station_raw(solar_history, args.seed + 2, station_scale=1.03, station_bias=1.5)
    forecast_4h = add_forecast_meta(make_weather_like_frame(args.start, periods_1h, 60, args.seed + 3, forecast_noise=18), 240)
    forecast_1d = add_forecast_meta(make_weather_like_frame(args.start, periods_15m, 15, args.seed + 4, forecast_noise=28), 1440)

    raw_sources = [
        ("solar_history.csv", solar_history),
        ("jiefang_station.csv", jiefang),
        ("qingda_station.csv", qingda),
        ("forecast_4h.csv", forecast_4h),
        ("forecast_1d.csv", forecast_1d),
    ]
    for file_name, frame in raw_sources:
        write_csv(frame, raw_dir / file_name)

    station_weather_label = "weather_history"
    train_specs = [
        ("train_solar_history.csv", solar_history, "GHI", "weather_history", "shared_weather_history", args.seed + 10),
        (
            "train_jiefang_station.csv",
            merge_station_with_weather(jiefang, solar_history, weather_label=station_weather_label),
            "observe_ghi",
            "station_observation",
            "解放",
            args.seed + 11,
        ),
        (
            "train_qingda_station.csv",
            merge_station_with_weather(qingda, solar_history, weather_label=station_weather_label),
            "observe_ghi",
            "station_observation",
            "庆达",
            args.seed + 12,
        ),
        ("train_forecast_4h.csv", forecast_4h, "GHI", "forecast_4h", "forecast_grid_or_station", args.seed + 13),
        ("train_forecast_1d.csv", forecast_1d, "GHI", "forecast_1d", "forecast_grid_or_station", args.seed + 14),
    ]

    for output_name, frame, ghi_column, source_type, station_name, seed in train_specs:
        physics = build_physics_features(
            frame,
            timestamp_column="dtime",
            ghi_column=ghi_column,
            config=config,
            source_type=source_type,
            station_name=station_name,
        )
        physics = add_targets(physics, seed)
        if train_dir is not None:
            write_csv(physics, train_dir / output_name)
        write_csv(physics, enriched_dir / output_name)
        write_csv(to_model_ready(physics), model_ready_dir / output_name)


if __name__ == "__main__":
    main()
