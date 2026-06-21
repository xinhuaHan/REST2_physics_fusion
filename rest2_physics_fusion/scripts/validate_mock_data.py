from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RAW_CONTRACTS = {
    "solar_history.csv": ["dtime", "GHI", "TEMP", "WS", "WD", "PREC", "PWAT", "SDWE"],
    "jiefang_station.csv": ["dtime", "observe_power", "observe_ghi"],
    "qingda_station.csv": ["dtime", "observe_power", "observe_ghi"],
    "forecast_4h.csv": [
        "dtime",
        "GHI",
        "TEMP",
        "WS",
        "WD",
        "PREC",
        "PWAT",
        "SDWE",
        "interval",
        "interval_day",
        "report_date",
        "report_time",
        "update_time",
        "govern_flag",
    ],
    "forecast_1d.csv": [
        "dtime",
        "GHI",
        "TEMP",
        "WS",
        "WD",
        "PREC",
        "PWAT",
        "SDWE",
        "interval",
        "interval_day",
        "report_date",
        "report_time",
        "update_time",
        "govern_flag",
    ],
}

TRAINING_REQUIRED = [
    "dtime",
    "input_ghi",
    "ghi_clear_target",
    "mu0_target",
    "pressure_pa_source",
    "pwv_cm_source",
    "aod700_source",
    "weather_join_status",
    "temp_c_is_observed",
    "weather_is_joined",
    "target_ghi_5min",
    "target_ghi_4h",
    "target_ghi_1d",
]

MODEL_READY_COLUMNS = [
    "dtime",
    "source_type",
    "station_name",
    "input_ghi",
    "target_ghi_5min",
    "target_ghi_4h",
    "target_ghi_1d",
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
    "pwv_cm_source",
    "pressure_pa_source",
    "aod700_source",
    "weather_join_status",
    "feature_quality_flags",
]


def assert_columns(path: Path, expected: list[str]) -> None:
    df = pd.read_csv(path, nrows=5)
    columns = list(df.columns)
    missing = [col for col in expected if col not in columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    print(f"{path.name}: rows_preview={len(df)} columns_ok={len(columns)}")


def assert_exact_columns(path: Path, expected: list[str]) -> None:
    df = pd.read_csv(path, nrows=5)
    columns = list(df.columns)
    if columns != expected:
        missing = [col for col in expected if col not in columns]
        extra = [col for col in columns if col not in expected]
        misplaced = [
            (idx, expected_col, columns[idx] if idx < len(columns) else None)
            for idx, expected_col in enumerate(expected)
            if idx >= len(columns) or columns[idx] != expected_col
        ][:10]
        raise ValueError(
            f"{path.name} schema mismatch: missing={missing}, extra={extra}, first_misplaced={misplaced}"
        )
    print(f"{path.name}: rows_preview={len(df)} exact_model_ready_columns={len(columns)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--raw-dir", default=str(root / "data" / "mock_raw"))
    parser.add_argument("--enriched-dir", default=str(root / "data" / "mock_enriched"))
    parser.add_argument("--training-dir", default="")
    parser.add_argument("--model-ready-dir", default=str(root / "data" / "mock_model_ready"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    enriched_dir = Path(args.enriched_dir)
    training_dir = Path(args.training_dir) if args.training_dir else None
    model_ready_dir = Path(args.model_ready_dir)
    for file_name, columns in RAW_CONTRACTS.items():
        assert_columns(raw_dir / file_name, columns)
    for path in sorted(enriched_dir.glob("train_*.csv")):
        assert_columns(path, TRAINING_REQUIRED)
    if training_dir is not None and training_dir.exists():
        for path in sorted(training_dir.glob("train_*.csv")):
            assert_columns(path, TRAINING_REQUIRED)
    for path in sorted(model_ready_dir.glob("train_*.csv")):
        assert_exact_columns(path, MODEL_READY_COLUMNS)
    print("mock_data_validation=passed")


if __name__ == "__main__":
    main()
