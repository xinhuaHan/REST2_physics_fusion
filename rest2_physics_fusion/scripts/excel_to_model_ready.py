from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rest2_physics_fusion.data.merge_sources import merge_station_with_weather
from rest2_physics_fusion.physics.physics_features import PhysicsConfig, build_physics_features
from rest2_physics_fusion.physics.solar_geometry import SiteConfig

from generate_mock_training_csv import MODEL_READY_COLUMNS, to_model_ready


EXCEL_SPECS = {
    "solar_history": {
        "file_name": "solar_history.xlsx",
        "output_name": "train_solar_history.csv",
        "ghi_column": "GHI",
        "source_type": "weather_history",
        "station_name": "shared_weather_history",
    },
    "jiefang_station": {
        "file_name": "解放电站数据.xlsx",
        "output_name": "train_jiefang_station.csv",
        "ghi_column": "observe_ghi",
        "source_type": "station_observation",
        "station_name": "解放",
        "join_weather": True,
    },
    "qingda_station": {
        "file_name": "庆达电站数据.xlsx",
        "output_name": "train_qingda_station.csv",
        "ghi_column": "observe_ghi",
        "source_type": "station_observation",
        "station_name": "庆达",
        "join_weather": True,
    },
    "forecast_4h": {
        "file_name": "天气预报（4小时）.xlsx",
        "output_name": "train_forecast_4h.csv",
        "ghi_column": "GHI",
        "source_type": "forecast_4h",
        "station_name": "forecast_grid_or_station",
    },
    "forecast_1d": {
        "file_name": "天气预报（一天）.xlsx",
        "output_name": "train_forecast_1d.csv",
        "ghi_column": "GHI",
        "source_type": "forecast_1d",
        "station_name": "forecast_grid_or_station",
    },
}


TARGET_HORIZONS = {
    "target_ghi_5min": pd.Timedelta(minutes=5),
    "target_ghi_4h": pd.Timedelta(hours=4),
    "target_ghi_1d": pd.Timedelta(days=1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert real Excel files with the sample schema into fixed-schema model_ready training CSV files."
    )
    parser.add_argument("--input-dir", default=r"C:\Users\ADMIN\Desktop\新建文件夹")
    parser.add_argument("--model-ready-output-dir", default=str(ROOT / "data" / "real_model_ready"))
    parser.add_argument("--enriched-output-dir", default=str(ROOT / "data" / "real_enriched"))
    parser.add_argument("--physics-output-dir", default=str(ROOT / "outputs" / "real_physics_csv"))
    parser.add_argument("--latitude", type=float, default=29.919)
    parser.add_argument("--longitude", type=float, default=100.641)
    parser.add_argument("--altitude-m", type=float, default=0.0)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument(
        "--clear-sky-backend",
        choices=["auto", "pvlib", "pvlib_simplified_solis", "rest2", "rest2_numpy", "rest2_like", "fallback", "fallback_rest2_like"],
        default="auto",
    )
    parser.add_argument("--station-weather-tolerance-minutes", type=int, default=60)
    parser.add_argument("--target-tolerance-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(EXCEL_SPECS),
        choices=sorted(EXCEL_SPECS),
        help="Subset of known Excel sources to convert.",
    )
    return parser.parse_args()


def read_excel(input_dir: Path, file_name: str) -> pd.DataFrame:
    path = input_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing Excel file: {path}")
    frame = pd.read_excel(path)
    if "dtime" not in frame.columns:
        raise ValueError(f"{path.name} is missing required dtime column")
    frame["dtime"] = pd.to_datetime(frame["dtime"])
    return frame.sort_values("dtime").reset_index(drop=True)


def infer_step(series: pd.Series) -> pd.Timedelta:
    times = pd.to_datetime(series).sort_values()
    diffs = times.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=15)
    return diffs.median()


def add_future_targets_from_observed_ghi(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "dtime",
    value_column: str = "input_ghi",
    tolerance_multiplier: float = 1.5,
) -> pd.DataFrame:
    out = frame.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column])
    step = infer_step(out[timestamp_column])
    tolerance = max(step * tolerance_multiplier, pd.Timedelta(minutes=1))
    target_lookup = out[[timestamp_column, value_column]].dropna().sort_values(timestamp_column)
    target_lookup = target_lookup.rename(columns={timestamp_column: "_target_time", value_column: "_target_value"})

    for target_column, horizon in TARGET_HORIZONS.items():
        query = pd.DataFrame(
            {
                "_row_id": out.index,
                "_target_time": out[timestamp_column] + horizon,
            }
        ).sort_values("_target_time")
        matched = pd.merge_asof(
            query,
            target_lookup,
            on="_target_time",
            direction="forward",
            tolerance=tolerance,
        ).sort_values("_row_id")
        target = matched["_target_value"].reset_index(drop=True)
        # Keep the last rows usable by falling back to the current observed GHI when the future horizon is outside the file.
        out[target_column] = target.fillna(pd.to_numeric(out[value_column], errors="coerce")).clip(lower=0.0)
        out[f"{target_column}_source"] = pd.Series(
            ["future_observed_or_forecast_ghi" if pd.notna(value) else "fallback_current_ghi" for value in target],
            index=out.index,
        )
    return out


def build_source_frame(
    *,
    key: str,
    frames: dict[str, pd.DataFrame],
    input_dir: Path,
    station_weather_tolerance_minutes: int,
) -> pd.DataFrame:
    spec = EXCEL_SPECS[key]
    frame = frames[key].copy()
    if spec.get("join_weather"):
        weather = frames.get("solar_history")
        if weather is None:
            weather = read_excel(input_dir, EXCEL_SPECS["solar_history"]["file_name"])
        frame = merge_station_with_weather(
            frame,
            weather,
            tolerance_minutes=station_weather_tolerance_minutes,
            weather_label="solar_history",
        )
    return frame


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote={path} rows={len(frame)} columns={len(frame.columns)}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    model_ready_dir = Path(args.model_ready_output_dir)
    enriched_dir = Path(args.enriched_output_dir)
    physics_dir = Path(args.physics_output_dir)
    config = PhysicsConfig(
        SiteConfig(args.latitude, args.longitude, args.altitude_m, args.timezone),
        clear_sky_backend=args.clear_sky_backend,
    )

    frames = {
        key: read_excel(input_dir, EXCEL_SPECS[key]["file_name"])
        for key in set(args.sources) | {"solar_history"}
        if key in EXCEL_SPECS
    }

    outputs = []
    for key in args.sources:
        spec = EXCEL_SPECS[key]
        raw = build_source_frame(
            key=key,
            frames=frames,
            input_dir=input_dir,
            station_weather_tolerance_minutes=args.station_weather_tolerance_minutes,
        )
        physics = build_physics_features(
            raw,
            timestamp_column="dtime",
            ghi_column=str(spec["ghi_column"]),
            config=config,
            source_type=str(spec["source_type"]),
            station_name=str(spec["station_name"]),
        )
        physics = add_future_targets_from_observed_ghi(
            physics,
            timestamp_column="dtime",
            value_column="input_ghi",
            tolerance_multiplier=args.target_tolerance_multiplier,
        )
        physics["source_file"] = str(spec["file_name"])
        output_name = str(spec["output_name"])
        write_csv(physics, enriched_dir / output_name)
        write_csv(physics, physics_dir / output_name.replace("train_", "physics_"))
        model_ready = to_model_ready(physics)
        write_csv(model_ready, model_ready_dir / output_name)
        outputs.append(model_ready_dir / output_name)

    print(f"model_ready_files={len(outputs)}")
    print("model_ready_columns=" + str(len(MODEL_READY_COLUMNS)))
    for path in outputs:
        print(f"model_ready={path}")
    print("note=physics features are deterministic from Excel inputs plus pvlib/REST2 clear-sky calculations; target columns are future shifted observed/forecast GHI, not random.")


if __name__ == "__main__":
    main()
