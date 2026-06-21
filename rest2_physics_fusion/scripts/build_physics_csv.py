from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.export.export_physics_features import export_excel_physics_csv
from rest2_physics_fusion.physics.physics_features import PhysicsConfig
from rest2_physics_fusion.physics.solar_geometry import SiteConfig


SPECS = [
    ("solar_history.xlsx", "physics_solar_history.csv", "weather_history", "shared_weather_history", "GHI"),
    ("解放电站数据.xlsx", "physics_jiefang_station.csv", "station_observation", "解放", "observe_ghi"),
    ("庆达电站数据.xlsx", "physics_qingda_station.csv", "station_observation", "庆达", "observe_ghi"),
    ("天气预报（4小时）.xlsx", "physics_forecast_4h.csv", "forecast_4h", "forecast_grid_or_station", "GHI"),
    ("天气预报（一天）.xlsx", "physics_forecast_1d.csv", "forecast_1d", "forecast_grid_or_station", "GHI"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=r"C:\Users\ADMIN\Desktop\新建文件夹")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "physics_csv"))
    parser.add_argument("--latitude", type=float, default=29.919)
    parser.add_argument("--longitude", type=float, default=100.641)
    parser.add_argument("--altitude-m", type=float, default=0.0)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument(
        "--clear-sky-backend",
        choices=["auto", "pvlib", "pvlib_simplified_solis", "rest2", "rest2_numpy", "rest2_like", "fallback", "fallback_rest2_like"],
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhysicsConfig(
        site=SiteConfig(
            latitude=args.latitude,
            longitude=args.longitude,
            altitude_m=args.altitude_m,
            timezone=args.timezone,
        ),
        clear_sky_backend=args.clear_sky_backend,
    )
    outputs = []
    for file_name, output_name, source_type, station_name, ghi_column in SPECS:
        path = export_excel_physics_csv(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_name=file_name,
            output_name=output_name,
            timestamp_column="dtime",
            ghi_column=ghi_column,
            source_type=source_type,
            station_name=station_name,
            config=config,
        )
        outputs.append(path)
        print(f"wrote={path}")
    print(f"files={len(outputs)}")


if __name__ == "__main__":
    main()
