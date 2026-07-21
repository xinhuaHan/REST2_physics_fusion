from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe import load_config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_header(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=5)
    return pd.read_csv(path, nrows=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the five station files without writing processed data.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "server_station.yaml"))
    parser.add_argument("--data-root")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config.data
    root = resolve_path(args.data_root or data.root_dir)
    specifications = {
        "power_ghi": (data.power_ghi_file, [data.timestamp_column, data.power_column, data.power_ghi_column]),
        "irradiance": (data.irradiance_file, [data.timestamp_column, data.irradiance_ghi_column, data.irradiance_dni_column, data.irradiance_dhi_column]),
        "forecast_4h": (data.forecast_4h_file, [data.timestamp_column, *data.forecast_columns, data.forecast_interval_column]),
        "forecast_1d": (data.forecast_1d_file, [data.timestamp_column, *data.forecast_columns, data.forecast_interval_column]),
        "weather": (data.weather_file, [data.timestamp_column, *data.weather_columns]),
    }
    report = {}
    ok = True
    for role, (filename, required) in specifications.items():
        path = root / filename
        if not path.exists():
            report[role] = {"path": str(path), "exists": False, "missing_columns": required}
            ok = False
            continue
        frame = read_header(path)
        missing = [name for name in required if name not in frame.columns]
        report[role] = {"path": str(path), "exists": True, "columns": list(map(str, frame.columns)), "missing_columns": missing}
        ok &= not missing
    print(json.dumps({"ok": ok, "files": report}, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
