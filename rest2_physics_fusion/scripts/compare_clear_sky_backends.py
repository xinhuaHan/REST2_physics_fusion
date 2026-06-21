from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.physics.physics_features import PhysicsConfig, build_physics_features
from rest2_physics_fusion.physics.solar_geometry import SiteConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "data" / "mock_raw" / "solar_history.csv"))
    parser.add_argument("--ghi-column", default="GHI")
    parser.add_argument("--timestamp-column", default="dtime")
    parser.add_argument("--latitude", type=float, default=29.919)
    parser.add_argument("--longitude", type=float, default=100.641)
    parser.add_argument("--altitude-m", type=float, default=0.0)
    return parser.parse_args()


def summarize_delta(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, column: str, candidate_name: str) -> dict[str, float]:
    delta = reference_df[column] - candidate_df[column]
    return {
        "candidate": candidate_name,
        "column": column,
        "mean_delta_vs_pvlib": float(delta.mean()),
        "mae_delta_vs_pvlib": float(delta.abs().mean()),
        "max_abs_delta_vs_pvlib": float(delta.abs().max()),
    }


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.csv)
    site = SiteConfig(args.latitude, args.longitude, args.altitude_m, "Asia/Shanghai")
    kwargs = {
        "timestamp_column": args.timestamp_column,
        "ghi_column": args.ghi_column,
        "source_type": "backend_compare",
        "station_name": "backend_compare",
    }
    pvlib_df = build_physics_features(raw, config=PhysicsConfig(site, clear_sky_backend="pvlib"), **kwargs)
    rest2_df = build_physics_features(raw, config=PhysicsConfig(site, clear_sky_backend="rest2_numpy"), **kwargs)
    fallback_df = build_physics_features(raw, config=PhysicsConfig(site, clear_sky_backend="fallback"), **kwargs)
    print(f"pvlib_backend={sorted(pvlib_df['clear_sky_backend'].unique().tolist())}")
    print(f"rest2_backend={sorted(rest2_df['clear_sky_backend'].unique().tolist())}")
    print(f"fallback_backend={sorted(fallback_df['clear_sky_backend'].unique().tolist())}")
    for candidate_name, candidate_df in (("rest2_numpy", rest2_df), ("fallback_rest2_like", fallback_df)):
        for column in ("ghi_clear_target", "dni_clear_target", "dhi_clear_target"):
            print(summarize_delta(pvlib_df, candidate_df, column, candidate_name))


if __name__ == "__main__":
    main()
