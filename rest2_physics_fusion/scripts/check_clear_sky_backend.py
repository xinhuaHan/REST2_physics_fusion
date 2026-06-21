from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.physics.physics_features import PhysicsConfig, build_physics_features
from rest2_physics_fusion.physics.solar_geometry import SiteConfig


def main() -> None:
    frame = pd.DataFrame(
        {
            "dtime": pd.date_range("2024-06-01 10:00:00", periods=4, freq="15min"),
            "GHI": [500.0, 620.0, 700.0, 680.0],
            "TEMP": [18.0, 19.0, 20.0, 20.0],
            "PWAT": [3.0, 3.1, 3.1, 3.2],
        }
    )
    out = build_physics_features(
        frame,
        timestamp_column="dtime",
        ghi_column="GHI",
        config=PhysicsConfig(SiteConfig(29.919, 100.641), clear_sky_backend="auto"),
        source_type="backend_check",
        station_name="check",
    )
    backend = sorted(out["clear_sky_backend"].unique().tolist())
    print(f"clear_sky_backend={backend}")


if __name__ == "__main__":
    main()
