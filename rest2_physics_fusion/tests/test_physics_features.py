from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.physics.physics_features import PHYSICS_FEATURES, PhysicsConfig, build_physics_features
from rest2_physics_fusion.physics.solar_geometry import SiteConfig


def test_build_physics_features_contains_contract() -> None:
    frame = pd.DataFrame(
        {
            "dtime": pd.date_range("2024-01-01", periods=8, freq="15min"),
            "GHI": [0, 0, 0, 10, 100, 200, 300, 350],
            "TEMP": [1] * 8,
            "PWAT": [2] * 8,
        }
    )
    out = build_physics_features(
        frame,
        timestamp_column="dtime",
        ghi_column="GHI",
        config=PhysicsConfig(SiteConfig(29.919, 100.641)),
        source_type="unit_test",
        station_name="unit",
    )
    for column in PHYSICS_FEATURES:
        assert column in out.columns
        assert out[column].isna().sum() == 0
    assert "clear_sky_backend" in out.columns
    assert out["clear_sky_backend"].notna().all()
