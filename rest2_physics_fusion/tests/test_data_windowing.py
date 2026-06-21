from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.data.schema import DataSchema
from rest2_physics_fusion.data.windowing import PhysicsWindowDataset
from rest2_physics_fusion.physics.physics_features import PHYSICS_FEATURES


def test_window_dataset_shapes() -> None:
    rows = []
    for i in range(20):
        row = {
            "dtime": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=15 * i),
            "target_ghi_5min": float(i),
            "input_ghi": float(i),
            "temp_c": 1.0,
            "wind_speed": 2.0,
            "wind_dir": 180.0,
            "precip": 0.0,
            "pwv_cm": 0.2,
            "temp_c_is_observed": 1.0,
            "wind_speed_is_observed": 1.0,
            "wind_dir_is_observed": 1.0,
            "precip_is_observed": 1.0,
            "pwv_cm_is_observed": 1.0,
            "pressure_pa_is_observed": 0.0,
            "aod700_is_observed": 0.0,
            "weather_is_joined": 1.0,
        }
        row.update({feature: 1.0 for feature in PHYSICS_FEATURES})
        rows.append(row)
    dataset = PhysicsWindowDataset(pd.DataFrame(rows), DataSchema(), window_steps=4)
    sample = dataset[0]
    assert sample.serial.shape == (4, 14)
    assert sample.weather.shape == (4, 11)
    assert sample.physics.shape == (len(PHYSICS_FEATURES),)
    assert sample.physics_raw.shape == (len(PHYSICS_FEATURES),)
    assert sample.target.shape == (1,)
