from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.physics.physics_features import PhysicsConfig, build_physics_features
from rest2_physics_fusion.physics.rest2_clear_sky import _rest2_components_numpy
from rest2_physics_fusion.physics.solar_geometry import SiteConfig
from rest2_physics_fusion.physics.torch_clear_sky import rest2_torch_clear_sky


def test_rest2_numpy_backend_is_available() -> None:
    frame = pd.DataFrame(
        {
            "dtime": pd.date_range("2024-01-01 00:00:00", periods=96, freq="15min"),
            "GHI": np.linspace(0.0, 700.0, 96),
            "TEMP": 5.0,
            "PWAT": 2.0,
        }
    )

    out = build_physics_features(
        frame,
        timestamp_column="dtime",
        ghi_column="GHI",
        config=PhysicsConfig(SiteConfig(29.919, 100.641), clear_sky_backend="rest2_numpy"),
        source_type="unit_test",
        station_name="unit",
    )

    assert sorted(out["clear_sky_backend"].unique().tolist()) == ["rest2_numpy"]
    assert out[["ghi_clear_target", "dni_clear_target", "dhi_clear_target"]].isna().sum().sum() == 0
    assert (out.loc[out["mu0_target"] <= 0.0, ["ghi_clear_target", "dni_clear_target", "dhi_clear_target"]] == 0.0).all().all()
    assert out.loc[out["mu0_target"] > 0.1, "ghi_clear_target"].max() > 0.0


def test_rest2_torch_matches_numpy_core() -> None:
    values = {
        "mu0": np.array([0.0, 0.2, 0.55, 0.9], dtype=np.float64),
        "pressure_pa": np.array([101325.0, 93000.0, 85000.0, 78000.0], dtype=np.float64),
        "pwv_cm": np.array([1.5, 0.8, 2.5, 4.0], dtype=np.float64),
        "aod700": np.array([0.08, 0.12, 0.2, 0.04], dtype=np.float64),
        "dni_extra": np.array([1400.0, 1380.0, 1360.0, 1340.0], dtype=np.float64),
    }
    numpy_out = _rest2_components_numpy(**values)
    torch_out = rest2_torch_clear_sky(**{key: torch.tensor(value, dtype=torch.float64) for key, value in values.items()})

    for key in ("ghi_clear_target", "dni_clear_target", "dhi_clear_target", "tau_rayleigh", "tau_water", "tau_aer"):
        np.testing.assert_allclose(torch_out[key].detach().numpy(), numpy_out[key], rtol=1e-6, atol=1e-6)


def test_rest2_torch_has_gradients_for_continuous_inputs() -> None:
    mu0 = torch.tensor([0.35, 0.65, 0.85], dtype=torch.float64)
    pressure_pa = torch.tensor([101325.0, 90000.0, 82000.0], dtype=torch.float64, requires_grad=True)
    pwv_cm = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    aod700 = torch.tensor([0.08, 0.12, 0.16], dtype=torch.float64, requires_grad=True)
    dni_extra = torch.tensor([1360.0, 1365.0, 1370.0], dtype=torch.float64)

    out = rest2_torch_clear_sky(
        mu0=mu0,
        pressure_pa=pressure_pa,
        pwv_cm=pwv_cm,
        aod700=aod700,
        dni_extra=dni_extra,
    )
    loss = out["ghi_clear_target"].sum() + 0.1 * out["dni_clear_target"].sum()
    loss.backward()

    assert pressure_pa.grad is not None and pressure_pa.grad.abs().sum() > 0.0
    assert pwv_cm.grad is not None and pwv_cm.grad.abs().sum() > 0.0
    assert aod700.grad is not None and aod700.grad.abs().sum() > 0.0
