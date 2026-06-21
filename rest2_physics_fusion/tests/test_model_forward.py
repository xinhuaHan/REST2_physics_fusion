from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.models.serial_physics_model import SerialPhysicsForecaster
from rest2_physics_fusion.physics.physics_features import PHYSICS_FEATURES


def test_model_forward_shape() -> None:
    model = SerialPhysicsForecaster(serial_input_dim=14, physics_input_dim=len(PHYSICS_FEATURES))
    batch = {
        "serial": torch.randn(3, 16, 14),
        "physics": torch.randn(3, len(PHYSICS_FEATURES)),
    }
    outputs = model(batch)
    assert outputs["prediction"].shape == (3, 1)


def test_model_forward_with_rest2_calibration() -> None:
    model = SerialPhysicsForecaster(
        serial_input_dim=14,
        physics_input_dim=len(PHYSICS_FEATURES),
        use_rest2_calibration=True,
        physics_feature_columns=tuple(PHYSICS_FEATURES),
        physics_norm={
            "mean": [0.0] * len(PHYSICS_FEATURES),
            "std": [1.0] * len(PHYSICS_FEATURES),
        },
    )
    physics = torch.randn(3, len(PHYSICS_FEATURES))
    physics_raw = torch.zeros(3, len(PHYSICS_FEATURES))
    index = {column: PHYSICS_FEATURES.index(column) for column in PHYSICS_FEATURES}
    physics_raw[:, index["mu0_target"]] = torch.tensor([0.25, 0.55, 0.85])
    physics_raw[:, index["dni_extra"]] = 1365.0
    physics_raw[:, index["pressure_pa"]] = 101325.0
    physics_raw[:, index["pwv_cm"]] = 1.8
    physics_raw[:, index["aod700"]] = 0.08
    physics_raw[:, index["ghi_clear_horizon_5min"]] = torch.tensor([20.0, 400.0, 800.0])
    physics_raw[:, index["mu0_horizon_5min"]] = torch.tensor([0.05, 0.55, 0.85])
    physics_raw[:, index["clear_sky_index_current"]] = torch.tensor([0.1, 0.8, 1.0])
    physics_raw[:, index["weather_attenuation_prior"]] = torch.tensor([0.5, 0.9, 1.0])
    batch = {
        "serial": torch.randn(3, 16, 14),
        "physics": physics,
        "physics_raw": physics_raw,
    }
    outputs = model(batch)
    loss = outputs["prediction"].sum()
    loss.backward()
    assert outputs["prediction"].shape == (3, 1)
    assert outputs["rest2_gate"].shape == (3, 1)
    assert outputs["rest2_effective_blend"].shape == (3, 1)
    assert outputs["rest2_gate"][2] > outputs["rest2_gate"][0]
    assert model.rest2_calibrator is not None
    assert model.rest2_calibrator.log_aod_scale.grad is not None


def test_model_forward_with_clear_sky_weather_head() -> None:
    model = SerialPhysicsForecaster(
        serial_input_dim=14,
        weather_input_dim=11,
        physics_input_dim=len(PHYSICS_FEATURES),
        use_clear_sky_weather_head=True,
        physics_feature_columns=tuple(PHYSICS_FEATURES),
    )
    physics = torch.randn(3, len(PHYSICS_FEATURES))
    physics_raw = torch.zeros(3, len(PHYSICS_FEATURES))
    index = {column: PHYSICS_FEATURES.index(column) for column in PHYSICS_FEATURES}
    physics_raw[:, index["ghi_clear_target"]] = torch.tensor([100.0, 500.0, 800.0])
    physics_raw[:, index["ghi_clear_horizon_4h"]] = torch.tensor([200.0, 600.0, 900.0])
    batch = {
        "serial": torch.randn(3, 16, 14),
        "weather": torch.randn(3, 16, 11),
        "physics": physics,
        "physics_raw": physics_raw,
    }
    outputs = model(batch)
    assert outputs["prediction"].shape == (3, 1)
    assert outputs["k_pred"].shape == (3, 1)
    assert outputs["residual_pred"].shape == (3, 1)
    assert outputs["clear_sky_base"].shape == (3, 1)


def test_clear_sky_weather_head_uses_target_horizon_base() -> None:
    model = SerialPhysicsForecaster(
        serial_input_dim=14,
        weather_input_dim=11,
        physics_input_dim=len(PHYSICS_FEATURES),
        use_clear_sky_weather_head=True,
        target_column="target_ghi_4h",
        physics_feature_columns=tuple(PHYSICS_FEATURES),
    )
    physics_raw = torch.zeros(2, len(PHYSICS_FEATURES))
    index = {column: PHYSICS_FEATURES.index(column) for column in PHYSICS_FEATURES}
    physics_raw[:, index["ghi_clear_target"]] = torch.tensor([10.0, 20.0])
    physics_raw[:, index["ghi_clear_horizon_4h"]] = torch.tensor([300.0, 400.0])
    batch = {
        "serial": torch.randn(2, 16, 14),
        "weather": torch.randn(2, 16, 11),
        "physics": torch.randn(2, len(PHYSICS_FEATURES)),
        "physics_raw": physics_raw,
    }
    outputs = model(batch)
    assert torch.allclose(outputs["clear_sky_base"].view(-1), torch.tensor([300.0, 400.0]))


def test_weather_prior_fusion_outputs_diagnostic_terms() -> None:
    model = SerialPhysicsForecaster(
        serial_input_dim=14,
        weather_input_dim=11,
        physics_input_dim=len(PHYSICS_FEATURES),
        use_clear_sky_weather_head=True,
        use_weather_prior_fusion=True,
        target_column="target_ghi_4h",
        physics_feature_columns=tuple(PHYSICS_FEATURES),
    )
    physics_raw = torch.zeros(2, len(PHYSICS_FEATURES))
    index = {column: PHYSICS_FEATURES.index(column) for column in PHYSICS_FEATURES}
    physics_raw[:, index["ghi_clear_horizon_4h"]] = torch.tensor([300.0, 500.0])
    physics_raw[:, index["weather_adjusted_clear_sky_ghi_horizon_4h"]] = torch.tensor([150.0, 400.0])
    physics_raw[:, index["weather_attenuation_prior"]] = torch.tensor([0.5, 0.8])
    batch = {
        "serial": torch.randn(2, 16, 14),
        "weather": torch.randn(2, 16, 11),
        "physics": torch.randn(2, len(PHYSICS_FEATURES)),
        "physics_raw": physics_raw,
    }
    outputs = model(batch)
    for name in ("prediction", "serial_pred", "prior_pred", "k_weather", "weather_k_prior", "prior_weight"):
        assert outputs[name].shape == (2, 1)
    assert torch.all(outputs["prediction"] >= 0.0)
    assert torch.allclose(outputs["clear_sky_base"].view(-1), torch.tensor([300.0, 500.0]))
    assert torch.allclose(outputs["weather_k_prior"].view(-1), torch.tensor([0.5, 0.8]))
