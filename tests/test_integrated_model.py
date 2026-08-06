from __future__ import annotations
import torch
from pv_physics_moe.config import ModelConfig
from pv_physics_moe.models.integrated import IntegratedPVPhysicsMoE
from pv_physics_moe.physics import Rest2FeatureBuilder, project_to_feasible_set


def small_config(target_mode: str = "power") -> ModelConfig:
    return ModelConfig(
        serial_input_dim=5, hidden_size=32, serial_layers=1, fusion_heads=4,
        moe_layers=1, moe_heads=4, num_experts=4, num_shared_experts=1,
        top_k_experts=2, pcd_hidden_size=16, pcd_layers=1,
        forecast_horizon=2, target_mode=target_mode, dropout=0.0,
        raw_state_residual_scale=100.0,
    )


def small_image_config() -> ModelConfig:
    config = small_config()
    config.use_image = True
    config.image_channels = 3
    config.image_size = 16
    config.image_cnn_width = 4
    config.image_temporal_layers = 1
    config.image_temporal_heads = 4
    config.validate()
    return config


def make_batch() -> dict:
    mu0 = torch.tensor([[0.8, 0.0], [0.35, 0.6]])
    return {
        "serial": torch.randn(2, 4, 5),
        "rest2_inputs": {
            "mu0": mu0,
            "input_ghi": torch.tensor([[600.0, 0.0], [180.0, 420.0]]),
            "pressure_pa": torch.full((2, 2), 101325.0),
            "pwv_cm": torch.full((2, 2), 1.5),
            "aod700": torch.full((2, 2), 0.08),
        },
        "future_cos_zenith": mu0,
    }


def test_rest2_feature_contract_and_clear_sky_closure():
    features = Rest2FeatureBuilder()(make_batch()["rest2_inputs"])
    assert features.shape == (2, 2, 26)
    ghi, dni, dhi = features[..., 0], features[..., 1], features[..., 2]
    mu0 = features[..., 3]
    assert torch.allclose(ghi, mu0 * dni + dhi, atol=1e-4)


def test_integrated_two_modal_shapes_attention_and_pcd_closure():
    model = IntegratedPVPhysicsMoE(small_config())
    assert model.image_encoder is None
    outputs = model(make_batch())
    assert outputs["prediction"].shape == (2, 2, 1)
    assert outputs["irradiance"].shape == (2, 2, 3)
    assert set(outputs["fusion_output"].modality_embeds) == {"serial", "physics"}
    assert set(outputs["fusion_output"].attention_maps) == {"serial_from_physics", "physics_from_serial"}
    ghi, dni, dhi = outputs["irradiance"].unbind(-1)
    cosz = outputs["future_cos_zenith"].clamp_min(0.0)
    assert torch.allclose(ghi, cosz * dni + dhi, atol=1e-4)
    assert torch.all(outputs["irradiance"] >= 0.0)
    assert torch.equal(outputs["irradiance"][0, 1], torch.zeros(3))
    assert outputs["power_prediction"][0, 1, 0] == 0.0


def test_optional_image_modality_supports_partial_and_fully_missing_samples():
    model = IntegratedPVPhysicsMoE(small_image_config())
    batch = make_batch()
    batch["images"] = torch.rand(2, 4, 3, 16, 16)
    batch["image_valid_mask"] = torch.tensor([[1, 1, 0, 1], [0, 0, 0, 0]])
    batch["image_time_offsets"] = torch.tensor([[-15.0, -10.0, -5.0, 0.0]]).expand(2, -1)
    outputs = model(batch)
    assert outputs["prediction"].shape == (2, 2, 1)
    assert set(outputs["fusion_output"].modality_embeds) == {"serial", "physics", "image"}
    assert torch.count_nonzero(outputs["fusion_output"].modality_embeds["image"][1]) == 0
    assert "serial_from_image" in outputs["fusion_output"].attention_maps
    assert torch.isfinite(outputs["prediction"]).all()


def test_enabled_image_modality_accepts_batch_without_image_tensor():
    model = IntegratedPVPhysicsMoE(small_image_config())
    outputs = model(make_batch())
    image_tokens = outputs["fusion_output"].modality_embeds["image"]
    assert image_tokens.shape[:2] == (2, 1)
    assert torch.count_nonzero(image_tokens) == 0
    assert torch.isfinite(outputs["prediction"]).all()


def test_end_to_end_backward_reaches_rest2_and_moe():
    model = IntegratedPVPhysicsMoE(small_config())
    batch = make_batch()
    batch["rest2_inputs"]["aod700"] = batch["rest2_inputs"]["aod700"].clone().requires_grad_(True)
    outputs = model(batch)
    loss = outputs["prediction"].mean() + 0.01 * outputs["moe_balance_loss"]
    loss.backward()
    assert batch["rest2_inputs"]["aod700"].grad is not None
    assert model.moe.layers[0].gate.weight.grad is not None
    assert model.pcd.blocks[0].state_embed.weight.grad is not None


def test_projection_operates_in_physical_units():
    raw = torch.tensor([[[500.0, 900.0, -20.0]]])
    projected = project_to_feasible_set(raw, torch.tensor([[0.5]]))
    ghi, dni, dhi = projected.unbind(-1)
    assert torch.allclose(ghi, 0.5 * dni + dhi, atol=1e-5)
    assert torch.all(projected >= 0.0)

