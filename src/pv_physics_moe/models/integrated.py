from __future__ import annotations
from typing import Any, Mapping
import torch
from torch import nn

from pv_physics_moe.config import IntegratedConfig, ModelConfig
from pv_physics_moe.physics import Rest2FeatureBuilder, SolarPhysicsStack
from .encoders import PhysicsTokenEncoder, SerialEncoder, SkyImageEncoder
from .fusion import PhysicsAwareCrossModalFusion
from .moe import PVMMoEDecoder


class IntegratedPVPhysicsMoE(nn.Module):
    """Two-modal station PVMMOE: serial + REST2 physics, followed by PCD."""
    def __init__(self, config: IntegratedConfig | ModelConfig) -> None:
        super().__init__()
        cfg = config.model if isinstance(config, IntegratedConfig) else config
        cfg.validate()
        self.config = cfg
        d, h = cfg.hidden_size, cfg.forecast_horizon
        self.rest2 = Rest2FeatureBuilder()
        self.serial_encoder = SerialEncoder(cfg.serial_input_dim, d, cfg.serial_layers, cfg.moe_heads, cfg.dropout)
        self.physics_encoder = PhysicsTokenEncoder(cfg.physics_input_dim, d, cfg.dropout)
        self.image_encoder = (
            SkyImageEncoder(
                cfg.image_channels,
                d,
                cfg.image_cnn_width,
                cfg.image_temporal_layers,
                cfg.image_temporal_heads,
                cfg.dropout,
                cfg.image_time_scale_minutes,
            )
            if cfg.use_image else None
        )
        input_dims = {"serial": d, "physics": d}
        modality_order = ["serial", "physics"]
        if cfg.use_image:
            input_dims["image"] = d
            modality_order.append("image")
        self.fusion = PhysicsAwareCrossModalFusion(
            input_dims, d, cfg.fusion_heads, cfg.dropout,
            modality_order=tuple(modality_order),
        )
        self.moe = PVMMoEDecoder(d, cfg.moe_layers, cfg.moe_heads, cfg.num_experts, cfg.num_shared_experts, cfg.top_k_experts, cfg.dropout)
        self.history_skip = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.LayerNorm(d))
        self.raw_state_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, h * 3))
        self.pcd_context_head = nn.Sequential(nn.Linear(d, h * cfg.pcd_hidden_size), nn.GELU())
        self.pcd = SolarPhysicsStack(cfg.pcd_hidden_size, cfg.pcd_hidden_size, cfg.pcd_layers, cfg.dropout, cfg.night_cos_threshold)
        self.power_head = nn.Sequential(
            nn.Linear(cfg.pcd_hidden_size + 3, cfg.pcd_hidden_size), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.pcd_hidden_size, 1), nn.Softplus(),
        )

    @staticmethod
    def _ensure_horizon(x: torch.Tensor, horizon: int, name: str) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"{name} must be [B,F] or [B,H,F]")
        if x.size(1) == 1 and horizon > 1:
            x = x.expand(-1, horizon, -1)
        if x.size(1) != horizon:
            raise ValueError(f"{name} horizon {x.size(1)} != configured horizon {horizon}")
        return x

    def _physics_inputs(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "physics" in batch:
            physics = batch["physics"].float()
            physics_raw = batch.get("physics_raw", physics).float()
        elif "rest2_inputs" in batch:
            physics_raw = self.rest2(batch["rest2_inputs"])
            physics = physics_raw
        else:
            raise KeyError("batch requires either 'physics' or 'rest2_inputs'")
        horizon = self.config.forecast_horizon
        return self._ensure_horizon(physics, horizon, "physics"), self._ensure_horizon(physics_raw, horizon, "physics_raw")

    def _future_cosine(self, batch: Mapping[str, Any], reference: torch.Tensor) -> torch.Tensor:
        if "future_cos_zenith" in batch:
            cosine = batch["future_cos_zenith"].to(reference).clamp(-1.0, 1.0)
        elif "future_zenith" in batch:
            cosine = torch.cos(torch.deg2rad(batch["future_zenith"].to(reference)))
        else:
            raise KeyError("batch requires 'future_zenith' in degrees or 'future_cos_zenith'")
        if cosine.ndim == 1:
            cosine = cosine[:, None]
        if cosine.size(1) == 1 and self.config.forecast_horizon > 1:
            cosine = cosine.expand(-1, self.config.forecast_horizon)
        if cosine.shape != (reference.size(0), self.config.forecast_horizon):
            raise ValueError("future zenith shape must be [B,forecast_horizon]")
        return cosine

    @staticmethod
    def _history_summary(serial: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # The station schema keeps observed GHI as serial feature 0.
        signal = serial[..., 0]
        weights = valid_mask.to(signal.dtype)
        mean = (signal * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        last_index = (valid_mask.sum(1) - 1).clamp_min(0)
        last = signal[torch.arange(signal.size(0), device=signal.device), last_index]
        first_index = valid_mask.float().argmax(1)
        first = signal[torch.arange(signal.size(0), device=signal.device), first_index]
        return torch.stack((last, mean, last - first), -1)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        serial = torch.nan_to_num(batch["serial"].float())
        serial_mask = batch.get("serial_valid_mask")
        serial_tokens, serial_mask = self.serial_encoder(serial, serial_mask)
        physics, physics_raw = self._physics_inputs(batch)
        physics_tokens, physics_mask = self.physics_encoder(physics, batch.get("physics_valid_mask"))
        modalities = {
            "serial": (serial_tokens, serial_mask),
            "physics": (physics_tokens, physics_mask),
        }
        if self.image_encoder is not None:
            images = batch.get("images")
            if images is None:
                images = serial.new_zeros(
                    (serial.size(0), 1, self.config.image_channels, self.config.image_size, self.config.image_size)
                )
                image_mask = torch.zeros((serial.size(0), 1), dtype=torch.long, device=serial.device)
                image_offsets = torch.zeros((serial.size(0), 1), dtype=serial.dtype, device=serial.device)
            else:
                images = images.to(serial.device)
                image_mask = batch.get("image_valid_mask")
                image_offsets = batch.get("image_time_offsets")
            image_tokens, image_mask = self.image_encoder(images, image_mask, image_offsets)
            modalities["image"] = (image_tokens, image_mask)
        fusion = self.fusion(modalities)
        moe = self.moe(fusion.prediction_embeds, fusion.prediction_mask)
        pooled = moe.pooled + self.history_skip(self._history_summary(serial, serial_mask))
        b, horizon = serial.size(0), self.config.forecast_horizon
        residual = torch.tanh(self.raw_state_head(pooled).reshape(b, horizon, 3)) * self.config.raw_state_residual_scale
        clear_sky_prior = physics_raw[..., :3]
        raw_state = clear_sky_prior + residual
        context = self.pcd_context_head(pooled).reshape(b, horizon, self.config.pcd_hidden_size)
        cos_zenith = self._future_cosine(batch, raw_state)
        irradiance, pcd_hidden = self.pcd(context, raw_state, cos_zenith)
        power = self.power_head(torch.cat((irradiance, pcd_hidden), -1))
        daylight = (cos_zenith > self.config.night_cos_threshold).unsqueeze(-1)
        power = torch.where(daylight, power, torch.zeros_like(power))
        prediction = power if self.config.target_mode == "power" else irradiance
        return {
            "prediction": prediction,
            "power_prediction": power,
            "irradiance": irradiance,
            "raw_irradiance": raw_state,
            "clear_sky_prior": clear_sky_prior,
            "future_cos_zenith": cos_zenith,
            "physics_features": physics,
            "fusion_output": fusion,
            "moe_hidden_states": moe.hidden_states,
            "moe_balance_loss": moe.balance_loss,
            "router_data": moe.router_data,
        }

