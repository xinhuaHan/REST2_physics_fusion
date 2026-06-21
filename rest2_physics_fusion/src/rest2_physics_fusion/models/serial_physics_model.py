from __future__ import annotations

import torch
from torch import nn

from rest2_physics_fusion.models.fusion import GatedSerialPhysicsFusion
from rest2_physics_fusion.models.physics_encoder import PhysicsEncoder
from rest2_physics_fusion.models.rest2_calibrator import Rest2PhysicsCalibrator


class SerialPhysicsForecaster(nn.Module):
    def __init__(
        self,
        serial_input_dim: int,
        physics_input_dim: int,
        weather_input_dim: int | None = None,
        serial_hidden: int = 64,
        weather_hidden: int = 32,
        physics_hidden: int = 64,
        fusion_hidden: int = 128,
        dropout: float = 0.1,
        use_rest2_calibration: bool = False,
        use_clear_sky_weather_head: bool = False,
        use_weather_prior_fusion: bool = False,
        use_clear_sky_power_prior: bool = False,
        weather_prior_weight_max: float = 1.0,
        sky_index_max: float = 2.0,
        target_column: str = "target_ghi_5min",
        physics_feature_columns: tuple[str, ...] | None = None,
        physics_norm: dict[str, list[float]] | None = None,
    ) -> None:
        super().__init__()
        self.use_clear_sky_weather_head = use_clear_sky_weather_head
        self.use_weather_prior_fusion = use_weather_prior_fusion
        self.use_clear_sky_power_prior = use_clear_sky_power_prior
        self.weather_prior_weight_max = float(weather_prior_weight_max)
        self.sky_index_max = float(sky_index_max)
        self.target_column = target_column
        self.physics_feature_columns = tuple(physics_feature_columns or ())
        self.clear_sky_index = self._feature_index(self._horizon_clear_sky_column(target_column))
        if self.clear_sky_index is None:
            self.clear_sky_index = self._feature_index("ghi_clear_target")
        self.weather_adjusted_clear_sky_index = self._feature_index(
            self._horizon_weather_adjusted_clear_sky_column(target_column)
        )
        self.weather_attenuation_index = self._feature_index("weather_attenuation_prior")
        self.serial_encoder = nn.GRU(
            input_size=serial_input_dim,
            hidden_size=serial_hidden,
            batch_first=True,
        )
        self.weather_encoder = None
        fusion_serial_dim = serial_hidden
        if weather_input_dim is not None and weather_input_dim > 0:
            self.weather_encoder = nn.GRU(
                input_size=weather_input_dim,
                hidden_size=weather_hidden,
                batch_first=True,
            )
            fusion_serial_dim += weather_hidden
        self.rest2_calibrator = None
        if use_rest2_calibration:
            if physics_feature_columns is None or physics_norm is None:
                raise ValueError("REST2 calibration requires physics_feature_columns and physics_norm")
            self.rest2_calibrator = Rest2PhysicsCalibrator(
                tuple(physics_feature_columns),
                physics_norm,
                target_column=target_column,
            )
        self.physics_encoder = PhysicsEncoder(physics_input_dim, physics_hidden, dropout)
        self.fusion = GatedSerialPhysicsFusion(fusion_serial_dim, physics_hidden, fusion_hidden, dropout)
        self.head = nn.Linear(fusion_hidden, 1)
        self.k_index_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Linear(fusion_hidden // 2, 1),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Linear(fusion_hidden // 2, 1),
        )
        self.weather_k_delta_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Linear(fusion_hidden // 2, 1),
            nn.Tanh(),
        )
        self.prior_weight_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Linear(fusion_hidden // 2, 1),
            nn.Sigmoid(),
        )
        self.log_power_scale = nn.Parameter(torch.zeros(1))
        self.power_bias = nn.Parameter(torch.zeros(1))
        self._initialize_weather_prior_heads()

    def _initialize_weather_prior_heads(self) -> None:
        nn.init.zeros_(self.weather_k_delta_head[2].weight)
        nn.init.zeros_(self.weather_k_delta_head[2].bias)
        nn.init.zeros_(self.prior_weight_head[2].weight)
        nn.init.constant_(self.prior_weight_head[2].bias, -2.0)

    def _feature_index(self, column: str) -> int | None:
        if not self.physics_feature_columns:
            return None
        if column not in self.physics_feature_columns:
            return None
        return self.physics_feature_columns.index(column)

    @staticmethod
    def _horizon_clear_sky_column(target_column: str) -> str:
        mapping = {
            "target_ghi_5min": "ghi_clear_horizon_5min",
            "target_ghi_4h": "ghi_clear_horizon_4h",
            "target_ghi_1d": "ghi_clear_horizon_1d",
        }
        return mapping.get(target_column, "ghi_clear_target")

    @staticmethod
    def _horizon_weather_adjusted_clear_sky_column(target_column: str) -> str:
        mapping = {
            "target_ghi_5min": "weather_adjusted_clear_sky_ghi_horizon_5min",
            "target_ghi_4h": "weather_adjusted_clear_sky_ghi_horizon_4h",
            "target_ghi_1d": "weather_adjusted_clear_sky_ghi_horizon_1d",
        }
        return mapping.get(target_column, "weather_adjusted_clear_sky_ghi")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        serial = torch.nan_to_num(batch["serial"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        physics = torch.nan_to_num(batch["physics"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        physics_raw = torch.nan_to_num(batch.get("physics_raw", physics).float(), nan=0.0, posinf=0.0, neginf=0.0)
        rest2_gate = None
        rest2_effective_blend = None
        if self.rest2_calibrator is not None:
            physics, rest2_gate, rest2_effective_blend = self.rest2_calibrator(physics, physics_raw)
        _, hidden = self.serial_encoder(serial)
        serial_embedding = hidden[-1]
        if self.weather_encoder is not None and "weather" in batch:
            weather = torch.nan_to_num(batch["weather"].float(), nan=0.0, posinf=0.0, neginf=0.0)
            _, weather_hidden = self.weather_encoder(weather)
            serial_embedding = torch.cat([serial_embedding, weather_hidden[-1]], dim=-1)
        physics_embedding = self.physics_encoder(physics)
        fused = self.fusion(serial_embedding, physics_embedding)
        if not self.use_clear_sky_weather_head:
            pred = self.head(fused)
            outputs = {"prediction": pred, "fused": fused, "baseline_prediction": pred}
            if rest2_gate is not None and rest2_effective_blend is not None:
                outputs["rest2_gate"] = rest2_gate
                outputs["rest2_effective_blend"] = rest2_effective_blend
            return outputs

        if self.clear_sky_index is None:
            raise ValueError("clear-sky weather head requires ghi_clear_target in physics_feature_columns")
        clear_sky_base = physics_raw[:, self.clear_sky_index : self.clear_sky_index + 1].clamp(min=0.0)
        weather_adjusted_prior = None
        if self.weather_adjusted_clear_sky_index is not None:
            weather_adjusted_prior = physics_raw[
                :, self.weather_adjusted_clear_sky_index : self.weather_adjusted_clear_sky_index + 1
            ].clamp(min=0.0)
        if weather_adjusted_prior is None:
            weather_adjusted_prior = clear_sky_base
        attenuation_prior = torch.ones_like(clear_sky_base)
        if self.weather_attenuation_index is not None:
            attenuation_prior = physics_raw[
                :, self.weather_attenuation_index : self.weather_attenuation_index + 1
            ].clamp(min=0.0, max=self.sky_index_max)
        attenuation_from_adjusted = weather_adjusted_prior / clear_sky_base.clamp(min=1.0)
        weather_k_prior = torch.where(
            clear_sky_base > 1.0,
            attenuation_from_adjusted,
            attenuation_prior,
        ).clamp(min=0.0, max=self.sky_index_max)

        if self.use_clear_sky_power_prior:
            power_scale = torch.exp(self.log_power_scale).clamp(min=0.0, max=10.0)
            clear_sky_power_prior = (clear_sky_base * power_scale + self.power_bias).clamp(min=0.0)
            residual_pred = self.residual_head(fused)
            pred = (clear_sky_power_prior + residual_pred).clamp(min=0.0)
            outputs = {
                "prediction": pred,
                "fused": fused,
                "clear_sky_base": clear_sky_base,
                "clear_sky_ghi": clear_sky_base,
                "clear_sky_power_prior": clear_sky_power_prior,
                "power_scale": power_scale.expand_as(pred),
                "power_bias": self.power_bias.expand_as(pred),
                "residual_pred": residual_pred,
                "baseline_prediction": pred,
                "weather_k_prior": weather_k_prior,
                "weather_adjusted_prior": weather_adjusted_prior,
            }
            if rest2_gate is not None and rest2_effective_blend is not None:
                outputs["rest2_gate"] = rest2_gate
                outputs["rest2_effective_blend"] = rest2_effective_blend
            return outputs

        if self.use_weather_prior_fusion:
            k_pred = self.k_index_head(fused) * self.sky_index_max
            residual_pred = self.residual_head(fused)
            serial_pred = (clear_sky_base * k_pred + residual_pred).clamp(min=0.0)
            k_weather = (weather_k_prior + 0.5 * self.weather_k_delta_head(fused)).clamp(
                min=0.0,
                max=self.sky_index_max,
            )
            prior_pred = clear_sky_base * k_weather
            prior_weight_raw = self.prior_weight_head(fused)
            prior_weight = (prior_weight_raw * self.weather_prior_weight_max).clamp(min=0.0, max=1.0)
            pred = (prior_weight * prior_pred + (1.0 - prior_weight) * serial_pred).clamp(min=0.0)
            outputs = {
                "prediction": pred,
                "fused": fused,
                "clear_sky_base": clear_sky_base,
                "clear_sky_ghi": clear_sky_base,
                "k_pred": k_pred,
                "residual_pred": residual_pred,
                "serial_pred": serial_pred,
                "baseline_prediction": serial_pred,
                "prior_pred": prior_pred,
                "weather_prior": prior_pred,
                "k_weather": k_weather,
                "weather_k_prior": weather_k_prior,
                "weather_adjusted_prior": weather_adjusted_prior,
                "prior_weight": prior_weight,
                "prior_weight_raw": prior_weight_raw,
            }
            if rest2_gate is not None and rest2_effective_blend is not None:
                outputs["rest2_gate"] = rest2_gate
                outputs["rest2_effective_blend"] = rest2_effective_blend
            return outputs

        k_pred = self.k_index_head(fused) * self.sky_index_max
        residual_pred = self.residual_head(fused)
        pred = (clear_sky_base * k_pred + residual_pred).clamp(min=0.0)
        outputs = {
            "prediction": pred,
            "fused": fused,
            "clear_sky_base": clear_sky_base,
            "clear_sky_ghi": clear_sky_base,
            "k_pred": k_pred,
            "residual_pred": residual_pred,
            "baseline_prediction": pred,
        }
        if rest2_gate is not None and rest2_effective_blend is not None:
            outputs["rest2_gate"] = rest2_gate
            outputs["rest2_effective_blend"] = rest2_effective_blend
        return outputs
