from __future__ import annotations

import torch
from torch import nn

from rest2_physics_fusion.physics.torch_clear_sky import rest2_torch_clear_sky


class Rest2PhysicsCalibrator(nn.Module):
    """Optional differentiable REST2 recalibration for the physics vector.

    The module recomputes clear-sky irradiance from raw physical inputs, then
    writes normalized REST2 values back into the existing physics feature vector.
    It keeps the CSV schema unchanged and can be disabled without changing the
    rest of the model.
    """

    REQUIRED_COLUMNS = (
        "ghi_clear_target",
        "dni_clear_target",
        "dhi_clear_target",
        "mu0_target",
        "dni_extra",
        "pressure_pa",
        "pwv_cm",
        "aod700",
        "t_clr_dd",
    )
    OPTIONAL_GATE_COLUMNS = (
        "clear_sky_index_current",
        "weather_attenuation_prior",
    )

    def __init__(
        self,
        physics_feature_columns: tuple[str, ...],
        physics_norm: dict[str, list[float]] | dict[str, torch.Tensor],
        target_column: str = "target_ghi_5min",
    ) -> None:
        super().__init__()
        missing = [column for column in self.REQUIRED_COLUMNS if column not in physics_feature_columns]
        if missing:
            raise ValueError(f"REST2 calibration requires physics columns: {missing}")

        self.indices = {column: physics_feature_columns.index(column) for column in self.REQUIRED_COLUMNS}
        self.physics_feature_columns = tuple(physics_feature_columns)
        self.target_column = target_column
        mean = torch.as_tensor(physics_norm["mean"], dtype=torch.float32)
        std = torch.as_tensor(physics_norm["std"], dtype=torch.float32).clamp_min(1e-6)
        self.register_buffer("physics_mean", mean)
        self.register_buffer("physics_std", std)
        self.log_pwv_scale = nn.Parameter(torch.zeros(()))
        self.log_aod_scale = nn.Parameter(torch.zeros(()))
        self.log_pressure_scale = nn.Parameter(torch.zeros(()))
        self.blend_logit = nn.Parameter(torch.tensor(2.0))
        # Gate input: normalized horizon clear-sky, horizon mu0, current sky index, weather attenuation.
        self.gate = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        self._initialize_gate()

    def _norm_value(self, column: str, value: torch.Tensor) -> torch.Tensor:
        idx = self.indices[column]
        return (value - self.physics_mean[idx]) / self.physics_std[idx]

    def _feature_index(self, column: str) -> int | None:
        if column not in self.physics_feature_columns:
            return None
        return self.physics_feature_columns.index(column)

    @staticmethod
    def _horizon_suffix(target_column: str) -> str:
        mapping = {
            "target_ghi_5min": "5min",
            "target_ghi_4h": "4h",
            "target_ghi_1d": "1d",
        }
        return mapping.get(target_column, "")

    def _horizon_column(self, prefix: str, fallback: str) -> str:
        suffix = self._horizon_suffix(self.target_column)
        candidate = f"{prefix}_{suffix}" if suffix else fallback
        return candidate if candidate in self.physics_feature_columns else fallback

    def _initialize_gate(self) -> None:
        first = self.gate[0]
        last = self.gate[2]
        nn.init.zeros_(first.weight)
        nn.init.zeros_(first.bias)
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 0.0)

    def _raw_column(self, physics_raw: torch.Tensor, column: str, default: float = 0.0) -> torch.Tensor:
        index = self._feature_index(column)
        if index is None:
            return torch.full_like(physics_raw[:, 0], default)
        return physics_raw[:, index]

    def gate_value(self, physics_raw: torch.Tensor) -> torch.Tensor:
        clear_column = self._horizon_column("ghi_clear_horizon", "ghi_clear_target")
        mu0_column = self._horizon_column("mu0_horizon", "mu0_target")
        clear_sky = self._raw_column(physics_raw, clear_column).clamp(min=0.0)
        mu0 = self._raw_column(physics_raw, mu0_column).clamp(min=0.0, max=1.0)
        sky_index = self._raw_column(physics_raw, "clear_sky_index_current").clamp(min=0.0, max=2.0)
        attenuation = self._raw_column(physics_raw, "weather_attenuation_prior", default=1.0).clamp(min=0.0, max=1.2)
        inputs = torch.stack(
            [
                torch.log1p(clear_sky) / 7.0,
                mu0,
                sky_index / 2.0,
                attenuation,
            ],
            dim=-1,
        )
        heuristic = (
            1.5 * (torch.log1p(clear_sky) / 7.0)
            + 1.2 * mu0
            + 1.0 * (sky_index / 2.0)
            + 0.6 * attenuation
            - 2.1
        ).unsqueeze(-1)
        learned = self.gate(inputs)
        return torch.sigmoid(heuristic + learned)

    def forward(self, physics: torch.Tensor, physics_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calibrated = physics.clone()
        mu0 = physics_raw[:, self.indices["mu0_target"]]
        dni_extra = physics_raw[:, self.indices["dni_extra"]]
        pressure_pa = physics_raw[:, self.indices["pressure_pa"]] * torch.exp(self.log_pressure_scale)
        pwv_cm = physics_raw[:, self.indices["pwv_cm"]] * torch.exp(self.log_pwv_scale)
        aod700 = physics_raw[:, self.indices["aod700"]] * torch.exp(self.log_aod_scale)

        rest2 = rest2_torch_clear_sky(
            mu0=mu0,
            pressure_pa=pressure_pa,
            pwv_cm=pwv_cm,
            aod700=aod700,
            dni_extra=dni_extra,
        )
        rest2_values = {
            "ghi_clear_target": rest2["ghi_clear_target"],
            "dni_clear_target": rest2["dni_clear_target"],
            "dhi_clear_target": rest2["dhi_clear_target"],
            "t_clr_dd": rest2["t_direct"],
        }
        global_blend = torch.sigmoid(self.blend_logit)
        rest2_gate = self.gate_value(physics_raw)
        blend = global_blend * rest2_gate
        for column, raw_value in rest2_values.items():
            idx = self.indices[column]
            normalized = self._norm_value(column, raw_value)
            calibrated[:, idx] = (1.0 - blend[:, 0]) * physics[:, idx] + blend[:, 0] * normalized
        return calibrated, rest2_gate, blend
