from __future__ import annotations
from typing import Mapping
import torch
from torch import nn
from pv_physics_moe.config import TrainingConfig


class IntegratedLoss(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Regression targets are in physical MW/W m^-2 units.  Squaring these
        # values in FP16 can overflow (max finite value is about 65504), so every
        # reduction intentionally uses FP32 even when the model forward uses AMP.
        prediction = outputs["prediction"].float()
        target = batch["target"].to(device=prediction.device, dtype=torch.float32)
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        task_mask = batch.get("target_valid_mask")
        if task_mask is None:
            task_mask = torch.ones_like(target[..., 0])
        task_mask = task_mask.to(device=prediction.device, dtype=torch.float32).unsqueeze(-1)
        task_denominator = task_mask.sum().clamp_min(1.0)
        task = (((prediction - target) ** 2) * task_mask).sum() / task_denominator
        irradiance_prediction = outputs["irradiance"].float()
        irradiance = irradiance_prediction.sum() * 0.0
        if "irradiance_target" in batch:
            irr_target = batch["irradiance_target"].to(
                device=irradiance_prediction.device, dtype=torch.float32
            )
            irr_mask = batch.get("irradiance_target_mask")
            if irr_mask is None:
                irr_mask = torch.ones_like(irr_target)
            irr_mask = irr_mask.to(device=irradiance_prediction.device, dtype=torch.float32)
            valid = irr_mask.sum()
            irradiance = torch.where(
                valid > 0,
                (((irradiance_prediction - irr_target) ** 2) * irr_mask).sum() / valid.clamp_min(1.0),
                irradiance_prediction.sum() * 0.0,
            )
        raw_correction = torch.mean(
            (outputs["raw_irradiance"].float() - outputs["clear_sky_prior"].float()) ** 2
        )
        total = (
            self.config.task_loss_weight * task
            + self.config.irradiance_loss_weight * irradiance
            + self.config.moe_balance_weight * outputs["moe_balance_loss"].float()
            + self.config.raw_correction_weight * raw_correction
        )
        return {"loss": total, "task_loss": task, "irradiance_loss": irradiance, "moe_balance_loss": outputs["moe_balance_loss"], "raw_correction_loss": raw_correction}

