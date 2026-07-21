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
        target = batch["target"].to(outputs["prediction"])
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        task = torch.nn.functional.mse_loss(outputs["prediction"], target)
        irradiance = torch.zeros((), device=task.device)
        if "irradiance_target" in batch:
            irr_target = batch["irradiance_target"].to(outputs["irradiance"])
            irradiance = torch.nn.functional.mse_loss(outputs["irradiance"], irr_target)
        raw_correction = torch.mean((outputs["raw_irradiance"] - outputs["clear_sky_prior"]) ** 2)
        total = (
            self.config.task_loss_weight * task
            + self.config.irradiance_loss_weight * irradiance
            + self.config.moe_balance_weight * outputs["moe_balance_loss"]
            + self.config.raw_correction_weight * raw_correction
        )
        return {"loss": total, "task_loss": task, "irradiance_loss": irradiance, "moe_balance_loss": outputs["moe_balance_loss"], "raw_correction_loss": raw_correction}

