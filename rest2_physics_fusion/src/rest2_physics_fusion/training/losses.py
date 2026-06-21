from __future__ import annotations

import torch
import torch.nn.functional as F


def regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target)


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    mae = error.abs().mean().item()
    rmse = torch.sqrt(torch.mean(error * error)).item()
    return {"mae": float(mae), "rmse": float(rmse)}
