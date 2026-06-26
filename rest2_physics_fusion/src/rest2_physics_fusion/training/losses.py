from __future__ import annotations

import torch
import torch.nn.functional as F


def regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target)


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    y_max: float | None = None,
) -> dict[str, float]:
    error = prediction - target
    mae = error.abs().mean().item()
    rmse = torch.sqrt(torch.mean(error * error)).item()
    if y_max is None:
        y_max = float(torch.max(target).detach().cpu())
    y_max = float(y_max)
    if y_max <= 0.0:
        nmae = float("nan")
        nrmse = float("nan")
    else:
        nmae = float(mae / y_max)
        nrmse = float(rmse / y_max)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "nmae": nmae,
        "nrmse": nrmse,
        "y_max": y_max,
    }
