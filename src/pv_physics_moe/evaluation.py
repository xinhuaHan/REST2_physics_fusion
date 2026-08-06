from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .parquet_config import EvaluationConfig


PREDICTION_COLUMNS = ["issue_time", "target_time", "horizon_minutes", "y_true", "y_pred"]


def prediction_records(batch: dict[str, Any], prediction: torch.Tensor) -> list[dict[str, Any]]:
    pred = prediction.detach().float().cpu().numpy()
    truth = batch["target"].detach().float().cpu().numpy()
    current = batch["current_power"].detach().float().cpu().numpy()
    records: list[dict[str, Any]] = []
    for row, issue_text in enumerate(batch["issue_time"]):
        issue = pd.Timestamp(issue_text)
        for step, target_text in enumerate(batch["target_times"][row]):
            target_time = pd.Timestamp(target_text)
            horizon = int(round((target_time - issue).total_seconds() / 60.0))
            records.append({
                "issue_time": issue.isoformat(),
                "target_time": target_time.isoformat(),
                "horizon_minutes": horizon,
                "y_true": float(truth[row, step, 0]),
                "y_pred": float(pred[row, step, 0]),
                "current_power": float(current[row, 0]),
            })
    return records


def finalize_prediction_frame(records: Iterable[dict[str, Any]], expected_steps: int) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("evaluation produced no prediction records")
    frame = frame.drop_duplicates(["issue_time", "target_time"], keep="first")
    frame = frame.sort_values(["issue_time", "target_time"]).reset_index(drop=True)
    counts = frame.groupby("issue_time").size()
    if not (counts == expected_steps).all():
        raise ValueError(f"each issue_time must have {expected_steps} rows; got {counts.to_dict()}")
    if int(frame["horizon_minutes"].max()) != 240:
        raise ValueError("the final exported target must be horizon_minutes=240")
    return frame


def _metric_block(frame: pd.DataFrame, denominator: float) -> dict[str, Any]:
    count = len(frame)
    if count == 0:
        return {"count": 0, "MSE": None, "RMSE": None, "MAE": None, "NRMSE": None, "NMAE": None}
    error = frame["y_pred"].to_numpy(float) - frame["y_true"].to_numpy(float)
    mse = float(np.mean(error ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(error)))
    return {
        "count": count, "MSE": mse, "RMSE": rmse, "MAE": mae,
        "NRMSE": rmse / denominator, "NMAE": mae / denominator,
    }


def official_metrics(frame: pd.DataFrame, config: EvaluationConfig) -> dict[str, Any]:
    denominator = config.nrmse_denominator
    if denominator is None:
        raise ValueError("official evaluation requires evaluation.nrmse_denominator; fill the verified capacity")
    result: dict[str, Any] = {"denominator": float(denominator), "horizons_minutes": {}}
    for horizon in config.horizons_minutes:
        selected = frame.loc[frame["horizon_minutes"] == horizon].copy()
        if selected.empty:
            raise ValueError(
                f"official horizon_minutes={horizon} is absent; evaluation selects by time, not array index"
            )
        overall = _metric_block(selected, float(denominator))
        delta = np.abs(selected["y_true"].to_numpy(float) - selected["current_power"].to_numpy(float))
        hard = selected.loc[delta >= config.hard_delta_fraction * float(denominator)].copy()
        hard_block = _metric_block(hard, float(denominator))
        hard_block["fraction"] = len(hard) / len(selected)
        directional_delta = hard["y_true"].to_numpy(float) - hard["current_power"].to_numpy(float)
        directional = np.abs(directional_delta) >= config.direction_min_delta_fraction * float(denominator)
        if directional.any():
            predicted_delta = hard["y_pred"].to_numpy(float) - hard["current_power"].to_numpy(float)
            hard_block["direction_accuracy"] = float(
                np.mean(np.sign(predicted_delta[directional]) == np.sign(directional_delta[directional]))
            )
        else:
            hard_block["direction_accuracy"] = None
        result["horizons_minutes"][str(horizon)] = {"overall": overall, "hard_delta": hard_block}
    return result


def pcd_diagnostics(
    irradiance: torch.Tensor, future_cos_zenith: torch.Tensor, irradiance_target: torch.Tensor | None = None,
    irradiance_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    values = irradiance.detach().float().cpu()
    cosine = future_cos_zenith.detach().float().cpu().clamp_min(0.0)
    ghi, dni, dhi = values.unbind(-1)
    result: dict[str, Any] = {
        "max_closure_error": float((ghi - cosine * dni - dhi).abs().max()),
        "negative_counts": {
            "ghi": int((ghi < 0).sum()), "dni": int((dni < 0).sum()), "dhi": int((dhi < 0).sum()),
        },
        "night_nonzero_count": int((values[cosine <= 0].abs() > 1e-6).sum()),
    }
    if irradiance_target is not None and irradiance_mask is not None:
        target = irradiance_target.detach().float().cpu()
        mask = irradiance_mask.detach().bool().cpu()
        names = ("ghi", "dni", "dhi")
        result["target_rmse"] = {}
        for index, name in enumerate(names):
            valid = mask[..., index]
            result["target_rmse"][name] = (
                float(torch.sqrt(torch.mean((values[..., index][valid] - target[..., index][valid]) ** 2)))
                if valid.any() else None
            )
    return result


def atomic_write_results(frame: pd.DataFrame, metrics: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "point_predictions.csv"
    json_path = output / "official_test_metrics.json"
    csv_tmp = output / f".{csv_path.name}.{os.getpid()}.tmp"
    json_tmp = output / f".{json_path.name}.{os.getpid()}.tmp"
    frame[PREDICTION_COLUMNS].to_csv(csv_tmp, index=False)
    with json_tmp.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(csv_tmp, csv_path)
    os.replace(json_tmp, json_path)
