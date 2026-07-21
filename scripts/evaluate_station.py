from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe import IntegratedPVPhysicsMoE, load_config
from pv_physics_moe.data.dataset import collate_solar_batch
from pv_physics_moe.data.station import StationFiveCsvDataset


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "server_station.yaml"))
    parser.add_argument("--data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config.training.device)
    model = IntegratedPVPhysicsMoE(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    dataset = StationFiveCsvDataset(resolve_path(args.data or config.data.processed_dir), args.split)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, num_workers=config.training.num_workers, collate_fn=collate_solar_batch)
    leads = dataset.metadata["forecast_lead_minutes"]
    power_sq_by_h = torch.zeros(len(leads), dtype=torch.float64)
    power_abs_by_h = torch.zeros(len(leads), dtype=torch.float64)
    irradiance_sq_by_h = torch.zeros((len(leads), 3), dtype=torch.float64)
    closure_max = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch)
            error = (output["prediction"] - batch["target"]).squeeze(-1).double().cpu()
            power_sq_by_h += error.square().sum(0)
            power_abs_by_h += error.abs().sum(0)
            irradiance_error = (output["irradiance"] - batch["irradiance_target"]).double().cpu()
            irradiance_sq_by_h += irradiance_error.square().sum(0)
            ghi, dni, dhi = output["irradiance"].unbind(-1)
            closure = torch.abs(ghi - output["future_cos_zenith"].clamp_min(0.0) * dni - dhi)
            closure_max = max(closure_max, float(closure.max()))
    denominator = max(1, len(dataset))
    power_rmse = torch.sqrt(power_sq_by_h / denominator)
    power_mae = power_abs_by_h / denominator
    irradiance_rmse = torch.sqrt(irradiance_sq_by_h / denominator)
    per_horizon = {
        str(lead): {
            "power_rmse": float(power_rmse[index]),
            "power_mae": float(power_mae[index]),
            "irradiance_rmse": irradiance_rmse[index].tolist(),
        }
        for index, lead in enumerate(leads)
    }
    result = {
        "split": args.split,
        "samples": len(dataset),
        "lead_minutes": leads,
        "per_horizon": per_horizon,
        "macro_power_rmse": float(power_rmse.mean()),
        "macro_power_mae": float(power_mae.mean()),
        "pcd_max_closure_error_wm2": closure_max,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
