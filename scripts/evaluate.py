from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pv_physics_moe import IntegratedPVPhysicsMoE, load_config
from pv_physics_moe.data import NpzSolarDataset, collate_solar_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config.training.device)
    model = IntegratedPVPhysicsMoE(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    loader = DataLoader(NpzSolarDataset(args.data), batch_size=config.training.batch_size, collate_fn=collate_solar_batch)
    squared, absolute, count, closure_max = 0.0, 0.0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch)
            target = batch["target"]
            error = outputs["prediction"] - target
            squared += float(error.square().sum())
            absolute += float(error.abs().sum())
            count += error.numel()
            ghi, dni, dhi = outputs["irradiance"].unbind(-1)
            closure = (ghi - outputs["future_cos_zenith"].clamp_min(0.0) * dni - dhi).abs().max()
            closure_max = max(closure_max, float(closure))
    print(f"rmse={(squared / count) ** 0.5:.6f} mae={absolute / count:.6f} closure_max={closure_max:.8f}")


if __name__ == "__main__":
    main()

