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


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train serial+REST2+MoE+PCD on the five station tables.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "server_station.yaml"))
    parser.add_argument("--data", help="Override data.processed_dir")
    parser.add_argument("--smoke", action="store_true", help="One CPU epoch over four station samples")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.model.target_mode != "power":
        raise ValueError("station training currently requires model.target_mode=power")
    if config.model.serial_input_dim != 13:
        raise ValueError("the five-table station schema produces exactly 13 serial features")
    if config.model.forecast_horizon != len(config.data.forecast_lead_minutes):
        raise ValueError("model.forecast_horizon must equal len(data.forecast_lead_minutes)")
    if args.smoke:
        config.model.hidden_size = 64
        config.model.serial_layers = 1
        config.model.fusion_heads = 4
        config.model.moe_layers = 1
        config.model.moe_heads = 4
        config.model.num_experts = 4
        config.model.num_shared_experts = 1
        config.model.top_k_experts = 2
        config.model.pcd_hidden_size = 32
        config.model.pcd_layers = 1
        config.training.batch_size = 2
        config.training.epochs = 1
        config.training.num_workers = 0
        config.training.device = "cpu"
    data_root = resolve_path(args.data or config.data.processed_dir)
    dataset = StationFiveCsvDataset(data_root, "train", limit=4 if args.smoke else None)
    metadata_horizon = int(dataset.metadata["forecast_horizon"])
    if metadata_horizon != config.model.forecast_horizon:
        raise ValueError(f"processed data has {metadata_horizon} horizons but model expects {config.model.forecast_horizon}")
    with (data_root / "normalization.json").open("r", encoding="utf-8") as handle:
        normalizers = json.load(handle)
    power_scale = torch.tensor(float(normalizers["target_power"]["scale"]), dtype=torch.float32)
    irradiance_scale = torch.tensor(normalizers["target_irradiance"]["scale"], dtype=torch.float32)

    torch.manual_seed(config.project.seed)
    device = torch.device(config.training.device)
    model = IntegratedPVPhysicsMoE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers, collate_fn=collate_solar_batch, pin_memory=device.type == "cuda")
    p_scale = power_scale.to(device).view(1, 1, 1)
    i_scale = irradiance_scale.to(device).view(1, 1, 3)
    for epoch in range(config.training.epochs):
        model.train()
        totals = {"loss": 0.0, "power": 0.0, "irradiance": 0.0}
        for batch in loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            power_loss = torch.mean(((outputs["prediction"] - batch["target"]) / p_scale) ** 2)
            irradiance_loss = torch.mean(((outputs["irradiance"] - batch["irradiance_target"]) / i_scale) ** 2)
            correction = torch.mean(((outputs["raw_irradiance"] - outputs["clear_sky_prior"]) / i_scale) ** 2)
            loss = (
                config.training.task_loss_weight * power_loss
                + config.training.irradiance_loss_weight * irradiance_loss
                + config.training.moe_balance_weight * outputs["moe_balance_loss"]
                + config.training.raw_correction_weight * correction
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["power"] += float(power_loss.detach())
            totals["irradiance"] += float(irradiance_loss.detach())
        denominator = max(1, len(loader))
        print(f"epoch={epoch + 1} loss={totals['loss']/denominator:.6f} power={totals['power']/denominator:.6f} irradiance={totals['irradiance']/denominator:.6f}", flush=True)

    output_dir = resolve_path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / ("checkpoint_smoke.pt" if args.smoke else "checkpoint_last.pt")
    torch.save({"model": model.state_dict(), "config": config.to_dict(), "normalization": normalizers}, checkpoint)
    with (output_dir / "config_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
    print(f"saved={checkpoint}")


if __name__ == "__main__":
    main()
