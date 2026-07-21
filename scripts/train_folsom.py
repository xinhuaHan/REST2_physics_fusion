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
from pv_physics_moe.data.folsom import FolsomAlignedDataset


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the integrated model on aligned Folsom data.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "folsom_15min.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "folsom_15min"))
    parser.add_argument("--smoke", action="store_true", help="One tiny CPU epoch over four real samples")
    parser.add_argument("--no-images", action="store_true", help="Ablation: omit the sky-image modality")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.model.target_mode != "irradiance" or config.model.serial_input_dim != 11:
        raise ValueError("Folsom config must use target_mode=irradiance and serial_input_dim=11")
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
        config.training.device = "cpu"
    data_root = Path(args.data)
    dataset = FolsomAlignedDataset(data_root, "train", use_images=not args.no_images, limit=4 if args.smoke else None)
    with (data_root / "normalization.json").open("r", encoding="utf-8") as handle:
        target_scale = torch.tensor(json.load(handle)["target"]["scale"], dtype=torch.float32)
    torch.manual_seed(config.project.seed)
    device = torch.device(config.training.device)
    model = IntegratedPVPhysicsMoE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=0, collate_fn=collate_solar_batch)
    scale = target_scale.to(device).view(1, 1, 3)
    for epoch in range(config.training.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            task = torch.mean(((outputs["prediction"] - batch["target"]) / scale) ** 2)
            correction = torch.mean(((outputs["raw_irradiance"] - outputs["clear_sky_prior"]) / scale) ** 2)
            loss = task + config.training.moe_balance_weight * outputs["moe_balance_loss"] + 0.001 * correction
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch={epoch + 1} scaled_loss={total / max(1, len(loader)):.6f}", flush=True)
    output_dir = ROOT / config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / ("checkpoint_smoke.pt" if args.smoke else "checkpoint_last.pt")
    torch.save({"model": model.state_dict(), "config": config.to_dict(), "target_scale": target_scale}, checkpoint)
    print(f"saved={checkpoint}", flush=True)


if __name__ == "__main__":
    main()

