from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe import IntegratedPVPhysicsMoE, load_config
from pv_physics_moe.data import NpzSolarDataset, collate_solar_batch
from pv_physics_moe.training import IntegratedLoss


class SyntheticDataset(Dataset):
    def __init__(self, count: int, serial_dim: int, horizon: int) -> None:
        self.count, self.serial_dim, self.horizon = count, serial_dim, horizon

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        mu0 = torch.rand(self.horizon, generator=generator) * 0.9 + 0.05
        zenith = torch.rad2deg(torch.arccos(mu0))
        serial = torch.randn(8, self.serial_dim, generator=generator)
        physics_raw = torch.zeros(self.horizon, 26)
        clear_dni, clear_dhi = 700.0 * mu0, 100.0 * mu0
        clear_ghi = clear_dni * mu0 + clear_dhi
        physics_raw[:, 0:3] = torch.stack((clear_ghi, clear_dni, clear_dhi), -1)
        physics_raw[:, 3], physics_raw[:, 4], physics_raw[:, 5] = mu0, zenith, 1361.0
        physics_raw[:, 6], physics_raw[:, 7], physics_raw[:, 8] = 101325.0, 1.5, 0.08
        target = (clear_ghi * 0.2).unsqueeze(-1)
        return {"serial": serial, "physics": physics_raw.clone(), "physics_raw": physics_raw, "future_zenith": zenith, "target": target}


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--data", help="Aligned NPZ dataset; omitted with --smoke")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
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
        config.training.epochs = 1
        config.training.batch_size = 2
        config.training.device = "cpu"
        dataset = SyntheticDataset(4, config.model.serial_input_dim, config.model.forecast_horizon)
    else:
        if not args.data:
            parser.error("--data is required unless --smoke is used")
        dataset = NpzSolarDataset(args.data)
    torch.manual_seed(config.project.seed)
    device = torch.device(config.training.device)
    model = IntegratedPVPhysicsMoE(config).to(device)
    criterion = IntegratedLoss(config.training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_solar_batch)
    for epoch in range(config.training.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = criterion(outputs, batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            total += float(losses["loss"].detach())
        print(f"epoch={epoch + 1} loss={total / max(1, len(loader)):.6f}")
    output_dir = ROOT / config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.to_dict()}, output_dir / "checkpoint_last.pt")
    (output_dir / "config_resolved.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    print(f"saved={output_dir / 'checkpoint_last.pt'}")


if __name__ == "__main__":
    main()

