from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe import IntegratedPVPhysicsMoE, load_parquet_config
from pv_physics_moe.data import build_parquet_datasets, collate_parquet_batch
from pv_physics_moe.training.losses import IntegratedLoss


def setup_distributed(requested_device: str) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = requested_device.startswith("cuda") and torch.cuda.is_available()
    if world_size > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device("cuda", local_rank)
    return rank, world_size, torch.device("cpu")


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Configuration-driven YLJ/Luoyang Parquet DDP training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--parquet-file", help="Override dataset.parquet_file")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    parser.add_argument("--smoke", action="store_true", help="One epoch over at most four real/synthetic configured samples")
    args = parser.parse_args()
    config = load_parquet_config(args.config)
    if args.parquet_file:
        config.dataset.parquet_file = args.parquet_file
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.smoke:
        config.training.epochs = 1
        config.training.batch_size = 2
        config.runtime.num_workers = 0
    rank, world_size, device = setup_distributed(config.training.device)
    torch.manual_seed(config.project.seed + rank)
    datasets = build_parquet_datasets(config)
    train_base = datasets["train"]
    train_data = Subset(train_base, range(min(4, len(train_base)))) if args.smoke else train_base
    train_sampler = DistributedSampler(train_data, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_data,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory and device.type == "cuda",
        collate_fn=collate_parquet_batch,
    )
    val_loader = None
    val_sampler = None
    if "val" in datasets:
        val_data = datasets["val"]
        val_sampler = DistributedSampler(val_data, shuffle=False) if world_size > 1 else None
        val_loader = DataLoader(
            val_data, batch_size=config.training.batch_size, sampler=val_sampler, shuffle=False,
            num_workers=config.runtime.num_workers, pin_memory=config.runtime.pin_memory and device.type == "cuda",
            collate_fn=collate_parquet_batch,
        )
    model = IntegratedPVPhysicsMoE(config.model).to(device)
    wrapped = DistributedDataParallel(model, device_ids=[device.index]) if world_size > 1 and device.type == "cuda" else (
        DistributedDataParallel(model) if world_size > 1 else model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    loss_fn = IntegratedLoss(config.training)
    amp_enabled = device.type == "cuda" and config.runtime.precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    for epoch in range(config.training.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        wrapped.train()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = wrapped(batch)
                losses = loss_fn(outputs, batch)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            totals += torch.tensor([float(losses["loss"].detach()), 1.0], dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(totals)
        if rank == 0:
            print(f"epoch={epoch + 1} train_loss={float(totals[0]/totals[1].clamp_min(1)):.6f}", flush=True)
        if val_loader is not None:
            wrapped.eval()
            validation = torch.zeros(2, dtype=torch.float64, device=device)
            with torch.no_grad():
                for batch in val_loader:
                    batch = move_batch(batch, device)
                    outputs = wrapped(batch)
                    loss = loss_fn(outputs, batch)["loss"]
                    validation += torch.tensor([float(loss), 1.0], dtype=torch.float64, device=device)
            if world_size > 1:
                dist.all_reduce(validation)
            if rank == 0:
                print(f"epoch={epoch + 1} val_loss={float(validation[0]/validation[1].clamp_min(1)):.6f}", flush=True)
    if rank == 0:
        output_dir = Path(config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "resolved_config": config.to_dict(),
            "field_order": list(config.dataset.serial_columns),
            "normalization": train_base.normalization.state_dict(),
            "capacity": config.normalization.rated_power,
            "site": config.to_dict()["site"],
            "data_schema": train_base.metadata,
        }
        checkpoint = output_dir / ("checkpoint_smoke.pt" if args.smoke else "checkpoint_last.pt")
        torch.save({"model": model.state_dict(), **metadata}, checkpoint)
        with (output_dir / "config_resolved.json").open("w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
        print(f"saved={checkpoint}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
