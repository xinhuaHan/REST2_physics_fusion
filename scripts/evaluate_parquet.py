from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe import IntegratedPVPhysicsMoE, load_parquet_config
from pv_physics_moe.data import ConfigurableParquetDataset, ParquetNormalization, collate_parquet_batch
from pv_physics_moe.evaluation import atomic_write_results, finalize_prediction_frame, official_metrics, prediction_records


def setup(requested_device: str) -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = requested_device.startswith("cuda") and torch.cuda.is_available()
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    if cuda:
        torch.cuda.set_device(local)
        return rank, world, torch.device("cuda", local)
    return rank, world, torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed YLJ/Luoyang Parquet evaluation and long-table export")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet-file")
    parser.add_argument("--output-dir", default="outputs/parquet_evaluation")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args()
    config = load_parquet_config(args.config)
    if args.parquet_file:
        config.dataset.parquet_file = args.parquet_file
    rank, world, device = setup(config.training.device)
    state = torch.load(args.checkpoint, map_location="cpu")
    normalization = ParquetNormalization.from_state_dict(state["normalization"])
    dataset = ConfigurableParquetDataset(config, args.split, normalization=normalization)
    sampler = DistributedSampler(dataset, shuffle=False) if world > 1 else None
    loader = DataLoader(
        dataset, batch_size=config.training.batch_size, sampler=sampler, shuffle=False,
        num_workers=config.runtime.num_workers, pin_memory=config.runtime.pin_memory and device.type == "cuda",
        collate_fn=collate_parquet_batch,
    )
    model = IntegratedPVPhysicsMoE(config.model).to(device)
    model.load_state_dict(state["model"])
    wrapped = DistributedDataParallel(model, device_ids=[device.index]) if world > 1 and device.type == "cuda" else (
        DistributedDataParallel(model) if world > 1 else model
    )
    wrapped.eval()
    local_records = []
    # closure max, negative x3, night count, irradiance squared-error x3, valid-count x3
    diag = torch.zeros(11, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            output = wrapped(moved)
            local_records.extend(prediction_records(batch, output["prediction"]))
            irr = output["irradiance"]
            cos = output["future_cos_zenith"].clamp_min(0)
            ghi, dni, dhi = irr.unbind(-1)
            diag[0] = torch.maximum(diag[0], (ghi - cos * dni - dhi).abs().max().double())
            diag[1] += (ghi < 0).sum()
            diag[2] += (dni < 0).sum()
            diag[3] += (dhi < 0).sum()
            diag[4] += (irr[cos <= 0].abs() > 1e-6).sum()
            irr_target = moved["irradiance_target"]
            irr_mask = moved["irradiance_target_mask"].to(torch.bool)
            for component in range(3):
                valid = irr_mask[..., component]
                diag[5 + component] += ((irr[..., component][valid] - irr_target[..., component][valid]) ** 2).sum().double()
                diag[8 + component] += valid.sum().double()
    if world > 1:
        gathered: list[list[dict] | None] = [None] * world if rank == 0 else []
        dist.gather_object(local_records, gathered if rank == 0 else None, dst=0)
        maximum = diag[0].clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        counts = diag[1:].clone()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        diag = torch.cat((maximum.view(1), counts))
        records = [record for part in gathered for record in (part or [])] if rank == 0 else []
    else:
        records = local_records
    if rank == 0:
        frame = finalize_prediction_frame(records, int(config.dataset.forecast_steps))
        metrics = official_metrics(frame, config.evaluation)
        metrics["pcd"] = {
            "max_closure_error": float(diag[0]),
            "negative_counts": {"ghi": int(diag[1]), "dni": int(diag[2]), "dhi": int(diag[3])},
            "night_nonzero_count": int(diag[4]),
            "irradiance_target_rmse": {
                name: (float(torch.sqrt(diag[5 + index] / diag[8 + index])) if diag[8 + index] > 0 else None)
                for index, name in enumerate(("ghi", "dni", "dhi"))
            },
        }
        atomic_write_results(frame, metrics, args.output_dir)
        print(f"predictions={Path(args.output_dir) / 'point_predictions.csv'}")
        print(f"metrics={Path(args.output_dir) / 'official_test_metrics.json'}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
