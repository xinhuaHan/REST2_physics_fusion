from __future__ import annotations

import shutil
import os
import socket
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe.models.integrated import IntegratedPVPhysicsMoE
from pv_physics_moe.config import ModelConfig


def worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    config = ModelConfig(
        serial_input_dim=4, hidden_size=32, serial_layers=1, fusion_heads=4,
        moe_layers=1, moe_heads=4, num_experts=2, num_shared_experts=1,
        top_k_experts=1, pcd_hidden_size=16, pcd_layers=1, forecast_horizon=2, dropout=0.0,
    )
    model = DistributedDataParallel(IntegratedPVPhysicsMoE(config))
    dataset = TensorDataset(torch.arange(4))
    sampler = DistributedSampler(dataset, shuffle=True)
    sampler.set_epoch(0)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for _ in loader:
        batch = {
            "serial": torch.randn(1, 3, 4),
            "physics": torch.randn(1, 2, 26),
            "physics_raw": torch.rand(1, 2, 26),
            "future_zenith": torch.tensor([[30.0, 60.0]]),
        }
        output = model(batch)
        loss = output["prediction"].mean() + output["moe_balance_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
    if rank == 0:
        print("2-process CPU/gloo DDP smoke passed", flush=True)
    dist.destroy_process_group()


def main() -> None:
    if sys.platform == "win32" and "GLOO_SOCKET_IFNAME" not in os.environ:
        names = {name for _, name in socket.if_nameindex()}
        if "loopback_0" in names:
            os.environ["GLOO_SOCKET_IFNAME"] = "loopback_0"
    temporary = Path(tempfile.mkdtemp(prefix="pv-ddp-smoke-"))
    try:
        init_method = (temporary / "store").resolve().as_uri()
        mp.spawn(worker, args=(2, init_method), nprocs=2, join=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
