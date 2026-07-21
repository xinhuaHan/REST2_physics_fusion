from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe import IntegratedPVPhysicsMoE, load_config
from pv_physics_moe.data.dataset import collate_solar_batch
from pv_physics_moe.data.folsom import FolsomAlignedDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "folsom_15min"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "folsom_15min.yaml"))
    args = parser.parse_args()
    data_root = Path(args.data)
    with (data_root / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    for split in ("train", "val", "test"):
        rows = np.load(data_root / "splits" / f"{split}_end_indices.npy", mmap_mode="r")
        if len(rows) != metadata["split_samples"][split]:
            raise AssertionError(f"{split} split count differs from metadata")
        if len(rows) and np.any(np.diff(rows) <= 0):
            raise AssertionError(f"{split} indices are not strictly increasing")
    dataset = FolsomAlignedDataset(data_root, "train", limit=2)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_solar_batch)))
    expected = {"serial": (2, 16, 11), "images": (2, 16, 3, 64, 64), "physics": (2, 1, 26), "physics_raw": (2, 1, 26), "future_zenith": (2, 1), "target": (2, 1, 3)}
    for name, shape in expected.items():
        if tuple(batch[name].shape) != shape or not torch.isfinite(batch[name]).all():
            raise AssertionError(f"invalid {name}: shape={tuple(batch[name].shape)}")
    config = load_config(args.config)
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
    model = IntegratedPVPhysicsMoE(config).eval()
    with torch.no_grad():
        output = model(batch)
    ghi, dni, dhi = output["irradiance"].unbind(-1)
    closure = torch.max(torch.abs(ghi - output["future_cos_zenith"].clamp_min(0.0) * dni - dhi)).item()
    if closure > 1e-3:
        raise AssertionError(f"PCD closure error is too large: {closure}")
    result = {
        "split_samples": metadata["split_samples"],
        "batch_shapes": {name: list(shape) for name, shape in expected.items()},
        "prediction_shape": list(output["prediction"].shape),
        "pcd_max_closure_error_wm2": closure,
        "modalities": sorted(output["fusion_output"].modality_embeds),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
