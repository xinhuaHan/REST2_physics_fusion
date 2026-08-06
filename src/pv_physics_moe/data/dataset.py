from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset


class NpzSolarDataset(Dataset):
    """Standalone aligned-data contract.

    Required arrays: serial [N,T,C], physics [N,H,26], physics_raw [N,H,26],
    future_zenith [N,H], target [N,H,1 or 3]. Optional: images [N,T,3,H,W],
    irradiance_target [N,H,3]. ``physics`` may be normalized; ``physics_raw`` must
    remain in the physical units defined by the REST2 contract.
    """
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.arrays = np.load(self.path, mmap_mode="r")
        required = {"serial", "physics", "physics_raw", "future_zenith", "target"}
        missing = required - set(self.arrays.files)
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {sorted(missing)}")
        lengths = {name: len(self.arrays[name]) for name in required}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"NPZ arrays have inconsistent sample counts: {lengths}")

    def __len__(self) -> int:
        return len(self.arrays["serial"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        names = (
            "serial", "physics", "physics_raw", "future_zenith", "target",
            "serial_valid_mask", "physics_valid_mask", "target_valid_mask",
            "images", "image_valid_mask", "image_time_offsets", "irradiance_target",
            "irradiance_target_mask", "current_power",
        )
        return {name: torch.from_numpy(np.asarray(self.arrays[name][index]).copy()).float() for name in names if name in self.arrays.files}


def collate_solar_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    keys = samples[0].keys()
    if any(sample.keys() != samples[0].keys() for sample in samples):
        raise ValueError("all samples in a batch must expose the same modalities")
    return {key: torch.stack([sample[key] for sample in samples]) for key in keys}
