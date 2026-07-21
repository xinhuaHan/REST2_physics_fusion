from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class StationFiveCsvDataset(Dataset):
    """Memory-mapped five-table dataset with one tensor row per forecast origin."""

    def __init__(self, root: str | Path, split: str, limit: int | None = None) -> None:
        self.root = Path(root)
        with (self.root / "metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.history_length = int(self.metadata["history_length"])
        timeline = self.root / "timeline"
        self.serial = np.load(timeline / "serial.npy", mmap_mode="r")
        self.physics = np.load(timeline / "physics.npy", mmap_mode="r")
        self.physics_raw = np.load(timeline / "physics_raw.npy", mmap_mode="r")
        self.target = np.load(timeline / "target_power.npy", mmap_mode="r")
        self.irradiance_target = np.load(timeline / "target_irradiance.npy", mmap_mode="r")
        self.future_zenith = np.load(timeline / "future_zenith.npy", mmap_mode="r")
        indices = np.load(self.root / "splits" / f"{split}_end_indices.npy", mmap_mode="r")
        self.indices = indices[:limit] if limit is not None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        end = int(self.indices[index])
        start = end - self.history_length + 1
        return {
            "serial": torch.from_numpy(np.asarray(self.serial[start : end + 1]).copy()).float(),
            "physics": torch.from_numpy(np.asarray(self.physics[end]).copy()).float(),
            "physics_raw": torch.from_numpy(np.asarray(self.physics_raw[end]).copy()).float(),
            "future_zenith": torch.from_numpy(np.asarray(self.future_zenith[end]).copy()).float(),
            "target": torch.from_numpy(np.asarray(self.target[end]).copy()).float(),
            "irradiance_target": torch.from_numpy(np.asarray(self.irradiance_target[end]).copy()).float(),
        }
