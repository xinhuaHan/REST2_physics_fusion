from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FolsomAlignedDataset(Dataset):
    """Memory-mapped Folsom timeline with lazily decoded sky images.

    The preprocessing output stores each minute once.  A sample is represented by
    an end-row index, from which the history slice and the +15 minute target are
    selected without duplicating large arrays on disk.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        history_length: int | None = None,
        image_size: int | None = None,
        use_images: bool = True,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        with (self.root / "metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.history_length = int(history_length or self.metadata["history_length"])
        self.image_size = int(image_size or self.metadata["image_size"])
        self.use_images = bool(use_images)
        timeline = self.root / "timeline"
        self.serial = np.load(timeline / "serial.npy", mmap_mode="r")
        self.physics = np.load(timeline / "physics.npy", mmap_mode="r")
        self.physics_raw = np.load(timeline / "physics_raw.npy", mmap_mode="r")
        self.target = np.load(timeline / "target.npy", mmap_mode="r")
        self.future_zenith = np.load(timeline / "future_zenith.npy", mmap_mode="r")
        self.image_ids = np.load(timeline / "image_ids.npy", mmap_mode="r")
        indices = np.load(self.root / "splits" / f"{split}_end_indices.npy", mmap_mode="r")
        self.indices = indices[:limit] if limit is not None else indices
        self.dataset_root = Path(self.metadata["source_dataset"]).resolve()
        self.image_paths: list[str] = []
        if self.use_images:
            with (self.root / "image_paths.txt").open("r", encoding="utf-8") as handle:
                self.image_paths = [line.rstrip("\n") for line in handle]

    def __len__(self) -> int:
        return len(self.indices)

    def _load_image(self, image_id: int) -> torch.Tensor:
        path = self.dataset_root / self.image_paths[image_id]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array.copy())

    def __getitem__(self, index: int) -> dict[str, Any]:
        end = int(self.indices[index])
        start = end - self.history_length + 1
        sample: dict[str, Any] = {
            "serial": torch.from_numpy(np.asarray(self.serial[start : end + 1]).copy()).float(),
            "physics": torch.from_numpy(np.asarray(self.physics[end : end + 1]).copy()).float(),
            "physics_raw": torch.from_numpy(np.asarray(self.physics_raw[end : end + 1]).copy()).float(),
            "future_zenith": torch.from_numpy(np.asarray(self.future_zenith[end : end + 1]).copy()).float(),
            "target": torch.from_numpy(np.asarray(self.target[end : end + 1]).copy()).float(),
        }
        if self.use_images:
            ids = np.asarray(self.image_ids[start : end + 1], dtype=np.int64)
            if np.any(ids < 0):
                raise RuntimeError("preprocessing contract violated: sample contains an unmatched image")
            sample["images"] = torch.stack([self._load_image(int(image_id)) for image_id in ids])
        return sample

