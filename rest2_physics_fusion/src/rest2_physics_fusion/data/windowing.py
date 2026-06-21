from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rest2_physics_fusion.data.preprocess import apply_norm
from rest2_physics_fusion.data.schema import DataSchema, validate_columns


@dataclass
class WindowSample:
    serial: torch.Tensor
    weather: torch.Tensor
    physics: torch.Tensor
    physics_raw: torch.Tensor
    target: torch.Tensor
    timestamp: str


class PhysicsWindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        schema: DataSchema,
        window_steps: int = 16,
        serial_norm: dict[str, np.ndarray] | None = None,
        physics_norm: dict[str, np.ndarray] | None = None,
    ) -> None:
        validate_columns(set(frame.columns), schema)
        self.frame = frame.reset_index(drop=True)
        self.schema = schema
        self.window_steps = max(1, int(window_steps))
        self.serial_columns = list(schema.serial_feature_columns)
        self.weather_columns = list(schema.weather_feature_columns)
        self.physics_columns = list(schema.physics_feature_columns)
        self.serial_norm = serial_norm
        self.physics_norm = physics_norm

    def __len__(self) -> int:
        return max(0, len(self.frame) - self.window_steps + 1)

    def __getitem__(self, index: int) -> WindowSample:
        end = index + self.window_steps
        window = self.frame.iloc[index:end]
        target_row = window.iloc[-1]
        serial = window[self.serial_columns].to_numpy(dtype=np.float32)
        weather = window[self.weather_columns].to_numpy(dtype=np.float32)
        physics_raw = target_row[self.physics_columns].to_numpy(dtype=np.float32)
        physics = physics_raw.copy()
        serial = np.nan_to_num(serial, nan=0.0, posinf=0.0, neginf=0.0)
        weather = np.nan_to_num(weather, nan=0.0, posinf=0.0, neginf=0.0)
        physics_raw = np.nan_to_num(physics_raw, nan=0.0, posinf=0.0, neginf=0.0)
        physics = np.nan_to_num(physics, nan=0.0, posinf=0.0, neginf=0.0)
        serial = apply_norm(serial, self.serial_norm)
        weather = apply_norm(weather, self.serial_norm_for(self.weather_columns))
        physics = apply_norm(physics, self.physics_norm)
        target = np.float32(target_row[self.schema.target_column])
        return WindowSample(
            serial=torch.from_numpy(serial).float(),
            weather=torch.from_numpy(weather).float(),
            physics=torch.from_numpy(physics).float(),
            physics_raw=torch.from_numpy(physics_raw).float(),
            target=torch.tensor([target], dtype=torch.float32),
            timestamp=str(target_row[self.schema.timestamp_column]),
        )

    def serial_norm_for(self, columns: list[str]) -> dict[str, np.ndarray] | None:
        if self.serial_norm is None:
            return None
        index = [self.serial_columns.index(column) for column in columns]
        return {
            "mean": self.serial_norm["mean"][index],
            "std": self.serial_norm["std"][index],
        }


def collate_window_samples(batch: list[WindowSample]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "serial": torch.stack([item.serial for item in batch]),
        "weather": torch.stack([item.weather for item in batch]),
        "physics": torch.stack([item.physics for item in batch]),
        "physics_raw": torch.stack([item.physics_raw for item in batch]),
        "target": torch.stack([item.target for item in batch]),
        "timestamp": [item.timestamp for item in batch],
    }
