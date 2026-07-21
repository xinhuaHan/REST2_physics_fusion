from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureStandardizer":
        mean = np.nanmean(values, axis=tuple(range(values.ndim - 1)))
        std = np.nanstd(values, axis=tuple(range(values.ndim - 1)))
        return cls(mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

