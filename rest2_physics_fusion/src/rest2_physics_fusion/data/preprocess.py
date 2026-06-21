from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def read_training_csv(path: str | Path, timestamp_column: str = "dtime") -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    return df.sort_values(timestamp_column).reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n = len(df)
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def compute_norm_stats(df: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    values = df[columns].to_numpy(dtype=np.float32)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_norm(values: np.ndarray, stats: dict[str, np.ndarray] | None) -> np.ndarray:
    if stats is None:
        return values
    return (values - stats["mean"]) / stats["std"]
