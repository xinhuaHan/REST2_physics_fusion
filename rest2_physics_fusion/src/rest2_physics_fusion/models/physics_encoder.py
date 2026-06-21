from __future__ import annotations

import torch
from torch import nn


class PhysicsEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, physics: torch.Tensor) -> torch.Tensor:
        return self.net(physics)
