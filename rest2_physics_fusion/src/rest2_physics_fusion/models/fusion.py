from __future__ import annotations

import torch
from torch import nn


class GatedSerialPhysicsFusion(nn.Module):
    def __init__(self, serial_dim: int, physics_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.serial_proj = nn.Linear(serial_dim, hidden_dim)
        self.physics_proj = nn.Linear(physics_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, serial_embedding: torch.Tensor, physics_embedding: torch.Tensor) -> torch.Tensor:
        serial = self.serial_proj(serial_embedding)
        physics = self.physics_proj(physics_embedding)
        weights = self.gate(torch.cat([serial, physics], dim=-1))
        fused = weights[:, :1] * serial + weights[:, 1:] * physics
        return self.out(fused)
