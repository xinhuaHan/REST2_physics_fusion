from __future__ import annotations
import math
import torch
from torch import nn


def sinusoidal_position(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    result = torch.zeros(1, length, dim, device=device, dtype=dtype)
    result[0, :, 0::2] = torch.sin(position * div)
    if dim > 1:
        result[0, :, 1::2] = torch.cos(position * div[: result[0, :, 1::2].shape[-1]])
    return result


class SerialEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, 4 * hidden_size, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, norm=nn.LayerNorm(hidden_size))

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3:
            raise ValueError("serial input must be [B,T,C]")
        mask = torch.ones(values.shape[:2], dtype=torch.long, device=values.device) if valid_mask is None else valid_mask.to(values.device).long()
        x = self.input_proj(torch.nan_to_num(values.float()))
        x = x + sinusoidal_position(x.size(1), x.size(2), x.device, x.dtype)
        x = self.encoder(x, src_key_padding_mask=(mask == 0))
        return x, mask


class PhysicsTokenEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size),
        )

    def forward(self, physics: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if physics.ndim == 2:
            physics = physics.unsqueeze(1)
        if physics.ndim != 3:
            raise ValueError("physics input must be [B,F] or [B,H,F]")
        x = self.net(torch.nan_to_num(physics.float()))
        mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device) if valid_mask is None else valid_mask.to(x.device).long()
        return x, mask
