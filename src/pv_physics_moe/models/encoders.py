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


class SkyImageEncoder(nn.Module):
    """Encode asynchronous sky frames as masked temporal tokens.

    The frame-level CNN and time-aware temporal encoder follow the role of the
    PVMMoE sky encoder, while the output contract is deliberately reduced to
    ``[B,T,D]`` tokens so it can plug into this project's existing fusion layer.
    Missing frames remain zero and are represented exclusively by ``valid_mask``.
    """

    def __init__(
        self,
        image_channels: int,
        hidden_size: int,
        cnn_width: int = 32,
        temporal_layers: int = 1,
        temporal_heads: int = 8,
        dropout: float = 0.1,
        time_scale_minutes: float = 5.0,
    ) -> None:
        super().__init__()
        if image_channels <= 0 or cnn_width <= 0:
            raise ValueError("image_channels and cnn_width must be positive")
        if temporal_layers < 1 or temporal_heads < 1:
            raise ValueError("image temporal_layers and temporal_heads must be positive")
        if hidden_size % temporal_heads:
            raise ValueError("hidden_size must be divisible by image temporal_heads")
        if time_scale_minutes <= 0:
            raise ValueError("image time_scale_minutes must be positive")
        self.image_channels = int(image_channels)
        self.hidden_size = int(hidden_size)
        self.time_scale_minutes = float(time_scale_minutes)
        width = int(cnn_width)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(self.image_channels, width, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, 2 * width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * width, 4 * width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * width, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(temporal_heads),
            dim_feedforward=4 * self.hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=int(temporal_layers), norm=nn.LayerNorm(self.hidden_size)
        )

    def forward(
        self,
        images: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        time_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("images must be [B,T,C,H,W]")
        batch, frames, channels, _, _ = images.shape
        if channels != self.image_channels:
            raise ValueError(
                f"image channel count {channels} does not match configured {self.image_channels}"
            )
        if frames < 1:
            raise ValueError("images must contain at least one padded or valid frame")
        if valid_mask is None:
            mask = torch.ones((batch, frames), dtype=torch.long, device=images.device)
        else:
            if valid_mask.shape != (batch, frames):
                raise ValueError("image_valid_mask must match images [B,T]")
            mask = valid_mask.to(device=images.device).long()
        if time_offsets is None:
            offsets = torch.arange(
                1 - frames, 1, device=images.device, dtype=torch.float32
            ).view(1, frames).expand(batch, -1)
        else:
            if time_offsets.shape != (batch, frames):
                raise ValueError("image_time_offsets must match images [B,T]")
            offsets = time_offsets.to(device=images.device, dtype=torch.float32)

        frame_tokens = self.frame_encoder(
            torch.nan_to_num(images.float()).reshape(batch * frames, channels, images.size(-2), images.size(-1))
        ).reshape(batch, frames, self.hidden_size)
        scaled = offsets.float() / self.time_scale_minutes
        time_features = torch.stack(
            (scaled, torch.sin(torch.pi * scaled), torch.cos(torch.pi * scaled)), dim=-1
        )
        tokens = frame_tokens + self.time_projection(time_features).to(frame_tokens.dtype)
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)

        # PyTorch attention cannot consume an all-True padding row. Temporarily
        # expose one zero token, then restore the true mask on the output.
        safe_mask = mask.clone()
        all_missing = safe_mask.sum(1) == 0
        if all_missing.any():
            safe_mask[all_missing, 0] = 1
        tokens = self.temporal_encoder(tokens, src_key_padding_mask=(safe_mask == 0))
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, mask
