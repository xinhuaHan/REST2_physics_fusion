from __future__ import annotations
import torch
from torch import nn


def project_to_feasible_set(raw_state: torch.Tensor, cos_zenith: torch.Tensor, night_cos_threshold: float = 0.0) -> torch.Tensor:
    """Euclidean projection onto GHI=c*DNI+DHI with nonnegative components."""
    if raw_state.shape[-1] != 3:
        raise ValueError("raw_state must have shape [...,3] ordered as [GHI,DNI,DHI]")
    g, n, d = raw_state.unbind(-1)
    c_raw = cos_zenith.to(raw_state).clamp(max=1.0)
    c = c_raw.clamp_min(0.0)
    zero, inf = torch.zeros_like(g), torch.full_like(g, float("inf"))
    n_i = (c * g + 2.0 * n - c * d) / (c.square() + 2.0)
    d_i = (g + (c.square() + 1.0) * d - c * n) / (c.square() + 2.0)
    interior = torch.stack((c * n_i + d_i, n_i, d_i), -1)
    n_b = ((c * g + n) / (c.square() + 1.0)).clamp_min(0.0)
    d_b = ((g + d) / 2.0).clamp_min(0.0)
    candidates = torch.stack((
        interior,
        torch.stack((c * n_b, n_b, zero), -1),
        torch.stack((d_b, zero, d_b), -1),
        torch.stack((zero, zero, zero), -1),
    ), -2)
    distance = (candidates - raw_state.unsqueeze(-2)).square().sum(-1)
    distance[..., 0] = torch.where((n_i >= 0.0) & (d_i >= 0.0), distance[..., 0], inf)
    best = distance.argmin(-1, keepdim=True)
    projected = candidates.gather(-2, best.unsqueeze(-1).expand(*best.shape, 3)).squeeze(-2)
    night = c_raw <= float(night_cos_threshold)
    return torch.where(night.unsqueeze(-1), torch.zeros_like(projected), projected)


class SolarPhysicsBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, night_cos_threshold: float) -> None:
        super().__init__()
        self.night_cos_threshold = night_cos_threshold
        self.state_embed = nn.Linear(3, hidden_size)
        self.cos_embed = nn.Linear(1, hidden_size)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, 1), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * hidden_size, hidden_size))
        self.norm2 = nn.LayerNorm(hidden_size)
        self.delta_head = nn.Linear(hidden_size, 3)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, hidden: torch.Tensor, state: torch.Tensor, cos_zenith: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.state_embed(state) + self.cos_embed(cos_zenith.unsqueeze(-1))
        delta = self.temporal((hidden + context).transpose(1, 2)).transpose(1, 2)
        hidden = self.norm1(hidden + delta)
        hidden = self.norm2(hidden + self.ffn(hidden))
        state = project_to_feasible_set(state + self.delta_head(hidden), cos_zenith, self.night_cos_threshold)
        return hidden, state


class SolarPhysicsStack(nn.Module):
    def __init__(self, context_dim: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1, night_cos_threshold: float = 0.0) -> None:
        super().__init__()
        self.night_cos_threshold = night_cos_threshold
        self.input = nn.Sequential(nn.Linear(context_dim, hidden_size), nn.LayerNorm(hidden_size))
        self.blocks = nn.ModuleList([SolarPhysicsBlock(hidden_size, dropout, night_cos_threshold) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, context: torch.Tensor, base_state: torch.Tensor, cos_zenith: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input(context)
        state = project_to_feasible_set(base_state, cos_zenith, self.night_cos_threshold)
        for block in self.blocks:
            hidden, state = block(hidden, state, cos_zenith)
        return state, self.output_norm(hidden)
