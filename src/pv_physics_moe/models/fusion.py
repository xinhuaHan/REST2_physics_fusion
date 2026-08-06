from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping
import torch
from torch import nn


@dataclass
class FusionOutput:
    prediction_embeds: torch.Tensor
    prediction_mask: torch.Tensor
    modality_embeds: Dict[str, torch.Tensor]
    attention_maps: Dict[str, torch.Tensor]


class PhysicsAwareCrossModalFusion(nn.Module):
    """Masked fusion for serial, REST2 physics and optional image modalities."""
    def __init__(self, input_dims: Mapping[str, int], hidden_size: int, num_heads: int = 8, dropout: float = 0.1, modality_order: tuple[str, ...] = ("serial", "physics")) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.modality_order = tuple(modality_order)
        self.projections = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size)) for name, dim in input_dims.items()})
        self.cross_attentions = nn.ModuleDict()
        for query in input_dims:
            for key in input_dims:
                if query != key:
                    self.cross_attentions[f"{query}_from_{key}"] = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_norms = nn.ModuleDict({name: nn.LayerNorm(hidden_size) for name in input_dims})
        self.ffn_norms = nn.ModuleDict({name: nn.LayerNorm(hidden_size) for name in input_dims})
        self.ffns = nn.ModuleDict({name: nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * hidden_size, hidden_size), nn.Dropout(dropout)) for name in input_dims})
        self.gates = nn.ModuleDict({name: nn.Sequential(nn.Linear(hidden_size, max(1, hidden_size // 2)), nn.GELU(), nn.Linear(max(1, hidden_size // 2), 1), nn.Sigmoid()) for name in input_dims})
        self.separators = nn.ParameterDict({name: nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02) for name in input_dims})

    @staticmethod
    def _safe_padding_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding = mask == 0
        all_masked = padding.all(1)
        if all_masked.any():
            padding = padding.clone()
            padding[all_masked, 0] = False
        return padding, all_masked

    def forward(self, modalities: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> FusionOutput:
        aligned: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        for name, (embeds, mask) in modalities.items():
            if name not in self.projections:
                raise KeyError(f"unregistered modality: {name}")
            if embeds.ndim != 3 or mask.shape != embeds.shape[:2]:
                raise ValueError(f"{name} embeds/mask must be [B,L,D] and [B,L]")
            aligned[name] = self.projections[name](embeds)
            masks[name] = mask.to(embeds.device).long()
        expected = set(self.modality_order)
        if set(aligned) != expected:
            raise ValueError(
                f"fusion requires configured modalities {sorted(expected)}, got {sorted(aligned)}"
            )
        if not {"serial", "physics"}.issubset(expected):
            raise ValueError("fusion always requires the serial and physics modalities")

        attention_maps: Dict[str, torch.Tensor] = {}
        enhanced: Dict[str, torch.Tensor] = {}
        for query_name, query in aligned.items():
            updates = []
            update_available = []
            for key_name, key_value in aligned.items():
                if query_name == key_name:
                    continue
                name = f"{query_name}_from_{key_name}"
                padding, all_masked = self._safe_padding_mask(masks[key_name])
                update, weights = self.cross_attentions[name](query, key_value, key_value, key_padding_mask=padding, need_weights=True)
                if all_masked.any():
                    update = update.masked_fill(all_masked[:, None, None], 0.0)
                    weights = weights.masked_fill(all_masked[:, None, None], 0.0)
                updates.append(update)
                update_available.append((~all_masked).to(update.dtype))
                attention_maps[name] = weights.detach()
            stacked_updates = torch.stack(updates, dim=0)
            availability = torch.stack(update_available, dim=0)
            aggregate = stacked_updates.sum(0) / availability.sum(0).clamp_min(1.0)[:, None, None]
            x = self.cross_norms[query_name](query + aggregate)
            x = self.ffn_norms[query_name](x + self.ffns[query_name](x))
            x = x * self.gates[query_name](x) * masks[query_name].unsqueeze(-1).to(x.dtype)
            enhanced[query_name] = x

        tokens, token_masks = [], []
        for name in self.modality_order:
            batch = enhanced[name].size(0)
            sep = self.separators[name].expand(batch, 1, -1)
            sep_mask = (masks[name].sum(1, keepdim=True) > 0).long()
            tokens.append(torch.cat((sep, enhanced[name]), 1))
            token_masks.append(torch.cat((sep_mask, masks[name]), 1))
        return FusionOutput(torch.cat(tokens, 1), torch.cat(token_masks, 1), enhanced, attention_maps)
