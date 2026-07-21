from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


class ExpertNetwork(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * hidden_size, hidden_size), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_experts: int, num_shared_experts: int, top_k: int, dropout: float) -> None:
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.norm1, self.norm2 = nn.LayerNorm(hidden_size), nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.experts = nn.ModuleList([ExpertNetwork(hidden_size, dropout) for _ in range(num_experts)])
        self.shared = nn.ModuleList([ExpertNetwork(hidden_size, dropout) for _ in range(num_shared_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.norm1(hidden)
        attention, _ = self.attention(x, x, x, key_padding_mask=(valid_mask == 0), need_weights=False)
        hidden = hidden + self.dropout(attention)
        x = self.norm2(hidden)
        probs = torch.softmax(self.gate(x), -1)
        top_prob, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_prob = top_prob / top_prob.sum(-1, keepdim=True).clamp_min(1e-8)
        routed = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            selected = top_idx == expert_id
            token_selected = selected.any(-1) & valid_mask.bool()
            if token_selected.any():
                weights = (top_prob * selected.to(top_prob.dtype)).sum(-1)
                routed[token_selected] += expert(x[token_selected]) * weights[token_selected].unsqueeze(-1)
        shared = torch.zeros_like(x)
        for expert in self.shared:
            shared = shared + expert(x)
        if len(self.shared) > 0:
            shared = shared / len(self.shared)
        hidden = hidden + self.dropout(routed + shared)
        hidden = hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)
        assignment = torch.nn.functional.one_hot(top_idx, self.num_experts).float().sum(-2) / self.top_k
        valid = valid_mask.bool()
        importance = probs[valid].mean(0)
        load = assignment[valid].mean(0)
        balance = self.num_experts * torch.sum(importance * load)
        return hidden, {"gate_probs": probs, "top_k_indices": top_idx, "balance_loss": balance}


@dataclass
class MoEOutput:
    hidden_states: torch.Tensor
    pooled: torch.Tensor
    balance_loss: torch.Tensor
    router_data: list[dict[str, torch.Tensor]]


class PVMMoEDecoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, num_experts: int, num_shared_experts: int, top_k: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MoELayer(hidden_size, num_heads, num_experts, num_shared_experts, top_k, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> MoEOutput:
        hidden, router = tokens, []
        for layer in self.layers:
            hidden, data = layer(hidden, valid_mask)
            router.append(data)
        hidden = self.norm(hidden)
        weights = valid_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        balance = torch.stack([item["balance_loss"] for item in router]).mean()
        return MoEOutput(hidden, pooled, balance, router)
