"""Architecture-only reference for OCLR.

This module exposes the component graph but intentionally omits the
end-to-end training objective, overlap-target construction, data pipeline,
inference aggregation, and checkpoint mapping used for paper experiments.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ArchitectureSpec:
    """Public module-level specification reported in the manuscript."""

    encoder_name: str = "DINOv2-reg ViT-B/14"
    target_layers: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)
    encoder_fusion: Tuple[Tuple[int, ...], ...] = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    )
    decoder_depth: int = 8
    decoder_heads: int = 12
    embedding_dim: int = 768
    local_lora_blocks: Tuple[int, ...] = (3, 4, 5, 6)
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.05


class BranchConditioner(nn.Module):
    """Independent normalization and embedding for global/local routes."""

    ROUTES = ("global", "local")

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norms = nn.ModuleDict(
            {route: nn.LayerNorm(dim) for route in self.ROUTES}
        )
        self.embeddings = nn.ParameterDict(
            {
                route: nn.Parameter(torch.zeros(1, 1, dim))
                for route in self.ROUTES
            }
        )

    def forward(self, tokens: torch.Tensor, route: str) -> torch.Tensor:
        if route not in self.ROUTES:
            raise ValueError(f"Unknown route: {route}")
        return self.norms[route](tokens) + self.embeddings[route]


class LocalOnlyQVLora(nn.Module):
    """Fused QKV projection with an optional low-rank update to Q and V."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if base.out_features != base.in_features * 3:
            raise ValueError("Expected a fused QKV linear projection")
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        dim = base.in_features
        self.q_down = nn.Linear(dim, rank, bias=False)
        self.q_up = nn.Linear(rank, dim, bias=False)
        self.v_down = nn.Linear(dim, rank, bias=False)
        self.v_up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.q_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.v_up.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        local_route: bool = False,
    ) -> torch.Tensor:
        qkv = self.base(tokens)
        if not local_route:
            return qkv
        dropped = self.dropout(tokens)
        delta_q = self.q_up(self.q_down(dropped)) * self.scale
        delta_v = self.v_up(self.v_down(dropped)) * self.scale
        return qkv + torch.cat(
            (delta_q, torch.zeros_like(delta_q), delta_v), dim=-1
        )


class LinearAttentionDecoderBlock(nn.Module):
    """The shared decoder block family used by both OCLR routes."""

    def __init__(self, dim: int = 768, heads: int = 12) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=1.0e-8)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1.0e-8)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        normalized = self.norm1(tokens)
        batch, count, channels = normalized.shape
        qkv = self.qkv(normalized).reshape(
            batch, count, 3, self.heads, channels // self.heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = F.elu(query) + 1.0
        key = F.elu(key) + 1.0
        accumulator_type = (
            torch.float32 if query.dtype in (torch.float16, torch.bfloat16)
            else query.dtype
        )
        query_acc = query.to(accumulator_type)
        key_acc = key.to(accumulator_type)
        value_acc = value.to(accumulator_type)
        context = torch.einsum("...sd,...se->...de", key_acc, value_acc)
        denominator = torch.einsum(
            "...sd,...d->...s", query_acc, key_acc.sum(dim=-2)
        ).clamp_min(1.0e-12)
        update = torch.einsum(
            "...de,...sd,...s->...se",
            context,
            query_acc,
            denominator.reciprocal(),
        ).to(query.dtype)
        update = update.transpose(1, 2).reshape(batch, count, channels)
        tokens = residual + self.proj(update)
        return tokens + self.mlp(self.norm2(tokens))


class OCLRArchitecture(nn.Module):
    """Inspectable OCLR component graph without an end-to-end forward."""

    def __init__(self, spec: ArchitectureSpec = ArchitectureSpec()) -> None:
        super().__init__()
        self.spec = spec
        dim = spec.embedding_dim
        self.route_conditioner = BranchConditioner(dim)
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 4, dim),
        )
        self.shared_decoder = nn.ModuleList(
            [
                LinearAttentionDecoderBlock(dim, spec.decoder_heads)
                for _ in range(spec.decoder_depth)
            ]
        )

    def describe(self) -> dict:
        return {
            **asdict(self.spec),
            "training_routes": ("global", "local"),
            "inference_route": "local",
            "public_scope": "architecture-only",
        }

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "This public release is architecture-only. The end-to-end route "
            "execution, objectives, aggregation, and experiment pipeline are "
            "not included."
        )
