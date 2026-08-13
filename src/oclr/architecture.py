"""Executable architecture-level reference for OCLR.

The public API begins at encoder-token features. It demonstrates the
global/local routes, route conditioning, local-only Q/V LoRA, feature fusion,
and the shared FP16-safe decoder. Image preprocessing, DINOv2 execution,
training objectives, overlap-target construction, anomaly aggregation,
evaluation, and checkpoint conversion are intentionally outside this release.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


TensorList = List[torch.Tensor]


@dataclass(frozen=True)
class ArchitectureSpec:
    """Public module-level specification reported in the manuscript."""

    encoder_name: str = "DINOv2-reg ViT-B/14"
    target_layers: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)
    encoder_fusion: Tuple[Tuple[int, ...], ...] = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    )
    decoder_fusion: Tuple[Tuple[int, ...], ...] = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    )
    decoder_depth: int = 8
    decoder_heads: int = 12
    embedding_dim: int = 768
    register_tokens: int = 4
    local_lora_blocks: Tuple[int, ...] = (3, 4, 5, 6)
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.05

    def validate(self) -> None:
        if self.embedding_dim <= 0 or self.decoder_depth <= 0:
            raise ValueError("Embedding dimension and decoder depth must be positive")
        if self.embedding_dim % self.decoder_heads:
            raise ValueError("embedding_dim must be divisible by decoder_heads")
        if not self.target_layers:
            raise ValueError("At least one target layer is required")
        if tuple(sorted(set(self.target_layers))) != self.target_layers:
            raise ValueError("target_layers must be unique and sorted")
        if tuple(sorted(set(self.local_lora_blocks))) != self.local_lora_blocks:
            raise ValueError("local_lora_blocks must be unique and sorted")
        for groups, upper, name in (
            (self.encoder_fusion, len(self.target_layers), "encoder_fusion"),
            (self.decoder_fusion, self.decoder_depth, "decoder_fusion"),
        ):
            if not groups or any(not group for group in groups):
                raise ValueError(f"{name} groups cannot be empty")
            if any(index < 0 or index >= upper for group in groups for index in group):
                raise ValueError(f"{name} index is out of range")


@dataclass
class RouteOutput:
    """Feature-level output shared by global and local routes."""

    route: str
    teacher_features: TensorList
    decoder_features: TensorList
    decoder_tokens: torch.Tensor

    def as_dict(self) -> Dict[str, object]:
        return {
            "route": self.route,
            "teacher_features": self.teacher_features,
            "decoder_features": self.decoder_features,
            "decoder_tokens": self.decoder_tokens,
        }


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
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        dim = base.in_features
        self.q_down = nn.Linear(dim, rank, bias=False)
        self.q_up = nn.Linear(rank, dim, bias=False)
        self.v_down = nn.Linear(dim, rank, bias=False)
        self.v_up = nn.Linear(rank, dim, bias=False)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        nn.init.kaiming_uniform_(self.q_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.v_up.weight)

    @property
    def adapter_parameters(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("base.")
        )

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


class BottleneckMLP(nn.Module):
    """Shared reconstruction bottleneck following the route conditioner."""

    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.dropout(tokens)
        tokens = self.fc1(tokens)
        tokens = self.activation(tokens)
        tokens = self.dropout(tokens)
        tokens = self.fc2(tokens)
        return self.dropout(tokens)


class LinearAttentionDecoderBlock(nn.Module):
    """Shared decoder block with FP32 reductions for FP16/BF16 inputs."""

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
        tokens = tokens + self.proj(update)
        return tokens + self.mlp(self.norm2(tokens))


class OCLRArchitecture(nn.Module):
    """Executable post-encoder reference for both OCLR routes.

    Inputs are target-layer token tensors produced by an external frozen
    encoder. Each tensor has shape B x N x C and includes class/register
    tokens. This boundary exposes the architecture without bundling the
    image-to-token or experiment pipeline.
    """

    ROUTES = ("global", "local")

    def __init__(self, spec: Optional[ArchitectureSpec] = None) -> None:
        super().__init__()
        self.spec = spec or ArchitectureSpec()
        self.spec.validate()
        dim = self.spec.embedding_dim
        self.route_conditioner = BranchConditioner(dim)
        self.bottleneck = BottleneckMLP(dim)
        self.shared_decoder = nn.ModuleList(
            [
                LinearAttentionDecoderBlock(dim, self.spec.decoder_heads)
                for _ in range(self.spec.decoder_depth)
            ]
        )

    @staticmethod
    def _mean_features(
        features: Sequence[torch.Tensor],
        indices: Sequence[int],
    ) -> torch.Tensor:
        selected = [features[index] for index in indices]
        return torch.stack(selected, dim=1).mean(dim=1)

    def _validate_features(
        self,
        encoder_features: Sequence[torch.Tensor],
    ) -> None:
        if len(encoder_features) != len(self.spec.target_layers):
            raise ValueError(
                f"Expected {len(self.spec.target_layers)} target features, "
                f"received {len(encoder_features)}"
            )
        expected_shape = encoder_features[0].shape
        if len(expected_shape) != 3:
            raise ValueError("Features must have shape B x N x C")
        if expected_shape[-1] != self.spec.embedding_dim:
            raise ValueError("Feature channels do not match embedding_dim")
        if any(feature.shape != expected_shape for feature in encoder_features):
            raise ValueError("All target features must share one shape")

    def forward(
        self,
        encoder_features: Sequence[torch.Tensor],
        route: str,
    ) -> RouteOutput:
        if route not in self.ROUTES:
            raise ValueError(f"Unknown route: {route}")
        self._validate_features(encoder_features)

        decoder_input = torch.stack(list(encoder_features), dim=1).mean(dim=1)
        tokens = self.route_conditioner(decoder_input, route)
        tokens = self.bottleneck(tokens)
        decoder_history = []
        for block in self.shared_decoder:
            tokens = block(tokens)
            decoder_history.append(tokens)
        reversed_history = list(reversed(decoder_history))

        teacher = [
            self._mean_features(encoder_features, group)
            for group in self.spec.encoder_fusion
        ]
        decoder = [
            self._mean_features(reversed_history, group)
            for group in self.spec.decoder_fusion
        ]
        return RouteOutput(route, teacher, decoder, tokens)

    def forward_global(
        self,
        encoder_features: Sequence[torch.Tensor],
    ) -> RouteOutput:
        return self.forward(encoder_features, route="global")

    def forward_local(
        self,
        teacher_features: Sequence[torch.Tensor],
        adapted_features: Optional[Sequence[torch.Tensor]] = None,
    ) -> RouteOutput:
        """Run local reconstruction from teacher and optional LoRA features."""

        source = (
            adapted_features
            if adapted_features is not None
            else teacher_features
        )
        output = self.forward(source, route="local")
        self._validate_features(teacher_features)
        output.teacher_features = [
            self._mean_features(teacher_features, group)
            for group in self.spec.encoder_fusion
        ]
        return output

    def describe(self) -> dict:
        return {
            **asdict(self.spec),
            "training_routes": self.ROUTES,
            "inference_route": "local",
            "input_boundary": "post-encoder target-layer tokens",
            "public_scope": "executable architecture reference",
            "omitted": (
                "image preprocessing",
                "encoder execution and LoRA injection",
                "training objectives",
                "overlap-target construction",
                "anomaly-map aggregation",
                "evaluation and checkpoints",
            ),
        }


def parameter_audit(
    architecture: OCLRArchitecture,
) -> Dict[str, int]:
    """Return transparent counts for public modules and paper additions."""

    conditioner = sum(
        parameter.numel()
        for parameter in architecture.route_conditioner.parameters()
    )
    bottleneck = sum(
        parameter.numel() for parameter in architecture.bottleneck.parameters()
    )
    decoder = sum(
        parameter.numel()
        for parameter in architecture.shared_decoder.parameters()
    )
    local_lora = (
        len(architecture.spec.local_lora_blocks)
        * 4
        * architecture.spec.embedding_dim
        * architecture.spec.lora_rank
    )
    return {
        "route_conditioner": conditioner,
        "bottleneck": bottleneck,
        "shared_decoder": decoder,
        "public_total": conditioner + bottleneck + decoder,
        "paper_local_lora": local_lora,
        "paper_route_conditioner": conditioner,
        "paper_total_added": local_lora + conditioner,
        "paper_trainable_with_lora": (
            conditioner + bottleneck + decoder + local_lora
        ),
    }
