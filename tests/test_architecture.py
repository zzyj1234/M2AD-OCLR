import torch
from torch import nn

from oclr import (
    ArchitectureSpec,
    BranchConditioner,
    LocalOnlyQVLora,
    OCLRArchitecture,
)


def test_architecture_description():
    architecture = OCLRArchitecture()
    description = architecture.describe()
    assert description["public_scope"] == "architecture-only"
    assert description["decoder_depth"] == 8
    assert description["inference_route"] == "local"


def test_route_conditioner_and_lora_shapes():
    conditioner = BranchConditioner(16)
    tokens = torch.randn(2, 5, 16)
    assert conditioner(tokens, "local").shape == tokens.shape
    projection = LocalOnlyQVLora(nn.Linear(16, 48), rank=2, alpha=4)
    assert projection(tokens, local_route=True).shape == (2, 5, 48)


def test_forward_is_intentionally_unavailable():
    architecture = OCLRArchitecture(
        ArchitectureSpec(embedding_dim=16, decoder_heads=4, decoder_depth=2)
    )
    try:
        architecture(torch.randn(1, 5, 16))
    except RuntimeError as error:
        assert "architecture-only" in str(error)
    else:
        raise AssertionError("Architecture-only forward must remain unavailable")
