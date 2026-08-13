import torch
from torch import nn

from oclr import (
    ArchitectureSpec,
    BranchConditioner,
    LinearAttentionDecoderBlock,
    LocalOnlyQVLora,
    OCLRArchitecture,
    parameter_audit,
)


def tiny_spec():
    return ArchitectureSpec(
        target_layers=(0, 1, 2, 3),
        encoder_fusion=((0, 1), (2, 3)),
        decoder_fusion=((0, 1), (2, 3)),
        decoder_depth=4,
        decoder_heads=4,
        embedding_dim=16,
        register_tokens=1,
        local_lora_blocks=(1, 2),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )


def fake_features(spec, batch=2, tokens=6):
    return [
        torch.randn(batch, tokens, spec.embedding_dim)
        for _ in spec.target_layers
    ]


def test_global_and_local_routes_execute():
    spec = tiny_spec()
    architecture = OCLRArchitecture(spec).eval()
    teacher = fake_features(spec)
    adapted = [feature + 0.01 for feature in teacher]
    global_output = architecture.forward_global(teacher)
    local_output = architecture.forward_local(teacher, adapted)
    assert global_output.route == "global"
    assert local_output.route == "local"
    assert len(global_output.teacher_features) == 2
    assert len(local_output.decoder_features) == 2
    assert local_output.decoder_tokens.shape == (2, 6, 16)
    assert all(item.shape == (2, 6, 16) for item in local_output.decoder_features)


def test_both_routes_use_the_same_decoder_parameters():
    spec = tiny_spec()
    architecture = OCLRArchitecture(spec).eval()
    features = fake_features(spec)
    decoder_ids_before = [id(block) for block in architecture.shared_decoder]
    architecture.forward_global(features)
    architecture.forward_local(features)
    decoder_ids_after = [id(block) for block in architecture.shared_decoder]
    assert decoder_ids_before == decoder_ids_after


def test_local_lora_changes_q_and_v_but_not_k():
    torch.manual_seed(3)
    base = nn.Linear(16, 48)
    adapter = LocalOnlyQVLora(base, rank=2, alpha=4, dropout=0.0)
    nn.init.constant_(adapter.q_up.weight, 0.2)
    nn.init.constant_(adapter.v_up.weight, -0.2)
    tokens = torch.randn(2, 5, 16)
    baseline = adapter(tokens, local_route=False)
    local = adapter(tokens, local_route=True)
    base_q, base_k, base_v = baseline.chunk(3, dim=-1)
    local_q, local_k, local_v = local.chunk(3, dim=-1)
    assert not torch.equal(base_q, local_q)
    torch.testing.assert_close(base_k, local_k)
    assert not torch.equal(base_v, local_v)
    assert all(not parameter.requires_grad for parameter in adapter.base.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in adapter.named_parameters()
        if not name.startswith("base.")
    )


def test_route_conditioners_are_independent():
    conditioner = BranchConditioner(16)
    with torch.no_grad():
        conditioner.embeddings["local"].fill_(1.0)
    tokens = torch.randn(2, 5, 16)
    global_tokens = conditioner(tokens, "global")
    local_tokens = conditioner(tokens, "local")
    assert not torch.equal(global_tokens, local_tokens)


def test_linear_attention_is_finite_and_backward_works():
    block = LinearAttentionDecoderBlock(16, 4)
    tokens = torch.randn(2, 7, 16, requires_grad=True)
    output = block(tokens)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_linear_attention_low_precision_forward_is_finite():
    for dtype in (torch.float16, torch.bfloat16):
        block = LinearAttentionDecoderBlock(16, 4).to(dtype=dtype)
        tokens = torch.randn(2, 7, 16, dtype=dtype)
        with torch.no_grad():
            output = block(tokens)
        assert output.dtype == dtype
        assert torch.isfinite(output).all()


def test_reported_added_parameter_count():
    architecture = OCLRArchitecture()
    audit = parameter_audit(architecture)
    assert audit["paper_local_lora"] == 49_152
    assert audit["paper_route_conditioner"] == 4_608
    assert audit["paper_total_added"] == 53_760
    assert audit["paper_trainable_with_lora"] == 61_479_168


def test_invalid_shapes_are_rejected():
    spec = tiny_spec()
    architecture = OCLRArchitecture(spec)
    features = fake_features(spec)
    try:
        architecture(features[:-1], route="global")
    except ValueError as error:
        assert "Expected 4" in str(error)
    else:
        raise AssertionError("Missing target features must be rejected")
