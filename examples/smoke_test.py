"""Run both public OCLR routes on synthetic post-encoder features."""

import json

import torch

from oclr import ArchitectureSpec, OCLRArchitecture, parameter_audit


def main():
    # A compact specification keeps this smoke test CPU-friendly. The public
    # API and execution graph are identical to the paper-scale specification.
    spec = ArchitectureSpec(
        target_layers=(0, 1, 2, 3),
        encoder_fusion=((0, 1), (2, 3)),
        decoder_fusion=((0, 1), (2, 3)),
        decoder_depth=4,
        decoder_heads=4,
        embedding_dim=32,
        register_tokens=1,
        local_lora_blocks=(1, 2),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    architecture = OCLRArchitecture(spec).eval()
    teacher = [
        torch.randn(1, 18, spec.embedding_dim)
        for _ in spec.target_layers
    ]
    adapted = [feature + 0.01 * torch.randn_like(feature) for feature in teacher]
    with torch.no_grad():
        global_output = architecture.forward_global(teacher)
        local_output = architecture.forward_local(teacher, adapted)

    print(json.dumps(architecture.describe(), indent=2))
    print("global:", tuple(global_output.decoder_tokens.shape))
    print("local: ", tuple(local_output.decoder_tokens.shape))
    print("shared decoder object:", id(architecture.shared_decoder))
    print("tiny audit:", parameter_audit(architecture))


if __name__ == "__main__":
    main()
