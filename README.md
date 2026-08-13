# OCLR

Executable architecture reference for **Overlap-consistent Local
Reconstruction (OCLR)** for high-resolution industrial anomaly localization.

This release sits between a static diagram and a full reproduction package:

- both Global and Local post-encoder routes execute end to end;
- both routes provably reuse the same reconstruction decoder;
- route conditioning, Local-only Q/V LoRA, feature fusion, and FP16-safe
  linear attention are implemented and tested;
- paper-scale parameter counts are audited automatically;
- data processing, training objectives, overlap-target construction, anomaly
  aggregation, evaluation, checkpoints, and weights are not released.

The code validates the manuscript architecture, but it cannot reproduce the
reported benchmark results by itself.

## Architecture

    Global view -> frozen encoder -> Global conditioner --+
                                                         |
                                              shared bottleneck
                                                         |
                                              shared 8-block decoder
                                                         |
    Local view  -> teacher target ----------------------> reconstruction
              -> Local Q/V LoRA -> Local conditioner --+
                                                         |
                                  omitted overlap agreement and map aggregation

The public boundary begins at eight target-layer token tensors from an
external frozen encoder. Each feature has shape B x N x C, including
class/register tokens.

| Manuscript component | Public implementation |
|---|---|
| Target blocks 2--9 | ArchitectureSpec.target_layers |
| Two groups of four encoder features | encoder_fusion |
| Global/Local route conditioning | BranchConditioner |
| Local-only Q/V adaptation | LocalOnlyQVLora |
| Shared reconstruction bottleneck | BottleneckMLP |
| Shared eight-block decoder | LinearAttentionDecoderBlock |
| Global feature route | OCLRArchitecture.forward_global |
| Local teacher/adapted route | OCLRArchitecture.forward_local |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the tensor contract and
[docs/RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md) for the release boundary.

## Installation

Python 3.9+ and PyTorch 2.1+ are required.

    git clone https://github.com/zzyj1234/M2AD-OCLR.git
    cd M2AD-OCLR
    pip install -e .

## Executable smoke test

The CPU-friendly example uses a reduced dimension while preserving the public
execution graph:

    python examples/smoke_test.py

Equivalent API usage:

    import torch
    from oclr import ArchitectureSpec, OCLRArchitecture

    spec = ArchitectureSpec(
        target_layers=(0, 1, 2, 3),
        encoder_fusion=((0, 1), (2, 3)),
        decoder_fusion=((0, 1), (2, 3)),
        decoder_depth=4,
        decoder_heads=4,
        embedding_dim=32,
        local_lora_blocks=(1, 2),
        lora_rank=2,
    )
    model = OCLRArchitecture(spec).eval()
    teacher = [torch.randn(1, 18, 32) for _ in spec.target_layers]
    adapted = [feature + 0.01 * torch.randn_like(feature) for feature in teacher]

    global_output = model.forward_global(teacher)
    local_output = model.forward_local(teacher, adapted)
    print(global_output.decoder_tokens.shape)  # torch.Size([1, 18, 32])
    print(local_output.decoder_tokens.shape)   # torch.Size([1, 18, 32])

No DINOv2 download, dataset, or checkpoint is needed for the smoke test.

## Parameter audit

Run:

    python examples/audit_paper_architecture.py

Expected paper-scale audit:

| Component | Parameters |
|---|---:|
| Route conditioner | 4,608 |
| Reconstruction bottleneck | 4,722,432 |
| Shared decoder | 56,702,976 |
| Local Q/V LoRA | 49,152 |
| Total trainable architecture | **61,479,168** |
| OCLR additions over the shared baseline | **53,760** |

The final two values agree with the manuscript's 61.479M trainable parameters
and 53,760 added parameters.

## Verification

    pip install -e ".[test]"
    pytest -q

Tests verify executable routes, shapes, shared decoder reuse, independent
conditioners, Local Q/V-only changes, frozen base projections, finite
attention outputs/gradients, exact parameter counts, and invalid inputs.

## Deliberately omitted

- raw-image preprocessing and crop sampling;
- DINOv2 loading/execution and in-place adapter injection;
- reconstruction and overlap-consistency objectives;
- training loop, optimizer, scheduler, and exact configuration;
- source-space anomaly-map generation and overlap aggregation;
- object/view grouping and AUROC/AP/mF1/AUPRO evaluation;
- checkpoint I/O, pretrained weights, and experiment outputs.

This repository is an **executable architecture reference**, not complete
source code or a reproducibility package.

## Plastic Gear data

- Archive: M2AD_Gear.rar
- Baidu Netdisk:
  https://pan.baidu.com/s/17cXv5vDcnG01WtFPnC4YIQ?pwd=aila
- Extraction code: aila

The archive is not included here. Users must comply with the terms applying
to the source images and dataset. This release has no loader or training
recipe for the archive.

| Dataset | Train normal | Test normal | Test anomalous | Categories |
|---|---:|---:|---:|---:|
| M2AD-Synergy | 35,880 | 33,079 | 50,800 | 10 |
| Real-IAD | 43,640 | 56,081 | 51,329 | 30 |
| Plastic Gear | 111 | 15 | 59 | 1 |

M2AD-Synergy counts are from an audit of 119,759 decodable records. Plastic
Gear uses 37 repaired normal sources for training and five for normal testing;
horizontal flip and 180-degree rotation yield 111 and 15 normal images. Its
anomalous test set contains 59 distinct 896x896 images with masks and five
labels: damage, dark spot, dirt, flash, and hair.

## Reported results

These percentages are transcribed from the manuscript. They are final-epoch,
single-seed results and are not reproduced by this architecture release.

| Dataset / method | Object AUROC | View AUROC | Pixel AUROC | Pixel mF1 | Pixel AP | AUPRO |
|---|---:|---:|---:|---:|---:|---:|
| M2AD / Dinomaly-FP16 | 93.000 | 86.442 | 95.643 | 16.380 | 9.346 | 80.785 |
| M2AD / OCLR | **93.660** | **87.470** | **97.720** | **36.236** | **29.586** | **91.341** |
| Real-IAD / Dinomaly-FP16 | 94.218 | 88.344 | 98.002 | 32.343 | 25.822 | 87.270 |
| Real-IAD / OCLR | **94.857** | **90.488** | **98.728** | **48.597** | **46.129** | **94.544** |
| Plastic Gear / Dinomaly-FP16 | 92.542 | 92.542 | 94.910 | 17.154 | 8.673 | 68.414 |
| Plastic Gear / OCLR | **97.610** | **97.610** | **98.467** | **28.011** | **20.618** | **84.732** |

On the reported system, OCLR has 148.063M total parameters, 61.479M trainable
parameters, a 234.564 MiB compact checkpoint, 3.797 GiB peak training memory,
and 16.223 FPS. Plastic Gear is exploratory and is not production validation.

## Manuscript status

The manuscript is currently **in preparation**. It has not yet been submitted,
so this repository does not call it "under submission" or assign a
provisional journal citation.

The intended journal uses double-anonymized review. Keep this identity-bearing
repository private during review or give reviewers a genuinely anonymized
snapshot with no account, commit-author, affiliation, acknowledgement, or
funding identifiers.

## Acknowledgements

OCLR was developed from the reconstruction architecture of
[Dinomaly](https://github.com/guojiajeremy/Dinomaly). The encoder design is
based on [DINOv2](https://github.com/facebookresearch/dinov2). Dataset
protocols build on [M2AD](https://github.com/hustCYQ/M2AD) and
[Real-IAD](https://realiad4ad.github.io/Real-IAD/).

No third-party source tree, dataset, pretrained weight, checkpoint, or
experiment artifact is included. See [THIRD_PARTY.md](THIRD_PARTY.md) and
[NOTICE](NOTICE).

## License

The published architecture reference is under the
[Apache License 2.0](LICENSE). Third-party projects and datasets retain their
own licenses and terms.
