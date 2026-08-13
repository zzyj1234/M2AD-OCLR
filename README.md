# OCLR

Architecture-only reference for **Overlap-consistent Local Reconstruction
(OCLR)**, a method for high-resolution industrial anomaly localization.

> **Release scope:** this repository is intended to communicate the model
> architecture and component relationships. It is not an official
> reproducibility package.

## What is public

- the global/local route structure;
- the route-specific conditioner;
- the shared linear-attention reconstruction decoder;
- the local-only Q/V LoRA module;
- the target-layer and fusion-group specification reported in the manuscript;
- reported benchmark results and dataset information.

## What is intentionally not public

- the end-to-end forward and backward execution path;
- training objectives and overlap-target construction;
- data preprocessing and sampling code;
- source-space inference aggregation;
- checkpoint loading or conversion code;
- experiment configurations, commands, and pretrained checkpoints;
- the original multi-method development framework.

The omitted components are represented explicitly by an exception in
OCLRArchitecture.forward. This makes the release scope unambiguous: the code
can be inspected as an architecture reference, but it cannot be used as a
drop-in training or inference package.

## Architecture

    Input
      |
      +-- Global route (training only)
      |     |
      |     +-- Frozen DINOv2-reg encoder
      |     +-- Global route conditioner
      |     -- Shared reconstruction decoder
      |
      -- Overlapping local routes (training and inference)
            |
            +-- Frozen teacher features
            +-- Local-only Q/V LoRA features
            +-- Local route conditioner
            -- Shared reconstruction decoder

During training, the global and local routes regularize the same decoder, and
overlapping local predictions are encouraged to agree. At inference, the
global route is removed and only overlapping local reconstruction is used.

The public component graph is implemented in
src/oclr/architecture.py:

- ArchitectureSpec records the model-level design.
- BranchConditioner implements independent route normalization and embeddings.
- LocalOnlyQVLora shows how the local route adapts Q and V while leaving K and
  the global route unchanged.
- LinearAttentionDecoderBlock shows the shared decoder block family and
  FP32 accumulation used for reduced-precision safety.
- OCLRArchitecture assembles the inspectable public modules but intentionally
  has no end-to-end forward.

## Installation and inspection

Python 3.9+ and PyTorch 2.1+ are required.

    pip install -e .

Inspect the architecture:

    from oclr import OCLRArchitecture

    architecture = OCLRArchitecture()
    print(architecture)
    print(architecture.describe())

Calling architecture(...) raises a RuntimeError by design.

Run the architecture-level tests:

    pip install -e ".[test]"
    pytest -q

## Plastic Gear data

The dataset bundle shared separately from this repository is:

- Archive: M2AD_Gear.rar
- Baidu Netdisk: https://pan.baidu.com/s/17cXv5vDcnG01WtFPnC4YIQ?pwd=aila
- Extraction code: aila

The archive is not part of the software distribution. Users must comply with
the terms applying to the source images and dataset. This architecture-only
repository does not include a loader or training recipe for the archive.

Dataset facts reported in the manuscript:

| Dataset | Train normal | Test normal | Test anomalous | Categories |
|---|---:|---:|---:|---:|
| M2AD-Synergy | 35,880 | 33,079 | 50,800 | 10 |
| Real-IAD | 43,640 | 56,081 | 51,329 | 30 |
| Plastic Gear | 111 | 15 | 59 | 1 |

M2AD-Synergy counts are from an audit of 119,759 decodable records. Plastic
Gear uses 37 repaired normal sources for training and five for normal testing;
horizontal flip and 180-degree rotation yield 111 and 15 normal images. Its
anomalous test set contains 59 distinct 896x896 images with pixel masks and
five defect labels: damage, dark spot, dirt, flash, and hair.

## Reported results

The numbers below are percentages transcribed from the manuscript. They are
final-epoch, single-seed results and are not reproduced by this
architecture-only repository.

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
and 16.223 FPS. Plastic Gear is an exploratory benchmark and should not be
interpreted as production validation.

## Manuscript status and citation

The manuscript is currently **in preparation**. It has not yet been submitted,
so this repository does not describe it as "under submission" and does not
provide a provisional paper citation with invented author or venue metadata.

A complete BibTeX record will be added when the manuscript is publicly
available or bibliographic information has been assigned. Until then, cite
this repository only if a repository URL, author list, release version, and
release date have been added by the maintainers.

Because the intended journal uses double-anonymized review, maintainers should
consider keeping this repository private during review or providing reviewers
with a genuinely anonymized snapshot that contains no account, commit, author,
affiliation, acknowledgement, or funding identifiers.

## Acknowledgements

OCLR was developed from the reconstruction architecture of
[Dinomaly](https://github.com/guojiajeremy/Dinomaly). The encoder design is
based on [DINOv2](https://github.com/facebookresearch/dinov2). Dataset
protocols build on [M2AD](https://github.com/hustCYQ/M2AD) and
[Real-IAD](https://realiad4ad.github.io/Real-IAD/).

No Dinomaly, DINOv2, M2AD, ADer, dataset, pretrained weight, checkpoint, or
experiment artifact is included here. See [THIRD_PARTY.md](THIRD_PARTY.md)
and [NOTICE](NOTICE).

## License

The architecture reference is released under the
[Apache License 2.0](LICENSE). Apache-2.0 permits use, modification, and
redistribution of the published files. The release is harder to reproduce
because implementation components are omitted, not because the license
prohibits legitimate reuse.
