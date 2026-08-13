# Release scope

This repository is an executable architecture reference, not a complete
reproduction package.

## Included

- paper-scale architecture specification;
- route-specific conditioning;
- local-only Q/V LoRA projection;
- shared bottleneck and FP16-safe linear-attention decoder;
- feature fusion and post-encoder Global/Local forward paths;
- parameter accounting;
- CPU smoke test and architecture-level unit tests;
- manuscript-reported benchmark tables.

## Not included

- raw-image transforms or dataset loaders;
- crop geometry implementation;
- DINOv2 construction, weight loading, or block hooks;
- training objectives, including overlap-target construction;
- training loop and exact experiment configuration;
- inference aggregation and Gaussian postprocessing;
- evaluation metrics and multi-view grouping;
- pretrained weights or checkpoints;
- checkpoint mapping and conversion.

The distinction is intentional and stated in documentation and code. Nothing
in this repository should be cited as reproducing the manuscript results.
