# Architecture contract

## Public input boundary

OCLRArchitecture consumes target-layer token features rather than raw images.
For the paper-scale specification:

- eight tensors correspond to encoder blocks 2 through 9;
- every tensor has shape B x N x 768;
- N includes one class token, four register tokens, and patch tokens;
- all tensors in one call share shape, dtype, and device.

This makes the Global/Local reconstruction architecture executable without
publishing the frozen-backbone execution and experiment pipeline.

## Global route

forward_global(features):

1. averages the eight target-layer features to form decoder input;
2. applies Global branch normalization and embedding;
3. passes tokens through the reconstruction bottleneck;
4. executes the shared eight-block decoder;
5. reverses decoder history and fuses it into two groups of four;
6. returns fused teacher features, reconstructed features, and final tokens.

## Local route

forward_local(teacher_features, adapted_features) uses frozen teacher features
as reconstruction targets and the externally adapted feature path as decoder
input. If adapted features are omitted, teacher features are used so the
public graph can be smoke-tested without an encoder.

LocalOnlyQVLora independently demonstrates the adapter rule:

- the base fused QKV projection is frozen;
- the Global route returns the base QKV output;
- the Local route adds trainable low-rank changes to Q and V;
- K is unchanged.

The up-projections are zero-initialized, so local and global outputs start
identically before optimization.

## Shared decoder guarantee

The model owns exactly one shared_decoder ModuleList. Both route methods call
the same forward implementation and cannot silently instantiate route-specific
decoder copies.

## Outputs

Every call returns RouteOutput:

| Field | Meaning |
|---|---|
| route | global or local |
| teacher_features | two fused reconstruction targets |
| decoder_features | two fused decoder outputs |
| decoder_tokens | final decoder token sequence |

The feature-to-anomaly transformation is deliberately outside this contract.
