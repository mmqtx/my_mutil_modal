# Fine-grained Temporal-Patch Alignment

TPA is the main local method built on the existing pretrained ECG signal encoder and CLIP ECG image encoder.

## Motivation

Existing signal-image ECG fusion methods usually aggregate each modality into a global representation before fusion. This coarse fusion can ignore the correspondence between temporal ECG segments and spatial regions in ECG images.

TPA keeps both token sequences:

- signal temporal tokens `S = [s1, ..., sT]`
- image patch tokens `P = [p1, ..., pN]`

It then performs cross-attention:

```text
S_aligned = CrossAttention(Q=S, K=P, V=P)
S_fused = S + S_aligned
F = concat(pool(S), pool(S_fused), pool(P))
```

`F` is sent to an MLP classifier.

## Configs

- `configs/ptbxl_hifuse_tpa.yaml`: single-direction signal-to-image TPA.
- `configs/ptbxl_hifuse_tpa_bidir.yaml`: bidirectional TPA, adding image-to-signal alignment.

Both configs reuse the existing pretrained signal checkpoint and CLIP vision encoder paths, initialize compatible weights from the adapter-only HiFuse run, and write outputs under `/data/ljq24358/mutil_modal_datasets/experiments/`.
