# Fine-grained Temporal-Patch Alignment

TPA is the main local method built on the existing pretrained ECG signal encoder and CLIP ECG image encoder.

## Motivation

Existing signal-image ECG fusion methods usually aggregate each modality into a global representation before fusion. This coarse fusion can ignore the correspondence between temporal ECG segments and spatial regions in ECG images.

TPA keeps both token sequences:

- signal temporal tokens `S = [s1, ..., sT]`
- image patch tokens `P = [p1, ..., pN]`

The current structure-aware TPA adds a temporal geometry prior before cross-attention:

- signal token time position: `t_s in [0, 1]`
- image patch time position from patch x-coordinate: `t_p in [0, 1]`
- attention bias: `bias(i,j) = -alpha * abs(t_s[i] - t_p[j])`

It then performs time-biased cross-attention:

```text
S_aligned = CrossAttention(Q=S, K=P, V=P, bias_time)
S_fused = S + S_aligned
F = concat(pool(S), pool(S_fused), pool(P))
```

`F` is sent to an MLP classifier.

The training objective adds time-region contrastive alignment. Image patches are pooled by x-position into the same number of temporal bins as signal tokens, producing `P_time = [p1, ..., pT]`. For each sample, `s_t` is pulled toward `p_t` and pushed away from other `p_k`.

```text
L = L_cls + lambda_align * L_time_align
```

## Configs

- `configs/ptbxl_hifuse_tpa.yaml`: single-direction structure-aware signal-to-image TPA.
- `configs/ptbxl_hifuse_tpa_timebias_only.yaml`: ablation that keeps time-biased attention and removes time-region contrastive alignment loss.
- `configs/ptbxl_hifuse_tpa_bidir.yaml`: bidirectional structure-aware TPA, adding image-to-signal alignment.
- `configs/ptbxl_hifuse_tpa_free.yaml`: archived free-attention TPA config for reproducing the completed non-structure-aware run.

Both configs reuse the existing pretrained signal checkpoint and CLIP vision encoder paths, initialize compatible weights from the adapter-only HiFuse run, and write outputs under `/data/ljq24358/mutil_modal_datasets/experiments/`.
