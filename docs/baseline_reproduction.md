# Baseline Reproduction Plan

This project treats the STFAC-ECGNet paper as the baseline reference, but separates paper-reported comparison numbers from models that must be rerun on our organized data.

This is not yet a bit-level reproduction of the paper implementation. The paper does not release full training code, its exact ECG image renderer, or every implementation detail. The current completed local runs are method-level reproductions on the organized PTB-XL/GenECG Dataset A data contract.

Because the paper explicitly uses generated single-channel grayscale ECG images, paper-aligned image runs must be tracked separately from RGB GenECG runs. The RGB runs remain useful for our multimodal dataset contract, but they must not be described as a complete reproduction of the paper baseline.

## What Must Be Reproduced Locally

Rows marked `(ours)` in the paper are considered local reproduction targets because the paper describes shared preprocessing, optimizer settings, and dataset splits for them.

PTB-XL targets:

- `CAMV-RNN`: signal branch only.
- `CBMV-CNN`: image branch only.
- `STFAC-ECGNet`: signal plus image fusion.

CPSC2018 targets:

- `CAMV-RNN`
- `CBMV-CNN`
- `STFAC-ECGNet`
- `STFAC-ECGNet(only contains CBAM, without CASSAN and SSAN)`

The non-ours rows in the paper tables are citation-labelled comparison rows. The paper does not provide enough implementation or same-rendering details to claim those were rerun by the authors, so they are recorded as `paper_reported` in `baselines/stfac_paper_results.yaml`.

## Current PTB-XL Reproduction Configs

All PTB-XL baseline outputs go under:

`/data/ljq24358/mutil_modal_datasets/experiments/baselines/ptbxl/`

Configs:

- `configs/baselines/ptbxl_camv_rnn.yaml`
- `configs/baselines/ptbxl_cbmv_cnn.yaml`
- `configs/baselines/ptbxl_stfac_ecgnet.yaml`

These use the paper-aligned settings: PTB-XL 100 Hz windows, 2.5 s windows with 50% overlap, official folds 1-8/9/10, batch size 128, 10 epochs, learning rate 3e-3, OneCycleLR, and F1-based model selection.

The first local image runs use GenECG Dataset A RGB images. Paper-aligned follow-up configs convert the same images to single-channel grayscale at load time:

- `configs/baselines/ptbxl_cbmv_cnn_gray.yaml`
- `configs/baselines/ptbxl_stfac_ecgnet_gray.yaml`

These grayscale runs still cannot guarantee exact reproduction of the paper renderer, but they remove the most obvious modality mismatch.

For paper-table comparison, use `*_accuracy_label`, not `*_accuracy_sample`. `accuracy_label` matches the paper's TP/TN/FP/FN-style binary accuracy over label decisions, while `accuracy_sample` is stricter exact-match accuracy over all labels in one ECG.

## Running

Single GPU:

```bash
scripts/launch_train.sh configs/baselines/ptbxl_camv_rnn.yaml 0
scripts/launch_train.sh configs/baselines/ptbxl_cbmv_cnn.yaml 0
scripts/launch_train.sh configs/baselines/ptbxl_stfac_ecgnet.yaml 0
```

DDP:

```bash
scripts/launch_ddp.sh configs/baselines/ptbxl_camv_rnn.yaml 0,1
scripts/launch_ddp.sh configs/baselines/ptbxl_cbmv_cnn.yaml 0,1
scripts/launch_ddp.sh configs/baselines/ptbxl_stfac_ecgnet.yaml 0,1
scripts/launch_ddp.sh configs/baselines/ptbxl_cbmv_cnn_gray.yaml 0,1
scripts/launch_ddp.sh configs/baselines/ptbxl_stfac_ecgnet_gray.yaml 0,1
```

Each run writes `best.pt`, `history.json`, `test_metrics.json`, and train logs to its configured output directory.
