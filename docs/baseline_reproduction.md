# Baseline Reproduction Plan

This project treats the STFAC-ECGNet paper as the baseline reference, but separates paper-reported comparison numbers from models that must be rerun on our organized data.

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

The image branch uses GenECG Dataset A RGB images. We do not force grayscale conversion because the project data contract is to reproduce the baseline method on our prepared multimodal dataset, not to recreate the paper's exact image renderer.

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
```

Each run writes `best.pt`, `history.json`, `test_metrics.json`, and train logs to its configured output directory.
