# My Multimodal ECG

This project trains multimodal ECG diagnosis models on the organized datasets in:

`/data/ljq24358/mutil_modal_datasets`

The main experiment is PTB-XL with the official folds:

- train: folds 1-8
- validation: fold 9
- test: fold 10
- labels: `NORM`, `CD`, `MI`, `HYP`, `STTC`

## Method

The baseline paper STFAC-ECGNet relies on hand-generated grayscale ECG images plus a signal branch. The main weakness is that the image modality is mostly a deterministic rendering of the signal, so shallow hand-designed fusion can overfit rendering artifacts instead of learning robust diagnosis features.

This project uses a stronger pretrained dual encoder:

- ECG signal encoder: GEM-compatible ECG Transformer, initialized from `cpt_wfep_epoch_20.pt`.
- ECG image encoder: local CLIP ViT-L/14-336.
- Fusion: gated residual fusion over global embeddings plus optional token-level ECG-image cross-attention.
- Training objective: asymmetric multi-label loss for label imbalance, plus optional signal-image contrastive alignment.
- Validation: per-class threshold calibration on the validation split, then fixed-threshold reporting on test.

## Paths

Pretrained weights are symlinked under `pretrained/` and ignored by git.

Default PTB-XL config:

```bash
conda activate pytorch
python scripts/train.py --config configs/ptbxl_hifuse.yaml
```

STFAC-ECGNet reproduction baseline on PTB-XL:

```bash
conda activate pytorch
python scripts/train.py --config configs/ptbxl_stfac_baseline.yaml
```

Background launch with logs on the dataset disk:

```bash
scripts/launch_train.sh configs/ptbxl_stfac_baseline.yaml 1
tail -f /data/ljq24358/mutil_modal_datasets/experiments/ptbxl_stfac_baseline/train.log
```

DDP launch:

```bash
scripts/launch_ddp.sh configs/ptbxl_hifuse.yaml 0,1
tail -f /data/ljq24358/mutil_modal_datasets/experiments/ptbxl_hifuse/train_ddp.log
```

The STFAC baseline uses the paper's PTB-XL protocol: 100Hz records, 2.5s windows with 50% overlap, official folds 1-8/9/10, multi-label one-hot targets, validation threshold tuning, and ECG-level test metrics after averaging window logits.

Quick data/model smoke test:

```bash
conda activate pytorch
python scripts/smoke_test.py --config configs/ptbxl_hifuse.yaml
```
