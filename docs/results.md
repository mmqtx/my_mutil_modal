# Experiment Results

This file is the running result table for local experiments. Outputs are stored on the dataset SSD under `/data/ljq24358/mutil_modal_datasets/experiments/`.

## Reproduction Policy

The STFAC-ECGNet paper is used as the baseline reference. Rows marked `(ours)` in the paper are local reproduction targets on our organized data. Citation-labelled comparison rows are treated as `paper_reported`, because the paper does not provide enough implementation or same-rendering details to establish that the authors reran them all on their generated grayscale images.

The intended local reproduction differs from the paper only in the image data source. Any remaining deviations are tracked explicitly in `docs/baseline_reproduction.md`.

## PTB-XL Table

Columns match STFAC-ECGNet Table 2. Local Accuracy is `accuracy_label`, i.e. `(TP + TN) / (TP + FP + FN + TN)` over binary label decisions. Recall, Precision, F1-score, and AUC are macro averages across the five PTB-XL superclass labels.

| Model | Source | Status | Accuracy | AUC | Recall | Precision | F1-score | Output |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Xresnet1d101 | paper-reported | reference | 0.885 | 0.929 | 0.705 | 0.780 | 0.741 | STFAC Table 2 |
| FCN-Wang | paper-reported | reference | 0.880 | 0.926 | 0.786 | 0.729 | 0.756 | STFAC Table 2 |
| LSTM | paper-reported | reference | 0.876 | 0.927 | 0.800 | 0.706 | 0.750 | STFAC Table 2 |
| ResNet-Wang | paper-reported | reference | 0.877 | 0.749 | 0.795 | 0.712 | 0.751 | STFAC Table 2 |
| Inception1d | paper-reported | reference | 0.876 | 0.926 | 0.788 | 0.711 | 0.748 | STFAC Table 2 |
| ECG-DNN | paper-reported | reference | 0.884 | 0.924 | 0.684 | 0.793 | 0.734 | STFAC Table 2 |
| DNN-zhu | paper-reported | reference | 0.890 | 0.918 | 0.774 | 0.758 | 0.766 | STFAC Table 2 |
| Resnet34_1d | paper-reported | reference | 0.882 | 0.908 | 0.691 | 0.778 | 0.732 | STFAC Table 2 |
| Resnet34_2d | paper-reported | reference | 0.879 | 0.911 | 0.706 | 0.722 | 0.714 | STFAC Table 2 |
| Image_CNN | paper-reported | reference | 0.888 | 0.921 | 0.764 | 0.721 | 0.742 | STFAC Table 2 |
| SincNet | paper-reported | reference | 0.765 | 0.910 | 0.662 | 0.714 | 0.687 | STFAC Table 2 |
| SE-ResNet1 | paper-reported | reference | 0.862 | 0.889 | 0.599 | 0.753 | 0.667 | STFAC Table 2 |
| SE-ResNet12 | paper-reported | reference | 0.880 | 0.923 | 0.694 | 0.772 | 0.731 | STFAC Table 2 |
| LightX3ECG | paper-reported | reference | 0.884 | 0.920 | 0.681 | 0.795 | 0.734 | STFAC Table 2 |
| 1D-ECGNet | paper-reported | reference | 0.884 | 0.919 | 0.696 | 0.780 | 0.736 | STFAC Table 2 |
| 2D-ECGNet | paper-reported | reference | 0.892 | 0.929 | 0.790 | 0.752 | 0.770 | STFAC Table 2 |
| CAMV-RNN | paper-method local reproduction | complete | 0.8110 | 0.7859 | 0.5508 | 0.3445 | 0.4061 | `/data/ljq24358/mutil_modal_datasets/experiments/baselines/ptbxl/camv_rnn` |
| CBMV-CNN | paper-method local reproduction | complete | 0.7843 | 0.8032 | 0.6578 | 0.3906 | 0.4557 | `/data/ljq24358/mutil_modal_datasets/experiments/baselines/ptbxl/cbmv_cnn` |
| STFAC-ECGNet | paper-method local reproduction | complete | 0.8170 | 0.7834 | 0.5363 | 0.3454 | 0.4049 | `/data/ljq24358/mutil_modal_datasets/experiments/baselines/ptbxl/stfac_ecgnet` |
| HiFuse DDP finetune | local method | complete | pending | 0.9069 | pending | pending | 0.7086 | `/data/ljq24358/mutil_modal_datasets/experiments/ptbxl_hifuse_ddp_finetune` |
| HiFuse adapter-only | local method | complete | pending | 0.9105 | pending | pending | 0.7154 | `/data/ljq24358/mutil_modal_datasets/experiments/ptbxl_hifuse_adapter_only` |
| HiFuse + TPA (free attention) | local method | complete | 0.8629 | 0.9070 | 0.7496 | 0.6758 | 0.7097 | `/data/ljq24358/mutil_modal_datasets/experiments/ptbxl_hifuse_tpa` |
| HiFuse + structure-aware TPA | local method | complete | 0.8679 | 0.9025 | 0.7054 | 0.7121 | 0.7063 | `/data/ljq24358/mutil_modal_datasets/experiments/ptbxl_hifuse_tpa_structure_aware` |
| HiFuse + structure-aware TPA bidirectional | local method | pending | pending | pending | pending | pending | pending | `/data/ljq24358/mutil_modal_datasets/experiments/ptbxl_hifuse_tpa_structure_aware_bidir` |

## Notes

- CAMV-RNN and STFAC-ECGNet were rerun after the CAMV branch was tightened to match the paper's batch-dimension global max/average fusion and skip-branch preprocessing.
- The intended mismatch is only image data: the paper used its own generated ECG images, while this project uses our organized 12-lead ECG images.
- Earlier approximate baseline runs were moved under `/data/ljq24358/mutil_modal_datasets/experiments/baselines/ptbxl/approx_v0/` and are not used for fair comparison.
