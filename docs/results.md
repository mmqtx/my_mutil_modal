# Experiment Results

This file is the running result table for local experiments. Outputs are stored on the dataset SSD under `/data/ljq24358/mutil_modal_datasets/experiments/`.

## Reproduction Policy

The STFAC-ECGNet paper is used as the baseline reference. Rows marked `(ours)` in the paper are local reproduction targets on our organized data. Citation-labelled comparison rows are treated as `paper_reported`, because the paper does not provide enough implementation or same-rendering details to establish that the authors reran them all on their generated grayscale images.

The intended local reproduction differs from the paper only in the image data source. Any remaining deviations are tracked explicitly in `docs/baseline_reproduction.md`.

## PTB-XL Table

Use `accuracy_label` for paper-table comparison. `accuracy_sample` is stricter exact-match accuracy and is tracked only as an auxiliary metric.

| Model | Source | Status | Best Val AUC | Best Val F1 | Test AUC | Test F1 | Test Accuracy | Output |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Xresnet1d101 | paper-reported | reference | 0.929 | 0.741 | 0.929 | 0.741 | 0.885 | STFAC Table 2 |
| FCN-Wang | paper-reported | reference | 0.926 | 0.756 | 0.926 | 0.756 | 0.880 | STFAC Table 2 |
| LSTM | paper-reported | reference | 0.927 | 0.750 | 0.927 | 0.750 | 0.876 | STFAC Table 2 |
| ResNet-Wang | paper-reported | reference | 0.749 | 0.751 | 0.749 | 0.751 | 0.877 | STFAC Table 2 |
| Inception1d | paper-reported | reference | 0.926 | 0.748 | 0.926 | 0.748 | 0.876 | STFAC Table 2 |
| ECG-DNN | paper-reported | reference | 0.924 | 0.734 | 0.924 | 0.734 | 0.884 | STFAC Table 2 |
| DNN-zhu | paper-reported | reference | 0.918 | 0.766 | 0.918 | 0.766 | 0.890 | STFAC Table 2 |
| Resnet34_1d | paper-reported | reference | 0.908 | 0.732 | 0.908 | 0.732 | 0.882 | STFAC Table 2 |
| Resnet34_2d | paper-reported | reference | 0.911 | 0.714 | 0.911 | 0.714 | 0.879 | STFAC Table 2 |
| Image_CNN | paper-reported | reference | 0.921 | 0.742 | 0.921 | 0.742 | 0.888 | STFAC Table 2 |
| SincNet | paper-reported | reference | 0.910 | 0.687 | 0.910 | 0.687 | 0.765 | STFAC Table 2 |
| SE-ResNet1 | paper-reported | reference | 0.889 | 0.667 | 0.889 | 0.667 | 0.862 | STFAC Table 2 |
| SE-ResNet12 | paper-reported | reference | 0.923 | 0.731 | 0.923 | 0.731 | 0.880 | STFAC Table 2 |
| LightX3ECG | paper-reported | reference | 0.920 | 0.734 | 0.920 | 0.734 | 0.884 | STFAC Table 2 |
| 1D-ECGNet | paper-reported | reference | 0.919 | 0.736 | 0.919 | 0.736 | 0.884 | STFAC Table 2 |
| 2D-ECGNet | paper-reported | reference | 0.929 | 0.770 | 0.929 | 0.770 | 0.892 | STFAC Table 2 |
| CAMV-RNN | paper-method local reproduction | complete | 0.8053 | 0.4435 | 0.7948 | 0.4061 | 0.8409 | `/data/ljq24358/mutil_modal_datasets/experiments/baselines/ptbxl/camv_rnn` |
| CBMV-CNN | paper-method local reproduction | complete | 0.8156 | 0.4691 | 0.8032 | 0.4557 | 0.7843 | `/data/ljq24358/mutil_modal_datasets/experiments/baselines/ptbxl/cbmv_cnn` |
| STFAC-ECGNet | paper-method local reproduction | pending | pending | pending | pending | pending | pending | `/data/ljq24358/mutil_modal_datasets/experiments/baselines/ptbxl/stfac_ecgnet` |
| HiFuse DDP finetune | local method | complete | 0.9112 | 0.7286 | 0.9069 | 0.7086 | pending | `/data/ljq24358/mutil_modal_datasets/experiments/ptbxl_hifuse_ddp_finetune` |
| HiFuse adapter-only | local method | complete | 0.9143 | 0.7324 | 0.9105 | 0.7154 | pending | `/data/ljq24358/mutil_modal_datasets/experiments/ptbxl_hifuse_adapter_only` |

## Notes

- The previous approximate baseline metrics were invalidated after architecture and training-protocol tightening, so the paper-method rows are pending rerun.
- The intended mismatch is only image data: the paper used its own generated ECG images, while this project uses our organized 12-lead ECG images.
- Earlier approximate baseline runs were moved under `/data/ljq24358/mutil_modal_datasets/experiments/baselines/ptbxl/approx_v0/` and are not used for fair comparison.
