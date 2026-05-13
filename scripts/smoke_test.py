#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_mutil_modal.data.ptbxl import PTBXLMultimodalDataset
from my_mutil_modal.models import HiFuseECG, load_gem_signal_weights
from my_mutil_modal.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ptbxl_hifuse.yaml")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ds = PTBXLMultimodalDataset(
        manifest=cfg["data"]["manifest"],
        root=cfg["data"]["root"],
        split="train",
        label_columns=cfg["data"]["label_columns"],
        signal_column=cfg["data"]["signal_column"],
        image_column=cfg["data"]["image_column"],
        signal_length=int(cfg["data"]["signal_length"]),
        image_size=int(cfg["data"]["image_size"]),
        limit=2,
    )
    sample = ds[0]
    print("dataset_len", len(ds))
    print("signal_shape", tuple(sample["signal"].shape))
    print("image_shape", tuple(sample["image"].shape))
    print("labels", sample["labels"].tolist())
    if args.skip_model:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiFuseECG(
        num_classes=len(cfg["data"]["label_columns"]),
        clip_model_path=cfg["model"]["clip_model_path"],
        fusion_dim=int(cfg["model"]["fusion_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        freeze_signal_encoder=True,
        freeze_image_encoder=True,
    )
    load_gem_signal_weights(model, cfg["model"]["signal_checkpoint"])
    model.to(device).eval()
    signal = torch.stack([ds[i]["signal"] for i in range(2)]).to(device)
    image = torch.stack([ds[i]["image"] for i in range(2)]).to(device)
    with torch.no_grad():
        out = model(signal, image)
    print("logits_shape", tuple(out["logits"].shape))
    print("gate_mean", float(out["fusion_gate"].mean().detach().cpu()))


if __name__ == "__main__":
    main()
