#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from my_mutil_modal.data import build_dataloaders
from scripts.train import build_model, class_pos_weight, run_epoch
from my_mutil_modal.training.metrics import compute_metrics
from my_mutil_modal.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ptbxl_hifuse.yaml")
    parser.add_argument("--checkpoint", default="outputs/ptbxl_hifuse/best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg, device, no_pretrained=True)
    model.load_state_dict(ckpt["model"])
    loaders = build_dataloaders(cfg)
    pos_weight = class_pos_weight(cfg).to(device)
    test_loss, targets, logits = run_epoch(model, loaders["test"], device, pos_weight, False, None, None, cfg)
    metrics, _ = compute_metrics(targets, logits, ckpt["thresholds"])
    print(json.dumps({"test_loss": test_loss, **metrics}, indent=2))


if __name__ == "__main__":
    main()
