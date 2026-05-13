#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_mutil_modal.data import build_dataloaders
from my_mutil_modal.models import HiFuseECG, STFACECGNet, load_gem_signal_weights
from my_mutil_modal.training.losses import multilabel_bce_loss, symmetric_contrastive_loss
from my_mutil_modal.training.metrics import compute_metrics, tune_thresholds
from my_mutil_modal.utils.config import ensure_dir, load_config
from my_mutil_modal.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ptbxl_hifuse.yaml")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def class_pos_weight(cfg: Dict) -> torch.Tensor:
    df = pd.read_csv(cfg["data"]["manifest"])
    train = df[df["split"] == "train"]
    labels = train[cfg["data"]["label_columns"]].astype(np.float32).to_numpy()
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    return torch.tensor(neg / np.clip(pos, 1.0, None), dtype=torch.float32)


def build_model(cfg: Dict, device: torch.device, no_pretrained: bool = False) -> HiFuseECG:
    model_cfg = cfg["model"]
    if model_cfg.get("name", "hifuse").lower() == "stfac":
        model = STFACECGNet(
            num_classes=len(cfg["data"]["label_columns"]),
            image_in_channels=int(model_cfg.get("image_in_channels", cfg["data"].get("image_channels", 3))),
            signal_hidden=int(model_cfg.get("signal_hidden", 64)),
            image_channels=int(model_cfg.get("image_channels", 128)),
            fusion_dim=int(model_cfg.get("fusion_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
        return model.to(device)
    model = HiFuseECG(
        num_classes=len(cfg["data"]["label_columns"]),
        clip_model_path=model_cfg["clip_model_path"],
        fusion_dim=int(model_cfg.get("fusion_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        freeze_signal_encoder=bool(model_cfg.get("freeze_signal_encoder", True)),
        freeze_image_encoder=bool(model_cfg.get("freeze_image_encoder", True)),
    )
    if not no_pretrained and model_cfg.get("signal_checkpoint"):
        load_gem_signal_weights(model, model_cfg["signal_checkpoint"])
    return model.to(device)


def optimizer_for(model: HiFuseECG, cfg: Dict) -> torch.optim.Optimizer:
    base_lr = float(cfg["train"]["lr"])
    encoder_lr = float(cfg["train"].get("encoder_lr", base_lr))
    weight_decay = float(cfg["train"].get("weight_decay", 0.05))
    encoder, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("signal_encoder", "image_encoder")):
            encoder.append(p)
        else:
            head.append(p)
    groups = [{"params": head, "lr": base_lr, "weight_decay": weight_decay}]
    if encoder:
        groups.append({"params": encoder, "lr": encoder_lr, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups)


def aggregate_by_ecg_id(targets: np.ndarray, logits: np.ndarray, ecg_ids: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    grouped: Dict[str, list[int]] = {}
    for i, ecg_id in enumerate(ecg_ids):
        grouped.setdefault(ecg_id, []).append(i)
    agg_targets, agg_logits = [], []
    for idxs in grouped.values():
        agg_targets.append(targets[idxs[0]])
        agg_logits.append(logits[idxs].mean(axis=0))
    return np.asarray(agg_targets), np.asarray(agg_logits)


def run_epoch(
    model: HiFuseECG,
    loader,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: Dict,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(train)
    logits_all, targets_all, ecg_ids_all = [], [], []
    losses = []
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    contrastive_weight = float(cfg["model"].get("contrastive_weight", 0.0))
    modality_dropout = float(cfg["model"].get("modality_dropout", 0.0)) if train else 0.0
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch in iterator:
        signal = batch["signal"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=amp):
            out = model(signal, image, modality_dropout=modality_dropout)
            loss = multilabel_bce_loss(out["logits"], targets, pos_weight=pos_weight)
            if contrastive_weight > 0 and "signal_z" in out and "image_z" in out:
                loss = loss + contrastive_weight * symmetric_contrastive_loss(
                    out["signal_z"], out["image_z"], model.logit_scale
                )
        if train:
            assert optimizer is not None
            if scaler is not None and amp:
                scaler.scale(loss).backward()
                if float(cfg["train"].get("grad_clip_norm", 0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(cfg["train"].get("grad_clip_norm", 0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        logits_all.append(out["logits"].detach().float().cpu().numpy())
        targets_all.append(targets.detach().float().cpu().numpy())
        ecg_ids_all.extend([str(x) for x in batch["ecg_id"]])
        iterator.set_postfix(loss=np.mean(losses))
    targets_np = np.concatenate(targets_all)
    logits_np = np.concatenate(logits_all)
    if (not train) and bool(cfg.get("eval", {}).get("aggregate_windows", False)):
        targets_np, logits_np = aggregate_by_ecg_id(targets_np, logits_np, ecg_ids_all)
    return float(np.mean(losses)), targets_np, logits_np


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 2026)))
    out_dir = ensure_dir(cfg["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(cfg, limit=args.limit_train)
    model = build_model(cfg, device, no_pretrained=args.no_pretrained)
    optimizer = optimizer_for(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")
    pos_weight = class_pos_weight(cfg).to(device)

    best_auc = -1.0
    stale = 0
    history = []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss, _, _ = run_epoch(model, loaders["train"], device, pos_weight, True, optimizer, scaler, cfg)
        val_loss, val_targets, val_logits = run_epoch(model, loaders["val"], device, pos_weight, False, None, None, cfg)
        thresholds = tune_thresholds(val_targets, 1.0 / (1.0 + np.exp(-val_logits)))
        val_metrics, _ = compute_metrics(val_targets, val_logits, thresholds)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_metrics["auc_macro"] > best_auc:
            best_auc = val_metrics["auc_macro"]
            stale = 0
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "thresholds": thresholds, "epoch": epoch, "val_metrics": val_metrics},
                out_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= int(cfg["train"].get("early_stop_patience", 6)):
                break

    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_loss, test_targets, test_logits = run_epoch(model, loaders["test"], device, pos_weight, False, None, None, cfg)
    test_metrics, _ = compute_metrics(test_targets, test_logits, best["thresholds"])
    result = {"best_epoch": best["epoch"], "test_loss": test_loss, **{f"test_{k}": v for k, v in test_metrics.items()}}
    print(json.dumps(result, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
