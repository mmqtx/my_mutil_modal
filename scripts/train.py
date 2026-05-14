#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_mutil_modal.data import build_dataloaders
from my_mutil_modal.models import CAMVRNNClassifier, CBMVCNNClassifier, HiFuseECG, STFACECGNet, load_gem_signal_weights
from my_mutil_modal.training.losses import asymmetric_multilabel_loss, multilabel_bce_loss, symmetric_contrastive_loss
from my_mutil_modal.training.metrics import compute_metrics, tune_thresholds
from my_mutil_modal.utils.config import ensure_dir, load_config
from my_mutil_modal.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ptbxl_hifuse.yaml")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def setup_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return distributed, rank, local_rank, world_size


def is_main_process(distributed: bool, rank: int) -> bool:
    return (not distributed) or rank == 0


def class_pos_weight(cfg: Dict) -> torch.Tensor | None:
    if not bool(cfg["train"].get("use_pos_weight", True)):
        return None
    df = pd.read_csv(cfg["data"]["manifest"])
    train = df[df["split"] == "train"]
    labels = train[cfg["data"]["label_columns"]].astype(np.float32).to_numpy()
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    return torch.tensor(neg / np.clip(pos, 1.0, None), dtype=torch.float32)


def build_model(cfg: Dict, device: torch.device, no_pretrained: bool = False) -> HiFuseECG:
    model_cfg = cfg["model"]
    model_name = model_cfg.get("name", "hifuse").lower()
    if model_name == "camv_rnn":
        model = CAMVRNNClassifier(
            num_classes=len(cfg["data"]["label_columns"]),
            signal_hidden=int(model_cfg.get("signal_hidden", 64)),
            fusion_dim=int(model_cfg.get("fusion_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
        return model.to(device)
    if model_name == "cbmv_cnn":
        model = CBMVCNNClassifier(
            num_classes=len(cfg["data"]["label_columns"]),
            image_in_channels=int(model_cfg.get("image_in_channels", cfg["data"].get("image_channels", 3))),
            image_channels=int(model_cfg.get("image_channels", 128)),
            fusion_dim=int(model_cfg.get("fusion_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            use_checkpoint=bool(model_cfg.get("use_checkpoint", False)),
        )
        return model.to(device)
    if model_name == "stfac":
        model = STFACECGNet(
            num_classes=len(cfg["data"]["label_columns"]),
            image_in_channels=int(model_cfg.get("image_in_channels", cfg["data"].get("image_channels", 3))),
            signal_hidden=int(model_cfg.get("signal_hidden", 64)),
            image_channels=int(model_cfg.get("image_channels", 128)),
            fusion_dim=int(model_cfg.get("fusion_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            use_checkpoint=bool(model_cfg.get("use_checkpoint", False)),
        )
        return model.to(device)
    model = HiFuseECG(
        num_classes=len(cfg["data"]["label_columns"]),
        clip_model_path=model_cfg["clip_model_path"],
        fusion_dim=int(model_cfg.get("fusion_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        freeze_signal_encoder=bool(model_cfg.get("freeze_signal_encoder", True)),
        freeze_image_encoder=bool(model_cfg.get("freeze_image_encoder", True)),
        signal_unlocked_groups=int(model_cfg.get("signal_unlocked_groups", 0)),
        image_unfreeze_last_n=int(model_cfg.get("image_unfreeze_last_n", 0)),
        token_fusion=bool(model_cfg.get("token_fusion", False)),
        token_fusion_layers=int(model_cfg.get("token_fusion_layers", 2)),
        token_fusion_heads=int(model_cfg.get("token_fusion_heads", 8)),
        label_query_fusion=bool(model_cfg.get("label_query_fusion", False)),
        label_query_layers=int(model_cfg.get("label_query_layers", 1)),
        label_query_heads=int(model_cfg.get("label_query_heads", 8)),
        label_query_weight=float(model_cfg.get("label_query_weight", 0.5)),
        signal_local_branch=bool(model_cfg.get("signal_local_branch", False)),
        signal_local_channels=int(model_cfg.get("signal_local_channels", 192)),
        signal_local_weight=float(model_cfg.get("signal_local_weight", 0.5)),
    )
    if not no_pretrained and model_cfg.get("signal_checkpoint"):
        load_gem_signal_weights(model, model_cfg["signal_checkpoint"])
    if model_cfg.get("init_checkpoint"):
        ckpt = torch.load(model_cfg["init_checkpoint"], map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"Loaded init checkpoint {model_cfg['init_checkpoint']} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    if bool(model_cfg.get("adapter_only_training", False)):
        model.freeze_base_for_adapter_training()
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
    if str(cfg["train"].get("optimizer", "adamw")).lower() == "adam":
        return torch.optim.Adam(groups)
    return torch.optim.AdamW(groups)


def scheduler_for(
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = str(cfg["train"].get("scheduler", "")).lower()
    if scheduler_name != "onecycle":
        return None
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in optimizer.param_groups],
        epochs=int(cfg["train"]["epochs"]),
        steps_per_epoch=steps_per_epoch,
        pct_start=float(cfg["train"].get("onecycle_pct_start", 0.3)),
        div_factor=float(cfg["train"].get("onecycle_div_factor", 25.0)),
        final_div_factor=float(cfg["train"].get("onecycle_final_div_factor", 1e4)),
    )


def aggregate_by_ecg_id(targets: np.ndarray, logits: np.ndarray, ecg_ids: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    grouped: Dict[str, list[int]] = {}
    for i, ecg_id in enumerate(ecg_ids):
        grouped.setdefault(ecg_id, []).append(i)
    agg_targets, agg_logits = [], []
    for idxs in grouped.values():
        agg_targets.append(targets[idxs[0]])
        agg_logits.append(logits[idxs].mean(axis=0))
    return np.asarray(agg_targets), np.asarray(agg_logits)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def run_epoch(
    model: HiFuseECG,
    loader,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: Dict,
    show_progress: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(train)
    logits_all, targets_all, ecg_ids_all = [], [], []
    losses = []
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    contrastive_weight = float(cfg["model"].get("contrastive_weight", 0.0))
    modality_dropout = float(cfg["model"].get("modality_dropout", 0.0)) if train else 0.0
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval", disable=not show_progress)
    for batch in iterator:
        signal = batch["signal"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=amp):
            out = model(signal, image, modality_dropout=modality_dropout)
            if cfg["train"].get("loss", "bce") == "asymmetric":
                loss = asymmetric_multilabel_loss(
                    out["logits"],
                    targets,
                    gamma_pos=float(cfg["train"].get("asl_gamma_pos", 0.0)),
                    gamma_neg=float(cfg["train"].get("asl_gamma_neg", 4.0)),
                    clip=float(cfg["train"].get("asl_clip", 0.05)),
                )
            else:
                loss = multilabel_bce_loss(out["logits"], targets, pos_weight=pos_weight)
            if contrastive_weight > 0 and "signal_z" in out and "image_z" in out:
                loss = loss + contrastive_weight * symmetric_contrastive_loss(
                    out["signal_z"], out["image_z"], out["logit_scale"]
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
            if scheduler is not None:
                scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if (not train) or show_progress:
            logits_all.append(out["logits"].detach().float().cpu().numpy())
            targets_all.append(targets.detach().float().cpu().numpy())
            ecg_ids_all.extend([str(x) for x in batch["ecg_id"]])
        if show_progress:
            iterator.set_postfix(loss=np.mean(losses))
    if not logits_all:
        return float(np.mean(losses)), np.empty((0, 0)), np.empty((0, 0))
    targets_np = np.concatenate(targets_all)
    logits_np = np.concatenate(logits_all)
    if (not train) and bool(cfg.get("eval", {}).get("aggregate_windows", False)):
        targets_np, logits_np = aggregate_by_ecg_id(targets_np, logits_np, ecg_ids_all)
    return float(np.mean(losses)), targets_np, logits_np


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = setup_distributed()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 2026)) + rank)
    out_dir = ensure_dir(cfg["output_dir"]) if is_main_process(distributed, rank) else Path(cfg["output_dir"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(cfg, limit=args.limit_train, distributed=distributed, rank=rank, world_size=world_size)
    model = build_model(cfg, device, no_pretrained=args.no_pretrained)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(cfg["train"].get("ddp_find_unused_parameters", False)),
        )
    optimizer = optimizer_for(model, cfg)
    scheduler = scheduler_for(optimizer, cfg, steps_per_epoch=len(loaders["train"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")
    pos_weight = class_pos_weight(cfg)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    selection_metric = str(cfg["train"].get("selection_metric", "auc_macro"))
    best_score = -1.0
    stale = 0
    history = []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        if distributed and hasattr(loaders["train"].sampler, "set_epoch"):
            loaders["train"].sampler.set_epoch(epoch)
        train_loss, _, _ = run_epoch(
            model, loaders["train"], device, pos_weight, True, optimizer, scheduler, scaler, cfg,
            show_progress=is_main_process(distributed, rank),
        )
        stop_now = torch.tensor(0, device=device)
        if is_main_process(distributed, rank):
            eval_model = model.module if distributed else model
            val_loss, val_targets, val_logits = run_epoch(
                eval_model, loaders["val"], device, pos_weight, False, None, None, None, cfg, show_progress=True,
            )
            thresholds = tune_thresholds(val_targets, 1.0 / (1.0 + np.exp(-val_logits)))
            val_metrics, _ = compute_metrics(val_targets, val_logits, thresholds)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            print(json.dumps(row, indent=2))
            current_score = float(val_metrics[selection_metric])
            if current_score > best_score:
                best_score = current_score
                stale = 0
                torch.save(
                    {
                        "model": eval_model.state_dict(),
                        "cfg": cfg,
                        "thresholds": thresholds,
                        "epoch": epoch,
                        "selection_metric": selection_metric,
                        "val_metrics": val_metrics,
                    },
                    out_dir / "best.pt",
                )
            else:
                stale += 1
                if stale >= int(cfg["train"].get("early_stop_patience", 6)):
                    stop_now.fill_(1)
        if distributed:
            dist.broadcast(stop_now, src=0)
        if int(stop_now.item()) == 1:
            break

    if not is_main_process(distributed, rank):
        if distributed:
            dist.destroy_process_group()
        return
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    eval_model = model.module if distributed else model
    eval_model.load_state_dict(best["model"])
    test_loss, test_targets, test_logits = run_epoch(eval_model, loaders["test"], device, pos_weight, False, None, None, None, cfg)
    test_metrics, _ = compute_metrics(test_targets, test_logits, best["thresholds"])
    result = {"best_epoch": best["epoch"], "test_loss": test_loss, **{f"test_{k}": v for k, v in test_metrics.items()}}
    print(json.dumps(result, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
