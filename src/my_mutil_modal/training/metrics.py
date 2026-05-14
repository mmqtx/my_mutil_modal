from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def macro_auc(targets: np.ndarray, probs: np.ndarray) -> float:
    scores: List[float] = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) < 2:
            continue
        scores.append(float(roc_auc_score(targets[:, i], probs[:, i])))
    return float(np.mean(scores)) if scores else float("nan")


def tune_thresholds(targets: np.ndarray, probs: np.ndarray, grid: Iterable[float] | None = None) -> np.ndarray:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    thresholds = []
    for i in range(targets.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred = (probs[:, i] >= t).astype(np.int64)
            score = f1_score(targets[:, i], pred, zero_division=0)
            if score > best_f1:
                best_t, best_f1 = float(t), float(score)
        thresholds.append(best_t)
    return np.asarray(thresholds, dtype=np.float32)


def compute_metrics(targets: np.ndarray, logits: np.ndarray, thresholds: np.ndarray | None = None) -> Tuple[Dict[str, float], np.ndarray]:
    probs = sigmoid_np(logits)
    if thresholds is None:
        thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    preds = (probs >= thresholds[None, :]).astype(np.int64)
    metrics = {
        "auc_macro": macro_auc(targets, probs),
        "f1_macro": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets, preds, average="macro", zero_division=0)),
        "accuracy_label": float((preds == targets).mean()),
        "accuracy_sample": float((preds == targets).all(axis=1).mean()),
    }
    return metrics, probs
