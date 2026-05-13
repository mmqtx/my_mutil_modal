from __future__ import annotations

import torch
import torch.nn.functional as F


def multilabel_bce_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def symmetric_contrastive_loss(signal_z: torch.Tensor, image_z: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    scale = logit_scale.exp().clamp(max=100)
    logits = scale * signal_z @ image_z.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
