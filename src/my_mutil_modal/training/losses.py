from __future__ import annotations

import torch
import torch.nn.functional as F


def multilabel_bce_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def asymmetric_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pos = probs
    neg = 1.0 - probs
    if clip > 0:
        neg = (neg + clip).clamp(max=1.0)
    loss = targets * torch.log(pos.clamp(min=eps)) + (1.0 - targets) * torch.log(neg.clamp(min=eps))
    pt = pos * targets + neg * (1.0 - targets)
    gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
    loss = loss * ((1.0 - pt) ** gamma)
    return -loss.mean()


def symmetric_contrastive_loss(signal_z: torch.Tensor, image_z: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    scale = logit_scale.exp().clamp(max=100)
    logits = scale * signal_z @ image_z.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
