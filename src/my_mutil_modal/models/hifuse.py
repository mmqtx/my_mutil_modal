from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import EcgTransformer

logger = logging.getLogger(__name__)


class CLIPVisionEncoder(nn.Module):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        from transformers import CLIPVisionModel

        self.vision_model = CLIPVisionModel.from_pretrained(model_path)
        self.hidden_size = int(self.vision_model.config.hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=images)
        return outputs.last_hidden_state[:, 0]

    def lock(self) -> None:
        for p in self.parameters():
            p.requires_grad = False


class GatedResidualFusion(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = dim * 4
        self.gate = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Linear(in_dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, signal_feat: torch.Tensor, image_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat(
            [signal_feat, image_feat, torch.abs(signal_feat - image_feat), signal_feat * image_feat],
            dim=-1,
        )
        gate = self.gate(x)
        fused = gate * signal_feat + (1.0 - gate) * image_feat + self.residual(x)
        return self.norm(fused), gate


class HiFuseECG(nn.Module):
    def __init__(
        self,
        num_classes: int,
        clip_model_path: str,
        fusion_dim: int = 512,
        dropout: float = 0.15,
        freeze_signal_encoder: bool = True,
        freeze_image_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.signal_encoder = EcgTransformer(
            seq_length=5000,
            lead_num=12,
            patch_size=50,
            width=768,
            layers=12,
            heads=12,
            mlp_ratio=4.0,
            output_dim=512,
        )
        self.image_encoder = CLIPVisionEncoder(clip_model_path)

        if freeze_signal_encoder:
            self.signal_encoder.lock()
        if freeze_image_encoder:
            self.image_encoder.lock()

        self.signal_proj = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.image_proj = nn.Sequential(
            nn.LayerNorm(self.image_encoder.hidden_size),
            nn.Linear(self.image_encoder.hidden_size, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = GatedResidualFusion(fusion_dim, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        self.signal_contrast = nn.Linear(512, fusion_dim)
        self.image_contrast = nn.Linear(self.image_encoder.hidden_size, fusion_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def forward(
        self,
        signal: torch.Tensor,
        image: torch.Tensor,
        modality_dropout: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        sig_raw = self.signal_encoder(signal)
        img_raw = self.image_encoder(image)
        sig = self.signal_proj(sig_raw)
        img = self.image_proj(img_raw)

        if self.training and modality_dropout > 0:
            keep = torch.rand(sig.size(0), 1, device=sig.device)
            sig = torch.where(keep < modality_dropout / 2, torch.zeros_like(sig), sig)
            img = torch.where(keep > 1 - modality_dropout / 2, torch.zeros_like(img), img)

        fused, gate = self.fusion(sig, img)
        logits = self.classifier(fused)
        sig_z = F.normalize(self.signal_contrast(sig_raw), dim=-1)
        img_z = F.normalize(self.image_contrast(img_raw), dim=-1)
        return {"logits": logits, "signal_z": sig_z, "image_z": img_z, "fusion_gate": gate}


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def load_gem_signal_weights(model: HiFuseECG, checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = strip_module_prefix(state)

    candidates: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("ecg."):
            candidates[key[len("ecg.") :]] = value
        elif key.startswith("signal_encoder."):
            candidates[key[len("signal_encoder.") :]] = value
        elif key.startswith("signal_backbone."):
            candidates[key[len("signal_backbone.") :]] = value

    if not candidates:
        sample = sorted(state.keys())[:20]
        raise RuntimeError(f"No ECG encoder keys found in {checkpoint_path}. Sample keys: {sample}")

    missing, unexpected = model.signal_encoder.load_state_dict(candidates, strict=False)
    useful_unexpected = [k for k in unexpected if not k.startswith(("text", "visual"))]
    logger.info(
        "Loaded GEM ECG weights from %s; missing=%d unexpected=%d",
        checkpoint_path,
        len(missing),
        len(useful_unexpected),
    )
