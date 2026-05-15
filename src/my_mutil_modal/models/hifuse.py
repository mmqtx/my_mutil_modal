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

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=images)
        return outputs.last_hidden_state

    def lock(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unlock_last_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        vision = self.vision_model.vision_model
        for layer in vision.encoder.layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        post_layernorm = getattr(vision, "post_layernorm", None)
        if post_layernorm is not None:
            for p in post_layernorm.parameters():
                p.requires_grad = True


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


class TokenCrossFusion(nn.Module):
    """Fuse ECG patch tokens and CLIP patch tokens with a compact transformer."""

    def __init__(self, dim: int, heads: int = 8, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.signal_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.image_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_type = nn.Parameter(torch.zeros(1, 1, dim))
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, signal_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        bsz = signal_tokens.size(0)
        cls = self.cls.expand(bsz, -1, -1) + self.cls_type
        x = torch.cat(
            [
                cls,
                signal_tokens + self.signal_type,
                image_tokens + self.image_type,
            ],
            dim=1,
        )
        return self.norm(self.encoder(x)[:, 0])


class TemporalPatchAlignment(nn.Module):
    """Fine-grained signal temporal token to image patch token alignment."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.signal_to_image = nn.ModuleList(
            [
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
                for _ in range(layers)
            ]
        )
        self.signal_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.signal_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        if bidirectional:
            self.image_to_signal = nn.ModuleList(
                [
                    nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
                    for _ in range(layers)
                ]
            )
            self.image_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
            self.image_ffn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
            )
        else:
            self.image_to_signal = None
            self.image_norms = None
            self.image_ffn = None
        self.output_dim = dim * (4 if bidirectional else 3)

    def forward(self, signal_tokens: torch.Tensor, image_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        aligned_signal = signal_tokens
        signal_attn = None
        for attn, norm in zip(self.signal_to_image, self.signal_norms):
            query = norm(aligned_signal)
            attended, signal_attn = attn(query, image_tokens, image_tokens, need_weights=False)
            aligned_signal = aligned_signal + attended
        aligned_signal = aligned_signal + self.signal_ffn(aligned_signal)

        f_signal = signal_tokens.mean(dim=1)
        f_aligned_signal = aligned_signal.mean(dim=1)
        f_image = image_tokens.mean(dim=1)
        features = [f_signal, f_aligned_signal, f_image]
        aux: Dict[str, torch.Tensor] = {"signal_aligned_tokens": aligned_signal}

        if self.bidirectional:
            assert self.image_to_signal is not None
            assert self.image_norms is not None
            assert self.image_ffn is not None
            aligned_image = image_tokens
            image_attn = None
            for attn, norm in zip(self.image_to_signal, self.image_norms):
                query = norm(aligned_image)
                attended, image_attn = attn(query, signal_tokens, signal_tokens, need_weights=False)
                aligned_image = aligned_image + attended
            aligned_image = aligned_image + self.image_ffn(aligned_image)
            features.append(aligned_image.mean(dim=1))
            aux["image_aligned_tokens"] = aligned_image
            if image_attn is not None:
                aux["image_to_signal_attn"] = image_attn
        if signal_attn is not None:
            aux["signal_to_image_attn"] = signal_attn
        return torch.cat(features, dim=-1), aux


class LabelQueryDecoder(nn.Module):
    """Use one learnable query per diagnosis to read cross-modal token evidence."""

    def __init__(self, num_classes: int, dim: int, heads: int = 8, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.label_queries = nn.Parameter(torch.zeros(1, num_classes, dim))
        self.global_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.signal_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.image_type = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        nn.init.trunc_normal_(self.label_queries, std=0.02)

    def forward(
        self,
        global_fused: torch.Tensor,
        signal_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        bsz = global_fused.size(0)
        context = torch.cat(
            [
                global_fused.unsqueeze(1) + self.global_type,
                signal_tokens + self.signal_type,
                image_tokens + self.image_type,
            ],
            dim=1,
        )
        queries = self.label_queries.expand(bsz, -1, -1) + global_fused.unsqueeze(1)
        decoded = self.decoder(queries, context)
        return self.head(self.norm(decoded)).squeeze(-1)


class LocalMorphologyBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 4
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=9, padding=padding, dilation=dilation, groups=channels, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SignalLocalEncoder(nn.Module):
    """Lightweight waveform branch for local morphology that a global CLS token can miss."""

    def __init__(self, out_dim: int, channels: int = 192, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            LocalMorphologyBlock(channels, dilation=1, dropout=dropout),
            nn.AvgPool1d(kernel_size=2, stride=2),
            LocalMorphologyBlock(channels, dilation=2, dropout=dropout),
            nn.AvgPool1d(kernel_size=2, stride=2),
            LocalMorphologyBlock(channels, dilation=4, dropout=dropout),
            LocalMorphologyBlock(channels, dilation=8, dropout=dropout),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(signal))
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return self.proj(pooled)


class HiFuseECG(nn.Module):
    def __init__(
        self,
        num_classes: int,
        clip_model_path: str,
        fusion_dim: int = 512,
        dropout: float = 0.15,
        freeze_signal_encoder: bool = True,
        freeze_image_encoder: bool = True,
        signal_unlocked_groups: int = 0,
        image_unfreeze_last_n: int = 0,
        token_fusion: bool = False,
        token_fusion_layers: int = 2,
        token_fusion_heads: int = 8,
        label_query_fusion: bool = False,
        label_query_layers: int = 1,
        label_query_heads: int = 8,
        label_query_weight: float = 0.5,
        signal_local_branch: bool = False,
        signal_local_channels: int = 192,
        signal_local_weight: float = 0.5,
        tpa_fusion: bool = False,
        tpa_heads: int = 8,
        tpa_layers: int = 1,
        tpa_bidirectional: bool = False,
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
        self.token_fusion_enabled = token_fusion
        self.label_query_enabled = label_query_fusion
        self.signal_local_enabled = signal_local_branch
        self.tpa_enabled = tpa_fusion

        if freeze_signal_encoder:
            self.signal_encoder.lock(unlocked_groups=signal_unlocked_groups)
        if freeze_image_encoder:
            self.image_encoder.lock()
            self.image_encoder.unlock_last_layers(image_unfreeze_last_n)

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
        self.signal_token_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, fusion_dim))
        self.image_token_proj = nn.Sequential(nn.LayerNorm(self.image_encoder.hidden_size), nn.Linear(self.image_encoder.hidden_size, fusion_dim))
        self.token_fusion = TokenCrossFusion(
            fusion_dim,
            heads=token_fusion_heads,
            layers=token_fusion_layers,
            dropout=dropout,
        )
        self.tpa = (
            TemporalPatchAlignment(
                fusion_dim,
                heads=tpa_heads,
                layers=tpa_layers,
                dropout=dropout,
                bidirectional=tpa_bidirectional,
            )
            if tpa_fusion
            else None
        )
        self.label_query_decoder = (
            LabelQueryDecoder(
                num_classes,
                fusion_dim,
                heads=label_query_heads,
                layers=label_query_layers,
                dropout=dropout,
            )
            if label_query_fusion
            else None
        )
        self.label_query_weight = (
            nn.Parameter(torch.tensor(float(label_query_weight), dtype=torch.float32))
            if label_query_fusion
            else None
        )
        self.signal_local_encoder = (
            SignalLocalEncoder(fusion_dim, channels=signal_local_channels, dropout=dropout)
            if signal_local_branch
            else None
        )
        self.signal_local_classifier = nn.Linear(fusion_dim, num_classes) if signal_local_branch else None
        self.signal_local_weight = (
            nn.Parameter(torch.tensor(float(signal_local_weight), dtype=torch.float32))
            if signal_local_branch
            else None
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        self.tpa_classifier = (
            nn.Sequential(
                nn.LayerNorm(self.tpa.output_dim),
                nn.Linear(self.tpa.output_dim, fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, num_classes),
            )
            if tpa_fusion and self.tpa is not None
            else None
        )
        self.signal_contrast = nn.Linear(512, fusion_dim)
        self.image_contrast = nn.Linear(self.image_encoder.hidden_size, fusion_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592), requires_grad=False)
        self.adapter_only_training = False
        if not (token_fusion or tpa_fusion):
            for module in (self.signal_token_proj, self.image_token_proj, self.token_fusion):
                for param in module.parameters():
                    param.requires_grad = False
        if tpa_fusion:
            for module in (self.signal_proj, self.image_proj, self.fusion, self.classifier, self.token_fusion):
                for param in module.parameters():
                    param.requires_grad = False

    def freeze_base_for_adapter_training(self) -> None:
        self.adapter_only_training = True
        trainable_prefixes = (
            "label_query_decoder",
            "label_query_weight",
            "signal_local_encoder",
            "signal_local_classifier",
            "signal_local_weight",
            "tpa",
            "tpa_classifier",
        )
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.adapter_only_training:
            frozen_modules = [
                self.signal_encoder,
                self.image_encoder,
                self.signal_proj,
                self.image_proj,
                self.fusion,
                self.signal_token_proj,
                self.image_token_proj,
                self.token_fusion,
                self.classifier,
                self.signal_contrast,
                self.image_contrast,
            ]
            for module in frozen_modules:
                module.eval()
        return self

    def _encode_signal_tokens(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.signal_encoder(signal, output_last_transformer_layer=True)
        norm_tokens = self.signal_encoder.ln_post(tokens)
        pooled = norm_tokens[:, 0] @ self.signal_encoder.proj
        return pooled, norm_tokens

    def forward(
        self,
        signal: torch.Tensor,
        image: torch.Tensor,
        modality_dropout: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        if self.token_fusion_enabled or self.tpa_enabled:
            sig_raw, sig_tokens_raw = self._encode_signal_tokens(signal)
            img_tokens_raw = self.image_encoder.forward_tokens(image)
            img_raw = img_tokens_raw[:, 0]
        else:
            sig_raw = self.signal_encoder(signal)
            img_raw = self.image_encoder(image)
        sig = self.signal_proj(sig_raw)
        img = self.image_proj(img_raw)

        if self.training and modality_dropout > 0:
            keep = torch.rand(sig.size(0), 1, device=sig.device)
            sig = torch.where(keep < modality_dropout / 2, torch.zeros_like(sig), sig)
            img = torch.where(keep > 1 - modality_dropout / 2, torch.zeros_like(img), img)

        fused, gate = self.fusion(sig, img)
        local_feat = None
        if self.signal_local_enabled:
            assert self.signal_local_encoder is not None
            assert self.signal_local_weight is not None
            local_feat = self.signal_local_encoder(signal)
            fused = fused + self.signal_local_weight * local_feat
        if self.token_fusion_enabled or self.tpa_enabled:
            sig_tokens = self.signal_token_proj(sig_tokens_raw[:, 1:])
            img_tokens = self.image_token_proj(img_tokens_raw[:, 1:])
        if self.tpa_enabled:
            assert self.tpa is not None
            assert self.tpa_classifier is not None
            tpa_feat, tpa_aux = self.tpa(sig_tokens, img_tokens)
            logits = self.tpa_classifier(tpa_feat)
            if local_feat is not None:
                assert self.signal_local_classifier is not None
                assert self.signal_local_weight is not None
                logits = logits + self.signal_local_weight * self.signal_local_classifier(local_feat)
            sig_z = F.normalize(self.signal_contrast(sig_raw), dim=-1)
            img_z = F.normalize(self.image_contrast(img_raw), dim=-1)
            return {
                "logits": logits,
                "signal_z": sig_z,
                "image_z": img_z,
                "fusion_gate": gate,
                "logit_scale": self.logit_scale,
                **tpa_aux,
            }
        if self.token_fusion_enabled:
            fused = fused + self.token_fusion(sig_tokens, img_tokens)
        logits = self.classifier(fused)
        if self.label_query_enabled:
            if not self.token_fusion_enabled:
                raise RuntimeError("label_query_fusion requires token_fusion=true")
            assert self.label_query_decoder is not None
            assert self.label_query_weight is not None
            label_logits = self.label_query_decoder(fused, sig_tokens, img_tokens)
            logits = logits + self.label_query_weight * label_logits
        if local_feat is not None:
            assert self.signal_local_classifier is not None
            assert self.signal_local_weight is not None
            logits = logits + self.signal_local_weight * self.signal_local_classifier(local_feat)
        sig_z = F.normalize(self.signal_contrast(sig_raw), dim=-1)
        img_z = F.normalize(self.image_contrast(img_raw), dim=-1)
        return {
            "logits": logits,
            "signal_z": sig_z,
            "image_z": img_z,
            "fusion_gate": gate,
            "logit_scale": self.logit_scale,
        }


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
