from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class CASSANBlock(nn.Module):
    """Lightweight coordinated self-attention approximation for CAMV-RNN."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax(q @ k.transpose(1, 2), dim=-1)
        out = attn @ v
        gate = self.channel_gate(x.transpose(1, 2)).unsqueeze(1)
        return self.norm(x + self.dropout(out * gate))


class CAMVRNNBlock(nn.Module):
    def __init__(self, in_channels: int = 12, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        dim = hidden * 2
        self.bigru_main = nn.GRU(in_channels, hidden, batch_first=True, bidirectional=True)
        self.bigru_skip = nn.GRU(in_channels, hidden, batch_first=True, bidirectional=True)
        self.bilstm = nn.LSTM(in_channels, hidden, batch_first=True, bidirectional=True)
        self.cassan = CASSANBlock(dim, dropout)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = signal.transpose(1, 2)
        main, _ = self.bigru_main(x)
        skip, _ = self.bigru_skip(x)
        lower, _ = self.bilstm(x)
        lower = self.cassan(lower)
        fused = main + skip + lower
        fused = self.bn(fused.transpose(1, 2)).transpose(1, 2)
        fused = self.act(self.dropout(fused))
        pooled = torch.cat([fused.max(dim=1).values, fused.mean(dim=1)], dim=-1)
        return fused, pooled


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(x.size(0), x.size(1), 1, 1)
        return x * gate


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class CBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float, pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CBMVCNNBlock(nn.Module):
    def __init__(self, in_channels: int = 1, channels: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32, dropout, pool=True),
            ConvBlock(32, 64, dropout, pool=True),
            ConvBlock(64, channels, dropout, pool=True),
        )
        self.cbam = CBAM(channels)
        self.tail = nn.Sequential(
            ConvBlock(channels, channels, dropout, pool=True),
            ConvBlock(channels, channels, dropout, pool=False),
        )

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(image)
        x = self.cbam(x) + x
        x = self.tail(x)
        height_pool = torch.cat([x.amax(dim=2), x.mean(dim=2)], dim=-1)
        spatial = torch.cat([height_pool.amax(dim=2), height_pool.mean(dim=2)], dim=-1)
        return x, spatial


class CAMVRNNClassifier(nn.Module):
    """Signal-only CAMV-RNN baseline from the STFAC-ECGNet paper."""

    def __init__(
        self,
        num_classes: int,
        signal_hidden: int = 64,
        fusion_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.camv = CAMVRNNBlock(in_channels=12, hidden=signal_hidden, dropout=dropout)
        temporal_dim = signal_hidden * 4
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(temporal_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, signal: torch.Tensor, image: torch.Tensor | None = None, **_: float) -> Dict[str, torch.Tensor]:
        _, temporal = self.camv(signal)
        return {"logits": self.classifier(temporal)}


class CBMVCNNClassifier(nn.Module):
    """Image-only CBMV-CNN baseline from the STFAC-ECGNet paper."""

    def __init__(
        self,
        num_classes: int,
        image_in_channels: int = 3,
        image_channels: int = 128,
        fusion_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cbmv = CBMVCNNBlock(in_channels=image_in_channels, channels=image_channels, dropout=dropout)
        spatial_dim = image_channels * 2
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(spatial_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, signal: torch.Tensor | None, image: torch.Tensor, **_: float) -> Dict[str, torch.Tensor]:
        _, spatial = self.cbmv(image)
        return {"logits": self.classifier(spatial)}


class STFACECGNet(nn.Module):
    """Reproduction-oriented STFAC-ECGNet baseline for PTB-XL superclass labels."""

    def __init__(
        self,
        num_classes: int,
        image_in_channels: int = 3,
        signal_hidden: int = 64,
        image_channels: int = 128,
        fusion_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.camv = CAMVRNNBlock(in_channels=12, hidden=signal_hidden, dropout=dropout)
        self.cbmv = CBMVCNNBlock(in_channels=image_in_channels, channels=image_channels, dropout=dropout)
        temporal_dim = signal_hidden * 4
        spatial_dim = image_channels * 2
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(temporal_dim + spatial_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_dim + spatial_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, signal: torch.Tensor, image: torch.Tensor, **_: float) -> Dict[str, torch.Tensor]:
        _, temporal = self.camv(signal)
        _, spatial = self.cbmv(image)
        logits = self.fusion(torch.cat([temporal, spatial], dim=-1))
        return {"logits": logits}
