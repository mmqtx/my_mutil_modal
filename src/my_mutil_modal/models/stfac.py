from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class CASSANBlock(nn.Module):
    """Coordinated adaptive simplified self-attention used by CAMV-RNN."""

    def __init__(self, dim: int, c: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % c != 0:
            raise ValueError(f"CASSAN dim={dim} must be divisible by c={c}")
        self.c = c
        self.h = dim // c
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, dim = x.shape
        q = self.q(x).view(b, w, self.c, self.h).permute(0, 2, 1, 3)
        k = self.k(x).view(b, w, self.c, self.h).permute(0, 2, 3, 1)
        v = self.v(x).view(b, w, self.c, self.h).permute(0, 2, 1, 3)
        qw = q.mean(dim=3, keepdim=True)
        kw = k.mean(dim=2, keepdim=True)
        qc = torch.cat([q, qw], dim=3)
        kc = torch.cat([k, kw], dim=2)
        sim = qc @ kc
        adaptive = qw.expand(-1, -1, -1, w) + kw.expand(-1, -1, w, -1)
        attn = torch.softmax(sim + adaptive, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, w, dim)
        return self.norm(x + self.dropout(out))


class CAMVRNNBlock(nn.Module):
    def __init__(self, in_channels: int = 12, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        dim = hidden * 2
        self.bigru_main = nn.GRU(in_channels, hidden, batch_first=True, bidirectional=True)
        self.bigru_skip = nn.GRU(in_channels, hidden, batch_first=True, bidirectional=True)
        self.bilstm = nn.LSTM(in_channels, hidden, batch_first=True, bidirectional=True)
        self.cassan = CASSANBlock(dim, c=8, dropout=dropout)
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


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, pool: str | None = None) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.pool == "max":
            x = nn.functional.max_pool2d(x, 2)
        elif self.pool == "avg":
            x = nn.functional.avg_pool2d(x, 2)
        elif self.pool == "maxavg":
            x = 0.5 * (nn.functional.max_pool2d(x, 2) + nn.functional.avg_pool2d(x, 2))
        return x


class CBMVCNNBlock(nn.Module):
    def __init__(self, in_channels: int = 1, channels: int = 576, dropout: float = 0.2, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        c1, c2, c3, c4, c5, c6 = 64, 128, 256, 512, channels, channels
        self.conv1 = ConvBNAct(in_channels, c1, pool="maxavg")
        self.conv2a = ConvBNAct(c1, c2)
        self.conv2b = ConvBNAct(c2, c2)
        self.conv3 = ConvBNAct(c2, c3, dropout=dropout)
        self.conv4 = ConvBNAct(c3, c4, dropout=dropout, pool="max")
        self.cbam = CBAM(c4)
        self.conv5 = ConvBNAct(c4, c5, dropout=dropout)
        self.conv6 = ConvBNAct(c5, c6)
        self.out_channels = c6

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_checkpoint:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._run(self.conv1, image)
        x = self._run(self.conv2a, x)
        x = self._run(self.conv2b, x)
        x = self._run(self.conv3, x)
        x = self._run(self.conv4, x)
        if self.training and self.use_checkpoint:
            x = checkpoint(lambda y: self.cbam(y) + y, x, use_reentrant=False)
        else:
            x = self.cbam(x) + x
        x = self._run(self.conv5, x)
        x = self._run(self.conv6, x)
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
        image_channels: int = 576,
        fusion_dim: int = 512,
        dropout: float = 0.2,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.cbmv = CBMVCNNBlock(
            in_channels=image_in_channels,
            channels=image_channels,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )
        spatial_dim = self.cbmv.out_channels * 2
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
        image_channels: int = 576,
        fusion_dim: int = 512,
        dropout: float = 0.2,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.camv = CAMVRNNBlock(in_channels=12, hidden=signal_hidden, dropout=dropout)
        self.cbmv = CBMVCNNBlock(
            in_channels=image_in_channels,
            channels=image_channels,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )
        temporal_dim = signal_hidden * 4
        spatial_dim = self.cbmv.out_channels * 2
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
