from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import wfdb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _record_base(path: Path) -> str:
    if path.suffix in {".dat", ".hea"}:
        path = path.with_suffix("")
    return str(path)


class PTBXLMultimodalDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        split: str,
        label_columns: Iterable[str],
        signal_column: str = "signal_hr_path",
        image_column: str = "image_path",
        signal_length: int = 5000,
        image_size: int = 336,
        use_images: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.label_columns = list(label_columns)
        self.signal_column = signal_column
        self.image_column = image_column
        self.signal_length = signal_length
        self.use_images = use_images

        df = pd.read_csv(manifest)
        if split != "all":
            df = df[df["split"] == split].copy()
        if use_images and "image_exists" in df.columns:
            df = df[df["image_exists"].astype(str).str.lower().isin(["true", "1"])].copy()
        if limit is not None:
            df = df.head(limit).copy()
        if df.empty:
            raise ValueError(f"No rows found for split={split} in {manifest}")
        missing = [c for c in self.label_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns in manifest: {missing}")
        self.df = df.reset_index(drop=True)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_signal(self, rel_path: str) -> torch.Tensor:
        path = self.root / rel_path
        signal = wfdb.rdrecord(_record_base(path)).p_signal.astype(np.float32)
        if signal.ndim != 2 or signal.shape[1] != 12:
            raise ValueError(f"Expected signal shape (T, 12), got {signal.shape} for {path}")
        if signal.shape[0] < self.signal_length:
            pad = self.signal_length - signal.shape[0]
            signal = np.pad(signal, ((0, pad), (0, 0)), mode="constant")
        elif signal.shape[0] > self.signal_length:
            signal = signal[: self.signal_length]
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True)
        signal = (signal - mean) / np.clip(std, 1e-6, None)
        return torch.from_numpy(signal.T.copy())

    def _load_image(self, rel_path: str) -> torch.Tensor:
        path = self.root / rel_path
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.image_transform(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        signal = self._load_signal(str(row[self.signal_column]))
        if self.use_images:
            image = self._load_image(str(row[self.image_column]))
        else:
            image = torch.zeros(3, 336, 336, dtype=torch.float32)
        labels = torch.tensor(row[self.label_columns].astype(np.float32).to_numpy(), dtype=torch.float32)
        return {
            "signal": signal,
            "image": image,
            "labels": labels,
            "ecg_id": str(row["ecg_id"]),
        }


class PTBXLWindowDataset(Dataset):
    """PTB-XL 2.5s sliding-window dataset used for STFAC-style reproduction."""

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        split: str,
        label_columns: Iterable[str],
        signal_column: str = "signal_lr_path",
        image_column: str = "image_path",
        image_size: int = 224,
        image_channels: int = 3,
        use_images: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.label_columns = list(label_columns)
        self.signal_column = signal_column
        self.image_column = image_column
        self.use_images = use_images

        df = pd.read_csv(manifest)
        if split != "all":
            df = df[df["split"] == split].copy()
        if limit is not None:
            df = df.head(limit).copy()
        if df.empty:
            raise ValueError(f"No rows found for split={split} in {manifest}")
        self.df = df.reset_index(drop=True)
        self.image_channels = image_channels
        normalize = transforms.Normalize(mean=(0.5,) * image_channels, std=(0.5,) * image_channels)
        pre = [] if image_channels == 3 else [transforms.Grayscale(num_output_channels=1)]
        self.image_transform = transforms.Compose(
            pre
            + [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _labels_from_row(self, row: pd.Series) -> torch.Tensor:
        names = {x.strip() for x in str(row["labels"]).split(";") if x.strip()}
        return torch.tensor([1.0 if c in names else 0.0 for c in self.label_columns], dtype=torch.float32)

    def _load_signal_window(self, row: pd.Series) -> torch.Tensor:
        path = self.root / str(row[self.signal_column])
        signal = wfdb.rdrecord(_record_base(path)).p_signal.astype(np.float32)
        start, end = int(row["start_sample"]), int(row["end_sample"])
        signal = signal[start:end]
        expected = end - start
        if signal.shape[0] < expected:
            signal = np.pad(signal, ((0, expected - signal.shape[0]), (0, 0)), mode="constant")
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True)
        signal = (signal - mean) / np.clip(std, 1e-6, None)
        return torch.from_numpy(signal.T.copy())

    def _load_image(self, rel_path: str) -> torch.Tensor:
        path = self.root / rel_path
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.image_transform(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        signal = self._load_signal_window(row)
        if self.use_images:
            image = self._load_image(str(row[self.image_column]))
        else:
            image = torch.zeros(self.image_channels, 224, 224, dtype=torch.float32)
        return {
            "signal": signal,
            "image": image,
            "labels": self._labels_from_row(row),
            "ecg_id": str(row["ecg_id"]),
            "window_id": str(row["window_id"]),
        }


def build_dataloaders(cfg: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, DataLoader]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    loaders: Dict[str, DataLoader] = {}
    dataset_type = data_cfg.get("dataset_type", "ptbxl_full")
    for split, shuffle in [("train", True), ("val", False), ("test", False)]:
        split_limit = limit if split == "train" else None
        if dataset_type == "ptbxl_windows":
            ds = PTBXLWindowDataset(
                manifest=data_cfg.get("window_manifest", data_cfg["manifest"]),
                root=data_cfg["root"],
                split=split,
                label_columns=data_cfg["label_columns"],
                signal_column=data_cfg.get("signal_column", "signal_lr_path"),
                image_column=data_cfg.get("image_column", "image_path"),
                image_size=int(data_cfg.get("image_size", 224)),
                image_channels=int(data_cfg.get("image_channels", 3)),
                use_images=bool(data_cfg.get("use_images", True)),
                limit=split_limit,
            )
        else:
            ds = PTBXLMultimodalDataset(
                manifest=data_cfg["manifest"],
                root=data_cfg["root"],
                split=split,
                label_columns=data_cfg["label_columns"],
                signal_column=data_cfg.get("signal_column", "signal_hr_path"),
                image_column=data_cfg.get("image_column", "image_path"),
                signal_length=int(data_cfg.get("signal_length", 5000)),
                image_size=int(data_cfg.get("image_size", 336)),
                use_images=bool(data_cfg.get("use_images", True)),
                limit=split_limit,
            )
        loaders[split] = DataLoader(
            ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=shuffle,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            drop_last=split == "train" and len(ds) >= int(train_cfg["batch_size"]),
            persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
        )
    return loaders
