"""Train the five-channel fusion baseline from an explicit dataset manifest.

Required CSV columns: optical_path, thermal_path, height_path, mask_path, and
class_id. Missing artifacts are fatal; random replacement data is forbidden.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.predictor import SimpleFusionNetwork


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class MultiSourceCoatingDataset(Dataset):
    REQUIRED_COLUMNS = {
        "optical_path", "thermal_path", "height_path", "mask_path", "class_id"
    }

    def __init__(self, metadata_df: pd.DataFrame, data_dir: str, image_size: int):
        missing = self.REQUIRED_COLUMNS.difference(metadata_df.columns)
        if missing:
            raise ValueError(f"Dataset manifest is missing columns: {sorted(missing)}")
        self.df = metadata_df.reset_index(drop=True)
        self.data_dir = Path(data_dir).resolve()
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _resolve(self, value: str) -> Path:
        path = (self.data_dir / str(value)).resolve()
        if self.data_dir not in path.parents and path != self.data_dir:
            raise ValueError(f"Dataset path escapes data_dir: {value}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _load_scalar(path: Path) -> np.ndarray:
        array = (
            np.load(path, allow_pickle=False)
            if path.suffix.lower() == ".npy"
            else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        )
        if array is None:
            raise ValueError(f"Could not decode sensor artifact: {path}")
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        return array.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb_path = self._resolve(row["optical_path"])
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Could not decode RGB artifact: {rgb_path}")
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        thermal = self._load_scalar(self._resolve(row["thermal_path"]))
        height = self._load_scalar(self._resolve(row["height_path"]))
        thermal = cv2.resize(thermal, (self.image_size, self.image_size)).astype(np.float32)
        height = cv2.resize(height, (self.image_size, self.image_size)).astype(np.float32)
        thermal = (thermal - thermal.mean()) / (thermal.std() + 1e-6)
        height = (height - height.mean()) / (height.std() + 1e-6)

        mask_path = self._resolve(row["mask_path"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not decode mask artifact: {mask_path}")
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        ).astype(np.int64)

        fused = np.concatenate([rgb, thermal[..., None], height[..., None]], axis=-1)
        return (
            torch.from_numpy(fused).permute(2, 0, 1).float(),
            torch.tensor(int(row["class_id"]), dtype=torch.long),
            torch.from_numpy(mask).long(),
        )


def train(args) -> None:
    seed_everything(args.seed)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    metadata = pd.read_csv(args.metadata)
    if metadata.empty:
        raise ValueError("Dataset manifest contains no samples")

    num_classes = int(config.get("model", {}).get("num_classes", 5))
    dataset = MultiSourceCoatingDataset(metadata, args.data_dir, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = SimpleFusionNetwork(in_channels=5, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, labels, masks in loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            cls_out, seg_out = model(inputs)
            if seg_out.shape[-2:] != masks.shape[-2:]:
                raise RuntimeError(
                    f"Segmentation shape mismatch: output={seg_out.shape}, target={masks.shape}"
                )
            loss = criterion_cls(cls_out, labels) + criterion_seg(seg_out, masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        epoch_loss = total_loss / max(1, len(loader))
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {epoch_loss:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "in_channels": 5,
        "image_size": args.image_size,
        "seed": args.seed,
        "manifest": str(Path(args.metadata).resolve()),
        "history": history,
    }, output)
    output.with_suffix(".metadata.json").write_text(
        json.dumps({"seed": args.seed, "history": history}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the five-channel fusion baseline")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--metadata", required=True, help="CSV dataset manifest")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="outputs/baseline_model.pth")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    train(parser.parse_args())
