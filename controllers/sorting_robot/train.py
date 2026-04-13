"""
train.py - Offline CNN Trainer
==============================

Griffith 3003ICT - Programming for Robotics - Assessment 1 (Layer 3: AI)

Trains the SortingCNN from model.py on images collected by data_collector.py.

Expected data layout:

    controllers/data_collector/data/
        fragile/
            frame_0001.png
            frame_0002.png
            ...
        standard/
            ...
        hazardous/
            ...
        unknown/
            ...

This is NOT run by Webots - it's a plain offline Python script. Run it
from a terminal:

    cd controllers/sorting_robot
    python train.py --data ../data_collector/data --epochs 20

It writes `model.pt` next to this file, which inference.py picks up on
the next Webots launch.

Wk04-EmbeddedAI slide 12: "Rule-based and learning systems can coexist
in embedded robotics." This trainer is the "learning" half of that quote.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    from PIL import Image
except Exception as exc:
    print(f"[train] FATAL: required packages missing: {exc}", flush=True)
    print("[train] install with: pip install torch torchvision pillow numpy", flush=True)
    sys.exit(1)

# Import our shared model/category definitions
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model import SortingCNN, CATEGORIES, INPUT_SIZE  # noqa: E402


class FrameDataset(Dataset):
    """
    Loads PNG frames from a category-per-subdirectory layout.

    Simple on-the-fly augmentation: horizontal flip + mild colour jitter.
    No torchvision dependency needed.
    """

    def __init__(self, root: str, train: bool = True):
        self.root = root
        self.train = train
        self.samples: List[Tuple[str, int]] = []
        for idx, category in enumerate(CATEGORIES):
            category_dir = os.path.join(root, category)
            if not os.path.isdir(category_dir):
                continue
            for name in os.listdir(category_dir):
                if name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(category_dir, name), idx))
        if not self.samples:
            raise RuntimeError(
                f"No training images found under {root}. Run data_collector first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if img.size != (INPUT_SIZE, INPUT_SIZE):
            img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        if self.train:
            if random.random() < 0.5:
                arr = arr[:, ::-1, :].copy()  # horizontal flip
            jitter = 1.0 + (random.random() - 0.5) * 0.2  # +/- 10% brightness
            arr = np.clip(arr * jitter, 0.0, 1.0)

        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, label


def train(
    data_root: str,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_split: float = 0.15,
    out_path: str = os.path.join(_HERE, "model.pt"),
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[train] loading dataset from {data_root}", flush=True)
    full = FrameDataset(data_root, train=True)
    val_size = max(1, int(len(full) * val_split))
    train_size = len(full) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full, [train_size, val_size])
    # The val split shouldn't use augmentation; easiest is to reuse the
    # underlying dataset and toggle the flag at eval time (see loop below).

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}, train={len(train_ds)}, val={len(val_ds)}", flush=True)

    model = SortingCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        # --- train pass ------------------------------------------------------
        model.train()
        full.train = True
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()
            total_loss += float(loss) * x.size(0)
            correct += int((logits.argmax(1) == y).sum())
            total += x.size(0)
        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # --- val pass --------------------------------------------------------
        model.eval()
        full.train = False
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += int((logits.argmax(1) == y).sum())
                val_total += x.size(0)
        val_acc = val_correct / max(1, val_total)
        full.train = True

        print(
            f"[train] epoch {epoch:3d}/{epochs}  "
            f"loss={train_loss:.4f}  acc={train_acc:.3f}  val_acc={val_acc:.3f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"[train]   -> saved new best to {out_path} (val_acc={val_acc:.3f})",
                  flush=True)

    print(f"[train] done. best val_acc={best_val_acc:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(_HERE, "..", "data_collector", "data"),
        help="Path to training data root (directory with one subdir per category).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "model.pt"),
        help="Where to save the best checkpoint.",
    )
    args = parser.parse_args()
    train(
        data_root=os.path.abspath(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        out_path=os.path.abspath(args.out),
    )


if __name__ == "__main__":
    main()
