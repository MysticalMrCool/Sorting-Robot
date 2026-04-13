"""
model.py - CNN Definition
=========================

Griffith 3003ICT - Programming for Robotics - Assessment 1 (Layer 3: AI)

A small convolutional neural network for classifying 128x128 RGB camera
frames into four warehouse cargo categories:

    0 -> fragile     (jam jars, biscuit boxes - glass / crushable)
    1 -> standard    (cans, apples - regular stock)
    2 -> hazardous   (oil barrels - dangerous content)
    3 -> unknown     (anything the classifier isn't confident about)

The network is intentionally small (~150k parameters) so:
  - it trains in minutes on a laptop CPU
  - inference cost is negligible compared to the Webots physics step
  - students can understand every layer

Wk07-Vision pipeline ordering (for the report):
    Camera Image -> Grayscale (optional) -> Noise Filtering ->
    Thresholding -> Edge Detection -> Object Identification -> Decision

Our CNN collapses the middle stages into learned convolution filters but
the conceptual order is preserved.
"""

from typing import List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is optional at runtime
    _TORCH_AVAILABLE = False


CATEGORIES: List[str] = ["fragile", "standard", "hazardous", "unknown"]
NUM_CLASSES = len(CATEGORIES)
INPUT_SIZE = 128


if _TORCH_AVAILABLE:

    class SortingCNN(nn.Module):
        """
        Small CNN: 3 conv blocks + 2 fully-connected layers.

        Input:  (B, 3, 128, 128)  RGB image in [0, 1]
        Output: (B, 4)            class logits
        """

        def __init__(self, num_classes: int = NUM_CLASSES):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            # After three 2x pools: 128 -> 64 -> 32 -> 16, so 64 * 16 * 16 = 16384
            self.fc1 = nn.Linear(64 * 16 * 16, 64)
            self.fc2 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

else:

    class SortingCNN:  # type: ignore[no-redef]
        """Stub used when PyTorch is not available. inference.py will fall
        back to the colour-histogram classifier instead."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is not installed. Install it with `pip install torch torchvision` "
                "or use the colour-histogram fallback in inference.py."
            )
