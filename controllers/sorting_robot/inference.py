"""
inference.py - CNN Classifier (with graceful fallback)
======================================================

Griffith 3003ICT - Programming for Robotics - Assessment 1 (Layer 3: AI)

Loads the trained SortingCNN and wraps it in a simple classify(image) API
that the Behaviour Tree can call.

Three-tier loading strategy:
  1. PyTorch available + model.pt  → native CNN inference (fastest)
  2. numpy available + model_weights.npz → pure-numpy CNN forward pass
  3. Neither → colour-histogram fallback

Tier 2 exists because Webots uses system Python (which may lack PyTorch),
while training happens in a venv.  The numpy forward pass replicates the
SortingCNN architecture (3 conv blocks + 2 FC layers) using only numpy
operations, so the trained model works at runtime regardless.

Wk09 Drone lecture "fail-safe" principle: the system defaults to a safe,
still-useful state when something goes wrong.
"""

from __future__ import annotations

import os
from typing import Any, Optional

# Debug logging toggle — set True for per-frame classification output
DEBUG = False

from model import CATEGORIES, INPUT_SIZE

try:
    import numpy as np  # type: ignore
    _NUMPY_AVAILABLE = True
except Exception as _np_exc:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False
    print(f"[inference] WARNING: numpy not available ({_np_exc}); "
          f"classifier will return 'unknown' for every frame.", flush=True)

try:
    import torch  # type: ignore
    from model import SortingCNN
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure-numpy CNN forward pass (mirrors SortingCNN architecture exactly)
# ---------------------------------------------------------------------------

def _np_conv2d(x, weight, bias):
    """Conv2d with padding=1, kernel_size=3.  x: (C_in, H, W)."""
    c_out, c_in, kh, kw = weight.shape
    h, w = x.shape[1], x.shape[2]
    x_pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="constant")
    # im2col: extract every 3x3 patch → (H*W, C_in*kh*kw)
    patches = np.lib.stride_tricks.as_strided(
        x_pad,
        shape=(h, w, c_in, kh, kw),
        strides=(x_pad.strides[1], x_pad.strides[2],
                 x_pad.strides[0], x_pad.strides[1], x_pad.strides[2]),
    ).reshape(h * w, c_in * kh * kw)
    # weight reshaped to (C_out, C_in*kh*kw)
    out = patches @ weight.reshape(c_out, -1).T + bias  # (H*W, C_out)
    return out.reshape(h, w, c_out).transpose(2, 0, 1)  # (C_out, H, W)


def _np_relu(x):
    return np.maximum(x, 0, out=x)


def _np_maxpool2(x):
    """MaxPool2d(2, 2).  x: (C, H, W) → (C, H//2, W//2)."""
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def _np_softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _np_forward(x, weights):
    """Full forward pass matching SortingCNN. x: (3, 128, 128) float32."""
    x = _np_maxpool2(_np_relu(_np_conv2d(x, weights["conv1.weight"], weights["conv1.bias"])))
    x = _np_maxpool2(_np_relu(_np_conv2d(x, weights["conv2.weight"], weights["conv2.bias"])))
    x = _np_maxpool2(_np_relu(_np_conv2d(x, weights["conv3.weight"], weights["conv3.bias"])))
    x = x.ravel()
    x = _np_relu(x @ weights["fc1.weight"].T + weights["fc1.bias"])
    x = x @ weights["fc2.weight"].T + weights["fc2.bias"]
    return x  # logits (4,)


class Classifier:
    """
    Top-level classifier used by the behaviour tree.

    classify(image) -> str  (one of "fragile", "standard", "hazardous", "unknown")
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self._np_weights = None
        self.mode = "fallback"

        # --- Tier 1: PyTorch CNN ---
        if _TORCH_AVAILABLE and weights_path and os.path.exists(weights_path):
            try:
                self.model = SortingCNN()
                state = torch.load(weights_path, map_location=device)
                self.model.load_state_dict(state)
                self.model.eval()
                self.model.to(device)
                self.mode = "cnn"
                print(f"[inference] loaded CNN weights from {weights_path}", flush=True)
                return
            except Exception as e:
                print(f"[inference] failed to load CNN weights ({e})", flush=True)
                self.model = None

        # --- Tier 2: numpy CNN ---
        if _NUMPY_AVAILABLE:
            npz_path = None
            if weights_path:
                npz_path = os.path.join(os.path.dirname(weights_path), "model_weights.npz")
            if npz_path and os.path.exists(npz_path):
                try:
                    self._np_weights = dict(np.load(npz_path))
                    self.mode = "numpy_cnn"
                    print(f"[inference] loaded numpy CNN weights from {npz_path}", flush=True)
                    return
                except Exception as e:
                    print(f"[inference] failed to load numpy weights ({e})", flush=True)

        # --- Tier 3: histogram fallback ---
        if not _TORCH_AVAILABLE:
            print("[inference] PyTorch not available", flush=True)
        if self._np_weights is None:
            print("[inference] no numpy weights found", flush=True)
        print("[inference] using colour-histogram fallback", flush=True)

    # ---------- Public API ----------------------------------------------------

    def classify(self, image: Any) -> str:
        cat, _ = self.classify_with_confidence(image)
        return cat

    def classify_with_confidence(self, image: Any) -> tuple:
        """Return (category_str, confidence_float)."""
        if image is None:
            return ("unknown", 0.0)
        if not _NUMPY_AVAILABLE:
            return ("unknown", 0.0)
        if self.mode == "cnn" and self.model is not None:
            return self._classify_cnn(image)
        if self.mode == "numpy_cnn" and self._np_weights is not None:
            return self._classify_numpy_cnn(image)
        return self._classify_colour_histogram(image)

    # ---------- PyTorch CNN path ----------------------------------------------

    def _classify_cnn(self, image: Any) -> tuple:
        tensor = self._to_tensor(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            confidence, idx = torch.max(probs, dim=0)
        conf_f = float(confidence)
        pred = CATEGORIES[int(idx)]
        if conf_f < 0.5:
            return ("unknown", conf_f)
        return (pred, conf_f)

    def _to_tensor(self, image: Any) -> "torch.Tensor":
        img = image.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0
        if img.shape[0] != INPUT_SIZE or img.shape[1] != INPUT_SIZE:
            img = _resize_nearest(img, INPUT_SIZE, INPUT_SIZE)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    # ---------- Numpy CNN path ------------------------------------------------

    def _classify_numpy_cnn(self, image: Any) -> tuple:
        img = image.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0
        if img.shape[0] != INPUT_SIZE or img.shape[1] != INPUT_SIZE:
            img = _resize_nearest(img, INPUT_SIZE, INPUT_SIZE)
        # (H, W, 3) → (3, H, W) contiguous float32
        x = np.ascontiguousarray(img.transpose(2, 0, 1))
        logits = _np_forward(x, self._np_weights)
        probs = _np_softmax(logits)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        # Debug: log all class probabilities
        if DEBUG:
            prob_str = ", ".join(f"{CATEGORIES[i]}={probs[i]:.3f}" for i in range(len(CATEGORIES)))
            print(f"[inference] numpy_cnn: [{prob_str}] -> {CATEGORIES[idx]} ({conf:.3f})", flush=True)
        if conf < 0.5:
            return ("unknown", conf)
        return (CATEGORIES[idx], conf)

    # ---------- Colour-histogram fallback ------------------------------------

    def _classify_colour_histogram(self, image: Any) -> tuple:
        h, w = image.shape[:2]
        y0, y1 = h // 4, 3 * h // 4
        x0, x1 = w // 4, 3 * w // 4
        crop = image[y0:y1, x0:x1].astype(np.float32)
        if crop.max() > 1.5:
            crop /= 255.0

        r = float(crop[..., 0].mean())
        g = float(crop[..., 1].mean())
        b = float(crop[..., 2].mean())

        brightness = (r + g + b) / 3.0

        if brightness > 0.7 and abs(r - g) < 0.08 and abs(g - b) < 0.08:
            return ("unknown", 0.5)
        if r > 0.45 and r > g + 0.1 and r > b + 0.1:
            return ("hazardous", 0.5)
        if brightness > 0.35 and (g > 0.3 or r > 0.4):
            return ("standard", 0.5)
        if brightness < 0.45:
            return ("fragile", 0.5)

        return ("unknown", 0.5)


def _resize_nearest(img: Any, new_h: int, new_w: int) -> Any:
    """Nearest-neighbour resize without pulling in Pillow or OpenCV."""
    h, w = img.shape[:2]
    ys = (np.linspace(0, h - 1, new_h)).astype(np.int32)
    xs = (np.linspace(0, w - 1, new_w)).astype(np.int32)
    return img[ys][:, xs]
