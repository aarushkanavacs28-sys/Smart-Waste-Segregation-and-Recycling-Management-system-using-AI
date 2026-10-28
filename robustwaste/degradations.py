from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw


class RandomRealWorldDegradation:
    """Randomly apply one realistic visual degradation."""

    def __init__(self, p: float = 0.65):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        choice = random.choice(["low_light", "blur", "noise", "occlusion"])
        severity = random.uniform(0.35, 0.85)
        return apply_degradation(img, choice, severity)


def apply_degradation(
    img: Image.Image,
    kind: str,
    severity: float = 0.65,
) -> Image.Image:
    severity = float(np.clip(severity, 0.0, 1.0))
    img = img.convert("RGB")

    if kind == "normal":
        return img.copy()

    if kind == "low_light":
        factor = max(0.15, 1.0 - 0.8 * severity)
        return ImageEnhance.Brightness(img).enhance(factor)

    if kind == "blur":
        radius = 0.5 + 4.5 * severity
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    if kind == "noise":
        arr = np.asarray(img).astype(np.float32)
        sigma = 8.0 + 42.0 * severity
        noise = np.random.normal(0.0, sigma, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    if kind == "occlusion":
        out = img.copy()
        draw = ImageDraw.Draw(out)
        w, h = out.size
        frac = 0.10 + 0.25 * severity
        ow = max(1, int(w * frac))
        oh = max(1, int(h * frac))
        x0 = random.randint(0, max(0, w - ow))
        y0 = random.randint(0, max(0, h - oh))
        draw.rectangle([x0, y0, x0 + ow, y0 + oh], fill=(0, 0, 0))
        return out

    raise ValueError(f"Unknown degradation kind: {kind}")
