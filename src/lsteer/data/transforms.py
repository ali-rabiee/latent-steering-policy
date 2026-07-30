"""Crop augmentation. Crop-only, NO color jitter: box color IS the goal
identity in the multi-goal task — hue shifts would corrupt the supervision."""

from __future__ import annotations

import torch

from lsteer.data.schema import IMG_CROP_SIZE


def random_crop_params(h: int, w: int, size: int, generator: torch.Generator | None = None) -> tuple[int, int]:
    top = int(torch.randint(0, h - size + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - size + 1, (1,), generator=generator).item())
    return top, left


def crop(img: torch.Tensor, top: int, left: int, size: int = IMG_CROP_SIZE) -> torch.Tensor:
    """Crop (..., C, H, W) at given corner. The SAME (top, left) must be used
    for every frame of an observation window."""
    return img[..., top : top + size, left : left + size]
