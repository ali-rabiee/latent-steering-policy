"""Observation building shared by the converter, the training dataset, and the
Isaac rollout scripts. Any change here changes BOTH sides — that is the point.
"""

from __future__ import annotations

import numpy as np
import torch

from lsteer.data.schema import IMG_CROP_SIZE, IMG_STORE_SIZE, gripper_state_to_float
from lsteer.utils.rotations import quat_to_rot6d


def resize_to_storage(img: np.ndarray, size: int = IMG_STORE_SIZE) -> np.ndarray:
    """Resize an RGB uint8 HWC image (e.g. 640x640 camera frame) to the stored
    resolution using area interpolation."""
    assert img.dtype == np.uint8 and img.ndim == 3, f"expected uint8 HWC, got {img.dtype} {img.shape}"
    if img.shape[0] == size and img.shape[1] == size:
        return img
    try:
        import cv2

        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((size, size), Image.BOX))


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC [0,255] -> float32 CHW [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0


def center_crop(img: torch.Tensor, size: int = IMG_CROP_SIZE) -> torch.Tensor:
    """Center crop (..., C, H, W) -> (..., C, size, size). Used at eval/rollout."""
    h, w = img.shape[-2], img.shape[-1]
    top = (h - size) // 2
    left = (w - size) // 2
    return img[..., top : top + size, left : left + size]


def preprocess_image(img: np.ndarray, train: bool = False) -> torch.Tensor:
    """Full camera-frame -> network-input pipeline for a single frame.

    Rollout uses train=False (deterministic center crop). Training applies its
    own random crop in the dataset (see transforms.random_crop_params), so it
    calls resize_to_storage/image_to_tensor directly instead of this helper.
    """
    if train:
        raise ValueError("training-time cropping is handled by the dataset, not here")
    t = image_to_tensor(resize_to_storage(img))
    return center_crop(t)


def build_lowdim_obs(pos_b, quat_wxyz_b, gripper) -> np.ndarray:
    """(3,) pos + (4,) wxyz quat + gripper ('open'/'close' or float) -> (10,) f32.

    Layout: [pos(3), rot6d(6), gripper(1)] — see lsteer.data.schema.
    """
    if isinstance(gripper, str):
        gripper = gripper_state_to_float(gripper)
    q = torch.as_tensor(quat_wxyz_b, dtype=torch.float32)
    rot6d = quat_to_rot6d(q).numpy()
    return np.concatenate(
        [np.asarray(pos_b, dtype=np.float32), rot6d.astype(np.float32), [np.float32(gripper)]]
    )
