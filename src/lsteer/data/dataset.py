"""Zarr-backed chunk dataset with Chi et al. sequence sampling.

Window layout (pred_horizon=8, obs_horizon=2, act_horizon=4):
  window ticks   [s, s+8)
  observations   ticks s .. s+1           (images + states)
  actions        ticks s .. s+7           (all 8, training target)
  "current" tick s + obs_horizon - 1; at inference the executed actions are
  window indices [obs_horizon-1, obs_horizon-1+act_horizon).
pad_before = obs_horizon-1 (repeat first frame), pad_after = act_horizon-1
(repeat last action) — identical to the reference implementation.

Images are multi-camera: one zarr array per camera, returned as separate
`img_<cam>` batch entries. Each camera gets its OWN random crop (they are
different viewpoints, so there is no spatial correspondence to preserve), but
the crop is shared across the obs_horizon frames of that camera.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lsteer.data import schema
from lsteer.data.obs import image_to_tensor
from lsteer.data.transforms import crop, random_crop_params


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
    episode_mask: np.ndarray | None = None,
) -> np.ndarray:
    indices = []
    for i in range(len(episode_ends)):
        if episode_mask is not None and not episode_mask[i]:
            continue
        start_idx = 0 if i == 0 else int(episode_ends[i - 1])
        end_idx = int(episode_ends[i])
        episode_length = end_idx - start_idx
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start = max(idx, 0) + start_idx
            buffer_end = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end
            sample_start = start_offset
            sample_end = sequence_length - end_offset
            indices.append([buffer_start, buffer_end, sample_start, sample_end])
    return np.array(indices, dtype=np.int64)


def sample_sequence(
    arr: np.ndarray, sequence_length: int, buffer_start: int, buffer_end: int, sample_start: int, sample_end: int
) -> np.ndarray:
    sample = arr[buffer_start:buffer_end]
    if sample_start == 0 and sample_end == sequence_length:
        return sample
    out = np.zeros((sequence_length,) + arr.shape[1:], dtype=arr.dtype)
    out[sample_start:sample_end] = sample
    if sample_start > 0:
        out[:sample_start] = sample[0]
    if sample_end < sequence_length:
        out[sample_end:] = sample[-1]
    return out


class ZarrChunkDataset(Dataset):
    def __init__(
        self,
        zarr_path: str | Path,
        *,
        episode_ids: np.ndarray | list[int] | None = None,
        obs_horizon: int = 2,
        pred_horizon: int = 8,
        act_horizon: int = 4,
        crop_size: int = schema.IMG_CROP_SIZE,
        train: bool = True,
        camera_names=None,
    ):
        import zarr

        self.root = zarr.open(str(zarr_path), mode="r")
        # the zarr records which cameras it was converted with; an explicit
        # argument wins so a run can train on a subset of the stored cameras
        stored = self.root.attrs.get("camera_names")
        self.camera_names = tuple(camera_names or stored or schema.CAMERA_NAMES)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.crop_size = crop_size
        self.train = train

        self.episode_ends = np.asarray(self.root[schema.META_EPISODE_ENDS])
        n_episodes = len(self.episode_ends)
        mask = np.zeros(n_episodes, dtype=bool)
        if episode_ids is None:
            mask[:] = True
        else:
            mask[np.asarray(episode_ids, dtype=int)] = True
        self.episode_mask = mask

        # low-dim fields fully in RAM; images stay lazy in zarr (1-frame chunks)
        self.state = np.asarray(self.root[schema.DATA_STATE])
        self.action = np.asarray(self.root[schema.DATA_ACTION])
        missing = [c for c in self.camera_names if schema.camera_img_key(c) not in self.root]
        if missing:
            raise KeyError(
                f"{zarr_path} has no image array for camera(s) {missing}; "
                f"available: {[k for k in self.root['data'].array_keys() if k.startswith('img')]}. "
                "Single-camera zarrs from before M3.6 must be re-converted."
            )
        self.img = {c: self.root[schema.camera_img_key(c)] for c in self.camera_names}

        self.indices = create_sample_indices(
            self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=act_horizon - 1,
            episode_mask=mask,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def stats_arrays(self) -> dict[str, np.ndarray]:
        """State/action arrays restricted to this split, for normalizer fitting."""
        sel = np.zeros(len(self.state), dtype=bool)
        for i in range(len(self.episode_ends)):
            if self.episode_mask[i]:
                start = 0 if i == 0 else int(self.episode_ends[i - 1])
                sel[start : int(self.episode_ends[i])] = True
        return {"state": self.state[sel], "action": self.action[sel]}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        buffer_start, buffer_end, sample_start, sample_end = self.indices[idx]

        action = sample_sequence(
            self.action, self.pred_horizon, buffer_start, buffer_end, sample_start, sample_end
        )
        state = sample_sequence(
            self.state, self.pred_horizon, buffer_start, buffer_end, sample_start, sample_end
        )[: self.obs_horizon]

        # observation images: window indices [0, obs_horizon); ticks before the
        # episode start (sample_start > 0) repeat the first frame
        buf_indices = [
            min(buffer_start + max(0, w - sample_start), buffer_end - 1)
            for w in range(self.obs_horizon)
        ]

        out = {
            "state": torch.from_numpy(np.ascontiguousarray(state)),
            "action": torch.from_numpy(np.ascontiguousarray(action)),
        }
        for cam, arr in self.img.items():
            img = torch.stack([image_to_tensor(arr[i]) for i in buf_indices])  # (T_o,3,H,W)
            h, w_ = img.shape[-2], img.shape[-1]
            if self.train:
                top, left = random_crop_params(h, w_, self.crop_size)
            else:
                top, left = (h - self.crop_size) // 2, (w_ - self.crop_size) // 2
            out[schema.camera_obs_key(cam)] = crop(img, top, left, self.crop_size)
        return out


def split_episodes(n_episodes: int, val_fraction: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_val = max(1, int(round(n_episodes * val_fraction))) if n_episodes > 1 else 0
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])
