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
        goal_conditioned: bool = False,
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

        # E2: per-episode goal vector = target colour one-hot + target xy (base).
        # The collector recorded round-robin targets, so this is ground truth,
        # not a reconstruction from the gripper signal.
        self.goal_conditioned = goal_conditioned
        if goal_conditioned:
            colors = np.asarray(self.root[schema.META_TARGET_COLOR], dtype=np.int64)
            pos = np.asarray(self.root[schema.META_TARGET_POS], dtype=np.float32)
            if colors.min() < 0 or colors.max() >= schema.MAX_BOXES:
                raise ValueError(f"target colors outside [0,{schema.MAX_BOXES}): {np.unique(colors)}")
            goals = np.zeros((n_episodes, schema.GOAL_DIM), dtype=np.float32)
            goals[np.arange(n_episodes), colors] = 1.0
            goals[:, schema.MAX_BOXES:] = pos[:, 0:2]
            self.goal_vecs = goals

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
        if self.goal_conditioned:
            # buffer_start is always inside the source episode (padding only
            # repeats frames, it never crosses an episode boundary)
            ep = int(np.searchsorted(self.episode_ends, buffer_start, side="right"))
            out["goal"] = torch.from_numpy(self.goal_vecs[ep])
        for cam, arr in self.img.items():
            img = torch.stack([image_to_tensor(arr[i]) for i in buf_indices])  # (T_o,3,H,W)
            h, w_ = img.shape[-2], img.shape[-1]
            if self.train:
                top, left = random_crop_params(h, w_, self.crop_size)
            else:
                top, left = (h - self.crop_size) // 2, (w_ - self.crop_size) // 2
            out[schema.camera_obs_key(cam)] = crop(img, top, left, self.crop_size)
        return out


def layout_group_ids(box_positions: np.ndarray, tol: float = 0.02) -> np.ndarray:
    """Group episodes that share a box layout.

    The collector records demos in cycles: the SAME layout is used for one
    episode per box (round-robin targets) before the boxes are respawned. That
    grouping is the multimodal pairing the policy is supposed to learn — and it
    is exactly why episodes must not be split randomly. Layouts within a cycle
    differ only by physics settling (~1-3 mm), while different cycles differ by
    >=10 cm, so a 2 cm tolerance separates them cleanly.

    box_positions: (n_episodes, MAX_BOXES, 3), NaN-padded. Returns (n_episodes,)
    integer group ids.
    """
    pts = []
    for row in box_positions:
        v = np.asarray(row, dtype=np.float64)
        pts.append(v[~np.isnan(v[:, 0])][:, :2])

    def same_layout(a: np.ndarray, b: np.ndarray) -> bool:
        # Greedy nearest-neighbour matching, NOT a sorted-key compare: settling
        # jitter can reorder two boxes with similar x, which would make an
        # order-dependent key spuriously distinct.
        if a.shape != b.shape:
            return False
        if a.size == 0:
            return True
        used = np.zeros(len(b), dtype=bool)
        for p in a:
            d = np.linalg.norm(b - p, axis=1)
            d[used] = np.inf
            j = int(d.argmin())
            if d[j] >= tol:
                return False
            used[j] = True
        return True

    gid = np.full(len(pts), -1, dtype=np.int64)
    nxt = 0
    for i in range(len(pts)):
        if gid[i] >= 0:
            continue
        gid[i] = nxt
        for j in range(i + 1, len(pts)):
            if gid[j] < 0 and same_layout(pts[i], pts[j]):
                gid[j] = nxt
        nxt += 1
    return gid


def split_episodes(
    n_episodes: int,
    val_fraction: float = 0.1,
    seed: int = 42,
    group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/val episode split.

    `group_ids` (from `layout_group_ids`) splits by LAYOUT rather than by
    episode. Without it, the four episodes sharing a layout get scattered across
    both splits: measured on boxes_v0, 88% of val episodes had a layout twin in
    train, so val error measured recall on memorised layouts instead of
    generalisation — and the reported 4.5 mm was not evidence the policy works.
    """
    rng = np.random.default_rng(seed)
    if group_ids is None:
        perm = rng.permutation(n_episodes)
        n_val = max(1, int(round(n_episodes * val_fraction))) if n_episodes > 1 else 0
        return np.sort(perm[n_val:]), np.sort(perm[:n_val])

    group_ids = np.asarray(group_ids)
    groups = np.unique(group_ids)
    gperm = rng.permutation(len(groups))
    n_val_g = max(1, int(round(len(groups) * val_fraction))) if len(groups) > 1 else 0
    val_groups = set(groups[gperm[:n_val_g]].tolist())
    is_val = np.array([g in val_groups for g in group_ids])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)
