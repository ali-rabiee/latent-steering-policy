"""Convert kinova-isaac "data_collection_v0" episodes (JSONL + PNG) to zarr.

Inverts data_collection/core/logger.py:
- floats are stored as 4-decimal strings -> parsed back to float
- actions are RECOMPUTED from consecutive absolute `ee_pose_b` records
  (a_t = pose_{t+1} (-) pose_t), because the logged `action_from_prev` at tick
  t is the delta that LED TO t (null at t=0) and is quantized ~100x coarser.
  Recomputed deltas are validated against the logged ones as a sanity check.
- gripper action is the ABSOLUTE setpoint at t+1 (open=+1/close=-1), not the
  sparse transition event.

Episode validity: metadata + instruction + ticks present, per-camera image
count == tick count, and (by default) the last grasp_result event has ok == true.

Images live in one directory per camera (`images/<cam>/image_XXXXXX.png`) and
land in one zarr array per camera (`data/img_<cam>`). Pass `camera_names=()`
to parse the low-dim stream only (used by the legacy-log regression tests,
whose episodes predate the per-camera layout).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lsteer.data import schema
from lsteer.data.obs import build_lowdim_obs, resize_to_storage
from lsteer.utils.rotations import delta_rotvec




@dataclass
class EpisodeRecord:
    path: Path
    states: np.ndarray          # (T-1, 10)
    actions: np.ndarray         # (T-1, 7)
    joints: np.ndarray          # (T-1, 6)
    image_paths: dict[str, list[Path]]  # camera name -> T-1 entries
    success: bool
    attempts: int
    target_pos_b: np.ndarray    # (3,)
    target_color: int
    box_positions: np.ndarray   # (MAX_BOXES, 3), NaN padded
    box_colors: np.ndarray      # (MAX_BOXES,), -1 padded
    language_command: str
    target_label: str
    validation: dict = field(default_factory=dict)


def _to_floats(x):
    """Recursively convert the logger's 4-decimal string floats back to float."""
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    if isinstance(x, list):
        return [_to_floats(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_floats(v) for k, v in x.items()}
    return x


def _color_id(label: str) -> int:
    for i, color in enumerate(schema.COLOR_PALETTE):
        if color in label:
            return i
    return -1


def discover_episodes(logs_root: Path) -> list[Path]:
    eps = sorted(logs_root.glob("session_*/episode_*"))
    return [e for e in eps if (e / "ticks.jsonl").exists()]


def _camera_images(ep_dir: Path, camera_names) -> dict[str, list[Path]] | None:
    """{cam: sorted frame paths}, or None if any camera's directory is missing."""
    out = {}
    for cam in camera_names:
        cam_dir = ep_dir / "images" / cam
        if not cam_dir.is_dir():
            print(f"[skip] {ep_dir}: no images/{cam}/ directory")
            return None
        out[cam] = sorted(cam_dir.glob("image_*.png"))
    return out


def parse_episode(
    ep_dir: Path,
    *,
    require_success: bool = True,
    camera_names=schema.CAMERA_NAMES,
) -> EpisodeRecord | None:
    meta_path = ep_dir / "metadata.json"
    instr_path = ep_dir / "instruction.json"
    ticks_path = ep_dir / "ticks.jsonl"
    events_path = ep_dir / "events.jsonl"
    if not (meta_path.exists() and instr_path.exists() and ticks_path.exists()):
        print(f"[skip] {ep_dir}: missing metadata/instruction/ticks")
        return None

    meta = json.loads(meta_path.read_text())
    instr = json.loads(instr_path.read_text())
    ticks = [_to_floats(json.loads(line)) for line in ticks_path.read_text().splitlines() if line.strip()]
    if len(ticks) < 3:
        print(f"[skip] {ep_dir}: only {len(ticks)} ticks")
        return None

    # success from events: use the LAST grasp_result (episodes may retry)
    success = False
    attempts = 0
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("type") == "grasp_result":
                attempts += 1
                success = bool(ev.get("data", {}).get("ok", False))
    if require_success and not success:
        print(f"[skip] {ep_dir}: no successful grasp_result (attempts={attempts})")
        return None

    # images: 1:1 with ticks, per camera
    images = _camera_images(ep_dir, camera_names)
    if images is None:
        return None
    for cam, frames in images.items():
        if len(frames) != len(ticks):
            print(f"[skip] {ep_dir}: {len(frames)} {cam} images != {len(ticks)} ticks")
            return None

    # contiguity check: tick_idx must be gapless. (t_ms is WALL clock and stalls
    # whenever the sim pauses, e.g. during cuRobo planning — sim time between
    # logged ticks is constant at the log rate as long as tick_idx is contiguous.)
    tick_idx = np.array([t["tick_idx"] for t in ticks], dtype=np.int64)
    if (np.diff(tick_idx) != 1).any():
        n_bad = int((np.diff(tick_idx) != 1).sum())
        print(f"[skip] {ep_dir}: {n_bad} tick_idx gaps")
        return None

    pos = torch.tensor([t["robot"]["ee_pose_b"]["position_m"] for t in ticks], dtype=torch.float64)
    quat = torch.tensor([t["robot"]["ee_pose_b"]["orientation_wxyz"] for t in ticks], dtype=torch.float64)
    grip = np.array([schema.gripper_state_to_float(t["robot"]["gripper"]["state"]) for t in ticks], dtype=np.float32)
    joints = np.array([t["robot"]["joints"]["positions"] for t in ticks], dtype=np.float32)

    # actions: a_t = pose_{t+1} (-) pose_t, gripper setpoint at t+1
    dpos = (pos[1:] - pos[:-1]).float()
    drot = delta_rotvec(quat[:-1].float(), quat[1:].float())
    grip_action = grip[1:]
    actions = np.concatenate([dpos.numpy(), drot.numpy(), grip_action[:, None]], axis=1).astype(np.float32)

    # validation vs logged action_from_prev (delta that led to tick t+1)
    max_pos_err, max_rot_err, n_checked = 0.0, 0.0, 0
    for t_idx in range(1, len(ticks)):
        logged = ticks[t_idx].get("policy", {}).get("action_from_prev")
        if not logged:
            continue
        lp = np.array(logged["ee_delta_pos_b"], dtype=np.float32)
        lr = np.array(logged["ee_delta_rotvec_b"], dtype=np.float32)
        pos_err = float(np.abs(actions[t_idx - 1, 0:3] - lp).max())
        # logged rotvec may be the non-canonical (angle > pi) representation of
        # the same rotation; compare modulo that by wrapping to the shorter arc
        lr_angle = float(np.linalg.norm(lr))
        if lr_angle > np.pi:
            lr = lr * (1.0 - 2.0 * np.pi / lr_angle)
        rot_err = float(np.abs(actions[t_idx - 1, 3:6] - lr).max())
        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)
        n_checked += 1

    states = np.stack(
        [
            build_lowdim_obs(pos[i].numpy(), quat[i].numpy(), float(grip[i]))
            for i in range(len(ticks) - 1)
        ]
    ).astype(np.float32)

    # goal metadata from tick 0 objects + instruction
    target_leaf = instr.get("language_command_meta", {}).get("target_leaf") or instr.get(
        "target_prim", ""
    ).split("/")[-1]
    target_color = _color_id(instr.get("target_label", instr.get("language_command", "")))
    box_positions = np.full((schema.MAX_BOXES, 3), np.nan, dtype=np.float32)
    box_colors = np.full((schema.MAX_BOXES,), -1, dtype=np.int8)
    target_pos_b = np.full((3,), np.nan, dtype=np.float32)
    for j, obj in enumerate(ticks[0].get("objects", [])[: schema.MAX_BOXES]):
        p = np.array(obj["pose_b"]["position_m"], dtype=np.float32)
        box_positions[j] = p
        box_colors[j] = _color_id(obj.get("label", ""))
        if obj.get("id") == target_leaf:
            target_pos_b = p

    return EpisodeRecord(
        path=ep_dir,
        states=states,
        actions=actions,
        joints=joints[:-1],
        image_paths={cam: frames[:-1] for cam, frames in images.items()},
        success=success,
        attempts=attempts,
        target_pos_b=target_pos_b,
        target_color=target_color,
        box_positions=box_positions,
        box_colors=box_colors,
        language_command=instr.get("language_command", ""),
        target_label=instr.get("target_label", ""),
        validation={"max_pos_err": max_pos_err, "max_rot_err": max_rot_err, "n_checked": n_checked},
    )


def convert(
    logs_root: Path,
    out_path: Path,
    *,
    require_success: bool = True,
    img_size: int = schema.IMG_STORE_SIZE,
    camera_names=schema.CAMERA_NAMES,
) -> dict:
    import imageio.v3 as iio
    import zarr
    from numcodecs import Blosc

    camera_names = tuple(camera_names)
    episodes: list[EpisodeRecord] = []
    for ep_dir in discover_episodes(logs_root):
        rec = parse_episode(ep_dir, require_success=require_success, camera_names=camera_names)
        if rec is not None:
            episodes.append(rec)
    if not episodes:
        raise RuntimeError(f"no valid episodes under {logs_root}")

    lengths = [len(e.actions) for e in episodes]
    total = int(np.sum(lengths))
    print(f"converting {len(episodes)} episodes, {total} transitions -> {out_path}")

    store = zarr.DirectoryStore(str(out_path))
    root = zarr.group(store=store, overwrite=True)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    img_arrs = {
        cam: root.create_dataset(
            schema.camera_img_key(cam),
            shape=(total, img_size, img_size, 3),
            chunks=(1, img_size, img_size, 3),
            dtype="u1",
            compressor=compressor,
        )
        for cam in camera_names
    }
    root.create_dataset(schema.DATA_STATE, data=np.concatenate([e.states for e in episodes]), dtype="f4")
    root.create_dataset(schema.DATA_ACTION, data=np.concatenate([e.actions for e in episodes]), dtype="f4")
    root.create_dataset(schema.DATA_JOINTS, data=np.concatenate([e.joints for e in episodes]), dtype="f4")
    root.create_dataset(schema.META_EPISODE_ENDS, data=np.cumsum(lengths).astype(np.int64))
    root.create_dataset(schema.META_EPISODE_SUCCESS, data=np.array([e.success for e in episodes]))
    root.create_dataset(schema.META_EPISODE_ATTEMPTS, data=np.array([e.attempts for e in episodes], dtype=np.int16))
    root.create_dataset(schema.META_TARGET_POS, data=np.stack([e.target_pos_b for e in episodes]))
    root.create_dataset(schema.META_TARGET_COLOR, data=np.array([e.target_color for e in episodes], dtype=np.int8))
    root.create_dataset(schema.META_BOX_POSITIONS, data=np.stack([e.box_positions for e in episodes]))
    root.create_dataset(schema.META_BOX_COLORS, data=np.stack([e.box_colors for e in episodes]))

    idx = 0
    for e in episodes:
        n_frames = len(e.actions)
        for cam in camera_names:
            arr = img_arrs[cam]
            for j, p in enumerate(e.image_paths[cam]):
                img = iio.imread(p)[..., :3]  # drop alpha if present
                arr[idx + j] = resize_to_storage(img, img_size)
        idx += n_frames
        print(f"  wrote {e.path.parent.name}/{e.path.name}: {n_frames} frames x {len(camera_names)} cams")

    root.attrs.update(
        {
            "source_logs_root": str(logs_root),
            "camera_names": list(camera_names),
            "img_size": img_size,
            "fps": schema.FPS,
            "color_palette": schema.COLOR_PALETTE,
            "language_commands": [e.language_command for e in episodes],
            "target_labels": [e.target_label for e in episodes],
            "episode_paths": [str(e.path) for e in episodes],
            "require_success": require_success,
        }
    )

    stats = {
        "episodes": len(episodes),
        "transitions": total,
        "cameras": list(camera_names),
        "max_pos_err": max(e.validation["max_pos_err"] for e in episodes),
        "max_rot_err": max(e.validation["max_rot_err"] for e in episodes),
        "color_histogram": {
            schema.COLOR_PALETTE[c]: int(np.sum([e.target_color == c for e in episodes]))
            for c in range(len(schema.COLOR_PALETTE))
        },
        "length_min": int(np.min(lengths)),
        "length_median": int(np.median(lengths)),
        "length_max": int(np.max(lengths)),
    }
    print(json.dumps(stats, indent=2))
    return stats
