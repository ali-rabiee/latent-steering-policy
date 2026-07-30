"""Converter validation on the real kinova-isaac pilot logs (skipped if absent).

Checks the whole causal chain: integrating the recomputed actions from the
first pose must recover the logged pose sequence, and the recomputed deltas
must match the logged `action_from_prev` (modulo quaternion-sign canonics).

These pilot episodes predate the per-camera image layout (they have a flat
`images/` dir), so they are parsed with `camera_names=()` — the low-dim stream
is what is under test here. The per-camera image path is covered by
test_multicamera.py against the boxes_v0 logs.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lsteer.data.convert import _to_floats, discover_episodes, parse_episode
from lsteer.utils.rotations import apply_delta_rotvec

LOGS_ROOT = Path(__file__).resolve().parents[2] / "kinova-isaac" / "logs" / "data_collection"

pytestmark = pytest.mark.skipif(not LOGS_ROOT.exists(), reason="pilot logs not found")


def first_episode():
    for ep_dir in discover_episodes(LOGS_ROOT):
        rec = parse_episode(ep_dir, require_success=False, camera_names=())
        if rec is not None:
            return rec, ep_dir
    pytest.skip("no parseable episode in pilot logs")


def test_action_matches_logged_deltas():
    rec, _ = first_episode()
    assert rec.validation["n_checked"] > 0
    assert rec.validation["max_pos_err"] <= 2e-4, rec.validation
    assert rec.validation["max_rot_err"] <= 2e-3, rec.validation


def test_action_integration_recovers_poses():
    rec, ep_dir = first_episode()
    ticks = [_to_floats(json.loads(l)) for l in (ep_dir / "ticks.jsonl").read_text().splitlines() if l.strip()]
    pos = torch.tensor([t["robot"]["ee_pose_b"]["position_m"] for t in ticks], dtype=torch.float32)
    quat = torch.tensor([t["robot"]["ee_pose_b"]["orientation_wxyz"] for t in ticks], dtype=torch.float32)

    p = pos[0].clone()
    q = quat[0].clone()
    max_pos_err = 0.0
    max_quat_err = 0.0
    for t_idx in range(len(rec.actions)):
        a = torch.from_numpy(rec.actions[t_idx])
        p = p + a[0:3]
        q = apply_delta_rotvec(q, a[3:6])
        max_pos_err = max(max_pos_err, float((p - pos[t_idx + 1]).abs().max()))
        max_quat_err = max(max_quat_err, float(1.0 - abs(float((q * quat[t_idx + 1]).sum()))))
    assert max_pos_err <= 1e-3, max_pos_err
    assert max_quat_err <= 1e-4, max_quat_err


def test_goal_metadata_present():
    rec, _ = first_episode()
    assert rec.target_color >= 0
    assert not np.isnan(rec.target_pos_b).any()
    assert (~np.isnan(rec.box_positions[:, 0])).sum() >= 2
    assert rec.states.shape[1] == 10
    assert rec.actions.shape[1] == 7
