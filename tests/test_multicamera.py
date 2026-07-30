"""M3.6: the dual-camera observation path, model side + converter side.

The model half is pure PyTorch. The converter half runs against the real
two-camera box logs (`kinova-isaac/logs/boxes_v0`) and is skipped if absent.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from lsteer.data import schema
from lsteer.data.convert import discover_episodes, parse_episode
from lsteer.models.vision import ObsEncoder
from lsteer.policy import DiffusionPolicy, PolicyConfig

BOX_LOGS = Path(__file__).resolve().parents[2] / "kinova-isaac" / "logs" / "boxes_v0"


# ------------------------------------------------------------------ model side
def tiny_policy(camera_names=schema.CAMERA_NAMES) -> DiffusionPolicy:
    torch.manual_seed(0)
    cfg = PolicyConfig(
        camera_names=tuple(camera_names), down_dims=(32, 64), img_dim=16,
        num_keypoints=8, num_inference_steps=2,
    )
    policy = DiffusionPolicy(cfg).eval()
    policy.normalizer.fit(
        {
            "state": torch.randn(100, cfg.state_dim).numpy(),
            "action": torch.randn(100, cfg.action_dim).numpy(),
        }
    )
    return policy


def make_obs(cfg: PolicyConfig, seed: int = 7) -> dict:
    g = torch.Generator().manual_seed(seed)
    obs = {
        schema.camera_obs_key(c): torch.rand(cfg.obs_horizon, 3, 96, 96, generator=g)
        for c in cfg.camera_names
    }
    obs["state"] = torch.randn(cfg.obs_horizon, cfg.state_dim, generator=g)
    return obs


def test_encoder_dim_scales_with_cameras():
    one = ObsEncoder(state_dim=10, obs_horizon=2, img_dim=16, camera_names=("front",))
    two = ObsEncoder(state_dim=10, obs_horizon=2, img_dim=16, camera_names=("front", "wrist"))
    assert one.out_dim == 2 * (16 + 10)
    assert two.out_dim == 2 * (2 * 16 + 10)


def test_per_camera_encoders_are_independent():
    """No weight sharing: front and wrist must be separate parameter sets."""
    enc = ObsEncoder(state_dim=10, obs_horizon=2, img_dim=16, camera_names=("front", "wrist"))
    front = dict(enc.vision["front"].named_parameters())
    wrist = dict(enc.vision["wrist"].named_parameters())
    assert front.keys() == wrist.keys()
    assert all(front[k] is not wrist[k] for k in front)
    # and both are registered on the module (they get gradients / EMA updates)
    names = {n for n, _ in enc.named_parameters()}
    assert any(n.startswith("vision.front.") for n in names)
    assert any(n.startswith("vision.wrist.") for n in names)


def test_missing_camera_raises():
    policy = tiny_policy()
    obs = make_obs(policy.cfg)
    del obs["img_wrist"]
    with pytest.raises(KeyError, match="img_wrist"):
        policy.predict_action(obs)


def test_every_camera_affects_the_prediction():
    """Perturbing ONE camera must change the output — catches a dropped view."""
    policy = tiny_policy()
    z = torch.randn(1, policy.cfg.pred_horizon, policy.cfg.action_dim,
                    generator=torch.Generator().manual_seed(3))
    base = make_obs(policy.cfg)
    out_base = policy.predict_action(base, z=z.clone())["action_pred"]
    for cam in policy.cfg.camera_names:
        obs = {k: v.clone() for k, v in base.items()}
        obs[schema.camera_obs_key(cam)] = torch.rand_like(obs[schema.camera_obs_key(cam)])
        out = policy.predict_action(obs, z=z.clone())["action_pred"]
        assert not torch.allclose(out, out_base, atol=1e-6), f"{cam} view is ignored"


def test_checkpoint_roundtrips_camera_names(tmp_path):
    policy = tiny_policy(("front", "wrist"))
    path = tmp_path / "p.ckpt"
    policy.save(path)
    loaded = DiffusionPolicy.load(path)
    assert loaded.cfg.camera_names == ("front", "wrist")
    assert loaded.obs_encoder.out_dim == policy.obs_encoder.out_dim


# -------------------------------------------------------------- converter side
@pytest.mark.skipif(not BOX_LOGS.exists(), reason="boxes_v0 logs not found")
def test_boxes_logs_parse_with_both_cameras():
    parsed = [
        rec
        for ep in discover_episodes(BOX_LOGS)
        if (rec := parse_episode(ep, require_success=True)) is not None
    ]
    assert parsed, f"no episode under {BOX_LOGS} parsed with cameras {schema.CAMERA_NAMES}"
    for rec in parsed:
        assert set(rec.image_paths) == set(schema.CAMERA_NAMES)
        for cam, frames in rec.image_paths.items():
            # one frame per transition, and frames must be the camera's own dir
            assert len(frames) == len(rec.actions), f"{rec.path}: {cam}"
            assert all(p.parent.name == cam for p in frames)
        assert rec.states.shape == (len(rec.actions), schema.STATE_DIM)
        assert not np.isnan(rec.target_pos_b).any()


@pytest.mark.skipif(not BOX_LOGS.exists(), reason="boxes_v0 logs not found")
def test_missing_camera_dir_skips_episode():
    ep = discover_episodes(BOX_LOGS)[0]
    assert parse_episode(ep, require_success=False, camera_names=("front", "nope")) is None
