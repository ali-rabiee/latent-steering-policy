"""The z-determinism contract that Phase 2 (latent steering) depends on."""

import torch

from lsteer.policy import DiffusionPolicy, PolicyConfig


def tiny_policy() -> DiffusionPolicy:
    torch.manual_seed(0)
    cfg = PolicyConfig(down_dims=(32, 64), img_dim=16, num_keypoints=8, num_inference_steps=4)
    policy = DiffusionPolicy(cfg).eval()
    policy.normalizer.fit(
        {
            "state": torch.randn(100, cfg.state_dim).numpy(),
            "action": torch.randn(100, cfg.action_dim).numpy(),
        }
    )
    return policy


def make_obs(cfg: PolicyConfig) -> dict:
    g = torch.Generator().manual_seed(7)
    return {
        "img": torch.rand(cfg.obs_horizon, 3, 96, 96, generator=g),
        "state": torch.randn(cfg.obs_horizon, cfg.state_dim, generator=g),
    }


def test_same_z_same_action():
    policy = tiny_policy()
    obs = make_obs(policy.cfg)
    z = torch.randn(1, policy.cfg.pred_horizon, policy.cfg.action_dim,
                    generator=torch.Generator().manual_seed(3))
    out1 = policy.predict_action(obs, z=z.clone())
    out2 = policy.predict_action(obs, z=z.clone())
    assert torch.equal(out1["action_pred"], out2["action_pred"])
    assert torch.equal(out1["z"], z)


def test_different_z_different_action():
    policy = tiny_policy()
    obs = make_obs(policy.cfg)
    z1 = torch.randn(1, policy.cfg.pred_horizon, policy.cfg.action_dim,
                     generator=torch.Generator().manual_seed(3))
    z2 = torch.randn(1, policy.cfg.pred_horizon, policy.cfg.action_dim,
                     generator=torch.Generator().manual_seed(4))
    out1 = policy.predict_action(obs, z=z1)
    out2 = policy.predict_action(obs, z=z2)
    assert not torch.allclose(out1["action_pred"], out2["action_pred"])


def test_generator_reproducible_when_z_none():
    policy = tiny_policy()
    obs = make_obs(policy.cfg)
    out1 = policy.predict_action(obs, generator=torch.Generator().manual_seed(5))
    out2 = policy.predict_action(obs, generator=torch.Generator().manual_seed(5))
    assert torch.equal(out1["action_pred"], out2["action_pred"])


def test_k_samples_shapes_and_batching():
    policy = tiny_policy()
    cfg = policy.cfg
    obs = make_obs(cfg)
    out = policy.predict_action(obs, k=5, generator=torch.Generator().manual_seed(6))
    assert out["action"].shape == (5, cfg.act_horizon, cfg.action_dim)
    assert out["action_pred"].shape == (5, cfg.pred_horizon, cfg.action_dim)
    assert out["z"].shape == (5, cfg.pred_horizon, cfg.action_dim)
    # k-batched sampling must equal running each z separately (shared cond)
    single = policy.predict_action(obs, z=out["z"][2:3], k=1)
    assert torch.allclose(single["action_pred"][0], out["action_pred"][2], atol=1e-5)
