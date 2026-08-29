"""DiffusionPolicy: obs encoder + conditional U-Net + DDIM sampler + normalizer.

`predict_action` exposes the initial noise `z` as a first-class argument and
always returns the z it used — with eta=0 the map (obs, z) -> action chunk is
deterministic, which is the handle Phase 2 (steering encoder) drives and
Phase 3 (conformal gate, k>1 dispersion) measures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsteer.data import schema
from lsteer.data.normalizer import LinearNormalizer
from lsteer.diffusion import DDIMSampler, DiffusionSchedule
from lsteer.models import ConditionalUnet1D, ObsEncoder


@dataclass
class PolicyConfig:
    state_dim: int = schema.STATE_DIM
    action_dim: int = schema.ACTION_DIM
    camera_names: tuple[str, ...] = schema.CAMERA_NAMES
    obs_horizon: int = 2
    pred_horizon: int = 8
    act_horizon: int = 4
    img_dim: int = 128
    num_keypoints: int = 32
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    diffusion_step_embed_dim: int = 128
    n_groups: int = 8
    num_train_timesteps: int = 100
    num_inference_steps: int = 16
    clip_sample: bool = True
    # E2: >0 appends a goal vector (target colour one-hot + xy) to the global
    # conditioning. 0 keeps the original unconditioned architecture.
    goal_dim: int = 0
    # E3: relative weight of the gripper channel in the denoising loss. Under a
    # flat 7-channel MSE the gripper is learned as bimodal +/-1 noise -- in
    # rollout it flips shut ~1.6 s in and back open a moment later -- while the
    # demos are unambiguous, closing at EE height 0.047 m (p5 0.047, p95 0.048)
    # at the bottom of the descent, every one of 420 episodes.
    grip_loss_weight: float = 1.0
    # A2: take the gripper OUT of the diffused action vector and give it its own
    # head, trained as classification (BCE) on the conditioning alone, with a
    # monotone "once closed, stays closed" commitment applied across the chunk.
    #
    # Why, measured. As one of 7 regressed channels the gripper is denoised from
    # fresh noise at every replan, so it is re-decided independently each step and
    # cannot represent a latch. In rollout the champion closes 6 times per
    # successful episode and 17 times per failed one, first closing at 0.106 m,
    # where every one of 420 demos closes exactly once at 0.047 m and never
    # reopens. G0c then measured the consequence directly: median max finger
    # closure is 0.929-0.939 rad for the scripted expert (stopped by the box
    # between the fingers) and exactly 1.202 -- the travel limit, fully shut on
    # nothing -- for the policy on all three seeds.
    #
    # G0d2 ruled out the alternative explanation: aiming the expert 8 mm off the
    # box (the policy's own approach error) costs it NOTHING, 40/40, and 16 mm
    # also costs nothing. Degradation only starts at 32 mm, where the expert's
    # finger signature becomes 1.200 -- the policy's. So the gap is not where the
    # arm is, it is when the hand closes.
    #
    # False keeps the original architecture exactly, so existing checkpoints
    # (goalcond_v1, the champion) load and behave unchanged.
    grip_head: bool = False
    grip_head_hidden: int = 256


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_encoder = ObsEncoder(
            state_dim=cfg.state_dim,
            obs_horizon=cfg.obs_horizon,
            img_dim=cfg.img_dim,
            num_keypoints=cfg.num_keypoints,
            camera_names=cfg.camera_names,
        )
        # A2: with a separate gripper head the U-Net diffuses only the 6 pose
        # channels; the gripper is classified, not denoised.
        self.diffusion_dim = cfg.action_dim - 1 if cfg.grip_head else cfg.action_dim
        cond_dim = self.obs_encoder.out_dim + cfg.goal_dim
        if cfg.grip_head:
            # Deterministic in the observation: no diffusion noise enters the
            # close decision at all, which is what removes the per-replan
            # re-rolling that produces the flicker.
            self.grip_head = nn.Sequential(
                nn.Linear(cond_dim, cfg.grip_head_hidden),
                nn.Mish(),
                nn.Linear(cfg.grip_head_hidden, cfg.grip_head_hidden),
                nn.Mish(),
                nn.Linear(cfg.grip_head_hidden, cfg.pred_horizon),
            )
        self.unet = ConditionalUnet1D(
            input_dim=self.diffusion_dim,
            global_cond_dim=cond_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=tuple(cfg.down_dims),
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
        )
        self.schedule = DiffusionSchedule(cfg.num_train_timesteps)
        self.sampler = DDIMSampler(self.schedule, clip_sample=cfg.clip_sample)
        self.normalizer = LinearNormalizer()

    # ------------------------------------------------------------ obs plumbing
    def _encode_obs(self, obs: dict[str, torch.Tensor], state_n: torch.Tensor) -> torch.Tensor:
        """Build the encoder input from the img_<cam> entries + normalized state
        (+ the goal vector, appended raw, when the policy is goal-conditioned)."""
        enc_in = {schema.camera_obs_key(c): obs[schema.camera_obs_key(c)] for c in self.cfg.camera_names}
        enc_in["state"] = state_n
        cond = self.obs_encoder(enc_in)
        if self.cfg.goal_dim > 0:
            if "goal" not in obs:
                raise KeyError("policy is goal-conditioned (goal_dim>0) but obs has no 'goal'")
            goal = obs["goal"]
            if goal.dim() == 1:
                goal = goal.unsqueeze(0)
            cond = torch.cat([cond, goal.to(cond.dtype)], dim=-1)
        return cond

    # ---------------------------------------------------------------- train
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """batch: img_<cam> (B,T_o,3,H,W) in [0,1] per camera; state (B,T_o,D_s);
        action (B,T_p,A)."""
        state_n = self.normalizer.normalize("state", batch["state"])
        action_n = self.normalizer.normalize("action", batch["action"])
        cond = self._encode_obs(batch, state_n)

        if self.cfg.grip_head:
            # Diffuse the 6 pose channels only.
            target_n = action_n[..., : self.diffusion_dim]
            b = target_n.shape[0]
            t = torch.randint(0, self.schedule.num_train_timesteps, (b,), device=target_n.device)
            noise = torch.randn_like(target_n)
            x_t = self.schedule.q_sample(target_n, t, noise)
            eps_loss = F.mse_loss(self.unet(x_t, t, cond), noise)
            # Gripper as classification. The demos encode open=+1 / close=-1, so
            # "closed" is the negative half; BCE on that label is a far better
            # match than MSE to a bimodal setpoint, and it is the loss that can
            # express "this is the moment", which regression to +/-1 cannot.
            grip_target = (batch["action"][..., 6] < 0.0).to(cond.dtype)  # (B, T_p)
            grip_loss = F.binary_cross_entropy_with_logits(self.grip_head(cond), grip_target)
            return eps_loss + self.cfg.grip_loss_weight * grip_loss

        b = action_n.shape[0]
        t = torch.randint(0, self.schedule.num_train_timesteps, (b,), device=action_n.device)
        noise = torch.randn_like(action_n)
        x_t = self.schedule.q_sample(action_n, t, noise)
        pred = self.unet(x_t, t, cond)
        if self.cfg.grip_loss_weight != 1.0:
            w = torch.ones(self.cfg.action_dim, device=pred.device, dtype=pred.dtype)
            w[6] = self.cfg.grip_loss_weight
            return (w * (pred - noise) ** 2).mean()
        return F.mse_loss(pred, noise)

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_action(
        self,
        obs: dict[str, torch.Tensor],
        *,
        z: Optional[torch.Tensor] = None,
        k: int = 1,
        generator: Optional[torch.Generator] = None,
        num_steps: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """obs: img_<cam> (T_o,3,H,W) or (1,T_o,3,H,W) in [0,1] for every camera
        in cfg.camera_names; state (T_o,D_s) or (1,T_o,D_s).

        z: optional initial noise (k, T_p, A). If None it is drawn from N(0, I)
        (via `generator` if given). k > 1 batches K samples over ONE encoded
        observation — the encoder runs once (the future dispersion-gate path).

        Returns {action: (k,T_a,A), action_pred: (k,T_p,A), z: (k,T_p,A)},
        unnormalized, on the policy's device.
        """
        cfg = self.cfg
        device = next(self.parameters()).device

        imgs = {}
        for cam in cfg.camera_names:
            key = schema.camera_obs_key(cam)
            if key not in obs:
                raise KeyError(f"observation is missing '{key}' (cameras: {cfg.camera_names})")
            img = obs[key].to(device)
            if img.dim() == 4:
                img = img.unsqueeze(0)
            if img.shape[0] != 1:
                raise ValueError("predict_action takes a single observation; use k for multiple samples")
            imgs[key] = img
        state = obs["state"].to(device)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        if self.cfg.goal_dim > 0 and "goal" in obs:
            imgs["goal"] = obs["goal"].to(device)

        state_n = self.normalizer.normalize("state", state)
        cond = self._encode_obs(imgs, state_n)  # (1, C)
        cond_k = cond.expand(k, -1)

        d = self.diffusion_dim
        if z is None:
            z = torch.randn((k, cfg.pred_horizon, d), generator=generator, device=device)
        else:
            z = z.to(device)
            if z.shape != (k, cfg.pred_horizon, d):
                raise ValueError(f"z shape {tuple(z.shape)} != {(k, cfg.pred_horizon, d)}")

        x0 = self.sampler.sample(
            lambda x, t: self.unet(x, t, cond_k),
            z.shape,
            z=z,
            num_steps=num_steps or cfg.num_inference_steps,
        )
        if cfg.grip_head:
            # Pad the gripper slot so the normalizer (fitted on all 7 channels)
            # still applies; the placeholder is overwritten immediately below.
            x0 = torch.cat([x0, torch.zeros_like(x0[..., :1])], dim=-1)
        action_pred = self.normalizer.unnormalize("action", x0)
        if cfg.grip_head:
            closed = (self.grip_head(cond_k) > 0.0)  # (k, T_p), logit > 0 == p > 0.5
            # Monotone commitment ACROSS the chunk: once the head says close, it
            # stays closed for the rest of the predicted horizon. This is the
            # half of A2 that regression could not express at all -- the demos
            # close exactly once and never reopen, and the champion reopens after
            # its first close in 39 of 40 episodes.
            closed = torch.cummax(closed.to(torch.uint8), dim=1).values.bool()
            action_pred = action_pred.clone()
            action_pred[..., 6] = torch.where(
                closed,
                torch.full_like(action_pred[..., 6], schema.GRIPPER_CLOSE),
                torch.full_like(action_pred[..., 6], schema.GRIPPER_OPEN),
            )
        start = cfg.obs_horizon - 1
        action = action_pred[:, start : start + cfg.act_horizon]
        return {"action": action, "action_pred": action_pred, "z": z}

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path, extra: Optional[dict] = None) -> None:
        payload = {
            "config": asdict(self.cfg),
            "model": self.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu", weights: str = "auto") -> "DiffusionPolicy":
        """weights: 'auto' prefers EMA weights when present; 'model'/'ema' force."""
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg_dict = dict(ckpt["config"])
        cfg_dict["down_dims"] = tuple(cfg_dict["down_dims"])
        cfg_dict["camera_names"] = tuple(cfg_dict.get("camera_names", schema.CAMERA_NAMES))
        policy = cls(PolicyConfig(**cfg_dict))
        if weights == "auto":
            weights = "ema" if "ema" in ckpt else "model"
        policy.load_state_dict(ckpt[weights])
        policy.normalizer.load_state_dict(ckpt["normalizer"])
        policy.schedule.to(device)
        return policy.to(device).eval()
