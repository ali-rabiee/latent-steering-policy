"""Training loop: DDPM MSE + AdamW + cosine/warmup schedule + EMA.

The EMA weights are what gets frozen and shipped as the final policy.
Checkpoints carry model + EMA + normalizer + config + episode split so that
rollout needs nothing but the .ckpt file.
"""

from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lsteer.data.dataset import ZarrChunkDataset, split_episodes
from lsteer.models.ema import EMAModel
from lsteer.policy import DiffusionPolicy, PolicyConfig
from lsteer.training.config import TrainConfig


def _git_sha(repo_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _make_logger(cfg: TrainConfig, out_dir: Path):
    kind = cfg.log.logger
    if kind == "wandb":
        import wandb

        wandb.init(project=cfg.log.project, name=cfg.log.run_name, config=cfg.to_dict(), dir=str(out_dir))
        return lambda step, d: wandb.log(d, step=step)
    if kind == "tb":
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
        return lambda step, d: [writer.add_scalar(k, v, step) for k, v in d.items()]
    return lambda step, d: None


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        run_name = cfg.log.run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self.out_dir = Path(cfg.log.out_dir) / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # data --------------------------------------------------------------
        import zarr

        n_episodes = len(zarr.open(cfg.data.zarr_path, mode="r")["meta/episode_ends"])
        self.train_eps, self.val_eps = split_episodes(n_episodes, cfg.data.val_fraction, cfg.data.split_seed)
        common = dict(
            obs_horizon=cfg.data.obs_horizon,
            pred_horizon=cfg.data.pred_horizon,
            act_horizon=cfg.data.act_horizon,
            crop_size=cfg.data.crop,
            camera_names=tuple(cfg.data.camera_names) or None,
        )
        self.train_set = ZarrChunkDataset(cfg.data.zarr_path, episode_ids=self.train_eps, train=True, **common)
        self.val_set = (
            ZarrChunkDataset(cfg.data.zarr_path, episode_ids=self.val_eps, train=False, **common)
            if len(self.val_eps)
            else None
        )
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=cfg.optim.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=cfg.data.num_workers > 0,
            drop_last=True,
        )
        self.val_loader = (
            DataLoader(self.val_set, batch_size=cfg.optim.batch_size, shuffle=False, num_workers=2)
            if self.val_set
            else None
        )

        # model ---------------------------------------------------------------
        # the dataset is the authority on which cameras are actually present
        policy_cfg = PolicyConfig(
            camera_names=self.train_set.camera_names,
            obs_horizon=cfg.data.obs_horizon,
            pred_horizon=cfg.data.pred_horizon,
            act_horizon=cfg.data.act_horizon,
            img_dim=cfg.model.img_dim,
            num_keypoints=cfg.model.num_keypoints,
            down_dims=tuple(cfg.model.down_dims),
            kernel_size=cfg.model.kernel_size,
            diffusion_step_embed_dim=cfg.model.diffusion_step_embed_dim,
            num_train_timesteps=cfg.diffusion.train_steps,
            num_inference_steps=cfg.diffusion.infer_steps,
            clip_sample=cfg.diffusion.clip_sample,
        )
        self.policy = DiffusionPolicy(policy_cfg).to(cfg.device)
        self.policy.schedule.to(cfg.device)
        self.policy.normalizer.fit(self.train_set.stats_arrays())

        self.ema = EMAModel(self.policy, inv_gamma=cfg.optim.ema_inv_gamma, power=cfg.optim.ema_power)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
        self.log_fn = _make_logger(cfg, self.out_dir)
        self.git_sha = _git_sha(Path(__file__).resolve().parents[3])

    def _lr_at(self, step: int) -> float:
        o = self.cfg.optim
        if step < o.warmup_steps:
            return o.lr * step / max(1, o.warmup_steps)
        progress = (step - o.warmup_steps) / max(1, o.num_steps - o.warmup_steps)
        return o.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _save(self, step: int, name: str) -> None:
        self.policy.save(
            self.out_dir / name,
            extra={
                "ema": self.ema.averaged_model.state_dict(),
                "step": step,
                "train_config": self.cfg.to_dict(),
                "git_sha": self.git_sha,
                "train_episodes": self.train_eps.tolist(),
                "val_episodes": self.val_eps.tolist(),
            },
        )

    @torch.no_grad()
    def validate(self, max_batches: int = 8) -> dict[str, float]:
        """eps-MSE with the EMA weights + DDIM action MSE (pos/rot/gripper split)."""
        if self.val_loader is None:
            return {}
        ema_policy: DiffusionPolicy = self.ema.averaged_model
        ema_policy.normalizer = self.policy.normalizer
        device = self.cfg.device
        eps_losses, pos_err, rot_err, grip_err = [], [], [], []
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            eps_losses.append(float(ema_policy.compute_loss(batch)))

            # denoise a full chunk per sample and compare to GT actions
            state_n = ema_policy.normalizer.normalize("state", batch["state"])
            cond = ema_policy._encode_obs(batch, state_n)
            z = torch.randn(
                (cond.shape[0], ema_policy.cfg.pred_horizon, ema_policy.cfg.action_dim), device=device
            )
            x0 = ema_policy.sampler.sample(
                lambda x, t: ema_policy.unet(x, t, cond), z.shape, z=z,
                num_steps=ema_policy.cfg.num_inference_steps,
            )
            pred = ema_policy.normalizer.unnormalize("action", x0)
            gt = batch["action"]
            pos_err.append(float(((pred[..., 0:3] - gt[..., 0:3]) ** 2).mean()))
            rot_err.append(float(((pred[..., 3:6] - gt[..., 3:6]) ** 2).mean()))
            grip_err.append(float(((pred[..., 6:7] - gt[..., 6:7]) ** 2).mean()))
        return {
            "val/eps_mse": float(np.mean(eps_losses)),
            "val/action_pos_mse": float(np.mean(pos_err)),
            "val/action_rot_mse": float(np.mean(rot_err)),
            "val/action_grip_mse": float(np.mean(grip_err)),
        }

    def fit(self) -> Path:
        cfg = self.cfg
        device = cfg.device
        step = 0
        best_val = float("inf")
        self.policy.train()
        data_iter = iter(self.train_loader)
        t0 = time.time()

        while step < cfg.optim.num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            lr = self._lr_at(step)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            loss = self.policy.compute_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.ema.step(self.policy)
            step += 1

            if step % cfg.log.log_every == 0:
                sps = step / (time.time() - t0)
                self.log_fn(step, {"train/loss": float(loss), "train/lr": lr, "train/steps_per_s": sps})
                print(f"step {step:>7d}  loss {float(loss):.5f}  lr {lr:.2e}  {sps:.1f} it/s")

            if self.val_loader is not None and step % cfg.log.val_every == 0:
                self.policy.eval()
                metrics = self.validate()
                self.policy.train()
                self.log_fn(step, metrics)
                print("  " + "  ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
                val_key = metrics.get("val/action_pos_mse", float("inf"))
                if val_key < best_val:
                    best_val = val_key
                    self._save(step, "best_val.ckpt")

            if step % cfg.log.ckpt_every == 0:
                self._save(step, f"step_{step:07d}.ckpt")
                self._save(step, "latest.ckpt")

        self._save(step, "latest.ckpt")
        return self.out_dir / "latest.ckpt"
