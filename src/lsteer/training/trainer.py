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

from lsteer.data import schema
from lsteer.data.dataset import ZarrChunkDataset, layout_group_ids, split_episodes
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

        root = zarr.open(cfg.data.zarr_path, mode="r")
        n_episodes = len(root["meta/episode_ends"])

        # Split by LAYOUT, not by episode. Episodes come in round-robin cycles
        # sharing one layout; splitting randomly puts a layout's twins on both
        # sides and validation then measures recall, not generalisation.
        group_ids = None
        if cfg.data.split_by_layout and schema.META_BOX_POSITIONS in root:
            group_ids = layout_group_ids(np.asarray(root[schema.META_BOX_POSITIONS]))
        self.train_eps, self.val_eps = split_episodes(
            n_episodes, cfg.data.val_fraction, cfg.data.split_seed, group_ids=group_ids
        )
        if group_ids is not None:
            tr_g, va_g = set(group_ids[self.train_eps]), set(group_ids[self.val_eps])
            leaked = tr_g & va_g
            print(
                f"split: {len(np.unique(group_ids))} layouts -> "
                f"{len(self.train_eps)} train / {len(self.val_eps)} val episodes "
                f"({len(tr_g)}/{len(va_g)} layouts), leaked layouts: {len(leaked)}"
            )
            assert not leaked, f"{len(leaked)} layouts appear in BOTH splits"
        common = dict(
            obs_horizon=cfg.data.obs_horizon,
            pred_horizon=cfg.data.pred_horizon,
            act_horizon=cfg.data.act_horizon,
            crop_size=cfg.data.crop,
            camera_names=tuple(cfg.data.camera_names) or None,
            goal_conditioned=cfg.data.goal_conditioned,
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
            goal_dim=schema.GOAL_DIM if cfg.data.goal_conditioned else 0,
            grip_loss_weight=cfg.optim.grip_loss_weight,
            grip_head=cfg.optim.grip_head,
        )
        self.policy = DiffusionPolicy(policy_cfg).to(cfg.device)
        self.policy.schedule.to(cfg.device)
        self.policy.normalizer.fit(self.train_set.stats_arrays())

        self.ema = EMAModel(self.policy, inv_gamma=cfg.optim.ema_inv_gamma, power=cfg.optim.ema_power)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
        self.best_val = float("inf")
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
                # "ema" stays a plain model state_dict — DiffusionPolicy.load reads it
                "ema": self.ema.averaged_model.state_dict(),
                # State needed to CONTINUE training after preemption. AdamW's
                # state is ~2x the model, doubling a checkpoint from 710 MB to
                # 1.4 GB, and only latest.ckpt is ever resumed from -- so the
                # periodic step_*.ckpt archives skip it.
                "ema_optimization_step": self.ema.optimization_step,
                "best_val": self.best_val,
                **({"optimizer": self.optimizer.state_dict()} if name == "latest.ckpt" else {}),
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
                (cond.shape[0], ema_policy.cfg.pred_horizon, ema_policy.diffusion_dim), device=device
            )
            x0 = ema_policy.sampler.sample(
                lambda x, t: ema_policy.unet(x, t, cond), z.shape, z=z,
                num_steps=ema_policy.cfg.num_inference_steps,
            )
            if ema_policy.cfg.grip_head:
                x0 = torch.cat([x0, torch.zeros_like(x0[..., :1])], dim=-1)
            pred = ema_policy.normalizer.unnormalize("action", x0)
            gt = batch["action"]
            pos_err.append(float(((pred[..., 0:3] - gt[..., 0:3]) ** 2).mean()))
            rot_err.append(float(((pred[..., 3:6] - gt[..., 3:6]) ** 2).mean()))
            if ema_policy.cfg.grip_head:
                # A2: the gripper is a classifier now, so MSE against +/-1 is the
                # wrong readout. Report accuracy of the close decision instead --
                # and remember it is only a screen: G0b showed NO on-distribution
                # metric predicts closed-loop behaviour on this task.
                closed = ema_policy.grip_head(cond) > 0.0
                grip_err.append(float((closed == (gt[..., 6] < 0.0)).float().mean()))
            else:
                grip_err.append(float(((pred[..., 6:7] - gt[..., 6:7]) ** 2).mean()))
        return {
            "val/eps_mse": float(np.mean(eps_losses)),
            "val/action_pos_mse": float(np.mean(pos_err)),
            "val/action_rot_mse": float(np.mean(rot_err)),
            "val/action_grip_mse": float(np.mean(grip_err)),
        }

    def _maybe_resume(self) -> int:
        """Continue from `latest.ckpt` in this run's out_dir, if present.

        Without this a preempted job restarts from step 0, which makes the
        *-preempt partitions unusable for multi-hour training -- and on a
        contended cluster those are often the only ones with free GPUs.
        Requires Slurm `--requeue` so the job comes back at all.
        """
        path = self.out_dir / "latest.ckpt"
        if not (self.cfg.resume and path.exists()):
            return 0
        ck = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.policy.load_state_dict(ck["model"])
        self.policy.normalizer.load_state_dict(ck["normalizer"])
        self.ema.averaged_model.load_state_dict(ck["ema"])
        self.ema.optimization_step = int(ck.get("ema_optimization_step", ck["step"]))
        if "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
        self.best_val = float(ck.get("best_val", float("inf")))
        step = int(ck["step"])
        print(f"RESUMED from {path} at step {step}/{self.cfg.optim.num_steps}", flush=True)
        return step

    def fit(self) -> Path:
        cfg = self.cfg
        device = cfg.device
        step = self._maybe_resume()
        start_step = step
        best_val = self.best_val
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
                sps = (step - start_step) / (time.time() - t0)
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
                    best_val = self.best_val = val_key
                    self._save(step, "best_val.ckpt")

            if step % cfg.log.ckpt_every == 0:
                self._save(step, f"step_{step:07d}.ckpt")
                self._save(step, "latest.ckpt")

        self._save(step, "latest.ckpt")
        return self.out_dir / "latest.ckpt"
