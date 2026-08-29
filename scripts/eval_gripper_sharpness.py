"""Why does more data reach better and grasp worse?

Five datasets in a row (R1, R2, R3, E4, P7) have bought reaching precision and
paid for it in grasp. The 621-episode model has the best approach of any model
measured (4.45 mm vs the champion's 7.3) and lifts 15 points less.

HYPOTHESIS. The gripper channel is a +/-1 setpoint whose TIMING is the most
variable thing across demonstrations: every demo agrees on where the box is, but
they disagree by a frame or two on exactly when the hand shuts. Position averages
harmlessly across demos -- they all drive to the same place -- while a disagreement
about timing averages to something in the MIDDLE, which is not a valid gripper
command at all. More data means more demonstrations averaged per visual state, so
the position head sharpens and the gripper head blurs toward 0.

MEASUREMENT. For matched states from the same validation episodes, denoise a chunk
from each checkpoint and look at the gripper channel's distribution:
  - |g| near 1  => a decisive open/shut command
  - |g| near 0  => the blurred average of "shut now" and "not yet"
Report the fraction of predictions in the dead band |g| < 0.5, and the position
error on the same samples, so sharpening and blurring can be seen together.

REFUTED IF the 621-episode model's gripper predictions are as sharp as the
420-episode model's -- then averaging is not what breaks the grasp.

    python -u scripts/eval_gripper_sharpness.py \
        --ckpts A.ckpt B.ckpt --zarr data/boxes_v0.zarr --episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from _isaac_bootstrap import add_lsteer_to_path

add_lsteer_to_path()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--zarr", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--samples-per-episode", type=int, default=12)
    args = ap.parse_args()

    from lsteer.data import schema
    from lsteer.data.dataset import DiffusionPolicyDataset
    from lsteer.policy import DiffusionPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}  zarr {args.zarr}")

    for ck in args.ckpts:
        policy = DiffusionPolicy.load(ck, device=device)
        cfg = policy.cfg
        ds = DiffusionPolicyDataset(
            str(args.zarr),
            obs_horizon=cfg.obs_horizon,
            pred_horizon=cfg.pred_horizon,
            act_horizon=cfg.act_horizon,
            camera_names=cfg.camera_names,
            train=False,  # eval preprocessing: center crop, no augmentation
            goal_conditioned=getattr(cfg, "goal_dim", 0) > 0,
        )
        n = min(len(ds), args.episodes * args.samples_per_episode)
        idx = np.linspace(0, len(ds) - 1, n).astype(int)

        g_pred, g_true, pos_err = [], [], []
        for i in idx:
            b = ds[int(i)]
            obs = {}
            for cam in cfg.camera_names:
                obs[schema.camera_obs_key(cam)] = b[schema.camera_obs_key(cam)]
            obs["state"] = b["state"]
            if getattr(cfg, "goal_dim", 0) > 0 and "goal" in b:
                obs["goal"] = b["goal"]
            with torch.no_grad():
                out = policy.predict_action(obs, k=1)
            a = out["action"][0].cpu().numpy()          # (T_a, 7)
            t = b["action"].numpy()[: a.shape[0]]        # ground truth
            g_pred.append(a[:, 6])
            g_true.append(t[:, 6])
            pos_err.append(np.linalg.norm(a[:, 0:3] - t[:, 0:3], axis=1))

        gp = np.concatenate(g_pred)
        gt = np.concatenate(g_true)
        pe = np.concatenate(pos_err)
        dead = float((np.abs(gp) < 0.5).mean())
        print(f"\n=== {ck.parent.name}/{ck.name} ===  n={len(gp)} predicted gripper values")
        print(f"  mean |g|                    {np.abs(gp).mean():.4f}   (1.0 = fully decisive)")
        print(f"  FRACTION IN DEAD BAND |g|<0.5 {dead:.4f}   <-- the blurring measure")
        print(f"  fraction |g|<0.2            {float((np.abs(gp) < 0.2).mean()):.4f}")
        print(f"  gripper sign accuracy       {float((np.sign(gp) == np.sign(gt)).mean()):.4f}")
        print(f"  position error (m)          med {np.median(pe):.5f}  mean {pe.mean():.5f}")
        print(f"  ground-truth mean |g|       {np.abs(gt).mean():.4f}   (demos are exactly +/-1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
