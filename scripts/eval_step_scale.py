"""Does the policy command smaller steps than the data, ON DISTRIBUTION?

Rollouts show the policy travelling 0.28-0.46x the demonstrated distance per
observation interval, matched on distance to the commanded box. That could be the
model (it learned small steps) or the rollout (it visits states the data never
contained and answers badly there). This asks the question where the answer is
unambiguous: at VALIDATION frames, which are on-distribution by construction,
compare the model's sampled action chunk against the ground-truth chunk recorded
at that same frame.

    python scripts/eval_step_scale.py --ckpt <ckpt> --zarr <zarr> [--frames 400]

Reports per-action |dxy z| magnitude for prediction vs ground truth, and the ratio.
A ratio near 1 on distribution means the crawl is produced by the rollout, not by
the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsteer.data import schema
from lsteer.data.dataset import ZarrChunkDataset, split_episodes
from lsteer.policy import DiffusionPolicy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--zarr", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--k", type=int, default=1, help="samples per frame; 1 mirrors the rollout's chosen chunk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    policy = DiffusionPolicy.load(args.ckpt, device=args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    import zarr

    root = zarr.open(str(args.zarr), mode="r")
    episode_ends = np.asarray(root[schema.META_EPISODE_ENDS])
    val_eps = ckpt.get("val_episodes")
    if not val_eps:
        _, val = split_episodes(len(episode_ends))
        val_eps = val.tolist()
    print(f"{len(val_eps)} validation episodes; policy act_horizon={policy.cfg.act_horizon} "
          f"pred_horizon={policy.cfg.pred_horizon}")

    ds = ZarrChunkDataset(
        args.zarr,
        episode_ids=[int(e) for e in val_eps],
        obs_horizon=policy.cfg.obs_horizon,
        pred_horizon=policy.cfg.pred_horizon,
        act_horizon=policy.cfg.act_horizon,
        camera_names=policy.cfg.camera_names,
        train=False,
    )
    idx = np.linspace(0, len(ds) - 1, min(args.frames, len(ds))).astype(int)
    g = torch.Generator(device=args.device).manual_seed(args.seed)

    pred_per_act, gt_per_act, pred_chunk, gt_chunk = [], [], [], []
    for i in idx:
        sample = ds[int(i)]
        obs = {schema.camera_obs_key(c): sample[schema.camera_obs_key(c)] for c in policy.cfg.camera_names}
        obs["state"] = sample["state"]
        if getattr(policy.cfg, "goal_dim", 0) > 0 and "goal" in sample:
            obs["goal"] = sample["goal"]
        out = policy.predict_action(obs, k=args.k, generator=g)
        p = out["action_pred"][0].cpu().numpy()          # (T_p, 7)
        gt = sample["action"].numpy()                     # (T_p, 7) ground truth chunk
        n = min(len(p), len(gt))
        pred_per_act += list(np.linalg.norm(p[:n, 0:3], axis=1) * 1000)
        gt_per_act += list(np.linalg.norm(gt[:n, 0:3], axis=1) * 1000)
        pred_chunk.append(float(np.linalg.norm(p[:n, 0:3].sum(axis=0)) * 1000))
        gt_chunk.append(float(np.linalg.norm(gt[:n, 0:3].sum(axis=0)) * 1000))

    def line(name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        print(f"  {name:<26} pred {np.median(a):7.2f}   gt {np.median(b):7.2f}   "
              f"ratio {np.median(a)/np.median(b):5.2f}x   (n={len(a)})")

    print("\nON-DISTRIBUTION step magnitude, mm (validation frames)")
    line("per action", pred_per_act, gt_per_act)
    line("per chunk (net displacement)", pred_chunk, gt_chunk)
    a = np.asarray(pred_per_act); b = np.asarray(gt_per_act)
    print(f"\n  per-action percentiles  pred p25 {np.percentile(a,25):.2f} p75 {np.percentile(a,75):.2f}"
          f"   gt p25 {np.percentile(b,25):.2f} p75 {np.percentile(b,75):.2f}")
    print("\nRollout, matched on distance, measured 0.28-0.46x.")
    print("Ratio ~1 here => the crawl is produced by the ROLLOUT (off-distribution states),")
    print("not by the model. Ratio ~0.3 here => the model itself commands small steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
