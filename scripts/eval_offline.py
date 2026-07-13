"""Offline z-sweep multimodality readout (no Isaac needed).

Takes an early frame of a validation episode, samples K initial noises z,
denoises each into an action chunk, integrates the position deltas open-loop
from the current EE pose, and scatters the final XY colored by nearest box.
A steerable policy shows samples collapsing onto multiple distinct boxes.

Usage:
    python scripts/eval_offline.py --ckpt outputs/train/<run>/latest.ckpt \
        --zarr data/boxes_v0.zarr [--episode -1] [--k 64] [--frame 2]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=-1, help="episode id; -1 = first val episode")
    parser.add_argument("--frame", type=int, default=2, help="tick within the episode")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3, help="open-loop replan rounds to integrate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/z_sweep.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    policy = DiffusionPolicy.load(args.ckpt, device=args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    import zarr

    root = zarr.open(str(args.zarr), mode="r")
    episode_ends = np.asarray(root[schema.META_EPISODE_ENDS])
    if args.episode < 0:
        val_eps = ckpt.get("val_episodes")
        if not val_eps:
            _, val = split_episodes(len(episode_ends))
            val_eps = val.tolist()
        ep = int(val_eps[0])
    else:
        ep = args.episode

    ds = ZarrChunkDataset(
        args.zarr,
        episode_ids=[ep],
        obs_horizon=policy.cfg.obs_horizon,
        pred_horizon=policy.cfg.pred_horizon,
        act_horizon=policy.cfg.act_horizon,
        train=False,
    )
    sample = ds[min(args.frame, len(ds) - 1)]
    obs = {"img": sample["img"], "state": sample["state"]}

    box_pos = np.asarray(root[schema.META_BOX_POSITIONS][ep])  # (4,3) NaN-padded
    box_colors = np.asarray(root[schema.META_BOX_COLORS][ep])
    valid = ~np.isnan(box_pos[:, 0])
    box_pos, box_colors = box_pos[valid], box_colors[valid]

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    out = policy.predict_action(obs, k=args.k, generator=g)
    chunks = out["action_pred"].cpu().numpy()  # (K, T_p, 7)

    start_pos = sample["state"][-1, 0:3].numpy()
    finals = start_pos[None] + chunks[:, :, 0:3].sum(axis=1)  # (K, 3)

    dists = np.linalg.norm(finals[:, None, 0:2] - box_pos[None, :, 0:2], axis=-1)
    nearest = dists.argmin(axis=1)
    counts = np.bincount(nearest, minlength=len(box_pos))
    print("per-box sample counts:", {schema.COLOR_PALETTE[c]: int(n) for c, n in zip(box_colors, counts)})
    probs = counts / counts.sum()
    entropy = -(probs[probs > 0] * np.log(probs[probs > 0])).sum()
    print(f"coverage entropy {entropy:.3f} (uniform over {len(box_pos)} boxes = {np.log(len(box_pos)):.3f})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for j, (bp, c) in enumerate(zip(box_pos, box_colors)):
        name = schema.COLOR_PALETTE[c] if 0 <= c < len(schema.COLOR_PALETTE) else "gray"
        ax.scatter(*bp[0:2], marker="s", s=250, color=name, edgecolors="k", zorder=3)
        ax.scatter(finals[nearest == j, 0], finals[nearest == j, 1], s=12, color=name, alpha=0.6)
    ax.scatter(*start_pos[0:2], marker="*", s=200, color="k", label="EE start", zorder=4)
    ax.set_xlabel("x_b [m]")
    ax.set_ylabel("y_b [m]")
    ax.set_title(f"z-sweep, episode {ep}, K={args.k}, entropy={entropy:.2f}")
    ax.set_aspect("equal")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
