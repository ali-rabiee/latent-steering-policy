"""Is the frozen policy actually steerable by z? The Phase-1 acceptance gate.

Sampling K initial noises on ONE observation should produce trajectories that
land on DIFFERENT boxes -- that multimodality is the whole premise of latent
steering (Phase 2 selects among the modes; Phase 3 measures their dispersion).
A collapsed policy scores entropy ~0 and cannot be steered no matter how good
the steering encoder is.

Run on HELD-OUT layouts (the checkpoint's own val episodes by default): on a
layout seen in training a memorising model can recall the recorded target and
look deceptively fine.

    python scripts/eval_multimodality.py --ckpt run/step_0025000.ckpt \
        --zarr data/boxes_v0.zarr [--k 64] [--episodes 8] [--frame 0]

Reports, per checkpoint: mean per-box coverage entropy (0 = collapsed,
ln(n_boxes) = uniform) and the endpoint spread in mm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsteer.data import schema
from lsteer.data.dataset import ZarrChunkDataset
from lsteer.policy import DiffusionPolicy


def sweep_episode(policy, root, zarr_path, ep: int, k: int, frame: int, seed: int):
    """Returns (entropy, counts, box_colors, endpoint_spread_mm) or None."""
    box_pos = np.asarray(root[schema.META_BOX_POSITIONS][ep])
    box_col = np.asarray(root[schema.META_BOX_COLORS][ep])
    valid = ~np.isnan(box_pos[:, 0])
    box_pos, box_col = box_pos[valid], box_col[valid]
    if len(box_pos) < 2:
        return None

    ds = ZarrChunkDataset(
        zarr_path, episode_ids=[ep],
        obs_horizon=policy.cfg.obs_horizon, pred_horizon=policy.cfg.pred_horizon,
        act_horizon=policy.cfg.act_horizon, camera_names=policy.cfg.camera_names,
        train=False,
    )
    if len(ds) == 0:
        return None
    sample = ds[min(frame, len(ds) - 1)]
    obs = {schema.camera_obs_key(c): sample[schema.camera_obs_key(c)] for c in policy.cfg.camera_names}
    obs["state"] = sample["state"]

    g = torch.Generator(device=next(policy.parameters()).device).manual_seed(seed)
    out = policy.predict_action(obs, k=k, generator=g)
    chunks = out["action_pred"].cpu().numpy()          # (k, T_p, 7)

    start = sample["state"][-1, 0:3].numpy()
    finals = start[None] + chunks[:, :, 0:3].sum(axis=1)
    spread = float(np.linalg.norm(finals.std(axis=0))) * 1000.0

    d = np.linalg.norm(finals[:, None, :2] - box_pos[None, :, :2], axis=-1)
    counts = np.bincount(d.argmin(axis=1), minlength=len(box_pos))
    p = counts / counts.sum()
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return entropy, counts, box_col, spread, float(np.log(len(box_pos)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--zarr", type=Path, required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=8, help="how many val episodes to sweep")
    ap.add_argument("--frame", type=int, default=0, help="window index within the episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import zarr

    root = zarr.open(str(args.zarr), mode="r")
    print(f"{'checkpoint':<28} {'step':>7}  {'entropy':>8} {'max':>6}  {'spread':>9}  coverage")
    print("-" * 96)
    for path in args.ckpt:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        policy = DiffusionPolicy.load(path, device=args.device, weights="ema")
        val_eps = list(ck.get("val_episodes") or [])[: args.episodes]

        ents, spreads, maxent, agg = [], [], 0.0, {}
        for ep in val_eps:
            r = sweep_episode(policy, root, args.zarr, int(ep), args.k, args.frame, args.seed)
            if r is None:
                continue
            e, counts, cols, sp, mx = r
            ents.append(e); spreads.append(sp); maxent = max(maxent, mx)
            for c, n in zip(cols, counts):
                name = schema.COLOR_PALETTE[c] if 0 <= c < len(schema.COLOR_PALETTE) else "?"
                agg[name] = agg.get(name, 0) + int(n)
        if not ents:
            print(f"{path.name:<28} {'?':>7}  no usable val episodes")
            continue
        tot = sum(agg.values()) or 1
        cov = " ".join(f"{k}:{100*v/tot:.0f}%" for k, v in sorted(agg.items(), key=lambda x: -x[1]))
        print(f"{path.name:<28} {ck.get('step','?'):>7}  {np.mean(ents):>8.3f} {maxent:>6.3f}  "
              f"{np.mean(spreads):>7.1f}mm  {cov}")
    print("\nentropy 0 = collapsed (z does nothing); max = uniform over the boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
