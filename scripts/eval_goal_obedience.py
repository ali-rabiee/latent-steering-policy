"""Does a goal-conditioned policy actually obey the goal? (E2's offline screen.)

The un-conditioned screen (eval_multimodality.py) asks "do K random z's spread
over the boxes?". For a goal-conditioned policy that is the wrong question: z is
no longer supposed to pick the box, the goal input is. So instead, hold the
observation fixed, feed each box in turn as the commanded goal, and ask which
box the predicted endpoint lands nearest.

Run on HELD-OUT layouts (the checkpoint's own val episodes), same as the
multimodality sweep — on a training layout a memorising model can recall the
recorded target and look obedient without using the goal at all.

    python scripts/eval_goal_obedience.py --ckpt run/step_0080000.ckpt \
        --zarr data/boxes_v0.zarr [--episodes 8] [--k 4]

Reports per checkpoint: obedience (fraction of commands whose predicted endpoint
is nearest the commanded box) and the mean endpoint travel between the two most
distant commands (mm) — a model that ignores the goal scores ~1/n_boxes
obedience and ~0 mm travel.
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


def episode_obedience(policy, root, zarr_path, ep: int, k: int, frame: int, seed: int):
    """Returns (n_commands, n_obeyed, travel_mm) for one episode, or None."""
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
    start = sample["state"][-1, 0:3].numpy()

    finals = []
    obeyed = 0
    for i, (col, pos) in enumerate(zip(box_col, box_pos)):
        goal = np.zeros(policy.cfg.goal_dim, dtype=np.float32)
        goal[int(col)] = 1.0
        goal[schema.MAX_BOXES : schema.MAX_BOXES + 2] = pos[0:2]
        obs["goal"] = torch.from_numpy(goal)
        g = torch.Generator(device=next(policy.parameters()).device).manual_seed(seed + i)
        out = policy.predict_action(obs, k=k, generator=g)
        chunks = out["action_pred"].cpu().numpy()
        ends = start[None] + chunks[:, :, 0:3].sum(axis=1)  # (k,3)
        end = ends.mean(axis=0)
        finals.append(end)
        d = np.linalg.norm(end[None, :2] - box_pos[:, :2], axis=-1)
        obeyed += int(d.argmin() == i)

    finals = np.asarray(finals)
    travel = float(np.max([np.linalg.norm(a[:2] - b[:2]) for a in finals for b in finals])) * 1000.0
    return len(box_pos), obeyed, travel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--zarr", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4, help="samples averaged per command")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import zarr

    root = zarr.open(str(args.zarr), mode="r")
    print(f"{'checkpoint':<28} {'step':>7}  {'obedience':>10}  {'chance':>7}  {'travel':>9}")
    print("-" * 72)
    for path in args.ckpt:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        policy = DiffusionPolicy.load(path, device=args.device, weights="ema")
        if policy.cfg.goal_dim <= 0:
            print(f"{path.name:<28} {'-':>7}  not goal-conditioned (goal_dim=0)")
            continue
        val_eps = list(ck.get("val_episodes") or [])[: args.episodes]

        tot, hit, travels, chances = 0, 0, [], []
        for ep in val_eps:
            r = episode_obedience(policy, root, args.zarr, int(ep), args.k, args.frame, args.seed)
            if r is None:
                continue
            n, o, t = r
            tot += n; hit += o; travels.append(t); chances.append(1.0 / n)
        if not tot:
            print(f"{path.name:<28} {'?':>7}  no usable val episodes")
            continue
        print(f"{path.name:<28} {ck.get('step','?'):>7}  {hit/tot:>9.1%}  {np.mean(chances):>6.1%}  "
              f"{np.mean(travels):>7.1f}mm")
    print("\nobedience = predicted endpoint nearest the COMMANDED box; chance = 1/n_boxes")
    print("travel = spread between the most distant commanded endpoints (0 = goal ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
