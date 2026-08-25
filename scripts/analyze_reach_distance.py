"""Is closed-loop box choice governed by DISTANCE rather than colour?

Every rollout job spawns its own random box layout (fixed across the episodes
of that job, but NOT shared between jobs), so per-colour numbers from different
runs describe different table arrangements and must never be compared directly.
The comparable quantity is each box's distance from the arm's home pose.

Feed it one or more rollout directories (each holding episode_*.npz):

    python scripts/analyze_reach_distance.py outputs/rollouts/m7_* [--commanded]

For every run it prints each box's distance from home, how often it was reached,
and — for runs where the target was commanded round-robin — how often the
commanded box was actually the one reached.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_run(run: Path):
    files = sorted(run.glob("episode_*.npz"))
    if not files:
        return None
    eps = []
    ids = pos_b = home = None
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        ids = [str(x) for x in d["layout_ids"]]
        if "scene_origin" in d.files:
            org = np.asarray(d["scene_origin"], dtype=np.float64)[0:2]
        else:
            # runs before scene_origin was logged: recover it from the episodes
            # themselves (median offset between a reached box and the final EE)
            org = None
        pos_w = np.asarray(d["layout_pos_w"], dtype=np.float64)[:, 0:2]
        keys = sorted(k for k in d.files if k.endswith("_state"))
        home = np.asarray(d["replan000_state"])[-1, 0:2]
        final = np.asarray(d[keys[-1]])[-1, 0:2]
        eps.append({"reached": str(d["reached"]), "pos_w": pos_w, "org": org,
                    "home": home, "final": final, "ids": ids, "idx": i})
    if eps[0]["org"] is None:
        # estimate the world->base offset from every episode whose reached box
        # is known: the EE ends up near that box, so the median residual is the
        # offset. Coarse, but only used for runs predating the logged origin.
        res = []
        for e in eps:
            if e["reached"] in e["ids"]:
                res.append(e["pos_w"][e["ids"].index(e["reached"])] - e["final"])
        org = np.median(np.asarray(res), axis=0) if res else np.zeros(2)
        for e in eps:
            e["org"] = org
    pos_b = eps[0]["pos_w"] - eps[0]["org"]
    return eps, ids, pos_b, eps[0]["home"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--commanded", action="store_true",
                    help="runs used round-robin commanded targets (sorted(ids)[ep %% n])")
    ap.add_argument("--json", action="store_true", help="emit machine-readable rows")
    args = ap.parse_args()

    out_rows = []
    for run in args.runs:
        loaded = load_run(run)
        if loaded is None:
            print(f"{run.name}: no episodes")
            continue
        eps, ids, pos_b, home = loaded
        dists = {i: float(np.linalg.norm(p - home)) for i, p in zip(ids, pos_b)}
        reached = Counter(e["reached"] for e in eps)
        n = len(eps)
        print(f"\n{run.name}  ({n} episodes, home {np.round(home,3).tolist()})")
        print(f"  {'box':>8} {'dist_m':>7} {'reached':>9}" + ("  obeyed" if args.commanded else ""))
        for i in sorted(ids, key=lambda k: dists[k]):
            line = f"  {i:>8} {dists[i]:>7.3f} {reached.get(i,0)/n:>8.1%}"
            row = {"run": run.name, "box": i, "dist_m": round(dists[i], 3),
                   "reached_frac": round(reached.get(i, 0) / n, 3)}
            if args.commanded:
                sub = [e for e in eps if sorted(ids)[e["idx"] % len(ids)] == i]
                obeyed = sum(e["reached"] == i for e in sub)
                line += f"  {obeyed}/{len(sub)}"
                row["obeyed"] = f"{obeyed}/{len(sub)}"
                row["obey_frac"] = round(obeyed / max(1, len(sub)), 3)
            print(line)
            out_rows.append(row)
    if args.json:
        print("\nJSON:")
        print(json.dumps(out_rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
