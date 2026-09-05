"""Per-COMMANDED-box readout of one or more rollout directories.

Why this exists: `rollout_sim.py` overwrites `label` with the box the gripper
CLOSED ON (line ~1334), so `label` is an outcome, not a command, and `""` there
means "never closed" rather than "no command". Grouping a per-box table by it
silently groups by outcome -- that already produced one wrong conclusion.

Phase 4 stores `commanded_label` / `commanded_id` in the npz. For runs written
before that, the commanded box is recovered from `replan000_commit_xy`, which
lives in the same frame as `layout_pos_w[:, 0:2]` minus `scene_origin`; the match
residual is asserted below 10 mm. A third, independent grouping -- the
round-robin rule `sorted(layout_ids)[ep % n]` -- is computed too, and all
available groupings must agree on every episode or the script refuses to report.

    python scripts/analyze_per_box.py outputs/rollouts/A7_s2_rep{0,1,2,3}
    python scripts/analyze_per_box.py --json out.json outputs/rollouts/D1_*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESIDUAL_MAX_M = 0.01


def _med(vals) -> float:
    """Median ignoring nan; nan when a column is absent from an older run."""
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.median(a)) if a.size else float("nan")


def _commanded(d, ep_idx: int, ids: list[str]) -> tuple[str, dict]:
    """Return the commanded leaf id plus the per-grouping votes that produced it."""
    votes: dict[str, str] = {}

    if "commanded_id" in d.files:
        cid = str(d["commanded_id"])
        if cid and cid != "None":
            votes["stored"] = cid

    # round-robin: rollout_sim.py assigns leaves[ep % len(leaves)] for --expert,
    # --steer-to-box and goal-conditioned checkpoints alike
    votes["roundrobin"] = ids[ep_idx % len(ids)]

    if "replan000_commit_xy" in d.files:
        cxy = np.asarray(d["replan000_commit_xy"], dtype=np.float64)
        org = np.asarray(d["scene_origin"], dtype=np.float64)[0:2]
        P = np.asarray(d["layout_pos_w"], dtype=np.float64)[:, 0:2] - org[None]
        dist = np.linalg.norm(P - cxy[None], axis=1)
        j = int(np.argmin(dist))
        if dist[j] < RESIDUAL_MAX_M:
            votes["commit_xy"] = ids[j]
        else:
            votes["commit_xy"] = f"UNMATCHED({dist[j]:.4f}m)"

    return votes.get("stored", votes["roundrobin"]), votes


def load(dirs: list[Path]) -> tuple[list[dict], list[str], np.ndarray, np.ndarray]:
    eps, ids, pos_b, home = [], None, None, None
    for run in dirs:
        files = sorted(run.glob("episode_*.npz"))
        if not files:
            print(f"  (no episodes in {run})")
            continue
        for i, f in enumerate(files):
            d = np.load(f, allow_pickle=True)
            run_ids = [str(x) for x in d["layout_ids"]]
            org = np.asarray(d["scene_origin"], dtype=np.float64)
            pos = np.asarray(d["layout_pos_w"], dtype=np.float64) - org[None]
            if ids is None:
                ids, pos_b = run_ids, pos
            elif run_ids != ids or float(np.abs(pos - pos_b).max()) > 1e-4:
                raise SystemExit(
                    f"{f}: layout differs from the first run -- these directories are\n"
                    "different tables and must not be pooled. Group by seed."
                )
            cmd, votes = _commanded(d, i, ids)
            rkeys = sorted(k for k in d.files if k.endswith("_state"))
            zs = [float(np.asarray(d[k])[-1, 2]) for k in rkeys]
            if home is None and rkeys:
                home = np.asarray(d[rkeys[0]])[-1, 0:2]
            cl = {str(k): float(v) for k, v in zip(d["closest_ids"], d["closest_dist"])}
            eps.append(
                {
                    "file": str(f),
                    "commanded": cmd,
                    "votes": votes,
                    "success": bool(d["success"]),
                    "reached_label": str(d["label"]),
                    "never_closed": str(d["label"]) == "",
                    "max_finger_rad": float(d["max_finger_rad"]) if "max_finger_rad" in d.files else float("nan"),
                    "tgt_err_mm": (float(d["tgt_err_median_m"]) * 1000.0
                                   if "tgt_err_median_m" in d.files else float("nan")),
                    "max_lift_m": float(d["max_lift_m"]),
                    "closest_cmd_mm": cl.get(cmd, float("nan")) * 1000.0,
                    "min_ee_z": min(zs) if zs else float("nan"),
                    "n_replans": len(rkeys),
                }
            )
    if ids is None:
        raise SystemExit("no episodes found")
    return eps, ids, pos_b, home


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    eps, ids, pos_b, home = load(args.dirs)

    # ---- grouping agreement: refuse to report a per-box table we cannot trust
    disagree = [e for e in eps if len(set(e["votes"].values())) > 1]
    print(f"episodes: {len(eps)}   groupings present: "
          f"{sorted(set(k for e in eps for k in e['votes']))}")
    print(f"grouping agreement: {len(eps) - len(disagree)}/{len(eps)}")
    if disagree:
        for e in disagree[:5]:
            print(f"  MISMATCH {Path(e['file']).name}: {e['votes']}")
        raise SystemExit(
            f"{len(disagree)} episodes disagree on the commanded box -- not reporting."
        )
    if not any("stored" in e["votes"] for e in eps):
        print("note: no commanded_label in these npz (pre-Phase-4 run); "
              "commanded box recovered from replan000_commit_xy + round-robin.")

    # ---- per-box table
    max_finger = np.array([e["max_finger_rad"] for e in eps])
    rows = []
    for j, leaf in enumerate(ids):
        sel = [e for e in eps if e["commanded"] == leaf]
        if not sel:
            continue
        r_base = float(np.linalg.norm(pos_b[j, 0:2]))
        r_home = float(np.linalg.norm(pos_b[j, 0:2] - home)) if home is not None else float("nan")
        rows.append(
            {
                "box": leaf.rsplit("/", 1)[-1],
                "leaf": leaf,
                "n": len(sel),
                "success": float(np.mean([e["success"] for e in sel])),
                "r_base_m": round(r_base, 4),
                "r_home_m": round(r_home, 4),
                "closest_mm_median": round(_med([e["closest_cmd_mm"] for e in sel]), 2),
                "max_finger_median": round(_med([e["max_finger_rad"] for e in sel]), 4),
                "tgt_err_mm_median": round(_med([e["tgt_err_mm"] for e in sel]), 3),
                "never_closed": round(float(np.mean([e["never_closed"] for e in sel])), 3),
                "min_ee_z_median": round(_med([e["min_ee_z"] for e in sel]), 4),
                "below_8cm_frac": round(float(np.mean([e["min_ee_z"] < 0.08 for e in sel])), 3),
                "replans_median": float(np.median([e["n_replans"] for e in sel])),
            }
        )
    rows.sort(key=lambda r: -r["r_base_m"])

    hdr = ("box", "n", "success", "r_base", "r_home", "closest_mm", "finger",
           "tgterr_mm", "never_cl", "min_z", "<8cm", "replans")
    print()
    print(f"{hdr[0]:<10}{hdr[1]:>5}{hdr[2]:>9}{hdr[3]:>8}{hdr[4]:>8}"
          f"{hdr[5]:>11}{hdr[6]:>8}{hdr[7]:>11}{hdr[8]:>9}{hdr[9]:>8}{hdr[10]:>7}{hdr[11]:>9}")
    for r in rows:
        print(f"{r['box']:<10}{r['n']:>5}{r['success']*100:>8.1f}%{r['r_base_m']:>8.3f}"
              f"{r['r_home_m']:>8.3f}{r['closest_mm_median']:>11.2f}"
              f"{r['max_finger_median']:>8.3f}{r['tgt_err_mm_median']:>11.2f}"
              f"{r['never_closed']*100:>8.1f}%"
              f"{r['min_ee_z_median']:>8.3f}{r['below_8cm_frac']*100:>6.0f}%"
              f"{r['replans_median']:>9.0f}")
    n = len(eps)
    k = sum(e["success"] for e in eps)
    se = float(np.sqrt(k / n * (1 - k / n) / n)) * 100 if n else float("nan")
    print(f"\npooled: {k}/{n} = {100*k/n:.1f}%  (SE {se:.1f} pts)")
    if len(rows) > 2:
        r = np.corrcoef([x["r_base_m"] for x in rows], [x["success"] for x in rows])[0, 1]
        print(f"corr(success, radius from base) = {r:+.3f}  over {len(rows)} cells")

    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "n": n, "success": k / n}, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
