"""D2: did the model ever PROPOSE a good chunk, or did the selector throw one away?

Only the winning chunk used to be stored, so "a better ranking rule would have won
this episode" and "the model never offered a descent" were indistinguishable
offline. `--dump-candidates` stores all K; this reads them.

The definition is the one preregistered in experiments/runs/D2/PREREG.md, before
any episode finished. In the approach band (EE height 0.045-0.12 m) a candidate is
GOOD when its chunk endpoint is

  * within XY_TOL of the commanded box in xy  (the grasp basin G0d2 measured), and
  * at least DZ_TOL BELOW the current EE height (it descends, rather than holding
    station or retreating).

f = the fraction of in-band replans holding at least one good candidate.
f >= 0.50 -> SELECTION problem.  f <= 0.10 -> GENERATION problem.

    python scripts/analyze_candidates.py outputs/rollouts/D2_cands_s2_rep{0,1,2,3}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BAND = (0.045, 0.12)
XY_TOL = 0.016
DZ_TOL = 0.010
GRIP_IDX = 9          # schema.STATE_DIM == 10; the gripper is the last channel
GRASP_Z = 0.070       # "got into grasp territory" -- the close happens at ~0.055


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--xy-tol", type=float, default=XY_TOL)
    ap.add_argument("--dz-tol", type=float, default=DZ_TOL)
    args = ap.parse_args()

    per_box: dict[str, dict] = {}
    n_eps = n_dumped = 0
    for run in args.dirs:
        for f in sorted(run.glob("episode_*.npz")):
            d = np.load(f, allow_pickle=True)
            n_eps += 1
            cmd = str(d["commanded_id"])
            if cmd == "None" or "commanded_pos_w" not in d.files:
                raise SystemExit(f"{f}: no commanded box stored -- pre-Phase-4 run?")
            org = np.asarray(d["scene_origin"], dtype=np.float64)
            box_xy = (np.asarray(d["commanded_pos_w"], dtype=np.float64) - org)[0:2]

            # the frame check: commit_xy is what the selector actually ranked against
            if "replan000_commit_xy" in d.files:
                resid = float(np.linalg.norm(np.asarray(d["replan000_commit_xy"], dtype=np.float64) - box_xy))
                assert resid < 0.01, f"{f}: commit_xy is {resid:.4f} m from the commanded box"

            b = per_box.setdefault(
                cmd,
                {"eps": 0, "succ": 0, "inband": 0, "good": 0, "xy_only": 0,
                 "ranks": [], "best_dz": [], "best_xy": [], "chosen_dz": [],
                 "n_replan": 0, "eps_with_band": 0},
            )
            b["eps"] += 1
            b["succ"] += int(bool(d["success"]))

            keys = sorted(k for k in d.files if k.endswith("_cand_ends"))
            if keys:
                n_dumped += 1
            saw_band = False
            for k in keys:
                pre = k[: -len("_cand_ends")]
                cur = np.asarray(d[f"{pre}_state"])[-1, 0:3].astype(np.float64)
                if not (BAND[0] <= cur[2] <= BAND[1]):
                    continue
                b["inband"] += 1
                saw_band = True
                ends = np.asarray(d[k], dtype=np.float64)          # (K, 3)
                score = np.asarray(d[f"{pre}_cand_score"], dtype=np.float64)
                dxy = np.linalg.norm(ends[:, 0:2] - box_xy[None], axis=1)
                dz = ends[:, 2] - cur[2]
                near = dxy <= args.xy_tol
                good = near & (dz <= -args.dz_tol)
                b["xy_only"] += int(near.any())
                b["best_xy"].append(float(dxy.min()))
                b["best_dz"].append(float(dz.min()))
                b["chosen_dz"].append(float(dz[int(np.argmin(score))]))
                if good.any():
                    b["good"] += 1
                    # where the best good candidate sits in the CURRENT ranking
                    order = np.argsort(score)
                    b["ranks"].append(int(np.argmax(good[order])))
            if saw_band:
                b["eps_with_band"] += 1

    print(f"episodes {n_eps}, of which {n_dumped} carry candidate dumps")
    print(f"band {BAND[0]}-{BAND[1]} m | good = endpoint within {args.xy_tol*1000:.0f} mm in xy "
          f"AND >= {args.dz_tol*1000:.0f} mm below the current EE\n")
    hdr = f"{'box':<9}{'eps':>5}{'succ':>7}{'in-band':>9}{'f(good)':>9}{'f(xy)':>8}{'rank':>7}{'best dxy':>10}{'best dz':>9}{'chosen dz':>11}"
    print(hdr)
    print("-" * len(hdr))
    for cmd in sorted(per_box, key=lambda c: -per_box[c]["inband"]):
        b = per_box[cmd]
        ib = b["inband"]
        f = b["good"] / ib if ib else float("nan")
        fxy = b["xy_only"] / ib if ib else float("nan")
        rank = float(np.median(b["ranks"])) if b["ranks"] else float("nan")
        print(f"{cmd.rsplit('/',1)[-1]:<9}{b['eps']:>5}{100*b['succ']/b['eps']:>6.1f}%{ib:>9}"
              f"{f:>9.3f}{fxy:>8.3f}{rank:>7.1f}"
              f"{1000*np.median(b['best_xy']):>10.1f}{1000*np.median(b['best_dz']):>9.1f}"
              f"{1000*np.median(b['chosen_dz']):>11.1f}")
    print("\nf(good) = fraction of in-band replans with >=1 good candidate  [>=0.50 SELECTION, <=0.10 GENERATION]")
    print("f(xy)   = same, ignoring z entirely")
    print("rank    = median position of the best good candidate under the CURRENT score (0 = already chosen)")
    print("best dxy / best dz / chosen dz in mm, medians over in-band replans")

    path_table(args)


def path_table(args) -> None:
    """The corrected reading: score the PATH, not the endpoint, before the grasp.

    The preregistered table above is confounded and its own numbers show why. It
    filtered on the arm ALREADY being in the 0.045-0.12 m band, which on a
    succeeding cell is mostly the post-grasp LIFT -- so "the endpoint ascends by
    178 mm" is the model correctly proposing to lift the box it just grabbed. And
    on Obj_01 the arm reaches that band 17 times in 40 episodes, so the
    preregistered population barely exists on the cell the phase is about.

    The question the run was bought to answer is asked here instead: while the
    gripper is still OPEN, does any candidate's PATH dip to grasp height over the
    commanded box? A chunk that descends to the box and lifts away again has an
    endpoint in mid-air; only the path shows the descent.
    """
    per_box: dict[str, dict] = {}
    for run in args.dirs:
        for f in sorted(run.glob("episode_*.npz")):
            d = np.load(f, allow_pickle=True)
            cmd = str(d["commanded_id"])
            org = np.asarray(d["scene_origin"], dtype=np.float64)
            box_xy = (np.asarray(d["commanded_pos_w"], dtype=np.float64) - org)[0:2]
            b = per_box.setdefault(
                cmd, {"eps": 0, "succ": 0, "open": 0, "desc": 0, "ranks": [],
                      "minz": [], "dxy_at_min": [], "chosen_minz": []})
            b["eps"] += 1
            b["succ"] += int(bool(d["success"]))
            for k in sorted(x for x in d.files if x.endswith("_cands")):
                pre = k[: -len("_cands")]
                st = np.asarray(d[f"{pre}_state"])[-1]
                if float(st[GRIP_IDX]) <= 0.0:      # already closed: this is the lift
                    continue
                b["open"] += 1
                cur = st[0:3].astype(np.float64)
                c = np.asarray(d[k], dtype=np.float64)              # (K, T, 7)
                zs = cur[2] + np.cumsum(c[:, :, 2], axis=1)          # (K, T)
                xy = cur[0:2][None, None] + np.cumsum(c[:, :, 0:2], axis=1)
                t = np.argmin(zs, axis=1)                            # deepest step
                minz = zs[np.arange(len(zs)), t]
                dxy = np.linalg.norm(xy[np.arange(len(xy)), t] - box_xy[None], axis=1)
                desc = (minz <= GRASP_Z) & (dxy <= args.xy_tol)
                score = np.asarray(d[f"{pre}_cand_score"], dtype=np.float64)
                sel = int(np.argmin(score))
                b["minz"].append(float(minz.min()))
                b["dxy_at_min"].append(float(dxy.min()))
                b["chosen_minz"].append(float(minz[sel]))
                if desc.any():
                    b["desc"] += 1
                    b["ranks"].append(int(np.argmax(desc[np.argsort(score)])))

    print(f"\n\n=== CORRECTED: path, not endpoint, while the gripper is still OPEN ===")
    print(f"descent candidate = path dips to <= {GRASP_Z*1000:.0f} mm with its deepest point "
          f"within {args.xy_tol*1000:.0f} mm of the commanded box\n")
    hdr = (f"{'box':<9}{'eps':>5}{'succ':>7}{'open rp':>9}{'f(desc)':>9}{'rank':>7}"
           f"{'best minz':>11}{'chosen minz':>13}{'best dxy@min':>14}")
    print(hdr); print("-" * len(hdr))
    for cmd in sorted(per_box, key=lambda c: -per_box[c]["succ"] / max(per_box[c]["eps"], 1)):
        b = per_box[cmd]
        o = b["open"]
        print(f"{cmd.rsplit('/',1)[-1]:<9}{b['eps']:>5}{100*b['succ']/b['eps']:>6.1f}%{o:>9}"
              f"{b['desc']/o if o else float('nan'):>9.3f}"
              f"{(float(np.median(b['ranks'])) if b['ranks'] else float('nan')):>7.1f}"
              f"{1000*np.median(b['minz']):>11.1f}{1000*np.median(b['chosen_minz']):>13.1f}"
              f"{1000*np.median(b['dxy_at_min']):>14.1f}")
    print("\nf(desc) = fraction of gripper-open replans with >=1 descent candidate")
    print("rank    = median position of the best descent candidate under the CURRENT score (0 = already chosen)")
    print("mm, medians over gripper-open replans")


if __name__ == "__main__":
    main()
