"""Is this dataset actually fit to train on?

Run BEFORE converting a fresh collection into training, and again on the zarr.
A demonstration that does not lift the box is not a demonstration, and a bad
episode is far more expensive than a missing one: behaviour cloning will
happily learn whatever is in here.

    python scripts/validate_dataset.py --zarr data/boxes_v0.zarr
    python scripts/validate_dataset.py --logs-root logs/boxes_v1   # raw session dirs

Checks, in order of how badly each one bites:

  1. every episode is a SUCCESS (the box was lifted)
  2. state/action consistency -- action[t] must equal the pose delta from t to
     t+1, because the converter recomputes actions that way; a mismatch means
     the recorded actions do not describe the recorded motion (this is exactly
     how a mid-reach perturbation leaks in and teaches the policy to lurch)
  3. no NaN / non-finite values anywhere
  4. the gripper closes, once, near the end
  5. the grasp happens at a sane height (demos close within a millimetre of
     EE z = 0.047, i.e. box_top + 0.01)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsteer.data import schema

# from the demos that lift: EE height at the moment the gripper closes
GRASP_Z_EXPECTED = 0.047
GRASP_Z_TOL = 0.02


def _fail(msg: str) -> tuple[bool, str]:
    return False, msg


def check_zarr(path: Path, action_tol: float = 2e-3, max_step: float = 0.09) -> int:
    import zarr

    root = zarr.open(str(path), mode="r")
    state = np.asarray(root[schema.DATA_STATE])
    action = np.asarray(root[schema.DATA_ACTION])
    ends = np.asarray(root[schema.META_EPISODE_ENDS])
    starts = np.concatenate([[0], ends[:-1]])
    success = np.asarray(root[schema.META_EPISODE_SUCCESS])
    n_ep = len(ends)

    print(f"dataset: {path}")
    print(f"  {n_ep} episodes, {len(state)} frames\n")

    problems: list[str] = []

    # 1. every episode lifted
    n_ok = int(np.sum(success))
    line = f"  [{'PASS' if n_ok == n_ep else 'FAIL'}] every episode lifts the box: {n_ok}/{n_ep}"
    print(line)
    if n_ok != n_ep:
        bad = np.flatnonzero(~success)[:10].tolist()
        problems.append(f"{n_ep - n_ok} episodes did not lift (first: {bad})")

    # 2. actions describe the recorded motion
    worst, worst_ep = 0.0, -1
    for e in range(n_ep):
        s, t = int(starts[e]), int(ends[e])
        pos = state[s:t, 0:3]
        act = action[s:t, 0:3]
        if t - s < 2:
            continue
        err = np.abs((pos[:-1] + act[:-1]) - pos[1:]).max()
        if err > worst:
            worst, worst_ep = float(err), e
    ok = worst <= action_tol
    print(f"  [{'PASS' if ok else 'FAIL'}] actions match the motion: worst mismatch "
          f"{worst*1000:.2f} mm (episode {worst_ep}, tolerance {action_tol*1000:.0f} mm)")
    if not ok:
        problems.append(
            f"action/state mismatch up to {worst*1000:.1f} mm in episode {worst_ep} - the recorded "
            "actions do not describe the recorded motion, so training would learn the discrepancy")

    # 2b. no action is larger than a real motion can be
    #
    # The consistency check above CANNOT catch a teleport: if the logs skip a
    # gap, action[t] still equals pose[t+1]-pose[t], so it is perfectly
    # "consistent" while describing a motion that never happened. This is how a
    # recovery dataset acquired 210 fake 0.31 m jumps -- 0.88% of frames
    # carrying 20.9% of the squared position error, which collapsed training to
    # 0% success. Magnitude is the only thing that catches it.
    mag = np.linalg.norm(action[:, 0:3], axis=1)
    n_out = int((mag > max_step).sum())
    ok_mag = n_out == 0
    print(f"  [{'PASS' if ok_mag else 'FAIL'}] no impossible single-step motions: "
          f"{n_out} frames above {max_step*1000:.0f} mm (largest {mag.max()*1000:.1f} mm)")
    if not ok_mag:
        share = float((action[mag > max_step, 0:3] ** 2).sum() / (action[:, 0:3] ** 2).sum())
        eps_hit = set()
        for e in range(n_ep):
            s_, t_ = int(starts[e]), int(ends[e])
            if mag[s_:t_].max() > max_step:
                eps_hit.add(e)
        problems.append(
            f"{n_out} frames in {len(eps_hit)} episodes exceed {max_step*1000:.0f} mm in one step "
            f"and carry {share:.0%} of the squared position error - these are almost certainly "
            "logging gaps, not motions, and training will spend that fraction of its budget on them")

    # 2c. every episode starts where the others start
    #
    # R1's recovery data began each perturbed episode AT the off-course pose,
    # mid-flight, instead of at home. Nothing here caught it: the actions were
    # consistent, finite and in range, and every episode lifted. The model still
    # came out closing the gripper at 0.247 m -- home height -- and never
    # descending, because a demo set whose episodes start in scattered places no
    # longer agrees on what "early in the reach" looks like. A recovery design
    # is only safe if the episode stays continuous, so check the starts.
    first = np.stack([state[int(s), 0:3] for s in starts])
    home = np.median(first, axis=0)
    off_home = np.linalg.norm(first - home, axis=1)
    n_adrift = int((off_home > 0.05).sum())
    ok_start = n_adrift == 0
    print(f"  [{'PASS' if ok_start else 'FAIL'}] episodes start at the same home pose: "
          f"{n_adrift} start >50 mm away (worst {off_home.max()*1000:.0f} mm)")
    if not ok_start:
        problems.append(
            f"{n_adrift} episodes begin more than 50 mm from the common start pose - these are "
            "fragments beginning mid-motion, not demonstrations, and they destroy the agreement "
            "about how far through a reach the gripper should close")

    # 3. finite
    finite = bool(np.isfinite(state).all() and np.isfinite(action).all())
    print(f"  [{'PASS' if finite else 'FAIL'}] all values finite")
    if not finite:
        problems.append("NaN or inf in state/action")

    # 4/5. gripper closes once, late, at a sane height
    n_never, n_multi, heights, frac_through = 0, 0, [], []
    for e in range(n_ep):
        s, t = int(starts[e]), int(ends[e])
        g = action[s:t, 6]
        closed = g <= 0
        if not closed.any():
            n_never += 1
            continue
        first = int(np.flatnonzero(closed)[0])
        # count open->close transitions
        n_multi += int((np.diff(closed.astype(int)) > 0).sum() > 1)
        heights.append(float(state[s + first, 2]))
        frac_through.append(first / max(1, t - s - 1))
    print(f"  [{'PASS' if n_never == 0 else 'FAIL'}] gripper closes in every episode "
          f"({n_ep - n_never}/{n_ep})")
    if n_never:
        problems.append(f"{n_never} episodes never close the gripper")
    if heights:
        h = np.asarray(heights)
        off = np.abs(h - GRASP_Z_EXPECTED)
        ok_h = bool((off <= GRASP_Z_TOL).all())
        print(f"  [{'PASS' if ok_h else 'WARN'}] grasp height sane: mean {h.mean():.4f} m "
              f"(expected {GRASP_Z_EXPECTED}, worst off by {off.max()*1000:.0f} mm)")
        if not ok_h:
            problems.append(f"{int((off > GRASP_Z_TOL).sum())} episodes close far from the demo grasp height")
        f = np.asarray(frac_through)
        print(f"  [INFO] gripper closes {f.mean():.0%} of the way through the episode "
              f"(min {f.min():.0%}, max {f.max():.0%})")
        lens = ends - starts
        print(f"  [INFO] episode length: median {int(np.median(lens))} frames "
              f"(min {int(lens.min())}, max {int(lens.max())}) - a recovery set is expected to be "
              "bimodal, with the perturbed episodes longer")

        # How tightly does height predict "how soon do I close"?
        #
        # This is the diagnostic that explains R1 rather than merely flagging it.
        # Closing the gripper is a decision the policy has to read off the scene,
        # and height is the strongest cue it has. In clean data every frame that
        # sits N steps before the close is at nearly the same height, so the cue
        # is unambiguous. R1's recovery episodes started up to 568 mm away yet
        # still closed 60% of the way through, which taught "high and off course
        # -> closing shortly" -- and the trained model duly closed at 0.247 m and
        # never descended. Diversity of state is the POINT of recovery data, so
        # this is reported, not failed: what matters is that the spread stays
        # small close to the grasp, where the timing decision is actually made.
        buckets: dict[int, list[float]] = {}
        for e in range(n_ep):
            s, t = int(starts[e]), int(ends[e])
            g = action[s:t, 6]
            closed = np.flatnonzero(g <= 0)
            if not len(closed):
                continue
            c = int(closed[0])
            for tt in range(max(0, c - 20), c):
                buckets.setdefault(c - tt, []).append(float(state[s + tt, 2]))
        cells = []
        for d in (1, 2, 4, 8, 16):
            v = np.asarray(buckets.get(d, []))
            if len(v) < 10:
                continue
            spread = float(np.percentile(v, 90) - np.percentile(v, 10))
            cells.append(f"d={d}: {np.median(v):.3f}m +/-{spread*1000:.0f}mm")
        if cells:
            print(f"  [INFO] EE height N frames before the gripper closes -- {', '.join(cells)}. "
                  "A wide spread at small d means height no longer tells the policy when to close.")

        # Is the vertical correction BALANCED?
        #
        # R2's knocks were upward only, so every recovery it ever demonstrated was
        # a descent: 68% of travel-phase z-actions pointed down against 20% in the
        # clean demos. The policy learned "correcting = go down" and drove a median
        # 5 cm THROUGH the box, and the overshoot deepened with training (lowest EE
        # z +0.020 at 20k, -0.004 at 60k) -- a directional prior, not a skill. A
        # recovery set has to shove the arm both ways, so watch this number: near
        # the clean baseline means balanced, far above it means biased downward.
        travel = state[:, 2] > 0.20
        if travel.sum() > 50:
            share_down = float((action[travel, 2] < 0).mean())
            print(f"  [INFO] travel-phase vertical corrections pointing DOWN: {share_down:.0%} "
                  f"(clean demos are 20%; R2's biased set was 68% and over-descended by 5 cm)")

    print()
    if problems:
        print("VERDICT: NOT FIT TO TRAIN ON")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERDICT: fit to train on")
    return 0


def check_logs(root: Path) -> int:
    """Raw session dirs, before conversion: did each episode actually lift?"""
    eps = sorted(root.glob("session_*/episode_*"))
    if not eps:
        print(f"no episodes under {root}")
        return 1
    ok = bad = missing = 0
    bad_ids = []
    for ep in eps:
        ev = ep / "events.jsonl"
        if not ev.exists():
            missing += 1
            continue
        verdict = None
        for line in ev.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("event") == "grasp_result" or "ok" in d.get("data", {}):
                verdict = d.get("data", {}).get("ok", d.get("ok"))
        if verdict is True:
            ok += 1
        else:
            bad += 1
            bad_ids.append(ep.name)
    print(f"raw logs: {root}")
    print(f"  [{'PASS' if bad == 0 and missing == 0 else 'FAIL'}] episodes that lifted: {ok}/{len(eps)}")
    if bad:
        print(f"  - {bad} did not lift: {bad_ids[:10]}")
    if missing:
        print(f"  - {missing} have no events.jsonl")
    return 0 if (bad == 0 and missing == 0) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr", type=Path, default=None)
    ap.add_argument("--logs-root", type=Path, default=None)
    ap.add_argument("--action-tol", type=float, default=2e-3,
                    help="max allowed |pos[t] + action[t] - pos[t+1]| in metres")
    ap.add_argument("--max-step", type=float, default=0.09,
                    help="largest believable single-step motion [m]; the scripted expert peaks at 0.078")
    args = ap.parse_args()
    if not args.zarr and not args.logs_root:
        ap.error("give --zarr or --logs-root")
    rc = 0
    if args.logs_root:
        rc |= check_logs(args.logs_root)
    if args.zarr:
        rc |= check_zarr(args.zarr, args.action_tol, args.max_step)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
