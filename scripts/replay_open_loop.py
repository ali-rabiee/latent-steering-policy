"""M6 gate: replay a recorded demo's GT actions open-loop through the twist
controller. Isolates executor bugs (frames, dt, sign conventions) from policy
bugs: if the robot re-reaches the demo's box and the lift succeeds, the
actuation path is sound and any later closed-loop failure is the policy's.

Run under Isaac Lab python, e.g.:
    conda run -n kinova python scripts/replay_open_loop.py \
        --episode-dir ../kinova-isaac/logs/data_collection/session_X/episode_0000 \
        --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _isaac_bootstrap import add_lsteer_to_path

add_lsteer_to_path()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--kinova-root", type=Path, default=None)
    parser.add_argument("--num_objects", type=int, default=None, help="default: as in the demo")
    parser.add_argument("--box-size", dest="box_size", type=float, default=0.05)
    parser.add_argument("--pos-tol", type=float, default=0.05, help="final EE position tolerance [m]")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from _isaac_bootstrap import bootstrap_kinova

    bootstrap_kinova(args.kinova_root)

    import numpy as np
    import torch

    from lsteer.data.convert import _to_floats, parse_episode
    from lsteer.isaac import runtime

    # ---- load demo ------------------------------------------------------
    rec = parse_episode(args.episode_dir, require_success=False)
    if rec is None:
        print(f"ERROR: cannot parse episode {args.episode_dir}")
        return 2
    ticks = [
        _to_floats(json.loads(l))
        for l in (args.episode_dir / "ticks.jsonl").read_text().splitlines()
        if l.strip()
    ]
    demo_objects = ticks[0].get("objects", [])
    if args.num_objects is None:
        args.num_objects = max(1, len(demo_objects))
    final_ee_pos_b = np.array(ticks[-1]["robot"]["ee_pose_b"]["position_m"], dtype=np.float32)

    # ---- build sim ------------------------------------------------------
    h = runtime.build_sim(args)
    controller = runtime.make_twist_controller(h)
    provider = runtime.ChunkActionProvider(device=str(h.sim.device))
    controller.set_input_provider(provider)

    # settle physics + renderer before teleporting
    for _ in range(10):
        runtime.step_sim(h, controller, render=True)

    # teleport boxes to the demo layout (world poses from tick 0)
    for obj in demo_objects:
        leaf = obj["id"]
        pos_w = obj["pose_w"]["position_m"]
        for p in h.spawned_paths:
            if p.endswith(leaf):
                ok = runtime.teleport_box(h, p, tuple(pos_w))
                print(f"teleport {leaf} -> {pos_w} ({'ok' if ok else 'FAILED'})")
    for _ in range(20):
        runtime.step_sim(h, controller, render=True)

    boxes0 = runtime.box_snapshot(h)
    n_phys = max(1, round((1.0 / 5.0) / h.dt))  # physics steps per 5 Hz action step
    print(f"replaying {len(rec.actions)} actions, {n_phys} physics steps each")

    # ---- replay ---------------------------------------------------------
    for i, a in enumerate(rec.actions):
        provider.set_step(a[0:3], a[3:6], float(a[6]), n_phys)
        for _ in range(n_phys):
            runtime.step_sim(h, controller, render=True)
        if i % 10 == 0:
            pos_b, _ = runtime.get_ee_pose_b(h, controller)
            print(f"  step {i:3d}  ee_b={np.round(pos_b, 3).tolist()}")

    # let the grasp/lift settle
    for _ in range(2 * n_phys):
        runtime.step_sim(h, controller, render=True)

    # ---- verdict --------------------------------------------------------
    pos_b, _ = runtime.get_ee_pose_b(h, controller)
    ee_err = float(np.linalg.norm(pos_b - final_ee_pos_b))
    boxes1 = runtime.box_snapshot(h)
    lifted = {
        k: float(boxes1[k][2] - boxes0[k][2]) for k in boxes1 if k in boxes0
    }
    max_lift = max(lifted.values()) if lifted else 0.0
    target_leaf = rec.path and json.loads((args.episode_dir / "instruction.json").read_text()).get(
        "language_command_meta", {}
    ).get("target_leaf")

    print("\n=== replay verdict ===")
    print(f"final EE error vs demo: {ee_err:.4f} m (tol {args.pos_tol})")
    print(f"box lift heights: { {k: round(v, 3) for k, v in lifted.items()} } (target: {target_leaf})")
    ok = ee_err <= args.pos_tol and (not rec.success or max_lift > 0.04)
    print("PASS" if ok else "FAIL")

    simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
