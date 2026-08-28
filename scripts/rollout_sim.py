"""M7: closed-loop evaluation of the frozen diffusion policy in Isaac Sim.

Receding-horizon loop at 5 Hz: capture every trained camera (front + wrist) +
EE state (identical preprocessing to training via lsteer.data.obs), denoise an
action chunk from a fresh z, execute act_horizon steps through the twist
controller, repeat. The camera set comes from the checkpoint, so a policy can
never be rolled out against a different set of views than it was trained on.

Every replan's (state, z, predicted chunk) plus the episode outcome is dumped
to outputs/rollouts/<run>/episode_XXXX.npz — the Phase-2 steering-encoder and
Phase-3 gate data seam.

Multimodality eval: boxes stay on a FIXED layout across episodes (teleported
back each reset); only z varies. Per-box coverage of the reached box measures
steerability of the frozen policy.

Run under Isaac Lab python (use -u: Kit swallows buffered stdout on crashes):
    conda run -n kinova python -u scripts/rollout_sim.py \
        --ckpt outputs/train/<run>/latest.ckpt --episodes 50 --headless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

from _isaac_bootstrap import add_lsteer_to_path

add_lsteer_to_path()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--kinova-root", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_objects", type=int, default=4)
    parser.add_argument("--box-size", dest="box_size", type=float, default=0.05)
    parser.add_argument("--max-duration-s", type=float, default=20.0)
    parser.add_argument("--lift-thresh", type=float, default=0.06)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--mode-lock",
        type=int,
        default=0,
        metavar="K",
        help="E1: commit to the first replan's predicted endpoint; on every later "
        "replan sample K candidate chunks and execute the one whose endpoint is "
        "closest (xy) to the commitment. 0 = off (one fresh sample per replan).",
    )
    parser.add_argument(
        "--close-on-arrival",
        type=float,
        default=0.0,
        metavar="RADIUS_M",
        help="E3a: veto GRIPPER_CLOSE until the EE is within RADIUS of the commanded "
        "box. The demos close the gripper on arrival (distance ~0 in all 420), but the "
        "trained gripper channel is bimodal noise that flips closed ~1.6 s in, long "
        "before the arm gets there. 0 = off. Needs a commanded box (goal ckpt or "
        "--steer-to-box).",
    )
    parser.add_argument(
        "--close-max-height",
        type=float,
        default=0.0,
        metavar="Z_M",
        help="E3b: also require the EE to be at or below this height (base frame, "
        "table ~0, box top ~0.05) before the gripper may close. Without it the arm "
        "closes ~25 cm above the box while horizontally on target, and because the "
        "gripper state is part of the observation that early close derails the rest "
        "of the episode: 38/40 rollouts only descend AFTER closing, by which point "
        "they have drifted ~16 cm away. 0 = off.",
    )
    parser.add_argument(
        "--exec-diffik",
        action="store_true",
        help="STAGE 0c: execute through DifferentialIKController with ABSOLUTE pose targets, the "
        "way collect_boxes.py does, instead of feeding delta twists to the jog controller. The "
        "policy still outputs deltas; they are accumulated into a target pose. Collection lifts "
        "12/12 on this path while the twist path lifts 0/40 with the box never moving, so every "
        "grasp number measured through the twist path is suspect. Orientation is held at the "
        "episode-start reference with yaw free (top-down task).",
    )
    parser.add_argument(
        "--hold-orientation",
        action="store_true",
        help="HARNESS FIX: hold the wrist at its episode-start orientation, correcting roll and "
        "pitch only and leaving yaw free (this is a top-down task: the gripper yaws to fit its "
        "fingers to the box and must not tip in any other direction). Collection runs the "
        "controller in 'translate' mode, which calls hold_orientation() against a captured "
        "reference; rollout runs 'twist', which does quat_box_plus(current, drot) -- an integrator "
        "with no reference, so IK tracking error random-walks the wrist away with nothing "
        "restoring it. Measured on the scripted expert commanding ZERO rotation: 0.0007 from the "
        "demo orientation at episode start, 0.145 by the gripper close, 1.84 by the end.",
    )
    parser.add_argument(
        "--expert",
        action="store_true",
        help="STAGE 0: drive the arm with the scripted expert instead of the policy, through the "
        "SAME twist controller, success test and summary. The success detector has never fired in "
        "~200 policy episodes; the demos were collected in kinova-isaac's env while rollouts run in "
        "lsteer.isaac.runtime, so this checks whether this harness can register a lift at all before "
        "any conclusion is drawn from a zero. Waypoints mirror the collection expert "
        "(align xy at home height -> descend -> lift); --ckpt is still required for the env config "
        "but the network is never queried.",
    )
    parser.add_argument(
        "--retry-after",
        type=int,
        default=0,
        metavar="N",
        help="bail out and re-approach after N replans with no lift. 0 = off. Motivated by "
        "the champion's own control run: a success finishes in a median of 5 replans and "
        "stops with the EE at +0.022 m, while a failure burns all 25 and ends at -0.021, "
        "digging into the table. A failed grasp is a stuck state, not a near miss, so "
        "spending more replans on it cannot help -- retracting restores an observation the "
        "demonstrations actually cover.",
    )
    parser.add_argument(
        "--retry-height",
        type=float,
        default=0.20,
        help="EE height [m] to retract to on a bail-out, back inside the demonstrated travel band",
    )
    parser.add_argument("--max-retries", type=int, default=2, help="bail-outs allowed per episode")
    parser.add_argument("--retry-ticks", type=int, default=25, help="physics ticks spent retracting")
    parser.add_argument(
        "--clamp-height",
        type=float,
        default=0.0,
        metavar="Z_MAX",
        help="E5: refuse to command the EE above Z_MAX (base frame). The demos never "
        "exceed 0.256 m, yet the closed-loop policy spends 24-37%% of every rollout above "
        "that, reaching 0.881 m -- it drifts out of the training distribution during the "
        "reach, before any gripper action, and off-distribution its gripper and descent "
        "predictions are unreliable. Keeping it inside the demo envelope is the cheapest "
        "test of whether that drift is what breaks the grasp. 0 = off.",
    )
    parser.add_argument(
        "--latch-gripper",
        action="store_true",
        help="E3c: once the gripper has closed (under whatever gates are active), keep "
        "it closed for the rest of the episode. The predicted gripper channel is bimodal "
        "+/-1 noise, so it flips back open within a replan or two: in the 10 E3b episodes "
        "that closed at a correct grasp pose the arm then lifted 0.41-0.56 m -- far past "
        "the 0.06 m success threshold -- but had already dropped the box.",
    )
    parser.add_argument(
        "--steer-to-box",
        action="store_true",
        help="E1b: instead of committing to the policy's own first endpoint, "
        "command the target box round-robin over episodes and lock to ITS xy "
        "(requires --mode-lock K for the candidate count). Measures the upper "
        "bound of inference-time steering; summary gains a per_command block.",
    )
    parser.add_argument(
        "--exec-abs-target",
        action="store_true",
        help="P2a: accumulate the policy's deltas into a target anchored in SPACE "
        "(tgt <- tgt + a[0:3] from the episode-start pose) instead of re-anchoring to the "
        "arm every step (tgt = cur_pos + a[0:3]). P0 measured why this matters: replaying "
        "a demonstration's own actions through the re-anchored path drifts a median "
        "178.6 mm off the demonstrated trajectory and lifts 0/2, because the arm "
        "over-executes each command by ~1.3-1.6x during fast travel and the next target "
        "is measured from wherever it actually ended up, so the error is absorbed instead "
        "of corrected. Driving the demo's absolute pose instead lifts 2/2 at 12-16 mm -- "
        "and for a demonstration's own actions an accumulator IS that absolute pose, since "
        "the actions are pose differences by construction (data/convert.py). Requires "
        "--exec-diffik.",
    )
    parser.add_argument(
        "--replay-demo",
        type=Path,
        default=None,
        metavar="ZARR",
        help="P0: replay a recorded demonstration's own actions through THIS harness's "
        "execution path instead of querying the network. The policy is executed as "
        "tgt_pos = cur_pos + a[0:3], but the --expert validation that produced '12/12 "
        "lifts' took the ABSOLUTE branch (tgt_pos = expert_target_abs), so the delta "
        "path has never been validated end-to-end. Spawns the episode's own layout from "
        "meta/episode_box_positions and feeds data/action through the identical code. "
        "The network is never queried; --ckpt is still needed for the env config.",
    )
    parser.add_argument(
        "--replay-episode",
        type=int,
        default=0,
        help="first demo episode index to replay; --episodes consecutive episodes with "
        "--num_objects boxes are taken from here",
    )
    parser.add_argument(
        "--replay-mode",
        choices=("delta", "abs"),
        default="delta",
        help="'delta' drives tgt_pos = cur_pos + a[0:3], the path the POLICY takes. 'abs' "
        "drives the demo's recorded absolute pose, the path --expert takes and the one "
        "collect_boxes.py used to record the data. Running both on the same episodes "
        "isolates execution error from everything else: identical inputs, one line "
        "different in the executor.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="dump per-camera PNGs per policy step into <out>/episode_XXXX/images/<cam>/, "
        "the same layout kinova-isaac's make_episode_video.py reads",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    code = 1
    try:
        code = _run(args)
    except BaseException:
        # os._exit below bypasses normal exception handling, so a crash would
        # otherwise exit(1) with an EMPTY log and no traceback at all.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        code = 1
    finally:
        # Kit sometimes wedges in close() headless; give it 60 s then force-exit
        # with the outcome code (all results were already printed/written).
        import os
        import threading

        t = threading.Thread(target=simulation_app.close, daemon=True)
        t.start()
        t.join(timeout=60)
        os._exit(code)


def _run(args) -> int:
    from _isaac_bootstrap import bootstrap_kinova

    bootstrap_kinova(args.kinova_root)

    import numpy as np
    import torch

    from lsteer.data import schema
    from lsteer.data.obs import build_lowdim_obs, center_crop, image_to_tensor, resize_to_storage
    from lsteer.isaac import runtime
    from lsteer.policy import DiffusionPolicy
    from lsteer.utils.rotations import delta_rotvec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = DiffusionPolicy.load(args.ckpt, device=device)
    cfg = policy.cfg
    print(
        f"loaded policy: T_o={cfg.obs_horizon} T_p={cfg.pred_horizon} "
        f"T_a={cfg.act_horizon} cameras={list(cfg.camera_names)}"
    )

    out_dir = args.out_dir or Path("outputs/rollouts") / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    h = runtime.build_sim(args, camera_names=cfg.camera_names)
    controller = runtime.make_twist_controller(h)
    provider = runtime.ChunkActionProvider(device=str(h.sim.device))
    controller.set_input_provider(provider)
    diffik = runtime.make_diffik_driver(h, controller) if args.exec_diffik else None
    if args.exec_abs_target and diffik is None:
        raise SystemExit("--exec-abs-target requires --exec-diffik")
    print(f"executor: {'abs-target accumulator' if args.exec_abs_target else 'cur_pos + delta'}")

    n_phys = max(1, round((1.0 / schema.FPS) / h.dt))
    steps_per_episode = int(args.max_duration_s * schema.FPS)
    # render at ~60 Hz (vla_v1's throttle) — rendering every physics step tanks FPS
    render_stride = max(1, round((1.0 / 60.0) / h.dt))

    def run_phys_steps(n: int) -> None:
        """n physics steps, rendering at the throttled rate + always on the last
        step so a fresh camera frame is available for the next observe()."""
        for j in range(n):
            render = ((j + 1) == n) or ((j + 1) % render_stride == 0)
            runtime.step_sim(h, controller, render=render)

    run_phys_steps(20)

    # fixed layout for the whole eval: record spawn poses once
    layout_w = runtime.box_snapshot(h)
    print(f"fixed layout: { {k: np.round(v, 3).tolist() for k, v in layout_w.items()} }")

    # ---- P0: open-loop replay of recorded demonstrations -------------------
    # The point of this mode is that NOTHING below changes: the same observe(),
    # the same act_horizon chunking, the same `for a in actions:` executor. Only
    # the source of `actions` differs (a zarr instead of the network) and the
    # layout is the demo's instead of the seeded one. Any failure is therefore
    # attributable to execution, because the actions are known-good by
    # construction -- they are exactly what produced a successful lift when the
    # data was collected.
    replay = None
    if args.replay_demo is not None:
        import zarr

        zr = zarr.open(str(args.replay_demo), mode="r")
        ends = np.asarray(zr[schema.META_EPISODE_ENDS][:])
        starts = np.concatenate([[0], ends[:-1]])
        box_pos_b_all = np.asarray(zr[schema.META_BOX_POSITIONS][:])   # (E, MAX_BOXES, 3), base frame
        box_col_all = np.asarray(zr[schema.META_BOX_COLORS][:])        # (E, MAX_BOXES), -1 padded
        tgt_col_all = np.asarray(zr[schema.META_TARGET_COLOR][:])
        n_boxes = (~np.isnan(box_pos_b_all[:, :, 0])).sum(axis=1)
        # only episodes whose box count matches what this sim spawned: teleporting
        # 3 demo boxes onto a 4-box table would leave a stray box on the layout
        # that the demonstration never saw.
        pool = [
            int(e)
            for e in range(args.replay_episode, len(ends))
            if int(n_boxes[e]) == int(args.num_objects)
        ][: args.episodes]
        if len(pool) < args.episodes:
            print(
                f"WARNING: only {len(pool)} episodes at/after {args.replay_episode} have "
                f"{args.num_objects} boxes; replaying those"
            )
        # colour id -> spawned leaf, via the label the loader assigned each prim
        color_to_leaf = {}
        for leaf, lab in h.id_to_label.items():
            if lab:
                color_to_leaf[schema.COLOR_PALETTE.index(lab.split()[0])] = leaf
        replay = {
            "episodes": pool,
            "state": zr[schema.DATA_STATE],
            "action": zr[schema.DATA_ACTION],
            "starts": starts,
            "ends": ends,
            "box_pos_b": box_pos_b_all,
            "box_col": box_col_all,
            "target_col": tgt_col_all,
            "color_to_leaf": color_to_leaf,
        }
        args.episodes = len(pool)
        print(
            f"REPLAY mode={args.replay_mode} zarr={args.replay_demo} "
            f"episodes={pool} color_to_leaf={color_to_leaf}"
        )
        if diffik is None:
            raise SystemExit(
                "--replay-demo needs --exec-diffik: the delta path under test is the "
                "diff-IK one the champion rollouts use"
            )

    def observe(gripper_state: float, save_to: Path | None = None, tick: int = 0):
        """One observation: {cam: cropped tensor} + the 10-D low-dim state.

        With save_to set, the RAW camera frames are also written out (the
        network sees a 224 center crop of a 256 resize of these).
        """
        frames = runtime.capture_rgb(h)
        if save_to is not None:
            from PIL import Image

            for cam, arr in frames.items():
                d = save_to / "images" / cam
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr).save(str(d / f"image_{tick:06d}.png"))
        imgs = {
            cam: center_crop(image_to_tensor(resize_to_storage(frames[cam])))
            for cam in cfg.camera_names
        }
        pos_b, quat_b = runtime.get_ee_pose_b(h, controller)
        state = build_lowdim_obs(pos_b, quat_b, gripper_state)
        return imgs, torch.from_numpy(state), pos_b

    # ---- STAGE 0: scripted expert -----------------------------------------
    # Mirrors the collection expert (ScriptedPlanner + WaypointFollowerInput):
    # three position waypoints executed closed-loop off the CURRENT pose, with
    # zero rotation. Rotation really is unnecessary here -- across all 420 demos
    # the home orientation and the grasp orientation are the same to 1e-4, so
    # the collection expert never rotates during a box reach either.
    # Heights come from the demos rather than from the world->base transform:
    # they close at EE z = 0.047 (1 mm spread) and lift 0.207 m from there.
    # collect_boxes.py geometry (the path that lifts 12/12), computed from the
    # ACTUAL box rather than a constant:
    #   grasp_z = box_top + ee_z_offset(0.08) + grasp_depth(-0.07) = box_top + 0.01
    #   lift    = box_top + ee_z_offset + travel_height(0.14)      = box_top + 0.22
    # A hardcoded 0.047 closed ~1.8 cm high, on the box's top edge, and the box
    # stayed behind while the arm lifted 21 cm.
    EXPERT_EE_Z_OFFSET = 0.08
    EXPERT_GRASP_DEPTH = -0.07
    EXPERT_TRAVEL_H = 0.14
    EXPERT_STEP_M = 0.03      # demos travel ~0.030 m per 5 Hz tick
    EXPERT_TOL_ALIGN_M = 0.012
    # The descend tolerance must be tight: it is applied to the 3-D error, so a
    # loose value lets the descent stop that far ABOVE the grasp height, and the
    # demos close within 1 mm of 0.047.
    EXPERT_TOL_DESCEND_M = 0.0015
    # Demos wait 4-5 ticks between closing and lifting (156 wait 4, 264 wait 5);
    # the fingers need that long to reach the 1.2 rad closed target.
    EXPERT_CLOSE_HOLD = 8   # collection settles 0.3 s then closes over 0.6 s

    def orientation_correction(quat_now_np, quat_ref_np):
        """Rotvec that pulls roll/pitch back to the reference, leaving yaw alone.

        Top-down task: the gripper may yaw to fit its fingers to the box, but any
        roll or pitch is drift and gets corrected. The error is expressed in the
        BASE frame, so its z component is the yaw part and is simply dropped.
        """
        q_now = torch.from_numpy(np.asarray(quat_now_np, dtype=np.float32)).unsqueeze(0)
        q_ref = torch.from_numpy(np.asarray(quat_ref_np, dtype=np.float32)).unsqueeze(0)
        err = delta_rotvec(q_now, q_ref)[0].numpy()  # base-frame rotvec now -> ref
        err[2] = 0.0  # keep yaw: fingers must be free to align with the box
        return err

    def expert_action(pos_b, box_xy, phase, hold):
        """Return (target_pos_abs, gripper, phase, hold) for one policy tick.

        Absolute, not a delta. collect_boxes drives every phase to an absolute
        goal (_run_segment) and holds at that same absolute goal (_hold_at); a
        delta round-trip loses a few mm per tick, which showed up as closing
        7 mm off in xy and 8.5 mm high while collection lands sub-millimetre.
        """
        home_z = float(home_pose_z)
        grasp_z = box_top_z + EXPERT_EE_Z_OFFSET + EXPERT_GRASP_DEPTH
        lift_z = box_top_z + EXPERT_EE_Z_OFFSET + EXPERT_TRAVEL_H
        targets = {
            0: np.array([box_xy[0], box_xy[1], home_z], dtype=np.float64),   # align xy high
            1: np.array([box_xy[0], box_xy[1], grasp_z], dtype=np.float64),  # descend
            2: np.array([box_xy[0], box_xy[1], lift_z], dtype=np.float64),   # lift
        }
        if phase == "grip":  # hold the grasp pose while the fingers close
            hold += 1
            return targets[1], schema.GRIPPER_CLOSE, ("lift" if hold >= EXPERT_CLOSE_HOLD else "grip"), hold
        if phase == "done":  # hold the lifted pose so the success test can settle
            return targets[2], schema.GRIPPER_CLOSE, "done", hold
        idx = {"align": 0, "descend": 1, "lift": 2}[phase]
        tgt = targets[idx]
        err = tgt - np.asarray(pos_b, dtype=np.float64)
        tol = EXPERT_TOL_ALIGN_M if phase == "align" else EXPERT_TOL_DESCEND_M
        if float(np.linalg.norm(err)) < tol:
            if phase == "align":
                return targets[1], schema.GRIPPER_OPEN, "descend", hold
            if phase == "descend":
                return tgt, schema.GRIPPER_CLOSE, "grip", hold
            return tgt, schema.GRIPPER_CLOSE, "done", hold
        # rate-limit the approach, but always toward the ABSOLUTE waypoint
        n = float(np.linalg.norm(err))
        capped = np.asarray(pos_b, dtype=np.float64) + err * min(1.0, EXPERT_STEP_M / max(n, 1e-9))
        grip = schema.GRIPPER_OPEN if phase in ("align", "descend") else schema.GRIPPER_CLOSE
        return capped, grip, phase, hold

    home_pose_z = 0.248  # replaced with the measured home pose below
    box_top_z = 0.036   # replaced per episode from the actual box

    results = []
    for ep in range(args.episodes):
        # reset robot + teleport boxes back to the fixed layout.
        # set_home_pose BEFORE controller.reset: every demo starts at the home
        # pose, and reset_sim_and_robot alone does not land there. Skipping it
        # starts each episode out of distribution (see runtime.set_home_pose).
        runtime.reset_sim_and_robot(h)
        runtime.set_home_pose(h)
        controller.reset(h.robot)
        controller.set_mode("twist")
        provider.set_step(np.zeros(3), np.zeros(3), schema.GRIPPER_OPEN, 1)
        # in replay mode the table is the DEMO's table, rebuilt from the zarr
        layout_ep = layout_w
        demo_ep = None
        if replay is not None:
            demo_ep = replay["episodes"][ep]
            layout_ep = {}
            for j in range(int(args.num_objects)):
                c = int(replay["box_col"][demo_ep, j])
                leaf = replay["color_to_leaf"].get(c)
                if leaf is None:
                    raise SystemExit(f"demo colour id {c} has no spawned box")
                layout_ep[leaf] = runtime.base_to_world_pos(
                    h, replay["box_pos_b"][demo_ep, j]
                ).astype(np.float32)
        for leaf, pos_w in layout_ep.items():
            for p in h.spawned_paths:
                if p.endswith(leaf):
                    runtime.teleport_box(h, p, tuple(float(v) for v in pos_w))
        run_phys_steps(20)

        g_ep = torch.Generator(device=device).manual_seed(args.seed * 100_000 + ep)
        # ONE z for the whole episode. z selects the behavioural mode (which box,
        # and implicitly which phase of the reach->descend->close->lift script),
        # so redrawing it every replan lets the policy jump modes mid-episode --
        # observed closing the gripper at replan 1, 24 cm above the table, then
        # executing the lift. PLAN.md's M7 protocol is "only z varies" ACROSS
        # episodes, with the layout fixed: that is one sample per episode.
        z_ep = torch.randn(
            (1, cfg.pred_horizon, cfg.action_dim), generator=g_ep, device=device
        )
        gripper = schema.GRIPPER_OPEN
        frame_dir = (out_dir / f"episode_{ep:04d}") if args.save_frames else None
        imgs, state, _ = observe(gripper, save_to=frame_dir, tick=0)
        obs_hist = deque([(imgs, state)] * cfg.obs_horizon, maxlen=cfg.obs_horizon)

        replans = []
        reached_leaf = None
        success = False
        boxes0 = runtime.box_snapshot(h)
        tgt_pos, quat_ref = runtime.get_ee_pose_b(h, controller)  # top-down reference
        tgt_pos = np.asarray(tgt_pos, dtype=np.float64).copy()  # absolute target for --exec-diffik
        origin_b = np.asarray(h.scene_origins[0], dtype=np.float32)
        boxes_xy_b = {k: (v - origin_b)[0:2] for k, v in boxes0.items()}
        # closest approach per box over the whole episode. `reached_leaf` is
        # recorded at the first gripper close, which fires ~1.6 s in regardless
        # of how far the target is -- that makes it a measure of gripper timing,
        # not of which box the arm actually travelled to. This one is timing-free.
        closest = {k: float("inf") for k in boxes_xy_b}
        max_lift_seen = 0.0
        max_finger_rad = 0.0  # how far the fingers actually close (target 1.2)
        max_box_move = 0.0    # 3-D box displacement: 0 means the fingers never touched it
        latched = False
        commit_xy = None  # E1 mode-lock: xy endpoint committed at the first replan
        commanded_leaf = None
        goal_vec = None
        if args.expert:
            leaves = sorted(layout_w.keys())
            commanded_leaf = leaves[ep % len(leaves)]
            origin0 = np.asarray(h.scene_origins[0], dtype=np.float32)
            box_xy_b = (layout_w[commanded_leaf] - origin0)[0:2].astype(np.float32)
            # box top in the BASE frame, via the robot root pose (scene_origins
            # carries the env origin, whose z is not the table height)
            box_b = runtime.world_to_base_pos(h, layout_w[commanded_leaf])
            box_top_z = float(box_b[2]) + 0.5 * float(args.box_size)
            expert_phase, expert_hold = "align", 0
            home_pose_z = float(runtime.get_ee_pose_b(h, controller)[0][2])
        replay_actions = replay_pos_abs = None
        if replay is not None:
            s0, e0 = int(replay["starts"][demo_ep]), int(replay["ends"][demo_ep])
            replay_actions = np.asarray(replay["action"][s0:e0], dtype=np.float32)
            demo_states = np.asarray(replay["state"][s0:e0], dtype=np.float32)
            # pose AFTER action t. states[t] is the pose BEFORE it, and the final
            # pose is not stored, so the last one is reconstructed from its action
            # (a_t = pose_{t+1} - pose_t by construction, see data/convert.py).
            replay_pos_abs = np.concatenate(
                [demo_states[1:, 0:3], (demo_states[-1, 0:3] + replay_actions[-1, 0:3])[None]]
            ).astype(np.float64)
            commanded_leaf = replay["color_to_leaf"][int(replay["target_col"][demo_ep])]
            origin0 = np.asarray(h.scene_origins[0], dtype=np.float32)
            box_xy_b = (layout_ep[commanded_leaf] - origin0)[0:2].astype(np.float32)
            # Prove the layout actually landed where the demo had it, and that the
            # arm starts where the demo started. Either being wrong would make the
            # replay a different experiment than the one described.
            snap_b = {k: (v - origin0) for k, v in boxes0.items()}
            layout_err = max(
                float(np.linalg.norm(snap_b[replay["color_to_leaf"][int(replay["box_col"][demo_ep, j])]][0:2]
                                     - replay["box_pos_b"][demo_ep, j][0:2]))
                for j in range(int(args.num_objects))
            )
            home_err = float(np.linalg.norm(np.asarray(tgt_pos) - demo_states[0, 0:3]))
            print(
                f"[ep {ep:03d}] demo={demo_ep} n_act={len(replay_actions)} "
                f"target={commanded_leaf} layout_xy_err={layout_err*1000:.2f}mm "
                f"home_err={home_err*1000:.2f}mm"
            )
        if replay is None and (args.steer_to_box or getattr(cfg, "goal_dim", 0) > 0):
            leaves = sorted(layout_w.keys())
            commanded_leaf = leaves[ep % len(leaves)]
            origin0 = np.asarray(h.scene_origins[0], dtype=np.float32)
            box_xy_b = (layout_w[commanded_leaf] - origin0)[0:2].astype(np.float32)
        if args.steer_to_box:
            if args.mode_lock <= 0:
                raise SystemExit("--steer-to-box requires --mode-lock K")
            commit_xy = box_xy_b
        if replay is None and getattr(cfg, "goal_dim", 0) > 0:
            # E2 goal-conditioned ckpt: command the box explicitly, same
            # round-robin protocol as --steer-to-box
            label = h.id_to_label.get(commanded_leaf, "")
            color_idx = schema.COLOR_PALETTE.index(label.split()[0])
            gv = np.zeros(cfg.goal_dim, dtype=np.float32)
            gv[color_idx] = 1.0
            gv[schema.MAX_BOXES : schema.MAX_BOXES + 2] = box_xy_b
            goal_vec = torch.from_numpy(gv)

        step = 0
        retries = 0
        replans_since_retry = 0
        # per-step |achieved EE - demo EE| and the error at the gripper close
        replay_err: list[float] = []
        replay_close_err = replay_close_dist_xy = None
        while step < steps_per_episode:
            if replay is not None and step >= len(replay_actions):
                break
            # Bail out of a failing attempt instead of grinding to the cap.
            #
            # In the champion's own 40-episode control, a SUCCESS finishes in a
            # median of 5 replans and stops with the EE at +0.022 m; a FAILURE
            # burns all 25 replans and ends up at -0.021, i.e. digging into the
            # table. So a failed grasp is not a near miss that needs more time,
            # it is a stuck state that more time cannot fix -- the arm keeps
            # re-planning from a pose no demonstration ever shows. Lifting back
            # to travel height restores an in-distribution observation and lets
            # the policy start the reach over.
            if (
                args.retry_after > 0
                and diffik is not None
                and retries < args.max_retries
                and replans_since_retry >= args.retry_after
                and max_lift_seen < 0.01
            ):
                cur_pos, _ = runtime.get_ee_pose_b(h, controller)
                up = np.asarray(cur_pos, dtype=np.float64).copy()
                up[2] = float(args.retry_height)
                gripper = schema.GRIPPER_OPEN     # let go of whatever it is holding
                latched = False
                diffik.set_gripper(False)
                for _ in range(int(args.retry_ticks)):
                    diffik.step(up, quat_ref, render=False)
                tgt_pos = up  # a retract moves the arm: re-anchor the accumulator
                retries += 1
                replans_since_retry = 0
                if args.steer_to_box:
                    commit_xy = box_xy_b          # re-commit to the commanded box
                step += 1
                imgs, state, ee_now = observe(gripper, save_to=frame_dir, tick=step)
                obs_hist.append((imgs, state))
                continue
            replans_since_retry += 1
            obs = {
                schema.camera_obs_key(cam): torch.stack([f[cam] for f, _ in obs_hist])
                for cam in cfg.camera_names
            }
            obs["state"] = torch.stack([s for _, s in obs_hist])
            if goal_vec is not None:
                obs["goal"] = goal_vec
            cur_xy = obs["state"][-1, 0:2].numpy()
            if replay is not None:
                # same act_horizon chunking the policy gets, so the executor sees
                # an identically-shaped stream; the actions just come from disk
                actions = replay_actions[step : step + cfg.act_horizon]
                replans.append(
                    {
                        "step": step,
                        "state": obs["state"].numpy(),
                        "z": np.zeros((cfg.pred_horizon, cfg.action_dim), dtype=np.float32),
                        "action_pred": np.repeat(
                            actions[0:1], cfg.pred_horizon, axis=0
                        ).astype(np.float32),
                        "mode_lock_sel": np.int64(0),
                        "commit_xy": box_xy_b.astype(np.float32),
                    }
                )
            elif args.expert:
                # one expert tick per policy tick, same cadence as act_horizon
                pos_now, _ = runtime.get_ee_pose_b(h, controller)
                tgt_abs, g_exp, expert_phase, expert_hold = expert_action(
                    pos_now, box_xy_b, expert_phase, expert_hold
                )
                expert_target_abs = np.asarray(tgt_abs, dtype=np.float64)
                actions = np.zeros((1, 7), dtype=np.float32)
                actions[0, 0:3] = expert_target_abs - np.asarray(pos_now, dtype=np.float64)
                actions[0, 6] = g_exp
                replans.append(
                    {
                        "step": step,
                        "state": obs["state"].numpy(),
                        "z": np.zeros((cfg.pred_horizon, cfg.action_dim), dtype=np.float32),
                        "action_pred": np.repeat(actions, cfg.pred_horizon, axis=0),
                        "mode_lock_sel": np.int64(0),
                        "commit_xy": box_xy_b.astype(np.float32),
                    }
                )
            elif args.mode_lock > 0 and commit_xy is not None:
                # sample K fresh candidates and stay loyal to the commitment
                out = policy.predict_action(obs, k=args.mode_lock, generator=g_ep)
                preds = out["action_pred"].cpu().numpy()  # (K, T_p, 7)
                ends = cur_xy[None] + preds[:, :, 0:2].sum(axis=1)
                sel = int(np.argmin(np.linalg.norm(ends - commit_xy[None], axis=1)))
            else:
                out = policy.predict_action(obs, z=z_ep, k=1)  # z held fixed for the episode
                sel = 0
                if args.mode_lock > 0:
                    pred0 = out["action_pred"][0].cpu().numpy()
                    commit_xy = cur_xy + pred0[:, 0:2].sum(axis=0)
            if not args.expert and replay is None:
                actions = out["action"][sel].cpu().numpy()  # (T_a, 7)
                replans.append(
                    {
                        "step": step,
                        "state": obs["state"].numpy(),
                        "z": out["z"][sel].cpu().numpy(),
                        "action_pred": out["action_pred"][sel].cpu().numpy(),
                        "mode_lock_sel": np.int64(sel),
                        "commit_xy": (np.zeros(2, dtype=np.float32) if commit_xy is None else commit_xy.astype(np.float32)),
                    }
                )

            for a in actions:
                if args.hold_orientation:
                    _, quat_now = runtime.get_ee_pose_b(h, controller)
                    a = a.copy()
                    a[3:6] = a[3:6] + orientation_correction(quat_now, quat_ref)
                if args.clamp_height > 0.0:
                    pos_now, _ = runtime.get_ee_pose_b(h, controller)
                    if pos_now[2] + a[2] > args.clamp_height:
                        a = a.copy()
                        a[2] = max(0.0, float(args.clamp_height - pos_now[2]))
                g_cmd = schema.GRIPPER_OPEN if a[6] > 0.0 else schema.GRIPPER_CLOSE
                if g_cmd == schema.GRIPPER_CLOSE and (
                    (args.close_on_arrival > 0.0 and commanded_leaf is not None)
                    or args.close_max_height > 0.0
                ):
                    pos_now, _ = runtime.get_ee_pose_b(h, controller)
                    too_far = (
                        args.close_on_arrival > 0.0
                        and commanded_leaf is not None
                        and float(np.linalg.norm(pos_now[0:2] - box_xy_b)) > args.close_on_arrival
                    )
                    too_high = args.close_max_height > 0.0 and float(pos_now[2]) > args.close_max_height
                    if too_far or too_high:
                        g_cmd = schema.GRIPPER_OPEN  # not on the box yet
                if args.latch_gripper:
                    if g_cmd == schema.GRIPPER_CLOSE:
                        latched = True
                    elif latched:
                        g_cmd = schema.GRIPPER_CLOSE  # hold what we grabbed
                if g_cmd != gripper and g_cmd == schema.GRIPPER_CLOSE and reached_leaf is None:
                    # record which box we are committing to at first close
                    pos_b, _ = runtime.get_ee_pose_b(h, controller)
                    snap = runtime.box_snapshot(h)
                    origin0 = np.asarray(h.scene_origins[0], dtype=np.float32)
                    d = {
                        k: float(np.linalg.norm((v - origin0)[0:2] - pos_b[0:2]))
                        for k, v in snap.items()
                    }
                    reached_leaf = min(d, key=d.get)
                gripper = g_cmd
                if diffik is not None:
                    # Target = CURRENT pose + this step's delta, recomputed every
                    # tick. A free-running accumulator diverges from the arm: when
                    # tracking lags, the target keeps advancing and the arm
                    # overshoots, which showed up as the EE oscillating +/-0.10 m
                    # around the box (even below the table) instead of settling.
                    # Position stays relative because the policy emits deltas;
                    # only ORIENTATION is held absolutely, which is what the twist
                    # path got wrong.
                    if args.expert:
                        tgt_pos = expert_target_abs  # absolute, as collect_boxes does
                    elif args.exec_abs_target:
                        # anchored in space: the target does NOT follow the arm, so
                        # an over- or under-executed step is corrected on the next
                        # one instead of becoming the new origin
                        tgt_pos = tgt_pos + np.asarray(a[0:3], dtype=np.float64)
                    elif replay is not None and args.replay_mode == "abs":
                        # the control arm: the demo's own recorded pose, absolute.
                        # Identical inputs to the delta arm below -- this is the
                        # ONE line that differs between them.
                        tgt_pos = replay_pos_abs[step]
                    else:
                        cur_pos, _ = runtime.get_ee_pose_b(h, controller)
                        tgt_pos = np.asarray(cur_pos, dtype=np.float64) + np.asarray(a[0:3], dtype=np.float64)
                    diffik.set_gripper(gripper == schema.GRIPPER_CLOSE)
                    for j in range(n_phys):
                        render = ((j + 1) == n_phys) or ((j + 1) % render_stride == 0)
                        diffik.step(tgt_pos, quat_ref, render=render)
                else:
                    provider.set_step(a[0:3], a[3:6], gripper, n_phys)
                    run_phys_steps(n_phys)
                step += 1

                imgs, state, ee_now = observe(gripper, save_to=frame_dir, tick=step)
                obs_hist.append((imgs, state))
                if replay is not None:
                    # `step` was incremented above, so the action just executed
                    # was index step-1 and should have landed on replay_pos_abs[step-1]
                    e = float(np.linalg.norm(np.asarray(ee_now, dtype=np.float64)
                                             - replay_pos_abs[step - 1]))
                    replay_err.append(e)
                    if replay_close_err is None and gripper == schema.GRIPPER_CLOSE:
                        replay_close_err = e
                        replay_close_dist_xy = float(
                            np.linalg.norm(np.asarray(ee_now[0:2]) - box_xy_b)
                        )
                for leaf, bxy in boxes_xy_b.items():
                    closest[leaf] = min(closest[leaf], float(np.linalg.norm(ee_now[0:2] - bxy)))

                snap = runtime.box_snapshot(h)
                lifts = {k: float(snap[k][2] - boxes0[k][2]) for k in snap if k in boxes0}
                gj = getattr(controller.gripper, "_base_joint_ids", None)
                if gj:
                    max_finger_rad = max(
                        max_finger_rad, float(h.robot.data.joint_pos[0, gj].max())
                    )
                if lifts:
                    moved = max(
                        float(np.linalg.norm(np.asarray(snap[k]) - np.asarray(boxes0[k])))
                        for k in snap if k in boxes0
                    )
                    max_box_move = max(max_box_move, moved)
                    # how far the box ACTUALLY moved, whether or not it cleared
                    # the threshold: 0 means the fingers never gripped it, a
                    # small positive value means it was gripped and slipped.
                    max_lift_seen = max(max_lift_seen, max(lifts.values()))
                if lifts and max(lifts.values()) > args.lift_thresh:
                    success = True
                    if reached_leaf is None:
                        reached_leaf = max(lifts, key=lifts.get)
                    # a replay runs the demo to its end even after a lift is
                    # detected, so the per-step error curve covers the whole
                    # trajectory rather than stopping at the first success
                    if replay is None:
                        break
            if success and replay is None:
                break

        label = h.id_to_label.get(reached_leaf or "", "")
        approached = min(closest, key=closest.get) if closest else None
        results.append({"episode": ep, "success": success, "reached": reached_leaf, "label": label,
                        "commanded": commanded_leaf, "approached": approached,
                        "max_lift_m": round(float(max_lift_seen), 4),
                        "max_finger_rad": round(float(max_finger_rad), 4),
                        "max_box_move_m": round(float(max_box_move), 5),
                        "closest": {k: round(v, 4) for k, v in closest.items()}})
        if replay is not None:
            results[-1].update(
                {
                    "demo_episode": demo_ep,
                    "replay_err_median_m": float(np.median(replay_err)) if replay_err else float("nan"),
                    "replay_err_max_m": float(np.max(replay_err)) if replay_err else float("nan"),
                    "replay_err_final_m": float(replay_err[-1]) if replay_err else float("nan"),
                    "replay_close_err_m": replay_close_err,
                    "replay_close_dist_xy_m": replay_close_dist_xy,
                }
            )
        cmd_str = f" commanded={commanded_leaf}" if commanded_leaf else ""
        rp_str = ""
        if replay is not None:
            r = results[-1]
            rp_str = (
                f" err_med={r['replay_err_median_m']*1000:.1f}mm"
                f" err_max={r['replay_err_max_m']*1000:.1f}mm"
                f" err_at_close={'n/a' if replay_close_err is None else f'{replay_close_err*1000:.1f}mm'}"
                f" xy_to_box_at_close={'n/a' if replay_close_dist_xy is None else f'{replay_close_dist_xy*1000:.1f}mm'}"
            )
        print(f"[ep {ep:03d}] success={success} reached={reached_leaf} ({label}){cmd_str} lift={max_lift_seen:.3f} boxmove={max_box_move:.4f} fingers={max_finger_rad:.2f} steps={step} retries={retries}{rp_str}")

        np.savez_compressed(
            out_dir / f"episode_{ep:04d}.npz",
            success=success,
            reached=str(reached_leaf),
            approached=str(approached),
            max_lift_m=np.float32(max_lift_seen),
            closest_ids=np.array(sorted(closest.keys())),
            closest_dist=np.array([closest[k] for k in sorted(closest.keys())], dtype=np.float32),
            label=label,
            layout_ids=np.array(sorted(layout_w.keys())),
            layout_pos_w=np.stack([layout_w[k] for k in sorted(layout_w.keys())]),
            # world->base offset, so offline analysis can assign endpoints to boxes
            # exactly (E0 had to estimate this to +/-0.15m from behavior)
            scene_origin=np.asarray(h.scene_origins[0], dtype=np.float32),
            replay_err=np.asarray(replay_err, dtype=np.float32),
            replay_pos_abs=(
                np.zeros((0, 3), dtype=np.float32)
                if replay_pos_abs is None
                else replay_pos_abs.astype(np.float32)
            ),
            **{
                f"replan{i:03d}_{key}": val
                for i, r in enumerate(replans)
                for key, val in r.items()
            },
        )

    # ---- summary ---------------------------------------------------------
    n_success = sum(r["success"] for r in results)
    reached_counts: dict[str, int] = {}
    for r in results:
        if r["reached"]:
            reached_counts[r["label"] or r["reached"]] = reached_counts.get(r["label"] or r["reached"], 0) + 1
    approach_counts: dict[str, int] = {}
    for r in results:
        if r["approached"]:
            lab = h.id_to_label.get(r["approached"], "") or r["approached"]
            approach_counts[lab] = approach_counts.get(lab, 0) + 1
    summary = {
        "episodes": len(results),
        "success_rate": n_success / max(1, len(results)),
        "per_box_coverage": {k: v / max(1, len(results)) for k, v in reached_counts.items()},
        # Timing-free companion to per_box_coverage: which box the arm actually
        # travelled to, by closest approach over the episode. per_box_coverage is
        # recorded at the first gripper close (~1.6 s in), so on a far target it
        # names whichever box was nearest at that instant, not the destination.
        "per_box_closest_approach": {
            k: v / max(1, len(results)) for k, v in approach_counts.items()
        },
        # box displacement regardless of the threshold: separates "never gripped"
        # (0.000) from "gripped then slipped" (small positive)
        "max_lift_m_median": float(np.median([r["max_lift_m"] for r in results])),
        "max_lift_m_best": float(max(r["max_lift_m"] for r in results)),
        # finger closure actually achieved; the closed target is 1.2 rad
        "max_finger_rad_median": float(np.median([r["max_finger_rad"] for r in results])),
        "max_box_move_m_median": float(np.median([r["max_box_move_m"] for r in results])),
        "median_closest_approach_m": float(
            np.median([min(r["closest"].values()) for r in results if r["closest"]])
        ),
        "ckpt": str(args.ckpt),
        "seed": args.seed,
    }
    if replay is not None:
        errs = [r["replay_err_median_m"] for r in results]
        closes = [r["replay_close_err_m"] for r in results if r["replay_close_err_m"] is not None]
        xys = [r["replay_close_dist_xy_m"] for r in results if r["replay_close_dist_xy_m"] is not None]
        summary["replay"] = {
            "mode": args.replay_mode,
            "zarr": str(args.replay_demo),
            "demo_episodes": [r["demo_episode"] for r in results],
            "lifts": int(sum(r["success"] for r in results)),
            "n": len(results),
            # P0's refutation criterion: >=8/10 lifts AND median per-step error <5 mm
            "per_step_err_median_m": float(np.median(errs)),
            "per_step_err_p90_m": float(np.percentile([r["replay_err_max_m"] for r in results], 90)),
            "per_step_err_worst_m": float(np.max([r["replay_err_max_m"] for r in results])),
            "err_at_close_median_m": float(np.median(closes)) if closes else None,
            "xy_to_target_at_close_median_m": float(np.median(xys)) if xys else None,
            "episodes_that_closed": len(closes),
        }
    if any(r["commanded"] for r in results):
        per_cmd: dict[str, dict[str, int]] = {}
        for r in results:
            c = per_cmd.setdefault(r["commanded"], {"n": 0, "correct": 0})
            c["n"] += 1
            c["correct"] += int(r["reached"] == r["commanded"])
        summary["per_command"] = {
            k: {"n": v["n"], "correct": v["correct"], "accuracy": v["correct"] / max(1, v["n"])}
            for k, v in sorted(per_cmd.items())
        }
        per_cmd_app: dict[str, dict[str, float]] = {}
        for r in results:
            c = per_cmd_app.setdefault(r["commanded"], {"n": 0, "correct": 0, "dist": []})
            c["n"] += 1
            c["correct"] += int(r["approached"] == r["commanded"])
            c["dist"].append(r["closest"].get(r["commanded"], float("nan")))
        summary["per_command_closest_approach"] = {
            k: {
                "n": v["n"],
                "correct": v["correct"],
                "accuracy": v["correct"] / max(1, v["n"]),
                "mean_dist_to_commanded_m": round(float(np.nanmean(v["dist"])), 4),
            }
            for k, v in sorted(per_cmd_app.items())
        }
    print(json.dumps(summary, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"rollout data -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
