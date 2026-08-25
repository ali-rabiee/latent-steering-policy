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
    EXPERT_GRASP_Z = 0.047
    EXPERT_LIFT_Z = EXPERT_GRASP_Z + 0.207
    EXPERT_STEP_M = 0.03      # demos travel ~0.030 m per 5 Hz tick
    EXPERT_TOL_ALIGN_M = 0.012
    # The descend tolerance must be tight: it is applied to the 3-D error, so a
    # loose value lets the descent stop that far ABOVE the grasp height, and the
    # demos close within 1 mm of 0.047.
    EXPERT_TOL_DESCEND_M = 0.004
    # Demos wait 4-5 ticks between closing and lifting (156 wait 4, 264 wait 5);
    # the fingers need that long to reach the 1.2 rad closed target.
    EXPERT_CLOSE_HOLD = 6

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
        """Return (dpos, gripper, phase, hold) for one policy tick."""
        home_z = float(home_pose_z)
        targets = {
            0: np.array([box_xy[0], box_xy[1], home_z], dtype=np.float64),        # align xy high
            1: np.array([box_xy[0], box_xy[1], EXPERT_GRASP_Z], dtype=np.float64),  # descend
            2: np.array([box_xy[0], box_xy[1], EXPERT_LIFT_Z], dtype=np.float64),   # lift
        }
        if phase == "grip":  # sit still while the fingers close
            hold += 1
            return np.zeros(3), schema.GRIPPER_CLOSE, ("lift" if hold >= EXPERT_CLOSE_HOLD else "grip"), hold
        if phase == "done":  # hold the lifted pose so the success test can settle
            return np.zeros(3), schema.GRIPPER_CLOSE, "done", hold
        idx = {"align": 0, "descend": 1, "lift": 2}[phase]
        tgt = targets[idx]
        err = tgt - np.asarray(pos_b, dtype=np.float64)
        tol = EXPERT_TOL_ALIGN_M if phase == "align" else EXPERT_TOL_DESCEND_M
        if float(np.linalg.norm(err)) < tol:
            if phase == "align":
                return np.zeros(3), schema.GRIPPER_OPEN, "descend", hold
            if phase == "descend":
                return np.zeros(3), schema.GRIPPER_CLOSE, "grip", hold
            return np.zeros(3), schema.GRIPPER_CLOSE, "done", hold
        n = float(np.linalg.norm(err))
        step = err * min(1.0, EXPERT_STEP_M / max(n, 1e-9))
        grip = schema.GRIPPER_OPEN if phase in ("align", "descend") else schema.GRIPPER_CLOSE
        return step, grip, phase, hold

    home_pose_z = 0.248  # replaced with the measured home pose below

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
        for leaf, pos_w in layout_w.items():
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
        _, quat_ref = runtime.get_ee_pose_b(h, controller)  # top-down reference
        origin_b = np.asarray(h.scene_origins[0], dtype=np.float32)
        boxes_xy_b = {k: (v - origin_b)[0:2] for k, v in boxes0.items()}
        # closest approach per box over the whole episode. `reached_leaf` is
        # recorded at the first gripper close, which fires ~1.6 s in regardless
        # of how far the target is -- that makes it a measure of gripper timing,
        # not of which box the arm actually travelled to. This one is timing-free.
        closest = {k: float("inf") for k in boxes_xy_b}
        max_lift_seen = 0.0
        max_finger_rad = 0.0  # how far the fingers actually close (target 1.2)
        latched = False
        commit_xy = None  # E1 mode-lock: xy endpoint committed at the first replan
        commanded_leaf = None
        goal_vec = None
        if args.expert:
            leaves = sorted(layout_w.keys())
            commanded_leaf = leaves[ep % len(leaves)]
            origin0 = np.asarray(h.scene_origins[0], dtype=np.float32)
            box_xy_b = (layout_w[commanded_leaf] - origin0)[0:2].astype(np.float32)
            expert_phase, expert_hold = "align", 0
            home_pose_z = float(runtime.get_ee_pose_b(h, controller)[0][2])
        if args.steer_to_box or getattr(cfg, "goal_dim", 0) > 0:
            leaves = sorted(layout_w.keys())
            commanded_leaf = leaves[ep % len(leaves)]
            origin0 = np.asarray(h.scene_origins[0], dtype=np.float32)
            box_xy_b = (layout_w[commanded_leaf] - origin0)[0:2].astype(np.float32)
        if args.steer_to_box:
            if args.mode_lock <= 0:
                raise SystemExit("--steer-to-box requires --mode-lock K")
            commit_xy = box_xy_b
        if getattr(cfg, "goal_dim", 0) > 0:
            # E2 goal-conditioned ckpt: command the box explicitly, same
            # round-robin protocol as --steer-to-box
            label = h.id_to_label.get(commanded_leaf, "")
            color_idx = schema.COLOR_PALETTE.index(label.split()[0])
            gv = np.zeros(cfg.goal_dim, dtype=np.float32)
            gv[color_idx] = 1.0
            gv[schema.MAX_BOXES : schema.MAX_BOXES + 2] = box_xy_b
            goal_vec = torch.from_numpy(gv)

        step = 0
        while step < steps_per_episode:
            obs = {
                schema.camera_obs_key(cam): torch.stack([f[cam] for f, _ in obs_hist])
                for cam in cfg.camera_names
            }
            obs["state"] = torch.stack([s for _, s in obs_hist])
            if goal_vec is not None:
                obs["goal"] = goal_vec
            cur_xy = obs["state"][-1, 0:2].numpy()
            if args.expert:
                # one expert tick per policy tick, same cadence as act_horizon
                pos_now, _ = runtime.get_ee_pose_b(h, controller)
                dpos, g_exp, expert_phase, expert_hold = expert_action(
                    pos_now, box_xy_b, expert_phase, expert_hold
                )
                actions = np.zeros((1, 7), dtype=np.float32)
                actions[0, 0:3] = dpos
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
            if not args.expert:
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
                provider.set_step(a[0:3], a[3:6], gripper, n_phys)
                run_phys_steps(n_phys)
                step += 1

                imgs, state, ee_now = observe(gripper, save_to=frame_dir, tick=step)
                obs_hist.append((imgs, state))
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
                    # how far the box ACTUALLY moved, whether or not it cleared
                    # the threshold: 0 means the fingers never gripped it, a
                    # small positive value means it was gripped and slipped.
                    max_lift_seen = max(max_lift_seen, max(lifts.values()))
                if lifts and max(lifts.values()) > args.lift_thresh:
                    success = True
                    if reached_leaf is None:
                        reached_leaf = max(lifts, key=lifts.get)
                    break
            if success:
                break

        label = h.id_to_label.get(reached_leaf or "", "")
        approached = min(closest, key=closest.get) if closest else None
        results.append({"episode": ep, "success": success, "reached": reached_leaf, "label": label,
                        "commanded": commanded_leaf, "approached": approached,
                        "max_lift_m": round(float(max_lift_seen), 4),
                        "max_finger_rad": round(float(max_finger_rad), 4),
                        "closest": {k: round(v, 4) for k, v in closest.items()}})
        cmd_str = f" commanded={commanded_leaf}" if commanded_leaf else ""
        print(f"[ep {ep:03d}] success={success} reached={reached_leaf} ({label}){cmd_str} lift={max_lift_seen:.3f} fingers={max_finger_rad:.2f} steps={step}")

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
        "median_closest_approach_m": float(
            np.median([min(r["closest"].values()) for r in results if r["closest"]])
        ),
        "ckpt": str(args.ckpt),
        "seed": args.seed,
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
