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
        "--exec-converge",
        type=float,
        default=0.0,
        metavar="TOL_M",
        help="P2c: drive each target until the EE is within TOL of it, instead of for a fixed "
        "n_phys ticks. Measured motivation: replaying a demo's own actions, achieved displacement "
        "over a replan window divided by commanded displacement has a median of 1.43 (1.62, 1.47, "
        "1.39, 0.81, 4.29, 0.94, 2.17) -- asked to move d, the arm moves 1.43 d. The demonstrations "
        "were recorded by a controller that CONVERGED to each waypoint "
        "(collect_boxes.py::_run_segment, 8 mm tolerance), while this loop spends a fixed tick "
        "budget and moves on whether or not the arm arrived, so the next target is measured from a "
        "pose that is still in motion. 0 = off. Requires --exec-diffik.",
    )
    parser.add_argument(
        "--exec-arrive-and-hold",
        action="store_true",
        help="P2e: keep the FULL n_phys window, but once the arm has travelled the distance it "
        "was commanded, freeze the target at the pose it reached and hold there for the rest of "
        "the window. Fixes both defects found so far. (1) The overshoot: P2c measured that the arm "
        "needs 15 of its 48 ticks to arrive and spends the other 33 coasting past, giving a 1.5x "
        "displacement gain. (2) The dead zone --exec-converge introduced: its fixed 8 mm tolerance "
        "is larger than 22.6%% of the policy's own commanded steps and 35.7%% of the demos', so "
        "those actions counted as 'already arrived' before the arm moved and got one tick instead "
        "of 48 -- gain fell to 0.692 and champion success to 32.5%%. Stopping on DISTANCE TRAVELLED "
        "is scale-free: a 2 mm command stops after 2 mm. Holding for the rest of the window keeps "
        "the 5 Hz control rate and the observation cadence identical to the baseline, which "
        "early-exit did not.",
    )
    parser.add_argument(
        "--exec-ramp",
        action="store_true",
        help="P2d: interpolate the target smoothly from the window's starting pose to "
        "pos_before + delta across the tick budget, instead of jumping to the endpoint and holding "
        "it. MEASURED motivation (job 63737044, per-tick trace): fed a step, the arm reaches its "
        "target at about 60%% of the window and then sails straight through at constant speed -- on "
        "a 69.3 mm command it passes within 4.3 mm at tick 28 and ends 52.2 mm beyond. The "
        "normalised response has the same shape at every step size and the overshoot grows with it "
        "(gain 1.39 at 1.5 mm, 1.59 at 30.7 mm, 1.78 at 76.3 mm): an underdamped second-order "
        "system driven by a step. The demonstrations never did that -- collect_boxes drives "
        "QUINTIC-interpolated segments -- so this feeds the same smoothstep profile the data was "
        "recorded under. Unlike --exec-converge and --exec-arrive-and-hold it changes WHAT is "
        "commanded during the window rather than when the window ends, and it has no tolerance and "
        "no dead zone.",
    )
    parser.add_argument(
        "--dagger-out",
        type=str,
        default="",
        metavar="DIR",
        help="A4: DAgger collection. Drive with the POLICY, but at every executed action log the observation "
        "together with the action the STATELESS EXPERT would take from that state. This is the one approach that "
        "attacks the project's central finding head-on -- the failure is invisible on-distribution, so the fix "
        "has to be measured where the policy actually goes. It also sidesteps what killed R1/R2/R3: those "
        "injected synthetic knocks, and each carried a directional asymmetry, whereas here the states are the "
        "ones the policy reaches on its own and nothing is perturbed. Images are captured EVERY tick (not every "
        "replan) so the logged data is 5 Hz per-tick like the demonstrations, at the cost of 4x the rendering.",
    )
    parser.add_argument(
        "--expert-stateless",
        action="store_true",
        help="A4: derive the expert's phase from the current pose and gripper instead of carrying it forward. "
        "Required for DAgger relabelling -- the carried phase machine reflects the EXPERT's own history, which "
        "never happened in a policy rollout, so it cannot answer 'what would the expert do from HERE'. Validate "
        "it against the stateful expert's 120/120 before trusting it as a relabeller.",
    )
    parser.add_argument(
        "--expert-xy-offset",
        type=float,
        default=0.0,
        metavar="METRES",
        help="G0d: with --expert, aim the scripted expert at a point this far from the true box, in "
        "a direction fixed per episode. MEASURED motivation: on the same three tables the expert "
        "lifts 120/120 reaching a median 0.5-0.8 mm from the box, while the champion lifts 45% "
        "reaching 7-9 mm. Sweeping this offset turns 'the grasp is marginal' into a tolerance curve "
        "and says whether the policy's approach error is inside the basin or outside it -- which "
        "decides whether the remaining gap is WHERE it grasps or WHEN it closes. Success detection "
        "still uses the true box position, so this perturbs the grasp and not the metric.",
    )
    parser.add_argument(
        "--exec-gain-comp",
        type=float,
        default=0.0,
        metavar="GAIN",
        help="A0: divide the commanded position delta by GAIN before forming the target, to cancel "
        "the executor's steady-state displacement gain. MEASURED motivation: with --exec-ramp the "
        "champion's own summary.json reports displacement_gain_median 0.833, i.e. the arm delivers "
        "83%% of every delta it is asked for, every action -- and rollout_sim re-reads the MEASURED "
        "pose before each action, so this is a stable multiplicative shortfall rather than a drift. "
        "The demonstrations were recorded where commanded == achieved, so a policy trained on them "
        "is asking for a displacement it does not get. Same argument as --exec-ramp (execute the "
        "action the way the data was recorded), applied to amplitude instead of profile. Unlike "
        "P2a/P2c/P2e this changes neither the reference frame nor when the window ends -- only the "
        "commanded amplitude. Pass 0 to disable. CHECK FIRST that the ramped gain is flat in "
        "amplitude (cmd_mm/ach_mm are now in every episode npz): the PRE-ramp gain ran 1.39->1.78 "
        "with step size, and if the ramped one also varies then a single scalar is the wrong shape "
        "of correction.",
    )
    parser.add_argument(
        "--exec-act-horizon",
        type=int,
        default=0,
        metavar="N",
        help="A5: execute only the first N actions of each predicted chunk before re-planning, instead of the "
        "trained act_horizon of 4. The scripted expert re-observes after every single action and reaches 1.2 mm "
        "laterally at the moment it closes; the policy goes 4 actions blind and closes at 10-22 mm. 0 = off.",
    )
    parser.add_argument(
        "--exec-grip-lookahead",
        type=int,
        default=0,
        metavar="L",
        help="A6: let the gripper see past the position horizon. MEASURED motivation: shortening "
        "--exec-act-horizon drives lateral error from 4.0 mm to 1.3 mm and success from 69% DOWN to "
        "22.5% - aim and success move in OPPOSITE directions, so the close, not the position, is the "
        "binding constraint. The reason truncation destroys the grasp is that the model emits a "
        "SCHEDULE ('close at chunk index k'); executing only the first N steps never reaches index k, "
        "so the hand never shuts. This reads the gripper channel from the UNTRUNCATED chunk and closes "
        "if the model intends to close within the next L steps, turning that schedule back into a "
        "decision. Only ever flips OPEN -> CLOSE, and only on the model's own prediction. "
        "0 = off (gripper stays tied to the truncated chunk).",
    )
    parser.add_argument(
        "--exec-abs-max-step",
        type=float,
        default=0.0,
        metavar="METRES",
        help="A1b: with --absolute-actions checkpoints, clip the implied motion (target - current) to this "
        "magnitude. Measured motivation: the absolute policy reaches 112 mm from the commanded box by replan 3 "
        "and then diverges back to 306 mm, burning all 25 replans, where the delta champion converges to 12 mm. "
        "Off the demonstrated phase an absolute model falls back to a specific workspace pose and drives to it; "
        "a delta model falls back to ~0 and stops. The expert's own peak single-step motion is 78.2 mm, so "
        "anything larger is off-distribution by construction. 0 disables.",
    )
    parser.add_argument(
        "--exec-gain-comp-slope",
        type=float,
        default=0.0,
        metavar="PER_MM",
        help="A0b: make the gain compensation AFFINE in commanded amplitude, c(a) = GAIN + SLOPE*a_mm, "
        "clamped to [0.60, 1.00]. MEASURED motivation: on A0gain's successful episodes the raw executor gain "
        "falls from 0.838 at a 7.5 mm command to 0.741 at 52 mm (fit: 0.8386 - 0.001316*a_mm), so a single "
        "scalar under-compensates long steps. Long steps are the reaches to the FAR boxes, and that is exactly "
        "where seed 1 fails: purple ends 21.1 mm out, past the 16 mm grasp basin G0d2 measured, with coverage "
        "6.9%%. Pass 0 to keep the constant compensation.",
    )
    parser.add_argument(
        "--dump-ticks",
        type=int,
        default=0,
        metavar="N_ACTIONS",
        help="characterise the executor: log the EE pose at EVERY physics tick for the first "
        "N_ACTIONS of episode 0, into <out>/tick_trace.npz. Three fixes to the delta executor have "
        "now failed (P2a accumulator 17.5%%, P2c converge 32.5%%, P2e arrive-and-hold gain 1.555) and "
        "all three were designed against a GUESS about how the arm responds inside a 0.2 s action "
        "window. This measures it instead: does the arm converge on a held target, how fast, does it "
        "overshoot, and what is the steady-state error.",
    )
    parser.add_argument(
        "--converge-max-ticks",
        type=int,
        default=0,
        help="tick cap per action for --exec-converge (default: 3x n_phys). A cap is required or a "
        "target the arm cannot reach would stall the episode forever.",
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
    # printed so a null result can be traced to the flag actually firing: slurm/ is
    # gitignored, and two experiments have already run as silent no-ops because a
    # --close-* flag never reached the cluster's sbatch.
    print(
        f"executor: {'abs-target accumulator' if args.exec_abs_target else 'cur_pos + delta'}"
        f" | ramp={bool(args.exec_ramp)}"
        f" | gain_comp={args.exec_gain_comp if args.exec_gain_comp > 0 else 'off'}"
        f" | actions={'ABSOLUTE' if getattr(cfg, 'absolute_actions', False) else 'delta'}"
        f" | act_horizon={args.exec_act_horizon if args.exec_act_horizon > 0 else 'full'}"
        f" | grip_lookahead={args.exec_grip_lookahead if args.exec_grip_lookahead > 0 else 'off'}"
    )

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

    def expert_action_stateless(pos_b, box_xy, box_held, hold):
        """A4: the same expert, but with its phase DERIVED from the current pose
        and gripper instead of carried forward from its own past.

        DAgger needs the action the expert would take at a state the POLICY
        reached, and the carried phase machine cannot answer that -- its phase
        reflects the expert's own history, which never happened in a policy
        rollout. Deriving the phase from the observation is what makes the
        expert usable as a relabeller.

        The derivation mirrors the stateful machine exactly: closed gripper
        means the grasp is done and the job is to lift; otherwise align in xy
        while high, then descend, then close.
        """
        home_z = float(home_pose_z)
        grasp_z = box_top_z + EXPERT_EE_Z_OFFSET + EXPERT_GRASP_DEPTH
        lift_z = box_top_z + EXPERT_EE_Z_OFFSET + EXPERT_TRAVEL_H
        pos = np.asarray(pos_b, dtype=np.float64)
        xy_err = float(np.linalg.norm(pos[0:2] - np.asarray(box_xy, dtype=np.float64)))

        if box_held:
            # genuinely holding the box -- judged from the WORLD (the box has
            # actually risen), never from the gripper command. That distinction is
            # the whole point: the pilot showed that keying on the gripper made
            # the relabeller inherit the policy's mistake, labelling "close" at a
            # median 0.111 m where the demos close at 0.047, because a policy that
            # had wrongly shut in mid-air looked to it like a successful grasp.
            # Reading the box instead means a premature close is labelled
            # "open and descend", which is the correction DAgger exists to supply.
            tgt = np.array([box_xy[0], box_xy[1], lift_z], dtype=np.float64)
            hold += 1
            return _rate_limit(pos, tgt), schema.GRIPPER_CLOSE, hold
        if xy_err > EXPERT_TOL_ALIGN_M:
            # not over the box yet: line up at travel height, fingers open
            tgt = np.array([box_xy[0], box_xy[1], home_z], dtype=np.float64)
            return _rate_limit(pos, tgt), schema.GRIPPER_OPEN, hold
        if pos[2] > grasp_z + EXPERT_TOL_DESCEND_M:
            tgt = np.array([box_xy[0], box_xy[1], grasp_z], dtype=np.float64)
            return _rate_limit(pos, tgt), schema.GRIPPER_OPEN, hold
        # over the box and at grasp height: close
        tgt = np.array([box_xy[0], box_xy[1], grasp_z], dtype=np.float64)
        return tgt, schema.GRIPPER_CLOSE, hold + 1

    def expert_chunk(pos0, box_xy, grip0, horizon):
        """A4: the expert's next `horizon` actions from pos0, in the DEMOS' own
        format (per-tick delta + gripper setpoint).

        Rolled forward kinematically: the demos were generated by driving to each
        absolute waypoint, so the pose after action k IS the waypoint, and
        pos_{k+1} = expert_target(pos_k). No simulation needed, and it reproduces
        exactly the convention convert.py uses (a_t = pose_{t+1} - pose_t, with
        the gripper as the setpoint at t+1).
        """
        out = np.zeros((horizon, 7), dtype=np.float32)
        p = np.asarray(pos0, dtype=np.float64)
        held, hold = bool(grip0), 0
        for k in range(horizon):
            tgt, g_next, hold = expert_action_stateless(p, box_xy, held, hold)
            # rolling forward, a close AT grasp height means the box is now held
            if g_next == schema.GRIPPER_CLOSE:
                held = True
            out[k, 0:3] = (np.asarray(tgt, dtype=np.float64) - p).astype(np.float32)
            out[k, 3:6] = 0.0  # orientation: the demos vary by ~0 and diff-IK pins it
            out[k, 6] = g_next
            p = np.asarray(tgt, dtype=np.float64)
        return out

    def _rate_limit(pos, tgt):
        err = tgt - pos
        n = float(np.linalg.norm(err))
        if n <= EXPERT_STEP_M:
            return tgt
        return pos + err * (EXPERT_STEP_M / max(n, 1e-9))

    home_pose_z = 0.248  # replaced with the measured home pose below
    box_top_z = 0.036   # replaced per episode from the actual box

    dagger_dir = Path(args.dagger_out) if args.dagger_out else None
    if dagger_dir is not None:
        dagger_dir.mkdir(parents=True, exist_ok=True)
        print(f"DAGGER collection -> {dagger_dir} (policy states, stateless-expert labels)")

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
        dag_states: list = []
        dag_actions: list = []
        dag_ep_dir = (dagger_dir / f"episode_{ep:04d}") if dagger_dir is not None else None
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
        grip_full = None  # A6: untruncated gripper schedule (expert/replay never set it)
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
            expert_grip = schema.GRIPPER_OPEN  # A4: observed gripper for the stateless expert
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
        if args.expert and args.expert_xy_offset > 0.0:
            # G0d: aim the expert at a point OFFSET from the true box, in a
            # direction fixed for the episode and varying across episodes, so the
            # offset can be swept into a grasp-tolerance curve. Success detection
            # still uses the TRUE box, so this perturbs the grasp, not the metric.
            #
            # THIS MUST STAY AFTER the goal-conditioning block above. The first
            # attempt applied it inside `if args.expert:` and every cell of the
            # sweep returned byte-identical numbers, because the champion is a
            # goal-conditioned checkpoint and the `goal_dim > 0` branch reassigns
            # box_xy_b from the true layout, silently undoing the offset. The flag
            # had arrived and the output tags were right; the variable was
            # overwritten downstream. Assert the offset survives.
            _th = np.random.default_rng(args.seed * 100_000 + ep).uniform(0.0, 2.0 * np.pi)
            _true_xy = box_xy_b.copy()
            box_xy_b = box_xy_b + np.array(
                [np.cos(_th), np.sin(_th)], dtype=np.float32
            ) * np.float32(args.expert_xy_offset)
            _applied = float(np.linalg.norm(box_xy_b - _true_xy))
            assert abs(_applied - args.expert_xy_offset) < 1e-6, (
                f"expert-xy-offset did not take: asked {args.expert_xy_offset}, applied {_applied}"
            )
            if ep == 0:
                print(f"expert-xy-offset ACTIVE: {args.expert_xy_offset*1000:.1f} mm")
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
        # executor diagnostics, recorded on every action regardless of mode:
        # how far the arm ended from the target it was GIVEN, what it was asked to
        # travel, what it actually travelled, and how many ticks that took
        tgt_err: list[float] = []
        tick_trace: list = []
        cmd_mm: list[float] = []
        ach_mm: list[float] = []
        ticks_used: list[int] = []
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
                if args.expert_stateless:
                    # The gripper is genuine OBSERVED state -- it is channel 9 of
                    # the state vector the policy itself sees -- so deriving the
                    # phase from it is not the hidden phase counter coming back.
                    _snap = runtime.box_snapshot(h)
                    _held = bool(
                        commanded_leaf in _snap
                        and commanded_leaf in boxes0
                        and (_snap[commanded_leaf][2] - boxes0[commanded_leaf][2]) > 0.01
                    )
                    tgt_abs, g_exp, expert_hold = expert_action_stateless(
                        pos_now, box_xy_b, _held, expert_hold
                    )
                else:
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
                # A1: with ABSOLUTE targets the chunk's endpoint is simply its last
                # predicted pose. Summing them, which is right for deltas, is
                # meaningless for absolute poses -- it adds up positions. Getting
                # this wrong fed mode-lock a nonsense selection criterion and
                # collapsed commanded-box obedience to 0.0-0.9 (run A1_60k, void).
                if getattr(cfg, "absolute_actions", False):
                    ends = preds[:, -1, 0:2]
                else:
                    ends = cur_xy[None] + preds[:, :, 0:2].sum(axis=1)
                sel = int(np.argmin(np.linalg.norm(ends - commit_xy[None], axis=1)))
            else:
                out = policy.predict_action(obs, z=z_ep, k=1)  # z held fixed for the episode
                sel = 0
                if args.mode_lock > 0:
                    pred0 = out["action_pred"][0].cpu().numpy()
                    if getattr(cfg, "absolute_actions", False):
                        commit_xy = pred0[-1, 0:2]
                    else:
                        commit_xy = cur_xy + pred0[:, 0:2].sum(axis=0)
            if not args.expert and replay is None:
                actions = out["action"][sel].cpu().numpy()  # (T_a, 7)
                grip_full = None
                if args.exec_act_horizon > 0:
                    # A5: execute fewer of the predicted actions before re-planning.
                    # MEASURED motivation: the scripted expert re-observes after
                    # EVERY action (its chunk is length 1) and lifts 120/120 at a
                    # 1.2 mm lateral error; the policy executes 4 blind actions
                    # between observations and closes at 10-22 mm. During the
                    # descent, 4 actions is ~8 cm of unobserved motion, and the
                    # failures' own numbers say alignment is achieved high
                    # (2.7 mm at z=0.20) and LOST on the way down. This is the one
                    # structural difference between the 100% expert and the 70%
                    # policy that has never been tested.
                    #
                    # NOTE it changes the observation cadence, not the control
                    # rate: every action still gets its full tick budget, unlike
                    # P2c's early exit which silently ran the arm at 3x speed.
                    if args.exec_grip_lookahead > 0:
                        # A6: keep the WHOLE schedule's gripper channel. Captured
                        # before the slice, so index i here is the same instant as
                        # index i in the truncated position chunk.
                        grip_full = actions[:, 6].copy()
                    actions = actions[: args.exec_act_horizon]
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

            for _ai, a in enumerate(actions):
                if dagger_dir is not None:
                    # A4: log the state the POLICY reached, paired with the action
                    # the EXPERT would take from it. Captured BEFORE the policy's
                    # action executes, so the observation and the label describe
                    # the same instant.
                    # `gripper` is the POLICY's current gripper - the state the
                    # policy is actually in. Both the observation and the expert's
                    # label must use it, or the two describe different robots.
                    _di, _ds, _dp = observe(gripper, save_to=dag_ep_dir, tick=len(dag_states))
                    dag_states.append(_ds.numpy().astype(np.float32))
                    _snap = runtime.box_snapshot(h)
                    _held = bool(
                        commanded_leaf in _snap
                        and commanded_leaf in boxes0
                        and (_snap[commanded_leaf][2] - boxes0[commanded_leaf][2]) > 0.01
                    )
                    dag_actions.append(expert_chunk(_dp, box_xy_b, _held, 1)[0])
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
                if grip_full is not None and g_cmd == schema.GRIPPER_OPEN:
                    _hi = min(len(grip_full), _ai + 1 + args.exec_grip_lookahead)
                    if bool(np.any(grip_full[_ai:_hi] <= 0.0)):
                        g_cmd = schema.GRIPPER_CLOSE
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
                    pos_before, _ = runtime.get_ee_pose_b(h, controller)
                    pos_before = np.asarray(pos_before, dtype=np.float64)
                    # Default for the expert / replay / abs-target branches, which do
                    # not go through the policy path below: a[0:3] is a displacement
                    # in all of them, exactly as it always was, so the reported gain
                    # stays comparable with every historical run. The policy path
                    # overwrites this (identically in delta mode; with the implied
                    # motion in A1's absolute mode).
                    _d_cmd = np.asarray(a[0:3], dtype=np.float64)
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
                        # A0: the arm delivers `gain` x the commanded delta (0.833 with
                        # the ramp), so ask for delta/gain to land on the delta the
                        # policy actually predicted. Direction is untouched; only the
                        # amplitude changes. cmd_mm below still records the ORIGINAL
                        # commanded magnitude, so the reported gain stays comparable
                        # across runs with and without the flag.
                        # A1: with an absolute-pose policy the action IS the target,
                        # so the implied motion is (target - where we are). Everything
                        # downstream - gain compensation, the ramp - then applies
                        # unchanged to that motion. This is the whole point of the
                        # representation: a delta re-baselines on wherever the arm
                        # actually got to, while an absolute target is re-aimed at the
                        # same place every step and so corrects its own shortfall.
                        if getattr(cfg, "absolute_actions", False):
                            _d = np.asarray(a[0:3], dtype=np.float64) - np.asarray(
                                cur_pos, dtype=np.float64
                            )
                            # A1b: bound the implied motion to the expert's own peak
                            # single-step displacement. MEASURED motivation: with
                            # absolute targets the arm closes to 112 mm of the
                            # commanded box by replan 3 and then DIVERGES, ending at
                            # 306 mm - roughly where it started - while burning all
                            # 25 replans. A delta model's off-distribution fallback is
                            # a small delta ("stay put"); an absolute model's is a
                            # specific place in the workspace, so it drives there. The
                            # demos never move more than 78.2 mm in one step, so any
                            # implied motion past that is off-distribution by
                            # construction and can be clipped without touching
                            # anything the demonstrations actually do.
                            if args.exec_abs_max_step > 0.0:
                                _n = float(np.linalg.norm(_d))
                                if _n > args.exec_abs_max_step:
                                    _d = _d * (args.exec_abs_max_step / _n)
                        else:
                            _d = np.asarray(a[0:3], dtype=np.float64)
                        _d_cmd = _d.copy()  # commanded motion, BEFORE compensation
                        if args.exec_gain_comp > 0.0:
                            # A0b: the executor's gain is not constant in amplitude.
                            # Measured on A0gain's successful episodes, the RAW gain
                            # falls from 0.838 at a 7.5 mm command to 0.741 at 52 mm,
                            # so one scalar over-compensates short steps and
                            # UNDER-compensates long ones - and the long ones are
                            # exactly the reaches to the far boxes, which is where
                            # seed 1 fails (purple at 21.1 mm, outside the 16 mm
                            # grasp basin, coverage 6.9%). An affine c(a) tracks it.
                            _c = args.exec_gain_comp
                            if args.exec_gain_comp_slope != 0.0:
                                _amp_mm = float(np.linalg.norm(_d)) * 1000.0
                                _c = float(
                                    np.clip(_c + args.exec_gain_comp_slope * _amp_mm, 0.60, 1.00)
                                )
                            _d = _d / _c
                        tgt_pos = np.asarray(cur_pos, dtype=np.float64) + _d
                    diffik.set_gripper(gripper == schema.GRIPPER_CLOSE)
                    if args.exec_ramp:
                        # smoothstep (quintic) from pos_before to tgt_pos across the window,
                        # the same profile collect_boxes.py interpolates its segments with.
                        # Same tick budget, so the control rate is unchanged.
                        for j in range(n_phys):
                            u = (j + 1) / n_phys
                            w = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)  # 0->1, zero end slopes
                            sub = pos_before + w * (tgt_pos - pos_before)
                            render = ((j + 1) == n_phys) or ((j + 1) % render_stride == 0)
                            diffik.step(sub, quat_ref, render=render)
                            if args.dump_ticks and ep == 0 and len(tick_trace) < args.dump_ticks * n_phys:
                                tick_trace.append(np.concatenate([
                                    np.asarray(runtime.get_ee_pose_b(h, controller)[0], dtype=np.float64),
                                    tgt_pos, pos_before, [float(step)]]))
                        ticks_used.append(n_phys)
                    elif args.exec_arrive_and_hold:
                        # Same tick budget as the baseline, so the control rate and the
                        # observation cadence are unchanged. The only difference is that
                        # the arm stops being pushed once it has covered the commanded
                        # distance, instead of coasting on toward a target it has passed.
                        want = float(np.linalg.norm(np.asarray(a[0:3], dtype=np.float64)))
                        frozen = None
                        for j in range(n_phys):
                            if frozen is None:
                                p_now = np.asarray(
                                    runtime.get_ee_pose_b(h, controller)[0], dtype=np.float64
                                )
                                if float(np.linalg.norm(p_now - pos_before)) >= want:
                                    frozen = p_now  # arrived: hold here
                            render = ((j + 1) == n_phys) or ((j + 1) % render_stride == 0)
                            diffik.step(
                                tgt_pos if frozen is None else frozen, quat_ref, render=render
                            )
                        ticks_used.append(n_phys)
                    elif args.exec_converge > 0.0:
                        # Stop when the arm ARRIVES, the way collect_boxes drove every
                        # segment, instead of when a tick counter runs out. Always take
                        # at least one step, and render the last one so the next
                        # observe() sees a fresh frame.
                        cap = int(args.converge_max_ticks) or (3 * n_phys)
                        j = 0
                        while j < cap:
                            j += 1
                            pos_now, _ = runtime.get_ee_pose_b(h, controller)
                            arrived = (
                                float(np.linalg.norm(np.asarray(pos_now, dtype=np.float64) - tgt_pos))
                                <= args.exec_converge
                            )
                            render = arrived or (j == cap) or (j % render_stride == 0)
                            diffik.step(tgt_pos, quat_ref, render=render)
                            if arrived:
                                break
                        ticks_used.append(j)
                    else:
                        for j in range(n_phys):
                            render = ((j + 1) == n_phys) or ((j + 1) % render_stride == 0)
                            diffik.step(tgt_pos, quat_ref, render=render)
                            if args.dump_ticks and ep == 0 and len(tick_trace) < args.dump_ticks * n_phys:
                                tick_trace.append(
                                    np.concatenate([
                                        np.asarray(runtime.get_ee_pose_b(h, controller)[0], dtype=np.float64),
                                        tgt_pos, pos_before, [float(step)],
                                    ])
                                )
                        ticks_used.append(n_phys)
                    # how far the arm ended from the target it was actually given.
                    # Large => the window ends mid-flight and the next target is set
                    # from a moving pose; ~0 => the gain comes from somewhere else.
                    _p = np.asarray(runtime.get_ee_pose_b(h, controller)[0], dtype=np.float64)
                    tgt_err.append(float(np.linalg.norm(_p - tgt_pos)))
                    # A1: for absolute targets |a[0:3]| is a POSITION, not a
                    # displacement, so the gain must be measured against the motion
                    # the action actually implies. _d is that motion in both modes.
                    cmd_mm.append(float(np.linalg.norm(_d_cmd)))
                    ach_mm.append(float(np.linalg.norm(_p - pos_before)))
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
        # displacement gain, over the actions that actually asked for real motion
        # (below ~5 mm the ratio is dominated by settling noise, not by tracking)
        big = [(c, g) for c, g in zip(cmd_mm, ach_mm) if c > 0.005]
        gain = float(np.median([g / c for c, g in big])) if big else float("nan")
        results[-1].update(
            {
                "gain_median": round(gain, 3),
                "tgt_err_median_m": round(float(np.median(tgt_err)), 5) if tgt_err else None,
                "ticks_median": float(np.median(ticks_used)) if ticks_used else None,
            }
        )
        cmd_str = f" commanded={commanded_leaf}" if commanded_leaf else ""
        ex_str = (
            f" gain={gain:.2f} tgt_err={np.median(tgt_err)*1000:.1f}mm ticks={np.median(ticks_used):.0f}"
            if tgt_err
            else ""
        )
        rp_str = ""
        if replay is not None:
            r = results[-1]
            rp_str = (
                f" err_med={r['replay_err_median_m']*1000:.1f}mm"
                f" err_max={r['replay_err_max_m']*1000:.1f}mm"
                f" err_at_close={'n/a' if replay_close_err is None else f'{replay_close_err*1000:.1f}mm'}"
                f" xy_to_box_at_close={'n/a' if replay_close_dist_xy is None else f'{replay_close_dist_xy*1000:.1f}mm'}"
            )
        print(f"[ep {ep:03d}] success={success} reached={reached_leaf} ({label}){cmd_str} lift={max_lift_seen:.3f} boxmove={max_box_move:.4f} fingers={max_finger_rad:.2f} steps={step} retries={retries}{ex_str}{rp_str}")

        if dagger_dir is not None and dag_states:
            # Meta in the zarr's own schema so the converter is trivial.
            origin_b = np.asarray(h.scene_origins[0], dtype=np.float32)
            leaves = sorted(layout_w.keys())
            bpos = np.full((schema.MAX_BOXES, 3), np.nan, dtype=np.float32)
            bcol = np.full((schema.MAX_BOXES,), -1, dtype=np.int64)
            for j, leaf in enumerate(leaves[: schema.MAX_BOXES]):
                bpos[j] = (layout_w[leaf] - origin_b).astype(np.float32)
                lab = h.id_to_label.get(leaf, "")
                bcol[j] = schema.COLOR_PALETTE.index(lab.split()[0]) if lab else -1
            tlab = h.id_to_label.get(commanded_leaf, "")
            np.savez_compressed(
                dag_ep_dir / "episode.npz",
                state=np.stack(dag_states),
                action=np.stack(dag_actions),
                target_pos_b=(layout_w[commanded_leaf] - origin_b).astype(np.float32),
                target_color=np.int64(schema.COLOR_PALETTE.index(tlab.split()[0])),
                box_positions=bpos,
                box_colors=bcol,
                success=np.int64(1),  # DAgger labels are the EXPERT's, always valid
            )
            print(f"[ep {ep:03d}] dagger: {len(dag_states)} relabelled frames -> {dag_ep_dir}")

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
            # A0: per-action commanded vs achieved displacement magnitude. These
            # were already computed for the summary's single median gain figure and
            # then thrown away; keeping them makes the gain-vs-AMPLITUDE curve
            # available offline from any rollout, which is what decides whether a
            # scalar gain compensation is the right shape of correction at all.
            cmd_mm=np.asarray(cmd_mm, dtype=np.float32),
            ach_mm=np.asarray(ach_mm, dtype=np.float32),
            tick_trace=(np.asarray(tick_trace, dtype=np.float32) if tick_trace
                        else np.zeros((0, 10), dtype=np.float32)),
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
    _g = [r["gain_median"] for r in results if r.get("gain_median") == r.get("gain_median")]
    if _g:
        summary["executor"] = {
            "converge_tol_m": args.exec_converge,
            "arrive_and_hold": bool(args.exec_arrive_and_hold),
            "ramp": bool(args.exec_ramp),
            "abs_target": bool(args.exec_abs_target),
            # asked to move d, the arm moves gain*d. 1.0 is faithful execution.
            "displacement_gain_median": round(float(np.median(_g)), 3),
            # how far the arm ends from the target it was given; large means the
            # action window closes while the arm is still in flight
            "target_err_median_m": round(
                float(np.median([r["tgt_err_median_m"] for r in results if r.get("tgt_err_median_m") is not None])), 5
            ),
            "ticks_per_action_median": float(
                np.median([r["ticks_median"] for r in results if r.get("ticks_median") is not None])
            ),
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
