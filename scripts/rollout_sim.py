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

    results = []
    for ep in range(args.episodes):
        # reset robot + teleport boxes back to the fixed layout
        runtime.reset_sim_and_robot(h)
        controller.reset(h.robot)
        controller.set_mode("twist")
        provider.set_step(np.zeros(3), np.zeros(3), schema.GRIPPER_OPEN, 1)
        for leaf, pos_w in layout_w.items():
            for p in h.spawned_paths:
                if p.endswith(leaf):
                    runtime.teleport_box(h, p, tuple(float(v) for v in pos_w))
        run_phys_steps(20)

        g_ep = torch.Generator(device=device).manual_seed(args.seed * 100_000 + ep)
        gripper = schema.GRIPPER_OPEN
        frame_dir = (out_dir / f"episode_{ep:04d}") if args.save_frames else None
        imgs, state, _ = observe(gripper, save_to=frame_dir, tick=0)
        obs_hist = deque([(imgs, state)] * cfg.obs_horizon, maxlen=cfg.obs_horizon)

        replans = []
        reached_leaf = None
        success = False
        boxes0 = runtime.box_snapshot(h)

        step = 0
        while step < steps_per_episode:
            obs = {
                schema.camera_obs_key(cam): torch.stack([f[cam] for f, _ in obs_hist])
                for cam in cfg.camera_names
            }
            obs["state"] = torch.stack([s for _, s in obs_hist])
            out = policy.predict_action(obs, k=1, generator=g_ep)  # fresh z per replan
            actions = out["action"][0].cpu().numpy()  # (T_a, 7)
            replans.append(
                {
                    "step": step,
                    "state": obs["state"].numpy(),
                    "z": out["z"][0].cpu().numpy(),
                    "action_pred": out["action_pred"][0].cpu().numpy(),
                }
            )

            for a in actions:
                g_cmd = schema.GRIPPER_OPEN if a[6] > 0.0 else schema.GRIPPER_CLOSE
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

                imgs, state, _ = observe(gripper, save_to=frame_dir, tick=step)
                obs_hist.append((imgs, state))

                snap = runtime.box_snapshot(h)
                lifts = {k: float(snap[k][2] - boxes0[k][2]) for k in snap if k in boxes0}
                if lifts and max(lifts.values()) > args.lift_thresh:
                    success = True
                    if reached_leaf is None:
                        reached_leaf = max(lifts, key=lifts.get)
                    break
            if success:
                break

        label = h.id_to_label.get(reached_leaf or "", "")
        results.append({"episode": ep, "success": success, "reached": reached_leaf, "label": label})
        print(f"[ep {ep:03d}] success={success} reached={reached_leaf} ({label}) steps={step}")

        np.savez_compressed(
            out_dir / f"episode_{ep:04d}.npz",
            success=success,
            reached=str(reached_leaf),
            label=label,
            layout_ids=np.array(sorted(layout_w.keys())),
            layout_pos_w=np.stack([layout_w[k] for k in sorted(layout_w.keys())]),
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
    summary = {
        "episodes": len(results),
        "success_rate": n_success / max(1, len(results)),
        "per_box_coverage": {k: v / max(1, len(results)) for k, v in reached_counts.items()},
        "ckpt": str(args.ckpt),
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"rollout data -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
