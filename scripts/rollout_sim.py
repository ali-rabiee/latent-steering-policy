"""M7: closed-loop evaluation of the frozen diffusion policy in Isaac Sim.

Receding-horizon loop at 5 Hz: capture top-down RGB + EE state (identical
preprocessing to training via lsteer.data.obs), denoise an action chunk from a
fresh z, execute act_horizon steps through the twist controller, repeat.

Every replan's (state, z, predicted chunk) plus the episode outcome is dumped
to outputs/rollouts/<run>/episode_XXXX.npz — the Phase-2 steering-encoder and
Phase-3 gate data seam.

Multimodality eval: boxes stay on a FIXED layout across episodes (teleported
back each reset); only z varies. Per-box coverage of the reached box measures
steerability of the frozen policy.

Run under Isaac Lab python, e.g.:
    conda run -n kinova python scripts/rollout_sim.py \
        --ckpt outputs/train/<run>/latest.ckpt --episodes 50 --headless --enable_cameras
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

    from lsteer.data import schema
    from lsteer.data.obs import build_lowdim_obs, center_crop, image_to_tensor, resize_to_storage
    from lsteer.isaac import runtime
    from lsteer.policy import DiffusionPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = DiffusionPolicy.load(args.ckpt, device=device)
    cfg = policy.cfg
    print(f"loaded policy: T_o={cfg.obs_horizon} T_p={cfg.pred_horizon} T_a={cfg.act_horizon}")

    out_dir = args.out_dir or Path("outputs/rollouts") / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    h = runtime.build_sim(args)
    controller = runtime.make_twist_controller(h)
    provider = runtime.ChunkActionProvider(device=str(h.sim.device))
    controller.set_input_provider(provider)

    for _ in range(20):
        runtime.step_sim(h, controller, render=True)

    # fixed layout for the whole eval: record spawn poses once
    layout_w = runtime.box_snapshot(h)
    print(f"fixed layout: { {k: np.round(v, 3).tolist() for k, v in layout_w.items()} }")

    n_phys = max(1, round((1.0 / schema.FPS) / h.dt))
    steps_per_episode = int(args.max_duration_s * schema.FPS)

    def observe(gripper_state: float):
        rgb = runtime.capture_rgb(h)
        assert rgb is not None, "camera returned no frame"
        img = center_crop(image_to_tensor(resize_to_storage(rgb)))
        pos_b, quat_b = runtime.get_ee_pose_b(h, controller)
        state = build_lowdim_obs(pos_b, quat_b, gripper_state)
        return img, torch.from_numpy(state), pos_b

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
        for _ in range(20):
            runtime.step_sim(h, controller, render=True)

        g_ep = torch.Generator(device=device).manual_seed(args.seed * 100_000 + ep)
        gripper = schema.GRIPPER_OPEN
        img, state, _ = observe(gripper)
        obs_hist = deque([(img, state)] * cfg.obs_horizon, maxlen=cfg.obs_horizon)

        replans = []
        reached_leaf = None
        success = False
        boxes0 = runtime.box_snapshot(h)

        step = 0
        while step < steps_per_episode:
            obs = {
                "img": torch.stack([f for f, _ in obs_hist]),
                "state": torch.stack([s for _, s in obs_hist]),
            }
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
                for _ in range(n_phys):
                    runtime.step_sim(h, controller, render=True)
                step += 1

                img, state, _ = observe(gripper)
                obs_hist.append((img, state))

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

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
