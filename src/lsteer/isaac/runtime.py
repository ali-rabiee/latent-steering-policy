"""Shared Isaac-side runtime for policy replay/rollout in the kinova-isaac sim.

Everything here mirrors data_collection/profiles/vla_v1.py's setup so the
deployment-time observation/actuation path matches the one the demos were
logged under. All heavy imports are deferred to call time (Kit must be up).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

BOX_COLORS: list[tuple[str, tuple[float, float, float]]] = [
    ("red", (0.85, 0.20, 0.20)),
    ("blue", (0.20, 0.35, 0.90)),
    ("yellow", (0.95, 0.85, 0.20)),
    ("purple", (0.65, 0.25, 0.80)),
    ("orange", (0.95, 0.55, 0.15)),
    ("cyan", (0.15, 0.80, 0.85)),
]


@dataclass
class SimHandles:
    sim: object
    env: object
    robot: object
    camera_sensor: object | None
    dt: float
    scene_origins: object
    spawned_paths: list[str] = field(default_factory=list)
    id_to_label: dict[str, str] = field(default_factory=dict)
    tracker: object | None = None


def build_sim(args, *, spawn_min=(0.30, -0.30, 0.90), spawn_max=(0.55, 0.30, 0.95)) -> SimHandles:
    """Build scene + robot + top-down camera + boxes, then reset. Mirrors vla_v1.run()."""
    import carb
    from isaaclab.sensors import Camera, CameraCfg

    from data_collection.core.objects import ObjectsTracker
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import object_loader_kwargs_from_physix
    from environments.ycb_reach_to_grasp import DEFAULT_TOP_DOWN_CAMERA, YCBReachToGraspEnv

    enable_cameras = bool(getattr(args, "enable_cameras", True))
    carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", enable_cameras)

    env = YCBReachToGraspEnv(device=str(getattr(args, "device", "cuda:0")))
    sim = env.build_simulation()
    if not getattr(args, "headless", False):
        env.set_default_camera_view()
    env.design_scene()
    robot = env.robot
    if enable_cameras:
        env.attach_top_down_camera()

    # boxes (spawn_mode="box", fixed size, deterministic color per index)
    num_objects = int(getattr(args, "num_objects", 4))
    box_size = float(getattr(args, "box_size", 0.05))
    loader_cfg = ObjectLoaderConfig(
        dataset_dirs=[],
        bounds=SpawnBounds(min_xyz=tuple(spawn_min), max_xyz=tuple(spawn_max)),
        min_distance=0.20,
        min_distance_xy_only=True,
        spawn_mode="box",
        box_size_min=(box_size, box_size, box_size),
        box_size_max=(box_size, box_size, box_size),
        box_color_palette=[rgb for (_n, rgb) in BOX_COLORS],
        box_color_names=[n for (n, _r) in BOX_COLORS],
        **object_loader_kwargs_from_physix(env.physics_cfg),
    )
    loader = ObjectLoader(loader_cfg)
    spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=num_objects)
    id_to_label = {}
    for p in spawned_paths:
        leaf = str(p).split("/")[-1]
        idx = int(leaf.split("_")[-1])
        id_to_label[leaf] = f"{BOX_COLORS[(idx - 1) % len(BOX_COLORS)][0]} box {idx}"

    camera_sensor = None
    if enable_cameras and DEFAULT_TOP_DOWN_CAMERA is not None:
        camera_cfg = CameraCfg(
            prim_path=DEFAULT_TOP_DOWN_CAMERA.prim_path,
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=None,
            data_types=["rgb"],
            width=DEFAULT_TOP_DOWN_CAMERA.resolution[0],
            height=DEFAULT_TOP_DOWN_CAMERA.resolution[1],
        )
        camera_sensor = Camera(cfg=camera_cfg)

    handles = SimHandles(
        sim=sim,
        env=env,
        robot=robot,
        camera_sensor=camera_sensor,
        dt=float(sim.get_physics_dt()),
        scene_origins=env.scene_origins,
        spawned_paths=list(spawned_paths),
        id_to_label=id_to_label,
        tracker=ObjectsTracker(prim_paths=list(spawned_paths)),
    )
    reset_sim_and_robot(handles)
    if camera_sensor is not None:
        camera_sensor.reset()
    return handles


def reset_sim_and_robot(h: SimHandles) -> None:
    """Same as vla_v1's _reset_sim_and_robot."""
    h.sim.reset()
    origin0 = torch.tensor(h.scene_origins[0], device=h.sim.device)
    root_state = h.robot.data.default_root_state.clone()
    root_state[:, :3] += origin0
    h.robot.write_root_pose_to_sim(root_state[:, :7])
    h.robot.write_root_velocity_to_sim(root_state[:, 7:])
    h.robot.write_joint_state_to_sim(h.robot.data.default_joint_pos, h.robot.data.default_joint_vel)
    h.robot.reset()


def set_arm_joint_positions(h: SimHandles, names: list[str], positions: list[float]) -> None:
    """Write arm joint positions (e.g. a demo's tick-0 state) into the sim."""
    ids, _ = h.robot.find_joints(list(names), preserve_order=True)
    q = h.robot.data.joint_pos.clone()
    q[0, ids] = torch.tensor(positions, dtype=q.dtype, device=q.device)
    h.robot.write_joint_state_to_sim(q, torch.zeros_like(h.robot.data.joint_vel))
    h.robot.reset()


def make_twist_controller(h: SimHandles, *, ee_link: str = "j2n6s300_end_effector"):
    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController

    cfg = CartesianVelocityJogConfig(
        ee_link_name=ee_link,
        device=str(h.sim.device),
        use_relative_mode=True,
        workspace_min=(0.20, -0.45, 0.0),
        workspace_max=(0.6, 0.45, 1.20),
    )
    controller = CartesianVelocityJogController(cfg, num_envs=1, device=str(h.sim.device))
    controller.set_mode("twist")
    controller.reset(h.robot)
    return controller


class ChunkActionProvider:
    """InputProvider feeding one policy action step as per-physics-step deltas.

    A policy action step is a delta over one policy period (0.2 s at 5 Hz).
    `set_step` divides it across the physics steps of that period; after they
    are consumed, `advance` returns zeros (robot holds) until the next set_step.
    The gripper channel is passed through un-divided (it is a setpoint).
    """

    def __init__(self, device: str):
        self.device = device
        self._per_step = torch.zeros(1, 7, device=device)
        self._g = 0.0
        self._remaining = 0

    def set_step(self, dpos: np.ndarray, drot: np.ndarray, gripper: float, n_phys_steps: int) -> None:
        cmd = torch.zeros(1, 7, device=self.device)
        cmd[0, 0:3] = torch.as_tensor(dpos, dtype=torch.float32, device=self.device) / n_phys_steps
        cmd[0, 3:6] = torch.as_tensor(drot, dtype=torch.float32, device=self.device) / n_phys_steps
        cmd[0, 6] = float(gripper)
        self._per_step = cmd
        self._remaining = int(n_phys_steps)

    def advance(self) -> torch.Tensor:
        if self._remaining > 0:
            self._remaining -= 1
            return self._per_step.clone()
        hold = torch.zeros(1, 7, device=self.device)
        hold[0, 6] = float(self._per_step[0, 6])  # keep holding the gripper setpoint
        return hold


def get_ee_pose_b(h: SimHandles, controller) -> tuple[np.ndarray, np.ndarray]:
    """EE pose in base frame — the same computation as logger.py::write_tick."""
    from isaaclab.utils.math import subtract_frame_transforms

    ee_id = int(controller._ee_body_id)
    ee_pose_w = h.robot.data.body_pose_w[:, ee_id]
    root_pose_w = h.robot.data.root_pose_w
    pos_b, quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    return pos_b[0].cpu().numpy(), quat_b[0].cpu().numpy()


def capture_rgb(h: SimHandles) -> Optional[np.ndarray]:
    """Latest top-down RGB frame as uint8 HWC, or None."""
    if h.camera_sensor is None:
        return None
    data = h.camera_sensor.data
    rgb = data.output.get("rgb") if data.output is not None else None
    if rgb is None:
        return None
    arr = rgb[0].cpu().numpy() if rgb.dim() == 4 else rgb.cpu().numpy()
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr[..., :3]


def step_sim(h: SimHandles, controller, *, render: bool) -> None:
    controller.step(h.robot, h.dt)
    h.sim.step(render=render)
    h.robot.update(h.dt)
    if render and h.camera_sensor is not None:
        h.camera_sensor.update(h.dt)


def box_snapshot(h: SimHandles) -> dict[str, np.ndarray]:
    """{leaf id: world position (3,)} for all spawned boxes."""
    out = {}
    for o in h.tracker.snapshot():
        out[o.id] = np.asarray(o.pose.position_m, dtype=np.float32)
    return out


def teleport_box(h: SimHandles, prim_path: str, pos_w: tuple[float, float, float], yaw: float = 0.0) -> bool:
    """Teleport one box rigid body (vla_v1's RigidPrim path, compacted)."""
    from isaacsim.core.simulation_manager import SimulationManager

    half = 0.5 * yaw
    quat = (math.cos(half), 0.0, 0.0, math.sin(half))
    view = SimulationManager.get_physics_sim_view().create_rigid_body_view(str(prim_path))
    t0 = view.get_transforms()
    if t0.shape[0] == 0:
        view = SimulationManager.get_physics_sim_view().create_rigid_body_view(f"{prim_path}/*")
        t0 = view.get_transforms()
        if t0.shape[0] == 0:
            return False
    # RigidBodyView transform format: [x, y, z, qx, qy, qz, qw] (as in vla_v1)
    n = max(1, int(t0.shape[0]))
    tf = torch.tensor(
        [[pos_w[0], pos_w[1], pos_w[2], quat[1], quat[2], quat[3], quat[0]]], device=t0.device
    ).repeat(n, 1)
    idx = torch.arange(n, dtype=torch.int32, device=t0.device)
    view.set_transforms(tf, indices=idx)
    if hasattr(view, "set_velocities"):
        view.set_velocities(torch.zeros(n, 6, device=t0.device), indices=idx)
    return True
