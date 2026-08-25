"""Shared Isaac-side runtime for policy replay/rollout in the kinova-isaac sim.

Everything here mirrors data_collection/collect_boxes.py's setup so the
deployment-time observation/actuation path matches the one the demos were
logged under. All heavy imports are deferred to call time (Kit must be up).

Cameras: the SAME set the policy was trained on (front + wrist by default),
built through kinova-isaac's camera package so the geometry cannot drift from
the collector's. The wrist camera is NOT parented to the arm — `step_sim` must
call `sync_wrist_camera_to_ee` before every render or it silently produces a
static view (see PLAN.md gotcha 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from lsteer.data import schema

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
    cameras: dict[str, object]
    dt: float
    scene_origins: object
    spawned_paths: list[str] = field(default_factory=list)
    id_to_label: dict[str, str] = field(default_factory=dict)
    tracker: object | None = None
    wrist_cfg: object | None = None


def build_sim(
    args,
    *,
    spawn_min=(0.30, -0.30, 0.90),
    spawn_max=(0.55, 0.30, 0.95),
    camera_names=schema.CAMERA_NAMES,
) -> SimHandles:
    """Build scene + robot + cameras + boxes, then reset. Mirrors collect_boxes.py."""
    import carb

    from data_collection.core.objects import ObjectsTracker
    from environments.utils.camera import DEFAULT_WRIST_CAMERA, build_camera
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import object_loader_kwargs_from_physix
    from environments.ycb_reach_to_grasp import YCBReachToGraspEnv

    enable_cameras = bool(getattr(args, "enable_cameras", True))
    carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", enable_cameras)

    env = YCBReachToGraspEnv(device=str(getattr(args, "device", "cuda:0")))
    sim = env.build_simulation()
    if not getattr(args, "headless", False):
        env.set_default_camera_view()
    env.design_scene()
    robot = env.robot

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

    # cameras MUST be built before the first reset (a render product created
    # afterwards hard-crashes Kit, silently — see PLAN.md gotcha 4).
    # Deliberately no try/except around build_camera the way collect_boxes.py
    # has: at rollout a missing view is a train/deploy mismatch, not something
    # to continue past.
    cameras: dict[str, object] = {}
    if enable_cameras:
        for name in camera_names:
            cameras[name] = build_camera(name, robot=robot)

    handles = SimHandles(
        sim=sim,
        env=env,
        robot=robot,
        cameras=cameras,
        dt=float(sim.get_physics_dt()),
        scene_origins=env.scene_origins,
        spawned_paths=list(spawned_paths),
        id_to_label=id_to_label,
        tracker=ObjectsTracker(prim_paths=list(spawned_paths)),
        wrist_cfg=DEFAULT_WRIST_CAMERA,
    )
    reset_sim_and_robot(handles)
    for cam in cameras.values():
        cam.reset()
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


ARM_JOINT_NAMES = [f"j2n6s300_joint_{i}" for i in range(1, 7)]


def set_home_pose(h: SimHandles) -> None:
    """Put the arm at the SAME home pose every demo starts from.

    `reset_sim_and_robot` writes the articulation's default joint state, which
    is NOT the configured home pose the collector starts every episode at. A
    rollout that skips this begins out of distribution, and behaviour cloning
    compounds the error from the first step. Call this BEFORE `controller.reset`
    so the controller anchors its hold state to the correct pose.
    """
    from environments.base import default_jaco2_home_pose

    home = default_jaco2_home_pose()
    set_arm_joint_positions(h, ARM_JOINT_NAMES, [float(home[n]) for n in ARM_JOINT_NAMES])


def make_twist_controller(h: SimHandles, *, ee_link: str = "j2n6s300_end_effector"):
    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController

    # workspace_min x MUST sit below the demos' minimum EE x (0.1647 measured
    # over the boxes_v0 campaign), otherwise the controller fights the policy.
    #
    # The jog controller clamps its IK target to this box EVERY step, including
    # when the command is zero: pos_des = clamp(ee_pos_b + dpos). The robot's
    # home pose is x=0.165, so with the old x_min=0.20 the controller solved IK
    # toward a target 3.5 cm away and drove the arm to the boundary before the
    # policy ever acted -- 4.7 cm of drift at reset, 6.7 cm by the time replay
    # started executing. 10.6% of all demo ticks are below 0.20 and were logged
    # with workspace_clamped_axes[0] = true.
    #
    # Collection is immune because collect_boxes.py drives DifferentialIKController
    # directly with absolute poses and applies NO workspace clamp; the clamp is a
    # teleoperation safety feature that does not belong in the replay path.
    cfg = CartesianVelocityJogConfig(
        ee_link_name=ee_link,
        device=str(h.sim.device),
        use_relative_mode=True,
        workspace_min=(0.15, -0.45, 0.0),
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


def _sensor_rgb(sensor) -> Optional[np.ndarray]:
    """One sensor's latest RGB frame as uint8 HWC, or None. Same normalization
    as collect_boxes.py's _capture_and_log, so training/rollout pixels match."""
    data = sensor.data
    rgb = data.output.get("rgb") if data.output is not None else None
    if rgb is None:
        return None
    arr = rgb[0].cpu().numpy() if rgb.dim() == 4 else rgb.cpu().numpy()
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr


def capture_rgb(h: SimHandles) -> dict[str, np.ndarray]:
    """{camera name: latest RGB frame (uint8 HWC)} for every built camera.

    Raises if a camera has no frame yet: a silently-missing view would be a
    train/deploy mismatch the policy cannot report.
    """
    out = {}
    for name, sensor in h.cameras.items():
        arr = _sensor_rgb(sensor)
        if arr is None:
            raise RuntimeError(f"camera {name!r} returned no frame (was the last step rendered?)")
        out[name] = arr
    return out


def sync_cameras(h: SimHandles) -> None:
    """Pose-update pass that must run BEFORE the render that produces a frame.

    Only the wrist camera needs it, and it needs it EVERY render: the prim is
    unparented (PhysX never pushes link motion into USD), so without this it
    stays frozen at its spawn pose and the "wrist" view is a lie.
    """
    if "wrist" not in h.cameras:
        return
    from environments.utils.camera import sync_wrist_camera_to_ee

    sync_wrist_camera_to_ee(h.robot, h.cameras["wrist"], h.wrist_cfg)


def step_sim(h: SimHandles, controller, *, render: bool) -> None:
    controller.step(h.robot, h.dt)
    if render:
        sync_cameras(h)
    h.sim.step(render=render)
    h.robot.update(h.dt)
    if render:
        for sensor in h.cameras.values():
            sensor.update(h.dt)


_RB_VIEWS: dict[str, object] = {}


def _rigid_view(prim_path: str):
    """Cached PhysX rigid-body view for a spawned box."""
    from isaacsim.core.simulation_manager import SimulationManager

    if prim_path in _RB_VIEWS:
        return _RB_VIEWS[prim_path]
    view = None
    try:
        sim_view = SimulationManager.get_physics_sim_view()
        v = sim_view.create_rigid_body_view(str(prim_path))
        if v.get_transforms().shape[0] == 0:
            v = sim_view.create_rigid_body_view(f"{prim_path}/*")
        view = v if v.get_transforms().shape[0] > 0 else None
    except Exception:
        view = None
    _RB_VIEWS[prim_path] = view
    return view


def box_snapshot(h: SimHandles) -> dict[str, np.ndarray]:
    """{leaf id: world position (3,)} for all spawned boxes, read from PHYSICS.

    ObjectsTracker.snapshot() prefers PhysX but silently falls back to a USD
    XformCache read, which -- as its own comment says -- "may not reflect dynamic
    motion". In this rollout that fallback was being taken, so every box reported
    its SPAWN pose forever: `lift` and total displacement came back as exactly
    0.0000 in every episode of every experiment, including ones where the arm
    demonstrably closed on the box and lifted 21 cm. The success detector was
    blind, which is why nine experiments in a row scored zero.

    collect_boxes.py reads rb.get_world_poses() for exactly this reason ("unlike
    the OBB / USD read, this is always current after a set_world_poses
    teleport"). This does the same through the physics sim view, falling back to
    the tracker only if no rigid-body view exists.
    """
    out: dict[str, np.ndarray] = {}
    for path in h.spawned_paths:
        leaf = str(path).split("/")[-1]
        view = _rigid_view(path)
        if view is not None:
            t = view.get_transforms()  # [x, y, z, qx, qy, qz, qw]
            out[leaf] = np.asarray(
                [float(t[0][0]), float(t[0][1]), float(t[0][2])], dtype=np.float32
            )
    if out:
        return out
    for o in h.tracker.snapshot():  # last resort: whatever the tracker can see
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


class DiffIKDriver:
    """Absolute-pose execution, mirroring collect_boxes.py's proven grasp path.

    The jog controller integrates DELTA twists and, in "twist" mode, derives the
    desired orientation as quat_box_plus(current, drot) -- an integrator with no
    reference, so IK tracking error random-walks the wrist away with nothing
    restoring it. Measured on a scripted expert commanding zero rotation: 0.0007
    from the demo orientation at episode start, 0.145 by the gripper close, 1.84
    by the end, and the box never moved in 40/40 episodes.

    Collection instead drives DifferentialIKController with an ABSOLUTE pose
    target every step (data_collection/collect_boxes.py::_drive_ik_step) and
    holds the orientation constant through a grasp. That path lifts 12/12. This
    class reproduces it so a policy can be evaluated the way its data was made.
    """

    def __init__(self, h: SimHandles, controller, ee_link: str = "j2n6s300_end_effector"):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.h = h
        self.controller = controller
        self.robot = h.robot
        # the jog controller already resolved these against this robot
        self._ee_body_id = int(controller._ee_body_id)
        self._ee_jacobi_idx = int(controller._ee_jacobi_idx)
        self._arm_joint_ids = controller._arm_joint_ids
        self.diff_ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=str(h.sim.device),
        )
        self.diff_ik.reset()

    def step(self, pos_des_b, quat_des_b, *, render: bool = False) -> None:
        """One physics step driving the EE toward an ABSOLUTE base-frame pose."""
        import torch

        robot = self.robot
        jac = robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._arm_joint_ids]
        q_arm = robot.data.joint_pos[:, self._arm_joint_ids]
        pos_b, quat_b = get_ee_pose_b(self.h, self.controller)
        dev = robot.data.joint_pos.device
        p_des = torch.as_tensor(pos_des_b, dtype=torch.float32, device=dev).unsqueeze(0)
        q_des_t = torch.as_tensor(quat_des_b, dtype=torch.float32, device=dev).unsqueeze(0)
        cur_p = torch.as_tensor(pos_b, dtype=torch.float32, device=dev).unsqueeze(0)
        cur_q = torch.as_tensor(quat_b, dtype=torch.float32, device=dev).unsqueeze(0)
        self.diff_ik.ee_pos_des[:] = p_des
        self.diff_ik.ee_quat_des[:] = q_des_t
        q_cmd = self.diff_ik.compute(cur_p, cur_q, jac, q_arm)
        robot.set_joint_position_target(robot.data.joint_pos)
        robot.set_joint_position_target(q_cmd, joint_ids=self._arm_joint_ids)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        try:
            self.controller.gripper.apply_hold(robot)
        except Exception:
            pass
        try:
            robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
        except Exception:
            pass
        robot.write_data_to_sim()
        # Same render bookkeeping as step_sim: the wrist camera is NOT parented
        # to the arm, so it must be synced before every render or it silently
        # returns a static view and the policy sees a frozen wrist image.
        if render:
            sync_cameras(self.h)
        self.h.sim.step(render=render)
        robot.update(self.h.dt)
        if render:
            for sensor in self.h.cameras.values():
                sensor.update(self.h.dt)

    def set_gripper(self, closed: bool) -> None:
        """Command the fingers, the way collect_boxes does (not via the jog controller)."""
        if closed:
            self.controller.gripper.command_close(self.robot)
        else:
            self.controller.gripper.command_open(self.robot)


def make_diffik_driver(h: SimHandles, controller, ee_link: str = "j2n6s300_end_effector") -> DiffIKDriver:
    return DiffIKDriver(h, controller, ee_link)


def world_to_base_pos(h: SimHandles, pos_w) -> np.ndarray:
    """World position -> robot base frame, the same transform collect_boxes uses.

    `scene_origins` is the ENV origin, not the robot base, so subtracting it
    gives a correct xy but a z that is off by the table height -- which is why
    an early analysis reported box heights of 0.811 m. The robot's root pose is
    the authority.
    """
    root = h.robot.data.root_pose_w
    base_pos_w = root[0, 0:3].detach().cpu().numpy().astype(np.float64)
    return np.asarray(pos_w, dtype=np.float64) - base_pos_w
