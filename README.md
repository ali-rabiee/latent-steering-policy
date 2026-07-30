# latent-steering-policy

Phase 1 of **Calibrated Latent Steering** (see `../calibrated_latent_steering_concept.md`):
train a **frozen, multimodal diffusion policy** on sim demos collected with
[kinova-isaac](../kinova-isaac) and deploy it closed-loop in Isaac Sim on the
Kinova Jaco2. Later phases (steering encoder, conformal gate) build on the seam
this repo exposes: the DDIM sampler takes the initial noise `z` as a
first-class argument, so the map `(obs, z) -> action chunk` is deterministic
and externally steerable.

## Layout

- `src/lsteer/` — pure-PyTorch package (no Isaac imports in the core):
  - `data/` — JSONL+PNG → zarr converter, dataset, normalizer, shared obs preprocessing
  - `diffusion/` — hand-rolled DDPM schedule + DDIM sampler (parity-tested vs diffusers)
  - `models/` — ResNet18-GN + SpatialSoftmax encoder, FiLM Conditional 1D U-Net, EMA
  - `policy.py` — `DiffusionPolicy` with `predict_action(obs, z=..., k=...)`
  - `training/` — dataclass+YAML config, trainer
  - `isaac/` — Isaac-side runtime helpers (import only under Kit)
- `scripts/` — CLIs; `rollout_sim.py` / `replay_open_loop.py` run under Isaac Lab python
- `configs/boxes_v0.yaml` — the v0 experiment

## Setup

Training env (any CUDA machine):

```bash
pip install -e ".[train,dev]"
pytest            # DDIM/DDPM parity vs diffusers, rotation roundtrips, z-determinism
```

Isaac side: the `kinova` conda env (Isaac Sim + Isaac Lab + kinova-isaac) needs
only the light extras — torch/numpy already ship with Isaac:

```bash
conda run -n kinova pip install "zarr>=2.16,<3" numcodecs
# lsteer itself is imported via sys.path by the scripts; no install needed
```

## Workflow

```bash
# 1. collect demos with kinova-isaac (multi-goal colored boxes, front+wrist cams).
#    NB: the vla_v1 --planner backends stall at pregrasp; collect_boxes.py is
#    the working collector (diff-IK expert motion).
cd ../kinova-isaac && python -u -m data_collection.collect_boxes --headless \
  --num-objects 4 --num-episodes 40 --seed 0 --logs-root logs/boxes_v0

# 2. convert JSONL+PNG -> zarr (successes only by default). One image array per
#    camera; every episode needs an images/<cam>/ dir for each --cameras entry.
python scripts/convert_to_zarr.py \
  --logs-root ../kinova-isaac/logs/boxes_v0 --out data/boxes_v0.zarr \
  --cameras front,wrist

# 3. train (EMA weights are the frozen policy)
python scripts/train.py --config configs/boxes_v0.yaml

# 4. offline multimodality readout (no Isaac): K-z sweep on a val frame
python scripts/eval_offline.py --ckpt outputs/train/<run>/latest.ckpt --zarr data/boxes_v0.zarr

# 5. M6 gate — replay GT demo actions through the twist controller (executor check)
conda run -n kinova python scripts/replay_open_loop.py \
  --episode-dir ../kinova-isaac/logs/boxes_v0/session_X/episode_0000 --headless

# 6. closed-loop eval: success rate + per-box coverage on a fixed layout, z-only variation
conda run -n kinova python scripts/rollout_sim.py \
  --ckpt outputs/train/<run>/latest.ckpt --episodes 50 --headless
```

## Data conventions (single source of truth: `src/lsteer/data/schema.py`)

- 5 Hz policy rate; obs horizon 2, prediction horizon 8, execution horizon 4
- **cameras**: `front` + `wrist` (`schema.CAMERA_NAMES`), each with its own zarr
  array `data/img_<cam>`, its own `VisionEncoder` (no weight sharing) and its own
  random crop. The camera set is stored in the zarr attrs and in the checkpoint,
  and `rollout_sim.py` builds exactly the cameras the checkpoint names. Rollout
  MUST call `runtime.sync_cameras()` before each render or the wrist view
  silently freezes — `step_sim` does this for you
- state (10): `ee_pos_b(3) + ee_rot6d_b(6) + gripper(1)`; action (7):
  `ee_delta_pos_b(3) + ee_delta_rotvec_b(3) + gripper_setpoint(1)`
- actions are recomputed from consecutive absolute poses (the logged
  `action_from_prev` trails by one tick); rotvec deltas are base-frame,
  left-multiplied — the same convention `quat_box_plus` uses in the twist mode
  added to kinova-isaac's `CartesianVelocityJogController`
- crop-only image augmentation — **no color jitter** (box color is the goal)
