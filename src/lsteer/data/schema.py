"""Single source of truth for dataset field names, dims, and conventions.

Data flow: kinova-isaac JSONL+PNG episodes ("data_collection_v0", 5 Hz)
  -> lsteer.data.convert -> zarr (Chi et al. layout: data/* + meta/episode_ends).

State (10): ee_pos_b (3) + ee_rot6d_b (6) + gripper (1, open=+1 / close=-1)
Action (7): ee_delta_pos_b (3) + ee_delta_rotvec_b (3) + gripper_abs (1)
  Action a_t is the delta from tick t to tick t+1 (recomputed from absolute
  poses, NOT the logged `action_from_prev`, which trails by one tick).

Cameras: episodes are logged with one image directory per camera
(`images/<cam>/image_XXXXXX.png`, written by kinova-isaac's
`data_collection/collect_boxes.py`) and each camera gets its own zarr array
`data/img_<cam>` and its own vision encoder. `top_down` is still captured by
some kinova-isaac scripts but is NOT part of the trained observation.
"""

from __future__ import annotations

STATE_DIM = 10
ACTION_DIM = 7
FPS = 5.0

# E2 goal conditioning: target box colour one-hot over MAX_BOXES palette slots
# (ids 0..3 in boxes_v0) + target xy in base frame, from meta/episode_target_*.
GOAL_DIM = 6

IMG_STORE_SIZE = 256  # stored in zarr (resized from 640 with area interpolation)
IMG_CROP_SIZE = 224   # network input (random crop train / center crop eval)

MAX_BOXES = 4

# palette used by kinova-isaac vla_v1 box mode; index = color id in meta arrays
COLOR_PALETTE = ["red", "blue", "yellow", "purple", "orange", "cyan"]

# cameras that make up an observation, in a fixed order (encoder/zarr layout)
CAMERA_NAMES = ("front", "wrist")


def camera_img_key(name: str) -> str:
    """zarr array key for a camera's image stack, e.g. 'front' -> 'data/img_front'."""
    return f"data/img_{name}"


def camera_obs_key(name: str) -> str:
    """Batch/observation dict key for a camera, e.g. 'front' -> 'img_front'."""
    return f"img_{name}"


# zarr keys
DATA_STATE = "data/state"
DATA_ACTION = "data/action"
DATA_JOINTS = "data/joints"
META_EPISODE_ENDS = "meta/episode_ends"
META_EPISODE_SUCCESS = "meta/episode_success"
META_EPISODE_ATTEMPTS = "meta/episode_attempts"
META_TARGET_POS = "meta/episode_target_pos_b"
META_TARGET_COLOR = "meta/episode_target_color"
META_BOX_POSITIONS = "meta/episode_box_positions"
META_BOX_COLORS = "meta/episode_box_colors"

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0


def gripper_state_to_float(state: str) -> float:
    return GRIPPER_OPEN if state == "open" else GRIPPER_CLOSE
