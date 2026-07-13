"""Nested-dataclass config loaded from flat YAML, with dotted-key overrides.

Matches kinova-isaac's argparse+dataclass culture — no hydra. Example:
    train.py --config configs/boxes_v0.yaml --override optim.lr=3e-5 log.logger=none
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    zarr_path: str = "data/boxes_v0.zarr"
    obs_horizon: int = 2
    pred_horizon: int = 8
    act_horizon: int = 4
    crop: int = 224
    val_fraction: float = 0.1
    split_seed: int = 42
    num_workers: int = 4


@dataclass
class ModelConfig:
    img_dim: int = 128
    num_keypoints: int = 32
    down_dims: list[int] = field(default_factory=lambda: [256, 512, 1024])
    kernel_size: int = 5
    diffusion_step_embed_dim: int = 128


@dataclass
class DiffusionConfig:
    train_steps: int = 100
    infer_steps: int = 16
    clip_sample: bool = True


@dataclass
class OptimConfig:
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-6
    batch_size: int = 64
    num_steps: int = 200_000
    warmup_steps: int = 500
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75


@dataclass
class LogConfig:
    out_dir: str = "outputs/train"
    logger: str = "tb"  # tb | wandb | none
    project: str = "lsteer"
    run_name: Optional[str] = None
    log_every: int = 100
    val_every: int = 2500
    ckpt_every: int = 10_000


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 0
    device: str = "cuda"

    def to_dict(self) -> dict:
        return asdict(self)


def _set_dotted(cfg, dotted_key: str, raw_value: str) -> None:
    obj = cfg
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    current = getattr(obj, leaf)
    value = yaml.safe_load(raw_value)
    if current is not None and not isinstance(value, type(current)) and not is_dataclass(current):
        value = type(current)(value)
    setattr(obj, leaf, value)


def _apply_dict(cfg, d: dict) -> None:
    valid = {f.name for f in fields(cfg)}
    for k, v in d.items():
        if k not in valid:
            raise KeyError(f"unknown config key '{k}' for {type(cfg).__name__}")
        current = getattr(cfg, k)
        if is_dataclass(current) and isinstance(v, dict):
            _apply_dict(current, v)
        else:
            setattr(cfg, k, v)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> TrainConfig:
    cfg = TrainConfig()
    if path is not None:
        with open(path) as f:
            _apply_dict(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set_dotted(cfg, key.strip(), val.strip())
    return cfg
