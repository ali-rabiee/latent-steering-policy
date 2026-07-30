"""Per-dimension min-max normalizer to [-1, 1], serialized inside checkpoints.

Fitting happens once at train start on the training split; rollout code never
recomputes stats — it loads them from the checkpoint, guaranteeing that train
and deployment use identical transforms.
"""

from __future__ import annotations

import numpy as np
import torch


class LinearNormalizer:
    def __init__(self):
        self.params: dict[str, dict[str, torch.Tensor]] = {}

    def fit(self, data: dict[str, np.ndarray], eps: float = 1e-7) -> None:
        for key, arr in data.items():
            flat = torch.as_tensor(np.asarray(arr), dtype=torch.float32).reshape(-1, arr.shape[-1])
            vmin = flat.min(dim=0).values
            vmax = flat.max(dim=0).values
            rng = vmax - vmin
            degenerate = rng < eps
            scale = torch.where(degenerate, torch.ones_like(rng), 2.0 / rng.clamp_min(eps))
            offset = torch.where(degenerate, -vmin, -1.0 - vmin * scale)
            self.params[key] = {"scale": scale, "offset": offset}

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        p = self.params[key]
        return x * p["scale"].to(x.device) + p["offset"].to(x.device)

    def unnormalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        p = self.params[key]
        return (x - p["offset"].to(x.device)) / p["scale"].to(x.device)

    def state_dict(self) -> dict:
        return {k: {n: t.clone() for n, t in v.items()} for k, v in self.params.items()}

    def load_state_dict(self, sd: dict) -> None:
        self.params = {
            k: {n: torch.as_tensor(t, dtype=torch.float32) for n, t in v.items()}
            for k, v in sd.items()
        }
