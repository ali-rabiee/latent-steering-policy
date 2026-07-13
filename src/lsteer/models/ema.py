"""Exponential moving average of model weights (Chi-style warmup schedule).

The EMA weights are what gets frozen and shipped as the final policy.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMAModel:
    def __init__(
        self,
        model: nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1.0 - (1.0 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: nn.Module) -> None:
        self.decay = self.get_decay(self.optimization_step)
        ema_params = dict(self.averaged_model.named_parameters())
        for name, param in new_model.named_parameters():
            ema_params[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        ema_buffers = dict(self.averaged_model.named_buffers())
        for name, buf in new_model.named_buffers():
            ema_buffers[name].copy_(buf)
        self.optimization_step += 1

    def state_dict(self):
        return {
            "averaged_model": self.averaged_model.state_dict(),
            "optimization_step": self.optimization_step,
        }

    def load_state_dict(self, sd):
        self.averaged_model.load_state_dict(sd["averaged_model"])
        self.optimization_step = sd["optimization_step"]
