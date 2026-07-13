"""Noise schedule for DDPM training / DDIM sampling.

Matches diffusers' `squaredcos_cap_v2` (Nichol & Dhariwal cosine schedule with
betas capped at 0.999) — numerical parity is asserted in tests/test_schedule.py.
"""

from __future__ import annotations

import math

import torch


def betas_squaredcos_cap_v2(num_train_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


class DiffusionSchedule:
    """Precomputed schedule quantities (float32 buffers on CPU, indexable on any device)."""

    def __init__(self, num_train_timesteps: int = 100):
        self.num_train_timesteps = int(num_train_timesteps)
        betas64 = betas_squaredcos_cap_v2(self.num_train_timesteps)
        alphas64 = 1.0 - betas64
        alphas_cumprod64 = torch.cumprod(alphas64, dim=0)

        self.betas = betas64.float()
        self.alphas = alphas64.float()
        self.alphas_cumprod = alphas_cumprod64.float()
        self.sqrt_alphas_cumprod = alphas_cumprod64.sqrt().float()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod64).sqrt().float()

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: x_t = sqrt(acp_t) x0 + sqrt(1-acp_t) eps.

        t: (B,) int64 timesteps; x0/noise: (B, ...).
        """
        shape = (-1,) + (1,) * (x0.dim() - 1)
        sa = self.sqrt_alphas_cumprod.to(x0.device)[t].view(shape)
        so = self.sqrt_one_minus_alphas_cumprod.to(x0.device)[t].view(shape)
        return sa * x0 + so * noise
