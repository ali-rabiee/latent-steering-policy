"""DDIM sampler with the initial noise `z` as a first-class argument.

This is the architectural seam for latent steering (Phase 2): the map
z -> action trajectory is deterministic when eta=0, and callers may pass their
own z, request K batched samples, or capture intermediate latents.

Timestep spacing follows diffusers' "leading" convention so the schedule is
numerically comparable to `diffusers.DDIMScheduler` (asserted in tests).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from lsteer.diffusion.schedule import DiffusionSchedule


class DDIMSampler:
    def __init__(self, schedule: DiffusionSchedule, clip_sample: bool = True):
        self.schedule = schedule
        self.clip_sample = clip_sample

    def timesteps(self, num_steps: int) -> torch.Tensor:
        T = self.schedule.num_train_timesteps
        if not (1 <= num_steps <= T):
            raise ValueError(f"num_steps must be in [1, {T}], got {num_steps}")
        step_ratio = T // num_steps
        ts = (torch.arange(num_steps) * step_ratio).round().long().flip(0)
        return ts

    @torch.no_grad()
    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        *,
        z: Optional[torch.Tensor] = None,
        num_steps: int = 16,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device | str] = None,
        return_intermediates: bool = False,
    ):
        """Denoise from z (or fresh Gaussian noise) to a sample.

        model: callable (x_t, t_batch) -> eps prediction. Conditioning must be
               closed over by the caller (see DiffusionPolicy.predict_action).
        shape: full sample shape (B, ...). Ignored for z's shape if z is given.
        z:     optional initial noise (B, ...). Returned samples are a pure
               function of (z, conditioning) when eta == 0.
        """
        sched = self.schedule
        if z is None:
            if device is None:
                raise ValueError("device is required when z is None")
            z = torch.randn(shape, generator=generator, device=device)
        x = z.clone()
        device = x.device
        acp = sched.alphas_cumprod.to(device)

        ts = self.timesteps(num_steps).to(device)
        intermediates = [x.clone()] if return_intermediates else None

        for i, t in enumerate(ts):
            t_batch = t.expand(x.shape[0])
            eps = model(x, t_batch)

            acp_t = acp[t]
            acp_prev = acp[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)

            pred_x0 = (x - (1.0 - acp_t).sqrt() * eps) / acp_t.sqrt()
            if self.clip_sample:
                # clip x0 but keep the original eps in the direction term,
                # matching diffusers' default (use_clipped_model_output=False)
                pred_x0 = pred_x0.clamp(-1.0, 1.0)

            sigma = 0.0
            if eta > 0.0:
                sigma = (
                    eta
                    * ((1.0 - acp_prev) / (1.0 - acp_t)).sqrt()
                    * (1.0 - acp_t / acp_prev).sqrt()
                )
            dir_xt = (1.0 - acp_prev - sigma**2).clamp_min(0.0).sqrt() * eps
            x = acp_prev.sqrt() * pred_x0 + dir_xt
            if eta > 0.0:
                x = x + sigma * torch.randn(x.shape, generator=generator, device=device)

            if return_intermediates:
                intermediates.append(x.clone())

        if return_intermediates:
            return x, intermediates
        return x
