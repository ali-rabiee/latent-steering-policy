"""Numerical parity of the hand-rolled schedule/samplers vs diffusers.

diffusers is a dev-only dependency; these tests pin the math so the runtime
package can stay dependency-light.
"""

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from lsteer.diffusion import DDIMSampler, DiffusionSchedule


def test_betas_alphas_cumprod_parity():
    sched = DiffusionSchedule(100)
    ref = diffusers.DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    assert torch.allclose(sched.betas, ref.betas, atol=1e-6)
    assert torch.allclose(sched.alphas_cumprod, ref.alphas_cumprod, atol=1e-6)


def test_q_sample_parity():
    sched = DiffusionSchedule(100)
    ref = diffusers.DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(4, 8, 7, generator=g)
    noise = torch.randn(4, 8, 7, generator=g)
    t = torch.tensor([0, 13, 57, 99])
    ours = sched.q_sample(x0, t, noise)
    theirs = ref.add_noise(x0, noise, t)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_ddim_denoising_parity():
    """Full DDIM trajectory with a deterministic dummy eps-model matches diffusers."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(7, 7)

    def model(x, t):
        return lin(x)

    sched = DiffusionSchedule(100)
    sampler = DDIMSampler(sched, clip_sample=True)
    z = torch.randn(2, 8, 7, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        ours = sampler.sample(model, z.shape, z=z.clone(), num_steps=16)

    ref = diffusers.DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        timestep_spacing="leading",
    )
    ref.set_timesteps(16)
    x = z.clone()
    with torch.no_grad():
        for t in ref.timesteps:
            eps = model(x, t)
            x = ref.step(eps, t, x, eta=0.0).prev_sample

    assert torch.allclose(ours, x, atol=1e-5), (ours - x).abs().max()
