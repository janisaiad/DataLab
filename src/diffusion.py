"""Noise schedules, forward process, and ODE samplers for score-based diffusion."""

import torch
import numpy as np


def get_sigmas(num_steps, sigma_min=0.01, sigma_max=80.0):
    """Build the discrete noise schedule for sampling.

    Args:
        num_steps: Number of sampling steps.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.

    Returns:
        A tensor of noise levels of length `num_steps + 1`.
    """
    ramp = torch.linspace(0, 1, num_steps)
    sigmas = sigma_max ** (1 - ramp) * sigma_min**ramp
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def sample_sigma_train(batch_size, sigma_min=0.01, sigma_max=80.0, device="cpu"):
    """Sample training noise levels with a log-uniform distribution.

    Args:
        batch_size: Number of noise levels to sample.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        device: Device where the tensor is created.

    Returns:
        A tensor of sampled noise levels of shape (batch_size,).
    """
    log_sigma = torch.rand(batch_size, device=device) * (
        np.log(sigma_max) - np.log(sigma_min)
    ) + np.log(sigma_min)
    return log_sigma.exp()


def add_noise(x_0, sigma):
    """Add Gaussian noise to clean inputs.

    Args:
        x_0: Clean input tensor.
        sigma: Noise levels per sample.

    Returns:
        A tuple `(x_t, eps)` with noised inputs and sampled noise.
    """
    if sigma.dim() == 1:
        sigma = sigma[:, None, None, None]
    eps = torch.randn_like(x_0)
    return x_0 + sigma * eps, eps


@torch.no_grad()
def euler_sample(model, sigmas, x_init, return_trajectory=False):
    """Sample with Euler integration on the PF-ODE.

    Args:
        model: Callable noise predictor.
        sigmas: Discrete noise schedule.
        x_init: Initial noisy tensor.
        return_trajectory: Whether to keep intermediate states.

    Returns:
        A tuple `(samples, trajectory_or_none)`.
    """
    x = x_init
    traj = [x] if return_trajectory else None

    for i in range(len(sigmas) - 1):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]
        if float(sigma_cur) <= 0:
            break

        eps_pred = model(x, sigma_cur.expand(x.shape[0]).to(x.device))
        x = x + (sigma_next - sigma_cur) * eps_pred

        if return_trajectory:
            traj.append(x)

    return (x, traj) if return_trajectory else (x, None)


@torch.no_grad()
def heun_sample(model, sigmas, x_init, return_trajectory=False):
    """Sample with Heun integration on the PF-ODE.

    Args:
        model: Callable noise predictor.
        sigmas: Discrete noise schedule.
        x_init: Initial noisy tensor.
        return_trajectory: Whether to keep intermediate states.

    Returns:
        A tuple `(samples, trajectory_or_none)`.
    """
    x = x_init
    traj = [x] if return_trajectory else None

    for i in range(len(sigmas) - 1):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]
        if float(sigma_cur) <= 0:
            break

        s = sigma_cur.expand(x.shape[0]).to(x.device)
        d1 = model(x, s)
        x_euler = x + (sigma_next - sigma_cur) * d1

        if float(sigma_next) > 0:
            s_next = sigma_next.expand(x.shape[0]).to(x.device)
            d2 = model(x_euler, s_next)
            x = x + (sigma_next - sigma_cur) * (d1 + d2) / 2
        else:
            x = x_euler

        if return_trajectory:
            traj.append(x)

    return (x, traj) if return_trajectory else (x, None)
