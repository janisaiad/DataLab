"""Analysis and visualisation of MCL expert specialisation."""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import ScoreNet
from src.diffusion import get_sigmas, sample_sigma_train, add_noise
from src.utils import get_mnist_loaders, set_seed, save_image_grid
from src.sample import (
    load_mcl, generate_mcl, generate_baseline, load_baseline,
)


@torch.no_grad()
def expert_vs_digit(experts, K, loader, sigma_min, sigma_max, device,
                    num_batches=50):
    """Count expert wins for each digit label.

    Args:
        experts: Sequence of expert models.
        K: Number of experts.
        loader: Data loader that yields image and label batches.
        sigma_min: Minimum noise level used for sampling.
        sigma_max: Maximum noise level used for sampling.
        device: Torch device used for computation.
        num_batches: Maximum number of batches to process.

    Returns:
        A tensor of shape (K, 10) with win counts per expert and digit.
    """
    counts = torch.zeros(K, 10)
    for i, (images, labels) in enumerate(tqdm(loader, desc="Expert vs digit")):
        if i >= num_batches:
            break
        x_0 = images.to(device)
        B = x_0.shape[0]
        sigma = sample_sigma_train(B, sigma_min, sigma_max, device)
        x_t, eps = add_noise(x_0, sigma)

        losses = []
        for k in range(K):
            pred = experts[k](x_t, sigma)
            losses.append((pred - eps).pow(2).sum(dim=(1, 2, 3)))
        losses = torch.stack(losses, dim=1)
        winners = losses.argmin(dim=1).cpu()

        for b in range(B):
            counts[winners[b], labels[b]] += 1

    return counts


def plot_expert_vs_digit(counts, out_path):
    """Save a heatmap of expert usage per digit.

    Args:
        counts: Win counts per expert and digit.
        out_path: Output path for the saved figure.
    """
    K, C = counts.shape
    normed = counts / counts.sum(dim=0, keepdim=True).clamp(min=1)

    fig, ax = plt.subplots(figsize=(8, 0.6 * K + 1))
    im = ax.imshow(normed.numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Expert")
    ax.set_xticks(range(C))
    ax.set_yticks(range(K))
    plt.colorbar(im, ax=ax, label="Fraction of wins")
    ax.set_title("Expert specialisation by digit class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def expert_vs_sigma(experts, K, loader, sigma_min, sigma_max, device,
                    n_bins=20, num_batches=50):
    """Measure expert win frequency across noise-level bins.

    Args:
        experts: Sequence of expert models.
        K: Number of experts.
        loader: Data loader that yields image batches.
        sigma_min: Minimum noise level used for sampling.
        sigma_max: Maximum noise level used for sampling.
        device: Torch device used for computation.
        n_bins: Number of bins used to group noise levels.
        num_batches: Maximum number of batches to process.

    Returns:
        A tuple `(centres, usage)` where `centres` are bin centers and
        `usage` contains per-bin expert win fractions.
    """
    log_edges = np.linspace(np.log(sigma_min), np.log(sigma_max), n_bins + 1)
    centres = np.exp((log_edges[:-1] + log_edges[1:]) / 2)
    usage = np.zeros((n_bins, K))

    for i, (images, _) in enumerate(tqdm(loader, desc="Expert vs sigma")):
        if i >= num_batches:
            break
        x_0 = images.to(device)
        B = x_0.shape[0]
        sigma = sample_sigma_train(B, sigma_min, sigma_max, device)
        x_t, eps = add_noise(x_0, sigma)

        losses = []
        for k in range(K):
            pred = experts[k](x_t, sigma)
            losses.append((pred - eps).pow(2).sum(dim=(1, 2, 3)))
        losses = torch.stack(losses, dim=1)
        winners = losses.argmin(dim=1).cpu().numpy()

        log_s = sigma.log().cpu().numpy()
        bins = np.digitize(log_s, log_edges) - 1
        bins = np.clip(bins, 0, n_bins - 1)
        for b in range(B):
            usage[bins[b], winners[b]] += 1

    row_sums = usage.sum(axis=1, keepdims=True)
    usage = np.divide(usage, row_sums, where=row_sums > 0)
    return centres, usage


def plot_expert_vs_sigma(centres, usage, out_path):
    """Save expert usage curves across noise levels.

    Args:
        centres: Noise bin centers.
        usage: Expert win fractions for each bin.
        out_path: Output path for the saved figure.
    """
    K = usage.shape[1]
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(K):
        ax.plot(centres, usage[:, k], label=f"Expert {k}")
    ax.set_xscale("log")
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("Win fraction")
    ax.set_title("Expert specialisation by noise level")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def same_noise_multi_expert(experts, K, device, sigma_min, sigma_max,
                            num_samples=8, num_steps=200, seed=42):
    """Generate samples per expert from the same initial noise.

    Args:
        experts: Sequence of expert models.
        K: Number of experts.
        device: Torch device used for generation.
        sigma_min: Minimum noise level used for sampling.
        sigma_max: Maximum noise level used for sampling.
        num_samples: Number of samples generated per expert.
        num_steps: Number of denoising steps.
        seed: Random seed to keep initial noise consistent.

    Returns:
        A tensor of generated samples with shape (K, num_samples, C, H, W).
    """
    all_samples = []
    for k in range(K):
        samples = generate_mcl(
            experts, K, strategy="single_expert", expert_id=k,
            num_samples=num_samples, num_steps=num_steps,
            device=device, sigma_min=sigma_min, sigma_max=sigma_max,
            seed=seed,
        )
        all_samples.append(samples)
    return torch.stack(all_samples)


def plot_multi_expert_grid(all_samples, out_path):
    """Save a grid where rows are experts and columns are samples."""
    K, N, C, H, W = all_samples.shape
    grid = all_samples.reshape(K * N, C, H, W)
    save_image_grid(grid, out_path, nrow=N)


def plot_trajectory(traj, out_path, steps_to_show=10):
    """Save snapshots from a single denoising trajectory.

    Args:
        traj: Sequence of intermediate generation states.
        out_path: Output path for the saved figure.
        steps_to_show: Number of evenly spaced steps to display.
    """
    total = len(traj)
    indices = np.linspace(0, total - 1, steps_to_show, dtype=int)
    imgs = torch.stack([traj[i][0] for i in indices])
    save_image_grid(imgs, out_path, nrow=steps_to_show)


def compare_strategies(experts, K, device, sigma_min, sigma_max,
                       num_samples=8, num_steps=200, seed=42, baseline_model=None):
    """Generate samples for each routing strategy with shared initial noise.

    Args:
        experts: Sequence of expert models.
        K: Number of experts.
        device: Torch device used for generation.
        sigma_min: Minimum noise level used for sampling.
        sigma_max: Maximum noise level used for sampling.
        num_samples: Number of samples generated per strategy.
        num_steps: Number of denoising steps.
        seed: Random seed to align initial noise across strategies.
        baseline_model: Optional baseline model for comparison.

    Returns:
        A dictionary mapping strategy names to generated sample tensors.
    """
    results = {}
    if baseline_model is not None:
        results["baseline"] = generate_baseline(
            baseline_model, num_samples=num_samples, num_steps=num_steps,
            device=device, sigma_min=sigma_min, sigma_max=sigma_max,
            seed=seed,
        )
    strategies = ["single_expert", "random_expert", "best_expert", "mixture_score"]
    for s in strategies:
        kwargs = {"strategy": s, "expert_id": 0}
        samples = generate_mcl(
            experts, K, num_samples=num_samples, num_steps=num_steps,
            device=device, sigma_min=sigma_min, sigma_max=sigma_max,
            seed=seed, **kwargs,
        )
        results[s] = samples
    return results


def plot_strategy_comparison(results, out_path):
    """Save a figure comparing outputs across routing strategies.

    Args:
        results: Strategy name to generated sample tensor mapping.
        out_path: Output path for the saved figure.
    """
    names = list(results.keys())
    all_imgs = torch.cat([results[n] for n in names])
    N = results[names[0]].shape[0]

    fig, axes = plt.subplots(len(names), N, figsize=(N * 1.2, len(names) * 1.4))
    for r, name in enumerate(names):
        for c in range(N):
            img = results[name][c, 0].cpu().clamp(-1, 1) * 0.5 + 0.5
            axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(name.replace("_", "\n"), fontsize=8, rotation=0,
                               labelpad=60, va="center")
    plt.suptitle("Strategy comparison (same initial noise)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Run the analysis pipeline and save all output figures."""
    p = argparse.ArgumentParser(description="Analyse MCL expert specialisation")
    p.add_argument("--mcl_ckpt", required=True)
    p.add_argument("--baseline_ckpt", default=None)
    p.add_argument("--out_dir", default="outputs/analysis")
    p.add_argument("--num_batches", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    experts, K, mcl_args = load_mcl(args.mcl_ckpt, device)
    sigma_min, sigma_max = mcl_args["sigma_min"], mcl_args["sigma_max"]
    train_loader, _ = get_mnist_loaders(batch_size=256)

    print("1/5  Expert vs digit heatmap ...")
    counts = expert_vs_digit(experts, K, train_loader, sigma_min, sigma_max,
                             device, args.num_batches)
    plot_expert_vs_digit(counts, os.path.join(args.out_dir, "expert_vs_digit.png"))

    print("2/5  Expert vs sigma ...")
    centres, usage = expert_vs_sigma(experts, K, train_loader, sigma_min, sigma_max,
                                     device, num_batches=args.num_batches)
    plot_expert_vs_sigma(centres, usage, os.path.join(args.out_dir, "expert_vs_sigma.png"))

    print("3/5  Same-noise multi-expert grid ...")
    all_samples = same_noise_multi_expert(
        experts, K, device, sigma_min, sigma_max, num_samples=8, seed=args.seed,
    )
    plot_multi_expert_grid(all_samples, os.path.join(args.out_dir, "multi_expert_grid.png"))

    print("4/5  Denoising trajectory ...")
    samples, traj = generate_mcl(
        experts, K, strategy="single_expert", expert_id=0,
        num_samples=1, num_steps=200, device=device,
        sigma_min=sigma_min, sigma_max=sigma_max, seed=args.seed,
        return_trajectory=True,
    )
    plot_trajectory(traj, os.path.join(args.out_dir, "trajectory.png"))

    print("5/5  Strategy comparison ...")
    baseline_model = None
    if args.baseline_ckpt:
        baseline_model, _ = load_baseline(args.baseline_ckpt, device)
    results = compare_strategies(
        experts, K, device, sigma_min, sigma_max, seed=args.seed,
        baseline_model=baseline_model,
    )
    plot_strategy_comparison(results, os.path.join(args.out_dir, "strategy_comparison.png"))

    print(f"All analysis saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
