"""Inference-time routing strategies for multi-expert sampling."""

import argparse
import os
import torch
import torch.nn as nn

from src.model import ScoreNet, GatingNet
from src.diffusion import get_sigmas, euler_sample, heun_sample
from src.utils import save_image_grid, set_seed


def _make_single_expert_fn(experts, expert_id):
    """Always use the same expert."""
    def fn(x, sigma):
        return experts[expert_id](x, sigma)
    return fn


def _make_random_expert_fn(experts, K):
    """Pick a uniformly random expert at each call (= each ODE step)."""
    def fn(x, sigma):
        k = torch.randint(0, K, (1,)).item()
        return experts[k](x, sigma)
    return fn


def _make_best_expert_fn(experts, K):
    """Pick the per-sample prediction with the smallest norm."""
    def fn(x, sigma):
        preds = [experts[k](x, sigma) for k in range(K)]
        norms = torch.stack([p.flatten(1).norm(dim=1) for p in preds], dim=1)
        best = norms.argmin(dim=1)
        preds = torch.stack(preds, dim=1)
        return preds[torch.arange(x.shape[0]), best]
    return fn


def _make_mixture_fn(experts, K):
    """Average noise predictions from all experts (mixture score)."""
    def fn(x, sigma):
        return sum(experts[k](x, sigma) for k in range(K)) / K
    return fn


def _make_gated_fn(experts, gating_net, K):
    """Route each sample to the expert chosen by the gating network."""
    def fn(x, sigma):
        logits = gating_net(x, sigma)
        sel = logits.argmax(dim=1)
        preds = torch.stack([experts[k](x, sigma) for k in range(K)], dim=1)
        return preds[torch.arange(x.shape[0]), sel]
    return fn


@torch.no_grad()
def generate_baseline(model, num_samples=64, num_steps=200, device="cpu",
                      sigma_min=0.01, sigma_max=80.0, solver="heun",
                      seed=None, return_trajectory=False):
    """Generate images with a single diffusion model.

    Args:
        model: Baseline score model.
        num_samples: Number of images to generate.
        num_steps: Number of denoising steps.
        device: Device used for generation.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        solver: ODE solver name (`euler` or `heun`).
        seed: Optional random seed.
        return_trajectory: Whether to return intermediate states.

    Returns:
        Generated samples, and optionally the trajectory.
    """
    model.eval()
    sigmas = get_sigmas(num_steps, sigma_min, sigma_max).to(device)
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(num_samples, 1, 28, 28, device=device) * sigma_max
    fn = euler_sample if solver == "euler" else heun_sample
    samples, traj = fn(model, sigmas, x, return_trajectory=return_trajectory)
    return samples.clamp(-1, 1) if not return_trajectory else (samples.clamp(-1, 1), traj)


@torch.no_grad()
def generate_mcl(experts, K, strategy="single_expert", expert_id=0,
                 gating_net=None, num_samples=64, num_steps=200,
                 device="cpu", sigma_min=0.01, sigma_max=80.0,
                 solver="euler", seed=None, return_trajectory=False):
    """Generate images from MCL experts with a routing strategy.

    Args:
        experts: Sequence of expert models.
        K: Number of experts.
        strategy: Routing strategy name.
        expert_id: Expert index for `single_expert`.
        gating_net: Optional gating model for `gated` strategy.
        num_samples: Number of images to generate.
        num_steps: Number of denoising steps.
        device: Device used for generation.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        solver: ODE solver name (`euler` or `heun`).
        seed: Optional random seed.
        return_trajectory: Whether to return intermediate states.

    Returns:
        Generated samples, and optionally the trajectory.
    """
    for e in experts:
        e.eval()

    if strategy == "single_expert":
        score_fn = _make_single_expert_fn(experts, expert_id)
    elif strategy == "random_expert":
        score_fn = _make_random_expert_fn(experts, K)
    elif strategy == "best_expert":
        score_fn = _make_best_expert_fn(experts, K)
    elif strategy == "mixture_score":
        score_fn = _make_mixture_fn(experts, K)
    elif strategy == "gated":
        assert gating_net is not None, "gated strategy requires a gating_net"
        gating_net.eval()
        score_fn = _make_gated_fn(experts, gating_net, K)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    sigmas = get_sigmas(num_steps, sigma_min, sigma_max).to(device)

    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(num_samples, 1, 28, 28, device=device) * sigma_max

    fn = euler_sample if solver == "euler" else heun_sample
    samples, traj = fn(score_fn, sigmas, x, return_trajectory=return_trajectory)
    return samples.clamp(-1, 1) if not return_trajectory else (samples.clamp(-1, 1), traj)


def load_baseline(ckpt_path, device="cpu"):
    """Load a baseline checkpoint.

    Args:
        ckpt_path: Checkpoint file path.
        device: Device used to map checkpoint tensors.

    Returns:
        A tuple `(model, args_dict)`.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ScoreNet(
        base_ch=a["base_ch"], ch_mult=tuple(a["ch_mult"]),
        num_res_blocks=a["num_res_blocks"], time_dim=a["time_dim"],
        dropout=a.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    return model, a


def load_mcl(ckpt_path, device="cpu"):
    """Load an MCL checkpoint.

    Args:
        ckpt_path: Checkpoint file path.
        device: Device used to map checkpoint tensors.

    Returns:
        A tuple `(experts, K, args_dict)`.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    K = a["K"]
    experts = nn.ModuleList([
        ScoreNet(
            base_ch=a["base_ch"], ch_mult=tuple(a["ch_mult"]),
            num_res_blocks=a["num_res_blocks"], time_dim=a["time_dim"],
            dropout=a.get("dropout", 0.0),
        )
        for _ in range(K)
    ]).to(device)
    if "emas" in ckpt:
        for k in range(K):
            experts[k].load_state_dict(ckpt["emas"][k])
    else:
        experts.load_state_dict(ckpt["experts"])
    experts.eval()
    return experts, K, a


def main():
    """Generate and save samples from baseline or MCL checkpoints."""
    p = argparse.ArgumentParser(description="Generate samples from trained model(s)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["baseline", "mcl"], default="mcl")
    p.add_argument("--strategy", default="single_expert",
                   choices=["single_expert", "random_expert", "best_expert",
                            "mixture_score", "gated"])
    p.add_argument("--expert_id", type=int, default=0)
    p.add_argument("--gating_ckpt", default=None, help="Path to gating net checkpoint")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=200,
                   help="Number of ODE discretisation steps")
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "baseline":
        model, a = load_baseline(args.checkpoint, device)
        samples = generate_baseline(
            model, args.num_samples, args.num_steps, device,
            a["sigma_min"], a["sigma_max"], args.solver, args.seed,
        )
        tag = f"baseline_{args.solver}"
    else:
        experts, K, a = load_mcl(args.checkpoint, device)
        gating_net = None
        if args.strategy == "gated" and args.gating_ckpt:
            g_ckpt = torch.load(args.gating_ckpt, map_location=device, weights_only=False)
            gating_net = GatingNet(K).to(device)
            gating_net.load_state_dict(g_ckpt["gating_net"])

        samples = generate_mcl(
            experts, K, strategy=args.strategy, expert_id=args.expert_id,
            gating_net=gating_net, num_samples=args.num_samples,
            num_steps=args.num_steps, device=device,
            sigma_min=a["sigma_min"], sigma_max=a["sigma_max"],
            solver=args.solver, seed=args.seed,
        )
        tag = f"mcl_K{K}_{args.strategy}"
        if args.strategy == "single_expert":
            tag += f"_e{args.expert_id}"

    path = os.path.join(args.out_dir, f"{tag}_{args.solver}_n{args.num_samples}.png")
    save_image_grid(samples, path)
    print(f"Saved {args.num_samples} samples -> {path}")

    torch.save(samples.cpu(), path.replace(".png", ".pt"))


if __name__ == "__main__":
    main()
