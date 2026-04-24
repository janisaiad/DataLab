"""Training entry point for the baseline diffusion model and MCL variants."""

import argparse
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.model import ScoreNet, ScoringHead
from src.diffusion import sample_sigma_train, add_noise
from src.utils import get_mnist_loaders, save_image_grid, set_seed, EMA
from src.sample import generate_baseline, generate_mcl


def train_baseline(args):
    """Train the single-model baseline diffusion network.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device)
    set_seed(args.seed)

    train_loader, _ = get_mnist_loaders(args.batch_size)
    model = ScoreNet(
        base_ch=args.base_ch,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        time_dim=args.time_dim,
        dropout=args.dropout,
    ).to(device)
    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    log = {"loss": []}

    print(f"Baseline  |  params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in pbar:
            x_0 = images.to(device)
            B = x_0.shape[0]
            sigma = sample_sigma_train(B, args.sigma_min, args.sigma_max, device)
            x_t, eps = add_noise(x_0, sigma)

            eps_pred = model(x_t, sigma)
            loss = nn.functional.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        log["loss"].append(avg_loss)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"  -> avg loss: {avg_loss:.4f}  lr: {lr_now:.2e}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"baseline_ep{epoch}.pt"))

            samples = generate_baseline(
                ema.shadow,
                num_samples=64,
                num_steps=args.preview_steps,
                device=device,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                solver=args.preview_solver,
                seed=args.seed,
            )
            save_image_grid(
                samples, os.path.join(args.out_dir, f"baseline_samples_ep{epoch}.png")
            )

    torch.save(ckpt, os.path.join(args.out_dir, "baseline_final.pt"))
    with open(os.path.join(args.out_dir, "baseline_log.json"), "w") as f:
        json.dump(log, f)
    print("Baseline training complete.")


def _make_experts(args, device):
    """Create K expert ScoreNet instances."""
    experts = nn.ModuleList(
        [
            ScoreNet(
                base_ch=args.base_ch,
                ch_mult=tuple(args.ch_mult),
                num_res_blocks=args.num_res_blocks,
                time_dim=args.time_dim,
                dropout=args.dropout,
            )
            for _ in range(args.K)
        ]
    ).to(device)
    return experts


def _save_mcl_checkpoint(experts, emas, optimizers, args, epoch, tag, extra=None):
    """Save an MCL checkpoint.

    Args:
        experts: Expert models container.
        emas: EMA trackers for experts.
        optimizers: Optimizer states for experts.
        args: Parsed command-line arguments.
        epoch: Current training epoch.
        tag: Suffix used in checkpoint file name.
        extra: Optional extra fields to append to the checkpoint.
    """
    ckpt = {
        "epoch": epoch,
        "experts": experts.state_dict(),
        "emas": [e.state_dict() for e in emas],
        "optimizers": [o.state_dict() for o in optimizers],
        "args": vars(args),
        "mcl_variant": args.mcl_variant,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, os.path.join(args.out_dir, f"mcl_K{args.K}_{tag}.pt"))


def _preview_and_log(experts, emas, args, epoch, log, avg_loss, usage_frac, device):
    """Update logs and save periodic sample previews.

    Args:
        experts: Expert models container.
        emas: EMA trackers for experts.
        args: Parsed command-line arguments.
        epoch: Current training epoch.
        log: Logging dictionary updated in-place.
        avg_loss: Average epoch loss.
        usage_frac: Expert usage fractions.
        device: Torch device used for sampling previews.
    """
    log["loss"].append(avg_loss)
    log["expert_usage"].append(usage_frac)
    print(f"  -> avg loss: {avg_loss:.4f}  usage: {[f'{u:.2f}' for u in usage_frac]}")

    if epoch % args.save_every == 0 or epoch == args.epochs:
        # Preview checkpoints do not need optimizer state, so placeholders are used.
        _save_mcl_checkpoint(experts, emas, [None] * args.K, args, epoch, f"ep{epoch}")
        ema_experts = [emas[k].shadow for k in range(args.K)]
        samples = generate_mcl(
            ema_experts,
            args.K,
            strategy="mixture_score",
            num_samples=64,
            device=device,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
        save_image_grid(
            samples,
            os.path.join(args.out_dir, f"mcl_K{args.K}_samples_ep{epoch}.png"),
        )


def train_mcl_hard_wta(args):
    """Train MCL experts with hard winner-takes-all routing.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device)
    set_seed(args.seed)
    train_loader, _ = get_mnist_loaders(args.batch_size)

    experts = _make_experts(args, device)
    emas = [EMA(experts[k], decay=args.ema_decay) for k in range(args.K)]
    optimizers = [
        torch.optim.Adam(experts[k].parameters(), lr=args.lr) for k in range(args.K)
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    log = {"loss": [], "expert_usage": [], "variant": "hard_wta"}

    total_params = sum(p.numel() for p in experts.parameters())
    print(f"MCL (hard_wta)  |  K={args.K}  |  total params: {total_params:,}")

    for epoch in range(1, args.epochs + 1):
        experts.train()
        epoch_loss = 0.0
        usage = torch.zeros(args.K)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in pbar:
            x_0 = images.to(device)
            B = x_0.shape[0]
            sigma = sample_sigma_train(B, args.sigma_min, args.sigma_max, device)
            x_t, eps = add_noise(x_0, sigma)

            with torch.no_grad():
                # Routing is selected from detached losses to avoid cross-expert gradients.
                losses_det = torch.stack(
                    [
                        (experts[k](x_t, sigma) - eps).pow(2).sum(dim=(1, 2, 3))
                        for k in range(args.K)
                    ],
                    dim=1,
                )
                winners = losses_det.argmin(dim=1)

            batch_loss = 0.0
            for k in range(args.K):
                mask = winners == k
                n = mask.sum().item()
                if n == 0:
                    continue
                optimizers[k].zero_grad()
                eps_pred_k = experts[k](x_t[mask], sigma[mask])
                loss_k = F.mse_loss(eps_pred_k, eps[mask])
                loss_k.backward()
                nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                optimizers[k].step()
                emas[k].update(experts[k])
                batch_loss += loss_k.item() * n
                usage[k] += n

            epoch_loss += batch_loss / B
            pbar.set_postfix(loss=f"{batch_loss / B:.4f}")

        avg_loss = epoch_loss / len(train_loader)
        usage_frac = (usage / usage.sum()).tolist()
        _preview_and_log(experts, emas, args, epoch, log, avg_loss, usage_frac, device)

    _save_mcl_checkpoint(
        experts, emas, optimizers, args, args.epochs, f"K{args.K}_final"
    )
    with open(os.path.join(args.out_dir, f"mcl_K{args.K}_log.json"), "w") as f:
        json.dump(log, f)
    print("MCL (hard_wta) training complete.")


def train_mcl_annealed_wta(args):
    """Train MCL experts with annealed soft-to-hard routing.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device)
    set_seed(args.seed)
    train_loader, _ = get_mnist_loaders(args.batch_size)

    experts = _make_experts(args, device)
    emas = [EMA(experts[k], decay=args.ema_decay) for k in range(args.K)]
    optimizers = [
        torch.optim.Adam(experts[k].parameters(), lr=args.lr) for k in range(args.K)
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    log = {"loss": [], "expert_usage": [], "variant": "annealed_wta"}

    total_params = sum(p.numel() for p in experts.parameters())
    print(f"MCL (annealed_wta)  |  K={args.K}  |  total params: {total_params:,}")

    tau_max = args.anneal_tau_max
    tau_min = args.anneal_tau_min

    for epoch in range(1, args.epochs + 1):
        experts.train()
        epoch_loss = 0.0
        usage = torch.zeros(args.K)

        progress = (epoch - 1) / max(args.epochs - 1, 1)
        # Temperature decreases through training to move from soft to hard routing.
        tau = tau_max * (tau_min / tau_max) ** progress

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} τ={tau:.3f}")
        for images, _ in pbar:
            x_0 = images.to(device)
            B = x_0.shape[0]
            sigma = sample_sigma_train(B, args.sigma_min, args.sigma_max, device)
            x_t, eps = add_noise(x_0, sigma)

            with torch.no_grad():
                losses_det = torch.stack(
                    [
                        (experts[k](x_t, sigma) - eps).pow(2).sum(dim=(1, 2, 3))
                        for k in range(args.K)
                    ],
                    dim=1,
                )

            # Experts are weighted by current soft assignment probabilities.
            weights = F.softmax(-losses_det / (tau + 1e-8), dim=1)
            winners = losses_det.argmin(dim=1)

            batch_loss = 0.0
            for k in range(args.K):
                w_k = weights[:, k]
                if w_k.sum().item() < 1e-8:
                    continue
                optimizers[k].zero_grad()
                eps_pred_k = experts[k](x_t, sigma)
                per_sample_loss = (eps_pred_k - eps).pow(2).mean(dim=(1, 2, 3))
                loss_k = (w_k * per_sample_loss).sum() / (w_k.sum() + 1e-8)
                loss_k.backward()
                nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                optimizers[k].step()
                emas[k].update(experts[k])
                batch_loss += loss_k.item() * B / args.K
                usage[k] += (winners == k).sum().item()

            epoch_loss += batch_loss / B
            pbar.set_postfix(loss=f"{batch_loss / B:.4f}", tau=f"{tau:.3f}")

        avg_loss = epoch_loss / len(train_loader)
        usage_frac = (usage / usage.sum()).tolist()
        _preview_and_log(experts, emas, args, epoch, log, avg_loss, usage_frac, device)

    _save_mcl_checkpoint(
        experts, emas, optimizers, args, args.epochs, f"K{args.K}_final"
    )
    with open(os.path.join(args.out_dir, f"mcl_K{args.K}_log.json"), "w") as f:
        json.dump(log, f)
    print("MCL (annealed_wta) training complete.")


def train_mcl_relaxed_wta(args):
    """Train MCL experts with relaxed winner-takes-all routing.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device)
    set_seed(args.seed)
    train_loader, _ = get_mnist_loaders(args.batch_size)

    experts = _make_experts(args, device)
    emas = [EMA(experts[k], decay=args.ema_decay) for k in range(args.K)]
    optimizers = [
        torch.optim.Adam(experts[k].parameters(), lr=args.lr) for k in range(args.K)
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    log = {"loss": [], "expert_usage": [], "variant": "relaxed_wta"}

    alpha = args.relaxed_alpha
    total_params = sum(p.numel() for p in experts.parameters())
    print(
        f"MCL (relaxed_wta, α={alpha})  |  K={args.K}  |  total params: {total_params:,}"
    )

    for epoch in range(1, args.epochs + 1):
        experts.train()
        epoch_loss = 0.0
        usage = torch.zeros(args.K)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in pbar:
            x_0 = images.to(device)
            B = x_0.shape[0]
            sigma = sample_sigma_train(B, args.sigma_min, args.sigma_max, device)
            x_t, eps = add_noise(x_0, sigma)

            with torch.no_grad():
                # Winner assignment is detached; each expert update uses fixed routing.
                losses_det = torch.stack(
                    [
                        (experts[k](x_t, sigma) - eps).pow(2).sum(dim=(1, 2, 3))
                        for k in range(args.K)
                    ],
                    dim=1,
                )
                winners = losses_det.argmin(dim=1)

            batch_loss = 0.0
            for k in range(args.K):
                mask_win = winners == k
                mask_lose = ~mask_win
                n_win = mask_win.sum().item()
                n_lose = mask_lose.sum().item()

                if n_win == 0 and n_lose == 0:
                    continue

                optimizers[k].zero_grad()
                eps_pred_k = experts[k](x_t, sigma)
                per_sample = (eps_pred_k - eps).pow(2).mean(dim=(1, 2, 3))

                # Winners get full weight, losers keep a smaller gradient via alpha.
                loss_win = per_sample[mask_win].sum() if n_win > 0 else 0.0
                loss_lose = alpha * per_sample[mask_lose].sum() if n_lose > 0 else 0.0
                total_weight = n_win + alpha * n_lose
                loss_k = (loss_win + loss_lose) / (total_weight + 1e-8)

                loss_k.backward()
                nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                optimizers[k].step()
                emas[k].update(experts[k])

                batch_loss += loss_k.item() * n_win if n_win > 0 else 0.0
                usage[k] += n_win

            epoch_loss += batch_loss / B
            pbar.set_postfix(loss=f"{batch_loss / B:.4f}")

        avg_loss = epoch_loss / len(train_loader)
        usage_frac = (usage / max(usage.sum(), 1)).tolist()
        _preview_and_log(experts, emas, args, epoch, log, avg_loss, usage_frac, device)

    _save_mcl_checkpoint(
        experts, emas, optimizers, args, args.epochs, f"K{args.K}_final"
    )
    with open(os.path.join(args.out_dir, f"mcl_K{args.K}_log.json"), "w") as f:
        json.dump(log, f)
    print("MCL (relaxed_wta) training complete.")


def train_mcl_resilient(args):
    """Train resilient MCL with learned per-expert scoring heads.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device)
    set_seed(args.seed)
    train_loader, _ = get_mnist_loaders(args.batch_size)

    experts = _make_experts(args, device)
    scoring_heads = nn.ModuleList([ScoringHead() for _ in range(args.K)]).to(device)

    emas = [EMA(experts[k], decay=args.ema_decay) for k in range(args.K)]
    expert_optimizers = [
        torch.optim.Adam(experts[k].parameters(), lr=args.lr) for k in range(args.K)
    ]
    score_optimizer = torch.optim.Adam(scoring_heads.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    log = {"loss": [], "expert_usage": [], "variant": "resilient_mcl"}

    total_params = sum(p.numel() for p in experts.parameters()) + sum(
        p.numel() for p in scoring_heads.parameters()
    )
    print(f"MCL (resilient)  |  K={args.K}  |  total params: {total_params:,}")

    for epoch in range(1, args.epochs + 1):
        experts.train()
        scoring_heads.train()
        epoch_loss = 0.0
        usage = torch.zeros(args.K)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in pbar:
            x_0 = images.to(device)
            B = x_0.shape[0]
            sigma = sample_sigma_train(B, args.sigma_min, args.sigma_max, device)
            x_t, eps = add_noise(x_0, sigma)

            with torch.no_grad():
                # Scoring heads choose routing; experts are updated only on assigned samples.
                scores = torch.stack(
                    [scoring_heads[k](x_t, sigma) for k in range(args.K)], dim=1
                )
            winners = scores.argmax(dim=1)

            batch_loss = 0.0
            per_expert_errors = []
            for k in range(args.K):
                mask = winners == k
                n = mask.sum().item()
                if n == 0:
                    per_expert_errors.append(torch.zeros(B, device=device))
                    continue
                expert_optimizers[k].zero_grad()
                eps_pred_k = experts[k](x_t[mask], sigma[mask])
                loss_k = F.mse_loss(eps_pred_k, eps[mask])
                loss_k.backward()
                nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                expert_optimizers[k].step()
                emas[k].update(experts[k])
                batch_loss += loss_k.item() * n
                usage[k] += n

                # Keep per-sample expert errors to build a full error tensor for this batch.
                error_full = torch.zeros(B, device=device)
                with torch.no_grad():
                    error_full[mask] = (
                        (eps_pred_k.detach() - eps[mask]).pow(2).mean(dim=(1, 2, 3))
                    )
                per_expert_errors.append(error_full)

            with torch.no_grad():
                actual_errors = torch.stack(per_expert_errors, dim=1)
                # Target labels for scoring heads come from true best experts on this batch.
                all_losses = torch.stack(
                    [
                        (experts[k](x_t, sigma) - eps).pow(2).sum(dim=(1, 2, 3))
                        for k in range(args.K)
                    ],
                    dim=1,
                )
                best_expert = all_losses.argmin(dim=1)

            score_optimizer.zero_grad()
            score_logits = torch.stack(
                [scoring_heads[k](x_t, sigma) for k in range(args.K)], dim=1
            )
            score_loss = F.cross_entropy(score_logits, best_expert)
            score_loss.backward()
            nn.utils.clip_grad_norm_(scoring_heads.parameters(), 1.0)
            score_optimizer.step()

            epoch_loss += batch_loss / B
            pbar.set_postfix(loss=f"{batch_loss / B:.4f}")

        avg_loss = epoch_loss / len(train_loader)
        usage_frac = (usage / max(usage.sum(), 1)).tolist()
        _preview_and_log(experts, emas, args, epoch, log, avg_loss, usage_frac, device)

    _save_mcl_checkpoint(
        experts,
        emas,
        expert_optimizers,
        args,
        args.epochs,
        f"K{args.K}_final",
        extra={"scoring_heads": scoring_heads.state_dict()},
    )
    with open(os.path.join(args.out_dir, f"mcl_K{args.K}_log.json"), "w") as f:
        json.dump(log, f)
    print("MCL (resilient) training complete.")


MCL_VARIANTS = {
    "hard_wta": train_mcl_hard_wta,
    "annealed_wta": train_mcl_annealed_wta,
    "relaxed_wta": train_mcl_relaxed_wta,
    "resilient_mcl": train_mcl_resilient,
}


def train_mcl(args):
    """Dispatch to the selected MCL training variant.

    Args:
        args: Parsed command-line arguments.
    """
    variant = getattr(args, "mcl_variant", "hard_wta")
    if variant not in MCL_VARIANTS:
        raise ValueError(
            f"Unknown MCL variant '{variant}'. Choose from: {list(MCL_VARIANTS.keys())}"
        )
    MCL_VARIANTS[variant](args)


def parse_args():
    """Parse command-line arguments for training scripts."""
    p = argparse.ArgumentParser(description="Train diffusion model (baseline or MCL)")
    p.add_argument("--mode", choices=["baseline", "mcl"], default="baseline")
    p.add_argument("--K", type=int, default=5, help="Number of MCL experts")
    p.add_argument(
        "--mcl_variant",
        choices=["hard_wta", "annealed_wta", "relaxed_wta", "resilient_mcl"],
        default="hard_wta",
        help="MCL training variant (only used when --mode mcl)",
    )
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--preview_steps", type=int, default=200)
    p.add_argument("--preview_solver", choices=["euler", "heun"], default="heun")
    p.add_argument(
        "--anneal_tau_max",
        type=float,
        default=10.0,
        help="Initial (soft) temperature for annealed WTA",
    )
    p.add_argument(
        "--anneal_tau_min",
        type=float,
        default=0.01,
        help="Final (hard) temperature for annealed WTA",
    )
    p.add_argument(
        "--relaxed_alpha",
        type=float,
        default=0.1,
        help="Gradient weight for losing experts in relaxed WTA",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "baseline":
        train_baseline(args)
    else:
        train_mcl(args)
