#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6 pipeline: MCL diffusion with a margin-aware dynamic router.

This is a drop-in replacement for run_variant.py.  It keeps the 7-stage
training/sampling/evaluation structure, but changes the gating stage and the
routing diagnostics:

  1. Gate labels are collected on a geometric sigma grid, not only at random
     sigmas. This better matches the states seen by the probability-flow ODE.
  2. The gate is trained on hard WTA labels only when the winner is identifiable
     by a positive loss margin. Ambiguous labels are diagnosed instead of being
     forced into arbitrary classes.
  3. The sampler adds a deployable gated_confident strategy: use argmax gate
     only when confidence is high; otherwise fall back to soft mixture, best-live
     expert, expert 0, or best-norm.
  4. It saves router diagnostics: teacher entropy, margin quantiles, confident
     fraction, winner usage, gate accuracy, fallback rate, and mixture-ratio
     summaries.

Expected repo layout: this script is placed next to src/ with the same modules
used by run_variant.py: src.model, src.diffusion, src.utils, src.evaluate, etc.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from src.model import GatingNet, ScoreNet
from src.diffusion import add_noise, sample_sigma_train
from src.utils import EMA, get_mnist_loaders, save_image_grid, set_seed
from src.sample import generate_baseline, generate_mcl
from src.evaluate import (
    compute_fid,
    compute_precision_recall,
    extract_features,
    train_classifier,
)
from src.analyze import (
    compare_strategies,
    expert_vs_digit,
    expert_vs_sigma,
    plot_expert_vs_digit,
    plot_expert_vs_sigma,
    plot_multi_expert_grid,
    plot_strategy_comparison,
    plot_trajectory,
    same_noise_multi_expert,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm_entropy_from_counts(counts: torch.Tensor, eps: float = 1e-12) -> float:
    counts = counts.float()
    if float(counts.sum()) <= 0:
        return 0.0
    p = counts / counts.sum().clamp_min(eps)
    K = p.numel()
    if K <= 1:
        return 0.0
    return float((-(p.clamp_min(eps) * p.clamp_min(eps).log()).sum() / math.log(K)).cpu())


def norm_entropy_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    K = probs.shape[-1]
    if K <= 1:
        return torch.zeros(probs.shape[:-1], device=probs.device, dtype=probs.dtype)
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1) / math.log(K)


def mutual_info_norm(assign: torch.Tensor, labels: torch.Tensor, K: int, C: int = 10, eps: float = 1e-12) -> float:
    a = assign.detach().cpu().long()
    y = labels.detach().cpu().long()
    n = max(1, int(y.numel()))
    joint = torch.zeros(C, K, dtype=torch.float64)
    for c in range(C):
        for k in range(K):
            joint[c, k] = ((y == c) & (a == k)).sum().item()
    joint /= n
    pc = joint.sum(dim=1, keepdim=True)
    pk = joint.sum(dim=0, keepdim=True)
    den = pc @ pk
    mask = joint > 0
    mi = (joint[mask] * (joint[mask] / den[mask].clamp_min(eps)).log()).sum()
    hc = -(pc[pc > 0] * pc[pc > 0].log()).sum()
    hk = -(pk[pk > 0] * pk[pk > 0].log()).sum()
    return float((mi / torch.minimum(hc, hk).clamp_min(eps)).item())


def margin_stats(rel_margin: torch.Tensor) -> Dict[str, float]:
    rel = rel_margin.detach().float().cpu()
    if rel.numel() == 0:
        return {"mean": 0.0, "q10": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q90": 0.0, "max": 0.0}
    qs = torch.quantile(rel, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return {
        "mean": float(rel.mean()),
        "q10": float(qs[0]),
        "q25": float(qs[1]),
        "q50": float(qs[2]),
        "q75": float(qs[3]),
        "q90": float(qs[4]),
        "max": float(rel.max()),
    }


def make_sigma_grid(num_steps: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    """Geometric grid from sigma_max to sigma_min, with a final zero appended."""
    sigmas = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), num_steps, device=device))
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def amp_context(device: torch.device, enabled: bool):
    use = enabled and device.type == "cuda"
    try:
        return autocast("cuda", enabled=use)
    except TypeError:
        return autocast(enabled=use)


# -----------------------------------------------------------------------------
# V6 gating: margin-aware label collection and training
# -----------------------------------------------------------------------------


@dataclass
class GateCollection:
    xt: torch.Tensor
    sigma: torch.Tensor
    winner: torch.Tensor
    digit: torch.Tensor
    rel_margin: torch.Tensor
    entropy_norm: torch.Tensor
    losses: Optional[torch.Tensor]
    threshold: float
    confident_mask: torch.Tensor
    diagnostics: Dict[str, object]


@torch.no_grad()
def _expert_losses(experts: nn.ModuleList, xt: torch.Tensor, sigma: torch.Tensor, eps: torch.Tensor, use_amp: bool) -> torch.Tensor:
    """Return per-sample WTA losses [B, K] using sum of squared noise-prediction errors."""
    device = xt.device
    rows = []
    with amp_context(device, use_amp):
        for expert in experts:
            pred = expert(xt, sigma)
            rows.append((pred - eps).pow(2).sum(dim=(1, 2, 3)))
    return torch.stack(rows, dim=1)


@torch.no_grad()
def collect_gate_labels_v6(
    experts: nn.ModuleList,
    train_loader: DataLoader,
    *,
    K: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    collect_batches: int,
    grid_size: int,
    max_labels: int,
    label_tau: float,
    min_rel_margin: float,
    margin_quantile: float,
    min_confident: int,
    use_amp: bool,
    store_losses: bool = False,
) -> GateCollection:
    """Collect (x_t, sigma, winner) labels for the gate.

    The important difference from the original code is that labels with no clear
    WTA margin are not trusted blindly.  A relative margin is used:

        (loss_2 - loss_1) / (0.5 * (loss_1 + loss_2) + eps).

    """
    experts.eval()
    xs: List[torch.Tensor] = []
    ss: List[torch.Tensor] = []
    ws: List[torch.Tensor] = []
    ds: List[torch.Tensor] = []
    ms: List[torch.Tensor] = []
    hs: List[torch.Tensor] = []
    ls_store: List[torch.Tensor] = []

    if grid_size > 0:
        sigmas_grid = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), grid_size, device=device))
    else:
        sigmas_grid = None

    total = 0
    for batch_idx, (images, digits) in enumerate(train_loader):
        if batch_idx >= collect_batches or total >= max_labels:
            break
        x0 = images.to(device, non_blocking=True)
        digits_cpu = digits.cpu()
        sigma_list: Iterable[Optional[torch.Tensor]]
        if sigmas_grid is None:
            sigma_list = [None]
        else:
            sigma_list = [s for s in sigmas_grid]

        for s in sigma_list:
            if total >= max_labels:
                break
            B = x0.shape[0]
            if s is None:
                sigma = sample_sigma_train(B, sigma_min, sigma_max, device)
            else:
                sigma = torch.full((B,), float(s.item()), device=device)
            xt, eps = add_noise(x0, sigma)
            losses = _expert_losses(experts, xt, sigma, eps, use_amp=use_amp)
            sorted_losses, _ = torch.sort(losses, dim=1)
            winner = losses.argmin(dim=1)
            l1 = sorted_losses[:, 0]
            l2 = sorted_losses[:, 1] if K > 1 else sorted_losses[:, 0]
            rel_margin = (l2 - l1) / (0.5 * (l1 + l2).clamp_min(1e-12))
            q = torch.softmax(-losses / max(label_tau, 1e-8), dim=1)
            entropy = norm_entropy_probs(q)

            remaining = max_labels - total
            take = min(B, remaining)
            xs.append(xt[:take].detach().cpu())
            ss.append(sigma[:take].detach().cpu())
            ws.append(winner[:take].detach().cpu())
            ds.append(digits_cpu[:take].detach().cpu())
            ms.append(rel_margin[:take].detach().cpu())
            hs.append(entropy[:take].detach().cpu())
            if store_losses:
                ls_store.append(losses[:take].detach().cpu())
            total += take

    if not xs:
        raise RuntimeError("No gating labels were collected. Check dataloader and collect_batches.")

    X = torch.cat(xs, dim=0).float()
    S = torch.cat(ss, dim=0).float()
    W = torch.cat(ws, dim=0).long()
    D = torch.cat(ds, dim=0).long()
    M = torch.cat(ms, dim=0).float()
    H = torch.cat(hs, dim=0).float()
    L = torch.cat(ls_store, dim=0).float() if store_losses and ls_store else None

    q_thr = float(torch.quantile(M, torch.tensor(float(margin_quantile))).item()) if M.numel() else 0.0
    threshold = max(float(min_rel_margin), q_thr)
    confident = M >= threshold

    # Avoid a hard crash: if the threshold is too strict, relax it, but record this.
    relaxed = False
    if int(confident.sum()) < min_confident:
        relaxed = True
        # This still keeps the most separated labels, but makes the run diagnostic rather than dead.
        fallback_q = 0.75 if M.numel() >= min_confident else 0.0
        threshold = float(torch.quantile(M, torch.tensor(fallback_q)).item())
        confident = M >= threshold

    counts_all = torch.bincount(W, minlength=K).float()
    counts_conf = torch.bincount(W[confident], minlength=K).float() if confident.any() else torch.zeros(K)

    diagnostics: Dict[str, object] = {
        "num_labels_total": int(W.numel()),
        "num_labels_confident": int(confident.sum()),
        "confident_fraction": float(confident.float().mean()),
        "threshold_rel_margin": float(threshold),
        "threshold_relaxed_due_to_too_few_confident": bool(relaxed),
        "margin_stats": margin_stats(M),
        "teacher_entropy_norm_mean": float(H.mean()),
        "teacher_entropy_norm_q50": float(torch.quantile(H, torch.tensor(0.50))),
        "teacher_entropy_norm_q90": float(torch.quantile(H, torch.tensor(0.90))),
        "winner_usage_all": (counts_all / counts_all.sum().clamp_min(1)).tolist(),
        "winner_usage_confident": (counts_conf / counts_conf.sum().clamp_min(1)).tolist() if counts_conf.sum() > 0 else [0.0] * K,
        "winner_usage_entropy_all": norm_entropy_from_counts(counts_all),
        "winner_usage_entropy_confident": norm_entropy_from_counts(counts_conf),
        "winner_digit_mi_norm_all": mutual_info_norm(W, D, K=K, C=10),
        "winner_digit_mi_norm_confident": mutual_info_norm(W[confident], D[confident], K=K, C=10) if confident.any() else 0.0,
    }

    # A practical status used later by the sampler and the summary.
    tiny_margin_threshold = float(threshold) < 1e-3
    if (
        diagnostics["confident_fraction"] < 0.05
        or diagnostics["teacher_entropy_norm_mean"] > 0.98
        or tiny_margin_threshold
        or relaxed
    ):
        diagnostics["teacher_status"] = "near_uniform_or_low_margin_router_likely_uninformative"
    elif diagnostics["winner_usage_entropy_confident"] < 0.25:
        diagnostics["teacher_status"] = "collapsed_or_single_survivor"
    else:
        diagnostics["teacher_status"] = "usable_margin_signal"

    return GateCollection(
        xt=X,
        sigma=S,
        winner=W,
        digit=D,
        rel_margin=M,
        entropy_norm=H,
        losses=L,
        threshold=threshold,
        confident_mask=confident.cpu(),
        diagnostics=diagnostics,
    )


def train_gating_v6(
    gate_data: GateCollection,
    *,
    K: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    label_smoothing: float,
    train_on: str,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Train GatingNet on confident hard-WTA labels with balanced sampling."""
    if train_on not in {"confident", "all"}:
        raise ValueError("train_on must be 'confident' or 'all'")

    idx_all = torch.arange(gate_data.winner.numel())
    if train_on == "confident" and gate_data.confident_mask.any():
        idx_use = idx_all[gate_data.confident_mask]
    else:
        idx_use = idx_all

    # Train/val split.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    perm = idx_use[torch.randperm(idx_use.numel(), generator=gen)]
    n_val = max(1, int(0.10 * perm.numel())) if perm.numel() > 10 else max(1, min(perm.numel(), 1))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if perm.numel() > n_val else perm

    Xtr, Str, Wtr = gate_data.xt[train_idx], gate_data.sigma[train_idx], gate_data.winner[train_idx]
    Xva, Sva, Wva = gate_data.xt[val_idx], gate_data.sigma[val_idx], gate_data.winner[val_idx]

    counts = torch.bincount(Wtr, minlength=K).float()
    inv = 1.0 / counts.clamp_min(1.0)
    sample_weights = inv[Wtr]
    sampler = WeightedRandomSampler(sample_weights.double(), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(TensorDataset(Xtr, Str, Wtr), batch_size=batch_size, sampler=sampler, drop_last=False)
    val_loader = DataLoader(TensorDataset(Xva, Sva, Wva), batch_size=batch_size, shuffle=False)

    gating = GatingNet(K=K).to(device)
    opt = torch.optim.AdamW(gating.parameters(), lr=lr, weight_decay=1e-4)

    history: Dict[str, List[float]] = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    def eval_loader(loader: DataLoader) -> Tuple[float, float]:
        gating.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for xb, sb, wb in loader:
                xb = xb.to(device, non_blocking=True)
                sb = sb.to(device, non_blocking=True)
                wb = wb.to(device, non_blocking=True)
                logits = gating(xb, sb)
                loss = F.cross_entropy(logits, wb, label_smoothing=0.0)
                loss_sum += float(loss.item()) * xb.shape[0]
                correct += int((logits.argmax(dim=1) == wb).sum().item())
                total += int(xb.shape[0])
        return loss_sum / max(total, 1), correct / max(total, 1)

    for epoch in range(1, epochs + 1):
        gating.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for xb, sb, wb in train_loader:
            xb = xb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            logits = gating(xb, sb)
            loss = F.cross_entropy(logits, wb, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(gating.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            correct += int((logits.argmax(dim=1) == wb).sum().item())
            total += int(xb.shape[0])
        val_loss, val_acc = eval_loader(val_loader)
        history["train_loss"].append(loss_sum / max(total, 1))
        history["train_acc"].append(correct / max(total, 1))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  gate epoch {epoch:3d}/{epochs}  "
                f"train_acc={history['train_acc'][-1]:.3f}  val_acc={val_acc:.3f}  "
                f"train_loss={history['train_loss'][-1]:.4f}"
            )

    # Evaluate on all labels and confident labels separately.
    all_loader = DataLoader(TensorDataset(gate_data.xt, gate_data.sigma, gate_data.winner), batch_size=batch_size, shuffle=False)
    conf_mask = gate_data.confident_mask
    if conf_mask.any():
        conf_loader = DataLoader(
            TensorDataset(gate_data.xt[conf_mask], gate_data.sigma[conf_mask], gate_data.winner[conf_mask]),
            batch_size=batch_size,
            shuffle=False,
        )
        conf_loss, conf_acc = eval_loader(conf_loader)
    else:
        conf_loss, conf_acc = 0.0, 0.0
    all_loss, all_acc = eval_loader(all_loader)

    diag = {
        "gate_train_on": train_on,
        "gate_num_train": int(train_idx.numel()),
        "gate_num_val": int(val_idx.numel()),
        "gate_train_label_usage": (counts / counts.sum().clamp_min(1)).tolist(),
        "gate_final_train_acc": float(history["train_acc"][-1]),
        "gate_final_val_acc": float(history["val_acc"][-1]),
        "gate_acc_all_labels": float(all_acc),
        "gate_ce_all_labels": float(all_loss),
        "gate_acc_confident_labels": float(conf_acc),
        "gate_ce_confident_labels": float(conf_loss),
        "history": history,
    }
    gating.eval()
    return gating, diag


# -----------------------------------------------------------------------------
# V6 sampler: confident gate with fallback
# -----------------------------------------------------------------------------


@torch.no_grad()
def _predict_all_experts(experts: nn.ModuleList, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.stack([expert(x, sigma) for expert in experts], dim=1)


@torch.no_grad()
def generate_mcl_v6(
    experts: nn.ModuleList,
    *,
    K: int,
    strategy: str,
    num_samples: int,
    num_steps: int,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    gating_net: Optional[nn.Module] = None,
    gate_conf_threshold: float = 0.55,
    gate_fallback: str = "softmix",
    default_expert: int = 0,
    sample_batch_size: int = 128,
    seed: int = 0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Generate samples with MCL experts and optional V6 gate.

    strategy options: single_expert, random_expert, best_expert, mixture_score,
    gated, gated_confident, gated_softmix.
    """
    if strategy in {"gated", "gated_confident", "gated_softmix"} and gating_net is None:
        raise ValueError(f"strategy={strategy} requires gating_net")
    if gate_fallback not in {"softmix", "mixture", "best_live", "expert0", "best_norm"}:
        raise ValueError(f"unknown gate_fallback={gate_fallback}")

    experts.eval()
    if gating_net is not None:
        gating_net.eval()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    outputs: List[torch.Tensor] = []
    stats = {
        "strategy": strategy,
        "gate_fallback": gate_fallback,
        "gate_conf_threshold": gate_conf_threshold,
        "default_expert": int(default_expert),
        "num_samples": int(num_samples),
        "num_steps": int(num_steps),
        "gate_usage_counts": [0 for _ in range(K)],
        "fallback_count": 0,
        "total_route_decisions": 0,
        "mean_gate_maxprob": 0.0,
    }

    for start in range(0, num_samples, sample_batch_size):
        B = min(sample_batch_size, num_samples - start)
        x = torch.randn((B, 1, 28, 28), generator=gen, device=device) * sigma_max
        sigmas = make_sigma_grid(num_steps, sigma_min, sigma_max, device)

        for i in range(num_steps):
            s = sigmas[i]
            s_next = sigmas[i + 1]
            sigma_vec = torch.full((B,), float(s.item()), device=device)
            dt = float((s_next - s).item())

            if strategy == "single_expert":
                eps_hat = experts[int(default_expert)](x, sigma_vec)

            elif strategy == "random_expert":
                pred_all = _predict_all_experts(experts, x, sigma_vec)
                chosen = torch.randint(0, K, (B,), generator=gen, device=device)
                idx = chosen.view(B, 1, 1, 1, 1).expand(-1, 1, *pred_all.shape[2:])
                eps_hat = pred_all.gather(1, idx).squeeze(1)

            elif strategy == "best_expert":
                pred_all = _predict_all_experts(experts, x, sigma_vec)
                norms = pred_all.pow(2).flatten(2).sum(dim=2)
                chosen = norms.argmin(dim=1)
                idx = chosen.view(B, 1, 1, 1, 1).expand(-1, 1, *pred_all.shape[2:])
                eps_hat = pred_all.gather(1, idx).squeeze(1)

            elif strategy == "mixture_score":
                pred_all = _predict_all_experts(experts, x, sigma_vec)
                eps_hat = pred_all.mean(dim=1)

            elif strategy in {"gated", "gated_confident", "gated_softmix"}:
                pred_all = _predict_all_experts(experts, x, sigma_vec)
                logits = gating_net(x, sigma_vec)
                probs = torch.softmax(logits, dim=1)
                maxprob, chosen = probs.max(dim=1)
                stats["mean_gate_maxprob"] += float(maxprob.sum().cpu())
                stats["total_route_decisions"] += int(B)
                stats["gate_usage_counts"] = [
                    int(a + b)
                    for a, b in zip(
                        stats["gate_usage_counts"],
                        torch.bincount(chosen.detach().cpu(), minlength=K).tolist(),
                    )
                ]

                if strategy == "gated_softmix":
                    eps_hat = torch.einsum("bk,bkchw->bchw", probs, pred_all)
                else:
                    idx = chosen.view(B, 1, 1, 1, 1).expand(-1, 1, *pred_all.shape[2:])
                    eps_hard = pred_all.gather(1, idx).squeeze(1)
                    if strategy == "gated":
                        eps_hat = eps_hard
                    else:
                        confident = maxprob >= gate_conf_threshold
                        if confident.all():
                            eps_hat = eps_hard
                        else:
                            stats["fallback_count"] += int((~confident).sum().item())
                            if gate_fallback in {"softmix", "mixture"}:
                                eps_fb = torch.einsum("bk,bkchw->bchw", probs, pred_all) if gate_fallback == "softmix" else pred_all.mean(dim=1)
                            elif gate_fallback == "best_live":
                                eps_fb = pred_all[:, int(default_expert)]
                            elif gate_fallback == "expert0":
                                eps_fb = pred_all[:, 0]
                            elif gate_fallback == "best_norm":
                                norms = pred_all.pow(2).flatten(2).sum(dim=2)
                                c_fb = norms.argmin(dim=1)
                                idx_fb = c_fb.view(B, 1, 1, 1, 1).expand(-1, 1, *pred_all.shape[2:])
                                eps_fb = pred_all.gather(1, idx_fb).squeeze(1)
                            else:
                                raise AssertionError(gate_fallback)
                            eps_hat = torch.where(confident.view(B, 1, 1, 1), eps_hard, eps_fb)
            else:
                raise ValueError(f"unknown strategy={strategy}")

            x = x + dt * eps_hat

        outputs.append(x.detach().cpu().clamp(-1.0, 1.0))

    if stats["total_route_decisions"] > 0:
        stats["mean_gate_maxprob"] /= stats["total_route_decisions"]
        stats["gate_usage_fraction"] = (torch.tensor(stats["gate_usage_counts"]).float() / max(1, stats["total_route_decisions"])).tolist()
        stats["fallback_fraction"] = float(stats["fallback_count"] / max(1, stats["total_route_decisions"]))
        stats["gate_usage_entropy"] = norm_entropy_from_counts(torch.tensor(stats["gate_usage_counts"]))
    else:
        stats["gate_usage_fraction"] = [0.0] * K
        stats["fallback_fraction"] = 0.0
        stats["gate_usage_entropy"] = 0.0

    return torch.cat(outputs, dim=0)[:num_samples], stats


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6 MCL diffusion pipeline with margin-aware dynamic gating")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--variant", required=True, choices=["hard_wta", "annealed_wta", "relaxed_wta", "resilient_mcl"])
    parser.add_argument("--tag_suffix", type=str, default="v6")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--baseline_epochs", type=int, default=200)
    parser.add_argument("--mcl_epochs", type=int, default=200)
    parser.add_argument("--gating_epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gate_lr", type=float, default=1e-3)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--force_retrain_baseline", action="store_true")

    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--n_eval", type=int, default=2048)
    parser.add_argument("--sample_batch_size", type=int, default=128)

    # Architecture; defaults match run_variant.py.
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--ch_mult", type=str, default="1 2")
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)

    # WTA variant hyperparameters.
    parser.add_argument("--anneal_tau_max", type=float, default=10.0)
    parser.add_argument("--anneal_tau_min", type=float, default=0.01)
    parser.add_argument("--relaxed_alpha", type=float, default=0.1)

    # V6 gate label collection/training.
    parser.add_argument("--gating_collect_batches", type=int, default=80)
    parser.add_argument("--gating_grid_size", type=int, default=4, help="0 means random sigma only")
    parser.add_argument("--gating_max_labels", type=int, default=80000)
    parser.add_argument("--gate_label_tau", type=float, default=10.0)
    parser.add_argument("--gate_min_rel_margin", type=float, default=0.02)
    parser.add_argument("--gate_margin_quantile", type=float, default=0.50)
    parser.add_argument("--gate_min_confident", type=int, default=2048)
    parser.add_argument("--gate_train_on", choices=["confident", "all"], default="confident")
    parser.add_argument("--gate_label_smoothing", type=float, default=0.02)
    parser.add_argument("--gate_conf_threshold", type=float, default=0.55)
    parser.add_argument("--gate_fallback", choices=["softmix", "mixture", "best_live", "expert0", "best_norm"], default="softmix")

    # Expensive but useful diagnostics.
    parser.add_argument("--eval_each_expert", action="store_true")
    parser.add_argument("--skip_analysis", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled = bool(args.use_amp and device.type == "cuda")

    K = args.K
    ch_mult = tuple(int(z) for z in args.ch_mult.split())
    arch = dict(
        base_ch=args.base_ch,
        ch_mult=ch_mult,
        num_res_blocks=args.num_res_blocks,
        time_dim=args.time_dim,
        dropout=args.dropout,
    )

    tag = f"{args.variant}_K{K}_B{args.batch_size}_{args.tag_suffix}"
    ckpt_dir = ensure_dir(Path("checkpoints") / tag)
    out_dir = ensure_dir(Path("outputs") / tag)
    analysis_dir = ensure_dir(out_dir / "analysis")
    shared_ckpt = ensure_dir(Path("checkpoints") / "shared")

    set_seed(args.seed)
    print(f"[{tag}] device={device}, amp={amp_enabled}")
    if device.type == "cuda":
        print(f"[{tag}] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = get_mnist_loaders(args.batch_size, num_workers=args.num_workers)
    if device.type == "cpu":
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_loader.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    # ------------------------------------------------------------------ Stage 1
    print(f"\n{'=' * 60}\n[{tag}] 1/7 Baseline\n{'=' * 60}")
    baseline_ckpt_path = shared_ckpt / "baseline_final.pt"
    bl_log_path = shared_ckpt / "baseline_log.json"
    if baseline_ckpt_path.exists() and not args.force_retrain_baseline:
        print(f"[{tag}] Reusing shared baseline from {baseline_ckpt_path}")
        ckpt_bl = torch.load(baseline_ckpt_path, map_location=device)
        model = ScoreNet(**arch).to(device)
        model.load_state_dict(ckpt_bl["model"])
        ema = EMA(model, decay=args.ema_decay)
        ema.load_state_dict(ckpt_bl["ema"])
        bl_log = json.load(open(bl_log_path)) if bl_log_path.exists() else {"loss": []}
    else:
        t0 = time.time()
        model = ScoreNet(**arch).to(device)
        ema = EMA(model, decay=args.ema_decay)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = GradScaler(enabled=amp_enabled)
        bl_log = {"loss": []}
        print(f"[{tag}] baseline params={sum(p.numel() for p in model.parameters()):,}")
        for epoch in range(1, args.baseline_epochs + 1):
            model.train()
            total_loss = 0.0
            for images, _ in train_loader:
                x0 = images.to(device, non_blocking=True)
                sigma = sample_sigma_train(x0.shape[0], args.sigma_min, args.sigma_max, device)
                xt, eps = add_noise(x0, sigma)
                with amp_context(device, amp_enabled):
                    loss = F.mse_loss(model(xt, sigma), eps)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                total_loss += float(loss.item())
            avg = total_loss / len(train_loader)
            bl_log["loss"].append(avg)
            if epoch % 5 == 0 or epoch == args.baseline_epochs:
                print(f"[{tag}] baseline epoch {epoch:3d}/{args.baseline_epochs} loss={avg:.4f}")
        ckpt_data = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "args": {**arch, "sigma_min": args.sigma_min, "sigma_max": args.sigma_max, "ch_mult": list(ch_mult)},
        }
        torch.save(ckpt_data, baseline_ckpt_path)
        with open(bl_log_path, "w") as f:
            json.dump(bl_log, f)
        print(f"[{tag}] baseline trained in {(time.time() - t0) / 60:.1f} min")
    baseline_model = copy.deepcopy(ema.shadow).eval()

    # ------------------------------------------------------------------ Stage 2
    print(f"\n{'=' * 60}\n[{tag}] 2/7 MCL training K={K}, variant={args.variant}\n{'=' * 60}")
    t0 = time.time()
    experts = nn.ModuleList([ScoreNet(**arch) for _ in range(K)]).to(device)
    emas_mcl = [EMA(experts[k], decay=args.ema_decay) for k in range(K)]
    opts = [torch.optim.Adam(experts[k].parameters(), lr=args.lr) for k in range(K)]
    scalers = [GradScaler(enabled=amp_enabled) for _ in range(K)]
    print(f"[{tag}] total expert params={sum(p.numel() for p in experts.parameters()):,}")

    if args.variant == "resilient_mcl":
        from src.model import ScoringHead

        scoring_heads = nn.ModuleList([ScoringHead() for _ in range(K)]).to(device)
        score_opt = torch.optim.Adam(scoring_heads.parameters(), lr=args.lr)
    else:
        scoring_heads = None
        score_opt = None

    mcl_log: Dict[str, object] = {"loss": [], "expert_usage": [], "variant": args.variant}
    for epoch in range(1, args.mcl_epochs + 1):
        experts.train()
        if scoring_heads is not None:
            scoring_heads.train()
        usage = torch.zeros(K)
        epoch_loss = 0.0
        if args.variant == "annealed_wta":
            progress = (epoch - 1) / max(args.mcl_epochs - 1, 1)
            tau = args.anneal_tau_max * (args.anneal_tau_min / args.anneal_tau_max) ** progress
        else:
            tau = None

        for images, _ in train_loader:
            x0 = images.to(device, non_blocking=True)
            B = x0.shape[0]
            sigma = sample_sigma_train(B, args.sigma_min, args.sigma_max, device)
            xt, eps = add_noise(x0, sigma)
            with torch.no_grad(), amp_context(device, amp_enabled):
                ls = torch.stack([(experts[k](xt, sigma) - eps).pow(2).sum(dim=(1, 2, 3)) for k in range(K)], dim=1)

            if args.variant == "hard_wta":
                winners = ls.argmin(dim=1)
                batch_loss = 0.0
                for k in range(K):
                    mask = winners == k
                    n = int(mask.sum().item())
                    if n == 0:
                        continue
                    opts[k].zero_grad(set_to_none=True)
                    with amp_context(device, amp_enabled):
                        lk = F.mse_loss(experts[k](xt[mask], sigma[mask]), eps[mask])
                    scalers[k].scale(lk).backward()
                    scalers[k].unscale_(opts[k])
                    nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                    scalers[k].step(opts[k])
                    scalers[k].update()
                    emas_mcl[k].update(experts[k])
                    batch_loss += float(lk.item()) * n
                    usage[k] += n

            elif args.variant == "annealed_wta":
                weights = F.softmax(-ls / (float(tau) + 1e-8), dim=1)
                winners = ls.argmin(dim=1)
                batch_loss = 0.0
                for k in range(K):
                    w = weights[:, k]
                    if float(w.sum().item()) < 1e-8:
                        continue
                    opts[k].zero_grad(set_to_none=True)
                    with amp_context(device, amp_enabled):
                        pred = experts[k](xt, sigma)
                        per = (pred - eps).pow(2).mean(dim=(1, 2, 3))
                        lk = (w * per).sum() / w.sum().clamp_min(1e-8)
                    scalers[k].scale(lk).backward()
                    scalers[k].unscale_(opts[k])
                    nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                    scalers[k].step(opts[k])
                    scalers[k].update()
                    emas_mcl[k].update(experts[k])
                    batch_loss += float(lk.item()) * B / K
                    usage[k] += (winners == k).sum().item()

            elif args.variant == "relaxed_wta":
                winners = ls.argmin(dim=1)
                batch_loss = 0.0
                for k in range(K):
                    win = winners == k
                    lose = ~win
                    n_win = int(win.sum().item())
                    n_lose = int(lose.sum().item())
                    opts[k].zero_grad(set_to_none=True)
                    with amp_context(device, amp_enabled):
                        pred = experts[k](xt, sigma)
                        per = (pred - eps).pow(2).mean(dim=(1, 2, 3))
                        loss_w = per[win].sum() if n_win > 0 else torch.tensor(0.0, device=device)
                        loss_l = args.relaxed_alpha * per[lose].sum() if n_lose > 0 else torch.tensor(0.0, device=device)
                        lk = (loss_w + loss_l) / (n_win + args.relaxed_alpha * n_lose + 1e-8)
                    scalers[k].scale(lk).backward()
                    scalers[k].unscale_(opts[k])
                    nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                    scalers[k].step(opts[k])
                    scalers[k].update()
                    emas_mcl[k].update(experts[k])
                    batch_loss += float(lk.item()) * max(n_win, 1)
                    usage[k] += n_win

            elif args.variant == "resilient_mcl":
                assert scoring_heads is not None and score_opt is not None
                with torch.no_grad():
                    scores = torch.stack([scoring_heads[k](xt, sigma) for k in range(K)], dim=1)
                winners = scores.argmax(dim=1)
                batch_loss = 0.0
                for k in range(K):
                    mask = winners == k
                    n = int(mask.sum().item())
                    if n == 0:
                        continue
                    opts[k].zero_grad(set_to_none=True)
                    with amp_context(device, amp_enabled):
                        lk = F.mse_loss(experts[k](xt[mask], sigma[mask]), eps[mask])
                    scalers[k].scale(lk).backward()
                    scalers[k].unscale_(opts[k])
                    nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                    scalers[k].step(opts[k])
                    scalers[k].update()
                    emas_mcl[k].update(experts[k])
                    batch_loss += float(lk.item()) * n
                    usage[k] += n
                best_expert = ls.argmin(dim=1)
                score_opt.zero_grad(set_to_none=True)
                with amp_context(device, amp_enabled):
                    score_logits = torch.stack([scoring_heads[k](xt, sigma) for k in range(K)], dim=1)
                    s_loss = F.cross_entropy(score_logits, best_expert)
                s_loss.backward()
                nn.utils.clip_grad_norm_(scoring_heads.parameters(), 1.0)
                score_opt.step()
            else:
                raise ValueError(args.variant)

            epoch_loss += batch_loss / max(B, 1)

        avg = epoch_loss / len(train_loader)
        usage_frac = (usage / usage.sum().clamp_min(1)).tolist()
        mcl_log["loss"].append(avg)
        mcl_log["expert_usage"].append(usage_frac)
        if epoch % 5 == 0 or epoch == args.mcl_epochs:
            extra = f" tau={tau:.4f}" if tau is not None else ""
            print(f"[{tag}] MCL epoch {epoch:3d}/{args.mcl_epochs} loss={avg:.4f} usage={[f'{u:.2f}' for u in usage_frac]}{extra}")

    ckpt_mcl = {
        "experts": experts.state_dict(),
        "emas": [e.state_dict() for e in emas_mcl],
        "args": {**arch, "sigma_min": args.sigma_min, "sigma_max": args.sigma_max, "K": K, "ch_mult": list(ch_mult)},
        "mcl_variant": args.variant,
    }
    if scoring_heads is not None:
        ckpt_mcl["scoring_heads"] = scoring_heads.state_dict()
    torch.save(ckpt_mcl, ckpt_dir / f"mcl_K{K}_final.pt")
    with open(ckpt_dir / f"mcl_K{K}_log.json", "w") as f:
        json.dump(mcl_log, f)
    print(f"[{tag}] MCL trained in {(time.time() - t0) / 60:.1f} min")
    ema_experts = nn.ModuleList([emas_mcl[k].shadow for k in range(K)]).eval()

    # ------------------------------------------------------------------ Stage 3
    print(f"\n{'=' * 60}\n[{tag}] 3/7 V6 gating\n{'=' * 60}")
    t0 = time.time()
    gate_data = collect_gate_labels_v6(
        ema_experts,
        train_loader,
        K=K,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        device=device,
        collect_batches=args.gating_collect_batches,
        grid_size=args.gating_grid_size,
        max_labels=args.gating_max_labels,
        label_tau=args.gate_label_tau,
        min_rel_margin=args.gate_min_rel_margin,
        margin_quantile=args.gate_margin_quantile,
        min_confident=args.gate_min_confident,
        use_amp=amp_enabled,
        store_losses=False,
    )
    print(f"[{tag}] gate teacher status: {gate_data.diagnostics['teacher_status']}")
    print(
        f"[{tag}] gate labels: total={gate_data.diagnostics['num_labels_total']} "
        f"confident={gate_data.diagnostics['num_labels_confident']} "
        f"frac={gate_data.diagnostics['confident_fraction']:.3f} "
        f"thr={gate_data.diagnostics['threshold_rel_margin']:.4f}"
    )
    print(f"[{tag}] confident winner usage={gate_data.diagnostics['winner_usage_confident']}")

    # Default expert for best-live fallback = most frequent confident winner; falls back to 0.
    conf_usage = torch.tensor(gate_data.diagnostics["winner_usage_confident"]).float()
    default_expert = int(conf_usage.argmax().item()) if float(conf_usage.sum()) > 0 else 0
    gate_data.diagnostics["default_expert"] = default_expert
    if gate_data.diagnostics.get("teacher_status") != "usable_margin_signal":
        args.gate_conf_threshold = 1.0

    gating, gate_train_diag = train_gating_v6(
        gate_data,
        K=K,
        device=device,
        epochs=args.gating_epochs,
        batch_size=512,
        lr=args.gate_lr,
        label_smoothing=args.gate_label_smoothing,
        train_on=args.gate_train_on,
    )
    gate_diag = {**gate_data.diagnostics, **gate_train_diag}
    torch.save({"gating_net": gating.state_dict(), "K": K, "gate_diag": gate_diag}, ckpt_dir / f"gating_K{K}_v6.pt")
    with open(out_dir / "gating_diagnostics.json", "w") as f:
        json.dump(gate_diag, f, indent=2)
    print(f"[{tag}] V6 gating trained in {(time.time() - t0) / 60:.1f} min")

    # Free memory from collected labels before sampling.
    del gate_data

    # ------------------------------------------------------------------ Stage 4
    print(f"\n{'=' * 60}\n[{tag}] 4/7 Sampling N={args.n_eval}\n{'=' * 60}")
    t0 = time.time()
    gen: Dict[str, torch.Tensor] = {}
    route_stats: Dict[str, object] = {}

    print(f"[{tag}] baseline_euler")
    gen["baseline_euler"] = generate_baseline(baseline_model, args.n_eval, args.num_steps, device, args.sigma_min, args.sigma_max, "euler", args.seed)
    save_image_grid(gen["baseline_euler"][:64], out_dir / "baseline_euler.png")

    print(f"[{tag}] baseline_heun")
    gen["baseline_heun"] = generate_baseline(baseline_model, args.n_eval, args.num_steps, device, args.sigma_min, args.sigma_max, "heun", args.seed)

    # Standard strategies from the report.
    for strat in ["single_expert", "random_expert", "best_expert", "mixture_score", "gated", "gated_softmix", "gated_confident"]:
        print(f"[{tag}] mcl_{strat}")
        imgs, st = generate_mcl_v6(
            ema_experts,
            K=K,
            strategy=strat,
            num_samples=args.n_eval,
            num_steps=args.num_steps,
            device=device,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            gating_net=gating if strat.startswith("gated") else None,
            gate_conf_threshold=args.gate_conf_threshold,
            gate_fallback=args.gate_fallback,
            default_expert=default_expert if strat != "single_expert" else 0,
            sample_batch_size=args.sample_batch_size,
            seed=args.seed,
        )
        gen[strat] = imgs
        route_stats[strat] = st
        save_image_grid(imgs[:64], out_dir / f"mcl_{strat}.png")

    print(f"[{tag}] legacy sampler parity check (single_expert e0)")
    try:
        legacy_single = generate_mcl(
            ema_experts,
            K,
            strategy="single_expert",
            expert_id=0,
            num_samples=args.n_eval,
            num_steps=args.num_steps,
            device=device,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            seed=args.seed,
        )
        if isinstance(legacy_single, tuple):
            legacy_single = legacy_single[0]
        gen["legacy_single_expert_e0"] = legacy_single
        route_stats["legacy_single_expert_e0"] = {"strategy": "legacy_single_expert_e0"}
        save_image_grid(legacy_single[:64], out_dir / "legacy_single_expert_e0.png")
    except Exception as e:
        print(f"[{tag}] legacy sampler parity check failed: {e}")

    if args.eval_each_expert:
        for eid in range(K):
            print(f"[{tag}] single_expert_e{eid}")
            imgs, st = generate_mcl_v6(
                ema_experts,
                K=K,
                strategy="single_expert",
                num_samples=args.n_eval,
                num_steps=args.num_steps,
                device=device,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                default_expert=eid,
                sample_batch_size=args.sample_batch_size,
                seed=args.seed,
            )
            gen[f"single_expert_e{eid}"] = imgs
            route_stats[f"single_expert_e{eid}"] = st
            save_image_grid(imgs[:64], out_dir / f"expert_{eid}_n_eval_grid.png")

    # Always save small per-expert grids.
    for eid in range(K):
        imgs, _ = generate_mcl_v6(
            ema_experts,
            K=K,
            strategy="single_expert",
            num_samples=64,
            num_steps=args.num_steps,
            device=device,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            default_expert=eid,
            sample_batch_size=64,
            seed=args.seed,
        )
        save_image_grid(imgs, out_dir / f"expert_{eid}_grid.png")

    for name, imgs in gen.items():
        torch.save(imgs.cpu(), out_dir / f"{name}.pt")
    with open(out_dir / "router_trace_stats.json", "w") as f:
        json.dump(route_stats, f, indent=2)
    gate_route_diag = {}
    st_gc = route_stats.get("gated_confident", {})
    if isinstance(st_gc, dict):
        usage_frac = st_gc.get("gate_usage_fraction", [])
        gate_entropy = float(st_gc.get("gate_usage_entropy", 0.0))
        mean_maxprob = float(st_gc.get("mean_gate_maxprob", 0.0))
        fallback_fraction = float(st_gc.get("fallback_fraction", 0.0))
        max_usage = max(usage_frac) if usage_frac else 0.0
        gate_route_diag = {
            "gate_usage_max_fraction": float(max_usage),
            "gate_usage_entropy": gate_entropy,
            "mean_gate_maxprob": mean_maxprob,
            "fallback_fraction": fallback_fraction,
        }
        gate_route_diag["gate_status_on_generated_traj"] = (
            "collapsed_to_single_expert_on_generated_traj"
            if (max_usage > 0.98 and gate_entropy < 0.05 and mean_maxprob > 0.95)
            else "not_collapsed_or_mixed"
        )
    with open(out_dir / "gate_route_diagnostics.json", "w") as f:
        json.dump(gate_route_diag, f, indent=2)
    print(f"[{tag}] sampling done in {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------------ Stage 5
    print(f"\n{'=' * 60}\n[{tag}] 5/7 Evaluation\n{'=' * 60}")
    t0 = time.time()
    real_imgs = torch.cat([x for x, _ in test_loader])[: args.n_eval]
    clf = train_classifier(device)
    feats_real = extract_features(clf, real_imgs, device=str(device))

    all_metrics: Dict[str, Dict[str, float]] = {}
    for name, imgs in gen.items():
        feats_gen = extract_features(clf, imgs.cpu()[: args.n_eval], device=str(device))
        fid = compute_fid(feats_real, feats_gen)
        prec, rec = compute_precision_recall(feats_real, feats_gen, k=5)
        all_metrics[name] = {"fid": float(fid), "precision": float(prec), "recall": float(rec)}
        print(f"[{tag}] {name:22s} FID={fid:9.2f} P={prec:.4f} R={rec:.4f}")

    # Mixture-ratio diagnostics.
    ratio_diag: Dict[str, float] = {}
    if "single_expert" in all_metrics and "gated_confident" in all_metrics:
        ratio_diag["single0_fid_over_gated_confident_fid"] = all_metrics["single_expert"]["fid"] / max(all_metrics["gated_confident"]["fid"], 1e-12)
    if "baseline_heun" in all_metrics and "gated_confident" in all_metrics:
        ratio_diag["baseline_heun_fid_over_gated_confident_fid"] = all_metrics["baseline_heun"]["fid"] / max(all_metrics["gated_confident"]["fid"], 1e-12)
    if args.eval_each_expert:
        single_keys = [k for k in all_metrics if k.startswith("single_expert_e")]
        if single_keys:
            best_single = min(single_keys, key=lambda k: all_metrics[k]["fid"])
            ratio_diag["best_single_fid"] = all_metrics[best_single]["fid"]
            ratio_diag["best_single_key"] = best_single  # type: ignore[assignment]
            ratio_diag["best_single_fid_over_gated_confident_fid"] = all_metrics[best_single]["fid"] / max(all_metrics["gated_confident"]["fid"], 1e-12)
    if "legacy_single_expert_e0" in all_metrics and "single_expert" in all_metrics:
        ratio_diag["legacy_single_e0_fid_over_v6_single_e0_fid"] = all_metrics["legacy_single_expert_e0"]["fid"] / max(all_metrics["single_expert"]["fid"], 1e-12)
        ratio_diag["legacy_minus_v6_single_e0_fid"] = all_metrics["legacy_single_expert_e0"]["fid"] - all_metrics["single_expert"]["fid"]

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(out_dir / "v6_ratio_diagnostics.json", "w") as f:
        json.dump(ratio_diag, f, indent=2)
    print(f"[{tag}] evaluation done in {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------------ Stage 6
    if not args.skip_analysis:
        print(f"\n{'=' * 60}\n[{tag}] 6/7 Analysis plots\n{'=' * 60}")
        t0 = time.time()
        try:
            counts = expert_vs_digit(ema_experts, K, train_loader, args.sigma_min, args.sigma_max, device, 30)
            plot_expert_vs_digit(counts, analysis_dir / "expert_vs_digit.png")
            centres, usage_data = expert_vs_sigma(ema_experts, K, train_loader, args.sigma_min, args.sigma_max, device, num_batches=30)
            plot_expert_vs_sigma(centres, usage_data, analysis_dir / "expert_vs_sigma.png")
            all_s = same_noise_multi_expert(ema_experts, K, device, args.sigma_min, args.sigma_max, num_samples=8, num_steps=args.num_steps, seed=args.seed)
            plot_multi_expert_grid(all_s, analysis_dir / "multi_expert_grid.png")
            _, traj = generate_mcl(
                ema_experts,
                K,
                strategy="single_expert",
                expert_id=0,
                num_samples=1,
                num_steps=args.num_steps,
                device=device,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                seed=args.seed,
                return_trajectory=True,
            )
            plot_trajectory(traj, analysis_dir / "trajectory.png")
            results = compare_strategies(ema_experts, K, device, args.sigma_min, args.sigma_max, seed=args.seed, baseline_model=baseline_model)
            plot_strategy_comparison(results, analysis_dir / "strategy_comparison.png")
        except Exception as e:
            print(f"[{tag}] analysis failed but pipeline continues: {e}")
        print(f"[{tag}] analysis done in {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------------ Stage 7
    print(f"\n{'=' * 60}\n[{tag}] 7/7 Summary figures\n{'=' * 60}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, len(bl_log["loss"]) + 1), bl_log["loss"], "b-", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Baseline Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, len(mcl_log["loss"]) + 1), mcl_log["loss"], "r-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title(f"MCL {args.variant} K={K}")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    usage_arr = np.array(mcl_log["expert_usage"])
    if usage_arr.size:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in range(K):
            ax.plot(range(1, len(usage_arr) + 1), usage_arr[:, k], label=f"Expert {k}", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Usage fraction")
        ax.set_title("Expert usage during MCL training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "expert_usage_training.png", dpi=150, bbox_inches="tight")
        plt.close()

    names = list(all_metrics.keys())
    fids = [all_metrics[n]["fid"] for n in names]
    precs = [all_metrics[n]["precision"] for n in names]
    recs = [all_metrics[n]["recall"] for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = range(len(names))
    axes[0].bar(x, fids)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    axes[0].set_ylabel("FID ↓")
    axes[0].set_title("FID")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar(x, precs)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    axes[1].set_ylabel("Precision ↑")
    axes[1].set_title("Precision")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[2].bar(x, recs)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    axes[2].set_ylabel("Recall ↑")
    axes[2].set_title("Recall")
    axes[2].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Text summary for quick SLURM inspection.
    summary = {
        "tag": tag,
        "args": vars(args),
        "gating_diagnostics": gate_diag,
        "router_trace_stats": route_stats,
        "metrics": all_metrics,
        "ratio_diagnostics": ratio_diag,
    }
    with open(out_dir / "SUMMARY_v6.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "SUMMARY_v6.txt", "w") as f:
        f.write(f"V6 MCL diffusion run: {tag}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Gate teacher status: {gate_diag.get('teacher_status')}\n")
        if gate_route_diag:
            f.write(f"Gate route status on generated traj: {gate_route_diag.get('gate_status_on_generated_traj')}\n")
        f.write(f"Confident labels: {gate_diag.get('num_labels_confident')}/{gate_diag.get('num_labels_total')} "
                f"({gate_diag.get('confident_fraction'):.3f})\n")
        f.write(f"Margin threshold: {gate_diag.get('threshold_rel_margin'):.6f}\n")
        f.write(f"Confident winner usage: {gate_diag.get('winner_usage_confident')}\n")
        f.write(f"Gate val acc: {gate_diag.get('gate_final_val_acc'):.4f}; "
                f"confident acc: {gate_diag.get('gate_acc_confident_labels'):.4f}; "
                f"all acc: {gate_diag.get('gate_acc_all_labels'):.4f}\n\n")
        f.write("Metrics:\n")
        for n, m in all_metrics.items():
            f.write(f"  {n:24s} FID={m['fid']:10.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}\n")
        f.write("\nRatio diagnostics:\n")
        for k, v in ratio_diag.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nRouter trace stats:\n")
        for k, v in route_stats.items():
            if k.startswith("gated"):
                f.write(f"  {k}: {v}\n")

    print(f"\n{'=' * 60}")
    print(f"[{tag}] PIPELINE COMPLETE")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Outputs:     {out_dir}")
    print(f"Summary:     {out_dir / 'SUMMARY_v6.txt'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main(parse_args())
