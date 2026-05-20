#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
V7 Bayes-risk routing / beta_x(t) validation for diffusion MCL.

This script is intentionally self-contained and does NOT depend on the broken V6
smoke sampler.  It tests the theoretical pieces directly on MNIST/CIFAR latent
PCA features:

  1. Bayes posterior over classes for the forward channel
       x_t = exp(-t) x_0 + sqrt(1-exp(-2t)) eps
     with scalar within-class variance sigma0^2.

  2. The optimal MSE-energy inverse temperature
       beta_x(t) = d / (2 v_t),
       v_t = 1-exp(-2t) + exp(-2t) sigma0^2,
     when energies are averaged per coordinate.

  3. MCL experts trained with beta_x(t) annealing.

  4. A deployable Bayes-risk router trained on
       q_risk(k|x_t,t) \propto exp(-gamma(t) * E_c[p(c|x_t,t) A_{c,k}(t)])
     rather than on the non-deployable sample-wise oracle q(k|x_t, eps, t).

Outputs in --outdir:
  params.json
  data_info.json
  bayes_params.json
  train_log_experts.csv
  train_log_router.csv
  risk_table.pt
  checkpoint_v7.pt
  beta_sweep_x.csv
  routing_by_t.csv
  README_SUMMARY.md
  *.png if matplotlib is available

Typical runs:

MNIST, fast theory check:
  python scripts/run_variant_v7.py \
    --dataset mnist --classes all --pca-dim 64 --K 4 \
    --n-train 20000 --n-test 5000 --steps 12000 --router-steps 6000 \
    --device cuda --outdir outputs/v7_mnist_pca64_K4

CIFAR automobile/horse:
  python scripts/run_variant_v7.py \
    --dataset cifar10 --classes automobile,horse --pca-dim 128 --K 2 \
    --n-train 8000 --n-test 2000 --steps 15000 --router-steps 8000 \
    --device cuda --outdir outputs/v7_cifar_auto_horse_pca128_K2

CIFAR10 all classes:
  python scripts/run_variant_v7.py \
    --dataset cifar10 --classes all --pca-dim 192 --K 10 \
    --n-train 30000 --n-test 5000 --steps 25000 --router-steps 12000 \
    --device cuda --outdir outputs/v7_cifar10_pca192_K10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
except Exception:  # pragma: no cover
    torchvision = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def norm_entropy_from_counts(counts: torch.Tensor, eps: float = 1e-12) -> float:
    counts = counts.detach().float().cpu()
    if counts.numel() <= 1 or counts.sum() <= 0:
        return 0.0
    p = counts / counts.sum().clamp_min(eps)
    h = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum() / math.log(p.numel())
    return float(h.item())


def norm_entropy_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # returns per-row normalized entropy
    k = probs.shape[-1]
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(-1) / math.log(k)


def kl_categorical(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(-1)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def ramp_multiplier(step: int, warmup: int, ramp: int) -> float:
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return smoothstep((step - warmup) / ramp)


# -----------------------------------------------------------------------------
# CLI params
# -----------------------------------------------------------------------------


@dataclass
class Params:
    # Data
    dataset: str = "mnist"  # mnist | cifar10
    data_root: str = "./data"
    classes: str = "all"
    n_train: int = 20000
    n_test: int = 5000
    n_calib: int = 4096
    pca_dim: int = 64
    no_download: bool = False

    # Diffusion-time grid
    t_min: float = 0.5
    t_max: float = 3.0
    t_grid: int = 11

    # MCL experts
    K: int = 4
    hidden: int = 512
    layers: int = 3
    time_dim: int = 64
    residual_scale: float = 1.0
    steps: int = 12000
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    ramp_steps: int = 3000
    beta_train_mult: float = 1.0
    beta_max: float = 200.0
    grad_clip: float = 1.0
    eval_every: int = 500

    # Risk table / router
    risk_resamples: int = 2
    router_steps: int = 6000
    router_batch_size: int = 256
    router_lr: float = 2e-4
    router_weight_decay: float = 1e-4
    router_hidden: int = 512
    router_layers: int = 3
    router_gamma_mult: float = 1.0
    router_gamma_max: float = 200.0
    router_entropy_floor: float = 0.0
    router_balance_weight: float = 0.0

    # Evaluation
    eval_n: int = 2048
    beta_sweep_min_mult: float = 0.25
    beta_sweep_max_mult: float = 4.0
    beta_sweep_points: int = 41

    # System
    seed: int = 0
    device: str = "auto"
    outdir: str = "outputs/v7_bayes_risk_router"


# -----------------------------------------------------------------------------
# Data loading and PCA latent features
# -----------------------------------------------------------------------------


def parse_classes(dataset: str, classes: str) -> Tuple[List[int], List[str]]:
    classes = classes.strip()
    if dataset == "mnist":
        all_ids = list(range(10))
        all_names = [str(i) for i in range(10)]
    elif dataset == "cifar10":
        all_ids = list(range(10))
        all_names = CIFAR10_CLASSES
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if classes.lower() == "all":
        return all_ids, all_names

    ids: List[int] = []
    for token in [z.strip() for z in classes.split(",") if z.strip()]:
        if token.isdigit():
            ci = int(token)
        else:
            if dataset == "mnist":
                raise ValueError("MNIST classes must be digits or 'all'.")
            ci = CIFAR10_CLASSES.index(token)
        if ci not in all_ids:
            raise ValueError(f"Invalid class id {ci} for {dataset}.")
        ids.append(ci)
    names = [all_names[i] for i in ids]
    return ids, names


def _balanced_take(y: torch.Tensor, class_ids: List[int], n: int, seed: int) -> torch.Tensor:
    # Returns indices into y, balanced as much as possible across selected original labels.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    per = max(1, n // len(class_ids))
    chunks: List[torch.Tensor] = []
    leftovers: List[torch.Tensor] = []
    for c in class_ids:
        idx = torch.where(y == c)[0]
        idx = idx[torch.randperm(idx.numel(), generator=gen)]
        chunks.append(idx[: min(per, idx.numel())])
        if idx.numel() > per:
            leftovers.append(idx[per:])
    out = torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.long)
    if out.numel() < n and leftovers:
        rest = torch.cat(leftovers, dim=0)
        rest = rest[torch.randperm(rest.numel(), generator=gen)]
        out = torch.cat([out, rest[: n - out.numel()]], dim=0)
    out = out[torch.randperm(out.numel(), generator=gen)]
    return out[:n]


def load_raw_dataset(params: Params) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    if torchvision is None:
        raise ImportError("torchvision is required. Install torchvision or run in the project environment.")

    ids, names = parse_classes(params.dataset, params.classes)
    local = {orig: i for i, orig in enumerate(ids)}

    if params.dataset == "mnist":
        tr = torchvision.datasets.MNIST(params.data_root, train=True, download=not params.no_download)
        te = torchvision.datasets.MNIST(params.data_root, train=False, download=not params.no_download)
        Xtr_all = tr.data.float().view(len(tr.data), -1) / 255.0
        ytr_all = tr.targets.long()
        Xte_all = te.data.float().view(len(te.data), -1) / 255.0
        yte_all = te.targets.long()
        raw_shape = [1, 28, 28]
    elif params.dataset == "cifar10":
        tr = torchvision.datasets.CIFAR10(params.data_root, train=True, download=not params.no_download)
        te = torchvision.datasets.CIFAR10(params.data_root, train=False, download=not params.no_download)
        Xtr_all = torch.tensor(tr.data, dtype=torch.float32).permute(0, 3, 1, 2).reshape(len(tr.data), -1) / 255.0
        ytr_all = torch.tensor(tr.targets, dtype=torch.long)
        Xte_all = torch.tensor(te.data, dtype=torch.float32).permute(0, 3, 1, 2).reshape(len(te.data), -1) / 255.0
        yte_all = torch.tensor(te.targets, dtype=torch.long)
        raw_shape = [3, 32, 32]
    else:
        raise ValueError(params.dataset)

    train_idx = _balanced_take(ytr_all, ids, params.n_train, params.seed + 10)
    test_idx = _balanced_take(yte_all, ids, params.n_test, params.seed + 20)

    Xtr = Xtr_all[train_idx]
    ytr_orig = ytr_all[train_idx]
    Xte = Xte_all[test_idx]
    yte_orig = yte_all[test_idx]

    ytr = torch.tensor([local[int(y)] for y in ytr_orig], dtype=torch.long)
    yte = torch.tensor([local[int(y)] for y in yte_orig], dtype=torch.long)

    # Same spirit as your previous scripts: pixelwise centering and global scalar std.
    raw_mean = Xtr.mean(0, keepdim=True)
    raw_std = Xtr.std().clamp_min(1e-6)
    Xtr = (Xtr - raw_mean) / raw_std
    Xte = (Xte - raw_mean) / raw_std

    info = dict(
        dataset=params.dataset,
        classes=names,
        class_original_ids=ids,
        raw_dim=int(Xtr.shape[1]),
        raw_shape=raw_shape,
        train_n=int(Xtr.shape[0]),
        test_n=int(Xte.shape[0]),
        raw_global_std=float(raw_std.item()),
    )
    return Xtr, ytr, Xte, yte, info


@dataclass
class PCAState:
    raw_mean: torch.Tensor
    components: torch.Tensor
    latent_mean: torch.Tensor
    latent_std: torch.Tensor


def build_pca_latent(Xtr: torch.Tensor, Xte: torch.Tensor, pca_dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor, PCAState]:
    del seed  # deterministic enough given torch seed; kept for metadata symmetry.
    q = min(int(pca_dim), Xtr.shape[0] - 1, Xtr.shape[1])
    raw_mean = Xtr.mean(0, keepdim=True)
    Xc = (Xtr - raw_mean).cpu()
    # torch.pca_lowrank is much faster than full SVD on CIFAR.
    _, _, V = torch.pca_lowrank(Xc, q=q, center=False, niter=4)
    components = V[:, :q].contiguous()  # raw_dim x q
    Ztr = (Xtr.cpu() - raw_mean) @ components
    Zte = (Xte.cpu() - raw_mean) @ components
    latent_mean = Ztr.mean(0, keepdim=True)
    latent_std = Ztr.std(0, keepdim=True).clamp_min(1e-6)
    Ztr = (Ztr - latent_mean) / latent_std
    Zte = (Zte - latent_mean) / latent_std
    state = PCAState(raw_mean=raw_mean.cpu(), components=components.cpu(), latent_mean=latent_mean.cpu(), latent_std=latent_std.cpu())
    return Ztr.float(), Zte.float(), state


# -----------------------------------------------------------------------------
# Bayes posterior and beta_x(t)
# -----------------------------------------------------------------------------


@dataclass
class BayesMixture:
    class_means: torch.Tensor  # C x d, latent standardized space
    priors: torch.Tensor       # C
    sigma0_sq: float           # scalar within-class variance per coordinate

    @property
    def C(self) -> int:
        return int(self.class_means.shape[0])

    @property
    def d(self) -> int:
        return int(self.class_means.shape[1])


def fit_bayes_mixture(z: torch.Tensor, labels: torch.Tensor, C: int) -> BayesMixture:
    z = z.cpu()
    labels = labels.cpu().long()
    d = z.shape[1]
    means: List[torch.Tensor] = []
    priors: List[float] = []
    sq_res_sum = 0.0
    count = 0
    for c in range(C):
        mask = labels == c
        if not bool(mask.any()):
            raise RuntimeError(f"Class {c} has no examples.")
        zc = z[mask]
        mc = zc.mean(0)
        means.append(mc)
        priors.append(float(zc.shape[0] / z.shape[0]))
        sq_res_sum += float(((zc - mc) ** 2).sum().item())
        count += int(zc.shape[0] * d)
    sigma0_sq = max(sq_res_sum / max(1, count), 1e-8)
    return BayesMixture(
        class_means=torch.stack(means, dim=0).float(),
        priors=torch.tensor(priors, dtype=torch.float32),
        sigma0_sq=float(sigma0_sq),
    )


def a_t(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-t)


def b_t(t: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((1.0 - torch.exp(-2.0 * t)).clamp_min(1e-12))


def v_t(t: torch.Tensor, sigma0_sq: float) -> torch.Tensor:
    return (1.0 - torch.exp(-2.0 * t)).clamp_min(1e-12) + torch.exp(-2.0 * t) * float(sigma0_sq)


def beta_x_t(t: torch.Tensor, d: int, sigma0_sq: float, beta_max: float = 200.0) -> torch.Tensor:
    # Energies are MSE per coordinate: E_c = ||x_t - a_t m_c||^2 / d.
    beta = float(d) / (2.0 * v_t(t, sigma0_sq))
    if beta_max is not None and beta_max > 0:
        beta = beta.clamp_max(beta_max)
    return beta


def diffuse(x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x0: B x d, t: B or scalar tensor.
    if t.ndim == 0:
        t = t.expand(x0.shape[0])
    aa = a_t(t).view(-1, 1)
    bb = b_t(t).view(-1, 1)
    eps = torch.randn_like(x0)
    xt = aa * x0 + bb * eps
    return xt, eps


def bayes_class_posterior(xt: torch.Tensor, t: torch.Tensor, mix: BayesMixture, beta_max: float = 200.0) -> torch.Tensor:
    # p(c|x_t) under isotropic class-conditional Gaussian model.
    # Uses MSE energies and beta_x(t)=d/(2v_t).
    means = mix.class_means.to(xt.device)
    priors = mix.priors.to(xt.device)
    if t.ndim == 0:
        t = t.expand(xt.shape[0])
    aa = a_t(t).view(-1, 1, 1)
    energy_mse = ((xt[:, None, :] - aa * means[None, :, :]) ** 2).mean(-1)
    beta = beta_x_t(t, mix.d, mix.sigma0_sq, beta_max=beta_max).view(-1, 1)
    logits = priors.clamp_min(1e-12).log().view(1, -1) - beta * energy_mse
    return torch.softmax(logits, dim=1)


def class_energy_mse(xt: torch.Tensor, t: torch.Tensor, mix: BayesMixture) -> torch.Tensor:
    means = mix.class_means.to(xt.device)
    if t.ndim == 0:
        t = t.expand(xt.shape[0])
    aa = a_t(t).view(-1, 1, 1)
    return ((xt[:, None, :] - aa * means[None, :, :]) ** 2).mean(-1)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim == 0:
        t = t.view(1)
    half = dim // 2
    if half <= 0:
        return t[:, None]
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(1, half - 1))
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int, dropout: float = 0.0):
        super().__init__()
        layers = max(1, int(layers))
        mods: List[nn.Module] = []
        cur = in_dim
        for _ in range(layers):
            mods.append(nn.Linear(cur, hidden))
            mods.append(nn.SiLU())
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
            cur = hidden
        mods.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MCLDenoiser(nn.Module):
    """Residualized K-expert denoiser: eps_hat_k = base(x_t,t) + residual_k(x_t,t)."""

    def __init__(self, d: int, K: int, hidden: int, layers: int, time_dim: int, residual_scale: float = 1.0):
        super().__init__()
        self.d = int(d)
        self.K = int(K)
        self.time_dim = int(time_dim)
        self.residual_scale = float(residual_scale)
        in_dim = d + time_dim
        trunk_dim = hidden
        self.trunk = MLP(in_dim, hidden, trunk_dim, layers=max(1, layers - 1))
        self.base = nn.Linear(trunk_dim, d)
        self.residual = nn.Linear(trunk_dim, K * d)
        nn.init.zeros_(self.base.bias)
        nn.init.zeros_(self.residual.bias)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(xt.shape[0])
        emb = sinusoidal_time_embedding(t, self.time_dim)
        h = self.trunk(torch.cat([xt, emb], dim=1))
        base = self.base(h)
        res = self.residual(h).view(xt.shape[0], self.K, self.d)
        return base[:, None, :] + self.residual_scale * res


class RiskRouter(nn.Module):
    def __init__(self, d: int, K: int, hidden: int, layers: int, time_dim: int):
        super().__init__()
        self.d = int(d)
        self.K = int(K)
        self.time_dim = int(time_dim)
        self.net = MLP(d + time_dim, hidden, K, layers=layers)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(xt.shape[0])
        emb = sinusoidal_time_embedding(t, self.time_dim)
        return self.net(torch.cat([xt, emb], dim=1))


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------


def mcl_softmin_loss(costs: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # costs: B x K, beta: B
    K = costs.shape[1]
    beta = beta.view(-1)
    q = torch.empty_like(costs)
    loss_i = torch.empty(costs.shape[0], device=costs.device, dtype=costs.dtype)
    small = beta <= 1e-8
    if bool(small.any()):
        q[small] = 1.0 / K
        loss_i[small] = costs[small].mean(1)
    if bool((~small).any()):
        b = beta[~small].view(-1, 1)
        logits = -b * costs[~small]
        q[~small] = torch.softmax(logits, dim=1)
        loss_i[~small] = -(torch.logsumexp(logits, dim=1) - math.log(K)) / beta[~small]
    return loss_i.mean(), q


def soft_ce(target_probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return -(target_probs.detach() * torch.log_softmax(logits, dim=1)).sum(1).mean()


# -----------------------------------------------------------------------------
# Training experts
# -----------------------------------------------------------------------------


def sample_times(n: int, params: Params, device: torch.device, grid_only: bool = False) -> torch.Tensor:
    if grid_only:
        grid = torch.linspace(params.t_min, params.t_max, params.t_grid, device=device)
        idx = torch.randint(0, grid.numel(), (n,), device=device)
        return grid[idx]
    return params.t_min + (params.t_max - params.t_min) * torch.rand(n, device=device)


@torch.no_grad()
def eval_experts_snapshot(model: MCLDenoiser, z: torch.Tensor, mix: BayesMixture, params: Params, device: torch.device, n: int = 1024) -> Dict[str, float]:
    model.eval()
    idx = torch.randint(0, z.shape[0], (min(n, z.shape[0]),), device=device)
    x0 = z[idx]
    t = sample_times(x0.shape[0], params, device, grid_only=False)
    xt, eps = diffuse(x0, t)
    pred = model(xt, t)
    costs = ((pred - eps[:, None, :]) ** 2).mean(-1)
    beta = beta_x_t(t, mix.d, mix.sigma0_sq, beta_max=params.beta_max) * params.beta_train_mult
    _, q = mcl_softmin_loss(costs, beta)
    gap = costs.mean(1) - costs.min(1).values
    return dict(
        val_best_mse=float(costs.min(1).values.mean().item()),
        val_mean_mse=float(costs.mean(1).mean().item()),
        val_cost_gap_mean=float(gap.mean().item()),
        val_beta_gap=float((beta * gap).mean().item()),
        val_teacher_entropy=float(norm_entropy_probs(q).mean().item()),
        val_winner_usage_entropy=norm_entropy_from_counts(torch.bincount(costs.argmin(1).detach().cpu(), minlength=model.K)),
        beta_mean=float(beta.mean().item()),
        beta_min=float(beta.min().item()),
        beta_max=float(beta.max().item()),
    )


def train_experts(z_train: torch.Tensor, z_val: torch.Tensor, mix: BayesMixture, params: Params, device: torch.device, out: Path) -> Tuple[MCLDenoiser, List[Dict[str, object]]]:
    model = MCLDenoiser(
        d=mix.d,
        K=params.K,
        hidden=params.hidden,
        layers=params.layers,
        time_dim=params.time_dim,
        residual_scale=params.residual_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    z_train = z_train.to(device)
    z_val = z_val.to(device)
    rows: List[Dict[str, object]] = []

    for step in range(params.steps + 1):
        if step % params.eval_every == 0 or step == params.steps:
            snap = eval_experts_snapshot(model, z_val, mix, params, device)
            snap.update(step=step, beta_ramp=ramp_multiplier(step, params.warmup_steps, params.ramp_steps))
            rows.append(snap)
            write_csv(out / "train_log_experts.csv", rows)
            print(
                f"[experts {step:06d}] best={snap['val_best_mse']:.5f} "
                f"mean={snap['val_mean_mse']:.5f} beta_gap={snap['val_beta_gap']:.4g} "
                f"Hq={snap['val_teacher_entropy']:.3f}",
                flush=True,
            )
        if step == params.steps:
            break

        model.train()
        idx = torch.randint(0, z_train.shape[0], (params.batch_size,), device=device)
        x0 = z_train[idx]
        t = sample_times(params.batch_size, params, device, grid_only=False)
        xt, eps = diffuse(x0, t)
        pred = model(xt, t)
        costs = ((pred - eps[:, None, :]) ** 2).mean(-1)
        beta = beta_x_t(t, mix.d, mix.sigma0_sq, beta_max=params.beta_max)
        beta = beta * params.beta_train_mult * ramp_multiplier(step, params.warmup_steps, params.ramp_steps)
        loss, _ = mcl_softmin_loss(costs, beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if params.grad_clip and params.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
        opt.step()

    return model, rows


# -----------------------------------------------------------------------------
# Risk table and router target
# -----------------------------------------------------------------------------


def t_grid_tensor(params: Params, device: torch.device) -> torch.Tensor:
    return torch.linspace(params.t_min, params.t_max, params.t_grid, device=device)


def nearest_t_indices(t: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    # t: B, grid: T
    return torch.argmin((t[:, None] - grid[None, :]).abs(), dim=1)


@torch.no_grad()
def compute_risk_table(
    model: MCLDenoiser,
    z_calib: torch.Tensor,
    y_calib: torch.Tensor,
    params: Params,
    device: torch.device,
    out: Path,
) -> torch.Tensor:
    """Returns A[t_index, class, expert] = E[cost_k | class, t]."""
    model.eval()
    z_calib = z_calib.to(device)
    y_calib = y_calib.to(device).long()
    grid = t_grid_tensor(params, device)
    C = int(y_calib.max().item()) + 1
    A = torch.zeros(grid.numel(), C, params.K, device=device)
    counts = torch.zeros(grid.numel(), C, device=device)

    bs = max(64, min(1024, params.batch_size * 2))
    for ti, tt in enumerate(grid):
        for _ in range(max(1, params.risk_resamples)):
            for lo in range(0, z_calib.shape[0], bs):
                hi = min(z_calib.shape[0], lo + bs)
                x0 = z_calib[lo:hi]
                labels = y_calib[lo:hi]
                t = tt.expand(x0.shape[0])
                xt, eps = diffuse(x0, t)
                pred = model(xt, t)
                costs = ((pred - eps[:, None, :]) ** 2).mean(-1)
                for c in range(C):
                    mask = labels == c
                    if bool(mask.any()):
                        A[ti, c] += costs[mask].sum(0)
                        counts[ti, c] += int(mask.sum().item())

    A = A / counts.clamp_min(1.0)[:, :, None]
    # Fill any empty cells with global expert mean at that t.
    for ti in range(A.shape[0]):
        global_mean = A[ti][counts[ti] > 0].mean(0) if bool((counts[ti] > 0).any()) else A[ti].mean(0)
        for c in range(C):
            if counts[ti, c] <= 0:
                A[ti, c] = global_mean

    torch.save(dict(A=A.detach().cpu(), t_grid=grid.detach().cpu(), counts=counts.detach().cpu()), out / "risk_table.pt")
    return A.detach()


def risk_targets(
    xt: torch.Tensor,
    t: torch.Tensor,
    mix: BayesMixture,
    A: torch.Tensor,
    grid: torch.Tensor,
    params: Params,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns q_risk, expected_risk, p_class."""
    pc = bayes_class_posterior(xt, t, mix, beta_max=params.beta_max)
    tidx = nearest_t_indices(t, grid)
    A_batch = A[tidx]  # B x C x K
    risk = torch.einsum("bc,bck->bk", pc, A_batch)
    gamma = beta_x_t(t, mix.d, mix.sigma0_sq, beta_max=params.router_gamma_max) * params.router_gamma_mult
    q = torch.softmax(-gamma[:, None] * risk, dim=1)
    if params.router_entropy_floor > 0:
        # Optional smoothing; useful if the risk target gets too hard too early.
        eps = params.router_entropy_floor
        q = (1.0 - eps) * q + eps / q.shape[1]
    return q, risk, pc


# -----------------------------------------------------------------------------
# Router training/eval
# -----------------------------------------------------------------------------


@torch.no_grad()
def eval_router_snapshot(
    model: MCLDenoiser,
    router: RiskRouter,
    z: torch.Tensor,
    mix: BayesMixture,
    A: torch.Tensor,
    params: Params,
    device: torch.device,
    n: int = 1024,
) -> Dict[str, float]:
    model.eval()
    router.eval()
    z = z.to(device)
    grid = t_grid_tensor(params, device)
    idx = torch.randint(0, z.shape[0], (min(n, z.shape[0]),), device=device)
    x0 = z[idx]
    t = sample_times(x0.shape[0], params, device, grid_only=False)
    xt, eps = diffuse(x0, t)
    pred = model(xt, t)
    costs = ((pred - eps[:, None, :]) ** 2).mean(-1)
    beta = beta_x_t(t, mix.d, mix.sigma0_sq, beta_max=params.beta_max)
    q_eps = torch.softmax(-beta[:, None] * costs, dim=1)
    q_risk, risk, _ = risk_targets(xt, t, mix, A, grid, params)
    rp = torch.softmax(router(xt, t), dim=1)
    r_best = risk.min(1).values
    r_router_soft = (rp * risk).sum(1)
    r_router_hard = risk.gather(1, rp.argmax(1, keepdim=True)).squeeze(1)
    return dict(
        router_kl_qrisk=float(kl_categorical(q_risk, rp).mean().item()),
        eps_to_risk_kl=float(kl_categorical(q_eps, q_risk).mean().item()),
        eps_to_router_kl=float(kl_categorical(q_eps, rp).mean().item()),
        router_excess_soft=float((r_router_soft - r_best).mean().item()),
        router_excess_hard=float((r_router_hard - r_best).mean().item()),
        router_acc_to_risk=float((rp.argmax(1) == risk.argmin(1)).float().mean().item()),
        eps_acc_to_risk=float((costs.argmin(1) == risk.argmin(1)).float().mean().item()),
        router_usage_entropy=norm_entropy_from_counts(torch.bincount(rp.argmax(1).detach().cpu(), minlength=params.K)),
        risk_usage_entropy=norm_entropy_from_counts(torch.bincount(risk.argmin(1).detach().cpu(), minlength=params.K)),
        eps_usage_entropy=norm_entropy_from_counts(torch.bincount(costs.argmin(1).detach().cpu(), minlength=params.K)),
        qrisk_entropy=float(norm_entropy_probs(q_risk).mean().item()),
        qeps_entropy=float(norm_entropy_probs(q_eps).mean().item()),
    )


def train_router(
    model: MCLDenoiser,
    z_train: torch.Tensor,
    z_val: torch.Tensor,
    mix: BayesMixture,
    A: torch.Tensor,
    params: Params,
    device: torch.device,
    out: Path,
) -> Tuple[RiskRouter, List[Dict[str, object]]]:
    router = RiskRouter(mix.d, params.K, params.router_hidden, params.router_layers, params.time_dim).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=params.router_lr, weight_decay=params.router_weight_decay)
    z_train = z_train.to(device)
    z_val = z_val.to(device)
    A = A.to(device)
    grid = t_grid_tensor(params, device)
    rows: List[Dict[str, object]] = []
    model.eval()

    for step in range(params.router_steps + 1):
        if step % params.eval_every == 0 or step == params.router_steps:
            snap = eval_router_snapshot(model, router, z_val, mix, A, params, device)
            snap.update(step=step)
            rows.append(snap)
            write_csv(out / "train_log_router.csv", rows)
            print(
                f"[router  {step:06d}] KL(qrisk||r)={snap['router_kl_qrisk']:.4g} "
                f"excess={snap['router_excess_soft']:.4g} acc={snap['router_acc_to_risk']:.3f} "
                f"Huse={snap['router_usage_entropy']:.3f}",
                flush=True,
            )
        if step == params.router_steps:
            break

        router.train()
        idx = torch.randint(0, z_train.shape[0], (params.router_batch_size,), device=device)
        x0 = z_train[idx]
        t = sample_times(params.router_batch_size, params, device, grid_only=False)
        xt, _ = diffuse(x0, t)
        q_risk, _, _ = risk_targets(xt, t, mix, A, grid, params)
        logits = router(xt, t)
        loss = soft_ce(q_risk, logits)
        if params.router_balance_weight > 0:
            probs = torch.softmax(logits, dim=1)
            usage = probs.mean(0)
            loss = loss + params.router_balance_weight * ((usage - 1.0 / params.K) ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if params.grad_clip and params.grad_clip > 0:
            nn.utils.clip_grad_norm_(router.parameters(), params.grad_clip)
        opt.step()
    return router, rows


# -----------------------------------------------------------------------------
# Theory evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def beta_sweep_x(
    z: torch.Tensor,
    labels: torch.Tensor,
    mix: BayesMixture,
    params: Params,
    device: torch.device,
) -> List[Dict[str, object]]:
    z = z.to(device)
    labels = labels.to(device).long()
    n = min(params.eval_n, z.shape[0])
    idx = torch.randperm(z.shape[0], device=device)[:n]
    x0 = z[idx]
    y = labels[idx]
    grid = t_grid_tensor(params, device)
    factors = torch.exp(
        torch.linspace(
            math.log(params.beta_sweep_min_mult),
            math.log(params.beta_sweep_max_mult),
            params.beta_sweep_points,
            device=device,
        )
    )
    rows: List[Dict[str, object]] = []
    priors_log = mix.priors.to(device).clamp_min(1e-12).log().view(1, -1)
    for tt in grid:
        t = tt.expand(n)
        xt, _ = diffuse(x0, t)
        E = class_energy_mse(xt, t, mix)  # n x C
        beta_theory = beta_x_t(t, mix.d, mix.sigma0_sq, beta_max=1e12)
        nlls = []
        accs = []
        for f in factors:
            logits = priors_log - (f * beta_theory).view(-1, 1) * E
            nll = F.cross_entropy(logits, y, reduction="mean")
            acc = (logits.argmax(1) == y).float().mean()
            nlls.append(float(nll.item()))
            accs.append(float(acc.item()))
        j = int(np.argmin(nlls))
        # Theory point uses f=1 exactly, even if grid does not hit it.
        logits_th = priors_log - beta_theory.view(-1, 1) * E
        nll_th = float(F.cross_entropy(logits_th, y, reduction="mean").item())
        acc_th = float((logits_th.argmax(1) == y).float().mean().item())
        pc_th = torch.softmax(logits_th, dim=1)
        class_oracle_gap = float((-pc_th.gather(1, y.view(-1, 1)).clamp_min(1e-12).log()).mean().item())
        rows.append(
            dict(
                t=float(tt.item()),
                beta_theory=float(beta_theory.mean().item()),
                beta_emp=float((factors[j] * beta_theory.mean()).item()),
                beta_emp_over_theory=float(factors[j].item()),
                nll_theory=nll_th,
                nll_emp=float(nlls[j]),
                nll_gap_theory_minus_emp=float(nll_th - nlls[j]),
                acc_theory=acc_th,
                acc_emp=float(accs[j]),
                class_oracle_to_px_kl=class_oracle_gap,
                v_t=float(v_t(tt, mix.sigma0_sq).item()),
            )
        )
    return rows


@torch.no_grad()
def routing_eval_by_t(
    model: MCLDenoiser,
    router: RiskRouter,
    z: torch.Tensor,
    labels: torch.Tensor,
    mix: BayesMixture,
    A: torch.Tensor,
    params: Params,
    device: torch.device,
) -> List[Dict[str, object]]:
    model.eval()
    router.eval()
    z = z.to(device)
    labels = labels.to(device).long()
    n = min(params.eval_n, z.shape[0])
    idx = torch.randperm(z.shape[0], device=device)[:n]
    x0 = z[idx]
    y = labels[idx]
    grid = t_grid_tensor(params, device)
    A = A.to(device)
    rows: List[Dict[str, object]] = []

    for tt in grid:
        t = tt.expand(n)
        xt, eps = diffuse(x0, t)
        pred = model(xt, t)
        costs = ((pred - eps[:, None, :]) ** 2).mean(-1)
        beta = beta_x_t(t, mix.d, mix.sigma0_sq, beta_max=params.beta_max)
        q_eps = torch.softmax(-beta[:, None] * costs, dim=1)
        q_risk, risk, pc = risk_targets(xt, t, mix, A, grid, params)
        rp = torch.softmax(router(xt, t), dim=1)

        risk_best = risk.min(1).values
        risk_router_soft = (rp * risk).sum(1)
        risk_router_hard = risk.gather(1, rp.argmax(1, keepdim=True)).squeeze(1)
        risk_qrisk = (q_risk * risk).sum(1)
        risk_qeps = (q_eps * risk).sum(1)
        sample_best = costs.min(1).values
        sample_router_soft = (rp * costs).sum(1)
        sample_router_hard = costs.gather(1, rp.argmax(1, keepdim=True)).squeeze(1)

        rows.append(
            dict(
                t=float(tt.item()),
                beta_x=float(beta.mean().item()),
                v_t=float(v_t(tt, mix.sigma0_sq).item()),
                kl_qeps_to_qrisk=float(kl_categorical(q_eps, q_risk).mean().item()),
                kl_qeps_to_router=float(kl_categorical(q_eps, rp).mean().item()),
                kl_qrisk_to_router=float(kl_categorical(q_risk, rp).mean().item()),
                route_risk_oracle=float(risk_best.mean().item()),
                route_risk_qrisk=float(risk_qrisk.mean().item()),
                route_risk_qeps=float(risk_qeps.mean().item()),
                route_risk_router_soft=float(risk_router_soft.mean().item()),
                route_risk_router_hard=float(risk_router_hard.mean().item()),
                route_risk_router_soft_excess_vs_oracle=float((risk_router_soft - risk_best).mean().item()),
                route_risk_router_hard_excess_vs_oracle=float((risk_router_hard - risk_best).mean().item()),
                sample_oracle_mse=float(sample_best.mean().item()),
                sample_router_soft_mse=float(sample_router_soft.mean().item()),
                sample_router_hard_mse=float(sample_router_hard.mean().item()),
                sample_router_soft_excess_vs_oracle=float((sample_router_soft - sample_best).mean().item()),
                sample_router_hard_excess_vs_oracle=float((sample_router_hard - sample_best).mean().item()),
                router_acc_to_risk_oracle=float((rp.argmax(1) == risk.argmin(1)).float().mean().item()),
                qeps_acc_to_risk_oracle=float((costs.argmin(1) == risk.argmin(1)).float().mean().item()),
                router_acc_to_sample_oracle=float((rp.argmax(1) == costs.argmin(1)).float().mean().item()),
                px_acc_to_label=float((pc.argmax(1) == y).float().mean().item()),
                qeps_entropy=float(norm_entropy_probs(q_eps).mean().item()),
                qrisk_entropy=float(norm_entropy_probs(q_risk).mean().item()),
                router_entropy=float(norm_entropy_probs(rp).mean().item()),
                qeps_usage_entropy=norm_entropy_from_counts(torch.bincount(costs.argmin(1).detach().cpu(), minlength=params.K)),
                qrisk_usage_entropy=norm_entropy_from_counts(torch.bincount(risk.argmin(1).detach().cpu(), minlength=params.K)),
                router_usage_entropy=norm_entropy_from_counts(torch.bincount(rp.argmax(1).detach().cpu(), minlength=params.K)),
                class_oracle_to_px_kl=float((-pc.gather(1, y.view(-1, 1)).clamp_min(1e-12).log()).mean().item()),
            )
        )
    return rows


# -----------------------------------------------------------------------------
# Plotting/reporting
# -----------------------------------------------------------------------------


def _mean(rows: List[Dict[str, object]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return float(np.mean(vals)) if vals else float("nan")


def _max(rows: List[Dict[str, object]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return float(np.max(vals)) if vals else float("nan")


def make_plots(out: Path, beta_rows: List[Dict[str, object]], route_rows: List[Dict[str, object]]) -> None:
    if plt is None:
        return
    try:
        if beta_rows:
            t = [r["t"] for r in beta_rows]
            ratio = [r["beta_emp_over_theory"] for r in beta_rows]
            plt.figure()
            plt.plot(t, ratio, marker="o")
            plt.axhline(1.0, linestyle="--")
            plt.xlabel("t")
            plt.ylabel("beta_emp / beta_theory")
            plt.tight_layout()
            plt.savefig(out / "beta_emp_over_theory.png", dpi=180)
            plt.close()

            plt.figure()
            plt.plot(t, [r["class_oracle_to_px_kl"] for r in beta_rows], marker="o")
            plt.xlabel("t")
            plt.ylabel("-log p(true class | x_t)")
            plt.tight_layout()
            plt.savefig(out / "class_oracle_to_px_kl.png", dpi=180)
            plt.close()

        if route_rows:
            t = [r["t"] for r in route_rows]
            plt.figure()
            plt.plot(t, [r["route_risk_router_soft_excess_vs_oracle"] for r in route_rows], marker="o")
            plt.xlabel("t")
            plt.ylabel("router soft excess risk")
            plt.tight_layout()
            plt.savefig(out / "router_excess_risk.png", dpi=180)
            plt.close()

            plt.figure()
            plt.plot(t, [r["kl_qeps_to_qrisk"] for r in route_rows], marker="o", label="q_eps -> q_risk")
            plt.plot(t, [r["kl_qrisk_to_router"] for r in route_rows], marker="o", label="q_risk -> router")
            plt.xlabel("t")
            plt.ylabel("KL")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "routing_kls.png", dpi=180)
            plt.close()

            plt.figure()
            plt.plot(t, [r["qeps_usage_entropy"] for r in route_rows], marker="o", label="sample oracle")
            plt.plot(t, [r["qrisk_usage_entropy"] for r in route_rows], marker="o", label="risk oracle")
            plt.plot(t, [r["router_usage_entropy"] for r in route_rows], marker="o", label="router")
            plt.xlabel("t")
            plt.ylabel("usage entropy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "routing_usage_entropy.png", dpi=180)
            plt.close()
    except Exception as e:  # pragma: no cover
        print(f"[plot warning] {e}", flush=True)


def write_summary(
    out: Path,
    params: Params,
    data_info: Dict[str, object],
    mix: BayesMixture,
    beta_rows: List[Dict[str, object]],
    route_rows: List[Dict[str, object]],
) -> None:
    lines: List[str] = []
    lines.append("# V7 Bayes-risk router report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- dataset: `{params.dataset}`")
    lines.append(f"- classes: `{data_info['classes']}`")
    lines.append(f"- latent dimension: `{mix.d}`")
    lines.append(f"- C: `{mix.C}`")
    lines.append(f"- K experts: `{params.K}`")
    lines.append(f"- t range: `{params.t_min}` to `{params.t_max}` with `{params.t_grid}` grid points")
    lines.append(f"- sigma0_sq estimated: `{mix.sigma0_sq:.6g}`")
    lines.append("")
    lines.append("## Consolidated diagnostics")
    lines.append(f"- beta_emp_over_theory mean: `{_mean(beta_rows, 'beta_emp_over_theory'):.6g}`")
    lines.append(f"- beta_emp_over_theory max abs error: `{_max([{'x': abs(float(r['beta_emp_over_theory']) - 1.0)} for r in beta_rows], 'x'):.6g}`")
    lines.append(f"- class_oracle_to_px_kl mean: `{_mean(beta_rows, 'class_oracle_to_px_kl'):.6g}`")
    lines.append(f"- class_oracle_to_px_kl max: `{_max(beta_rows, 'class_oracle_to_px_kl'):.6g}`")
    lines.append(f"- KL(q_eps || q_risk) mean: `{_mean(route_rows, 'kl_qeps_to_qrisk'):.6g}`")
    lines.append(f"- KL(q_risk || router) mean: `{_mean(route_rows, 'kl_qrisk_to_router'):.6g}`")
    lines.append(f"- router soft excess Bayes risk mean: `{_mean(route_rows, 'route_risk_router_soft_excess_vs_oracle'):.6g}`")
    lines.append(f"- router hard excess Bayes risk mean: `{_mean(route_rows, 'route_risk_router_hard_excess_vs_oracle'):.6g}`")
    lines.append(f"- router acc to risk oracle mean: `{_mean(route_rows, 'router_acc_to_risk_oracle'):.6g}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- `beta_emp_over_theory` close to 1 validates the Bayes temperature law for MSE energies.")
    lines.append("- `class_oracle_to_px_kl` measures the information gap between a hidden-label/oracle view and the deployable posterior p(c|x_t).")
    lines.append("- `KL(q_eps || q_risk)` measures how far the sample-wise eps oracle is from the deployable Bayes-risk target.")
    lines.append("- The actual deployable criterion is the excess risk of the router relative to `argmin_k E[cost_k | x_t]`, not agreement with the eps oracle.")
    lines.append("")
    lines.append("## Files")
    lines.append("- `beta_sweep_x.csv`")
    lines.append("- `routing_by_t.csv`")
    lines.append("- `risk_table.pt`")
    lines.append("- `checkpoint_v7.pt`")
    lines.append("- `train_log_experts.csv`")
    lines.append("- `train_log_router.csv`")
    (out / "README_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def save_checkpoint(
    out: Path,
    params: Params,
    pca: PCAState,
    mix: BayesMixture,
    expert: MCLDenoiser,
    router: RiskRouter,
    A: torch.Tensor,
    data_info: Dict[str, object],
) -> None:
    torch.save(
        dict(
            params=asdict(params),
            data_info=data_info,
            pca=dict(
                raw_mean=pca.raw_mean,
                components=pca.components,
                latent_mean=pca.latent_mean,
                latent_std=pca.latent_std,
            ),
            bayes=dict(
                class_means=mix.class_means.cpu(),
                priors=mix.priors.cpu(),
                sigma0_sq=mix.sigma0_sq,
            ),
            expert_state=expert.state_dict(),
            router_state=router.state_dict(),
            risk_table=A.detach().cpu(),
            t_grid=torch.linspace(params.t_min, params.t_max, params.t_grid).cpu(),
        ),
        out / "checkpoint_v7.pt",
    )


def run(params: Params) -> None:
    set_seed(params.seed)
    device = get_device(params.device)
    out = ensure_dir(params.outdir)
    write_json(out / "params.json", asdict(params))
    print(f"V7 Bayes-risk router on {device}; outdir={out}", flush=True)

    # Data and latent PCA
    Xtr, ytr, Xte, yte, data_info = load_raw_dataset(params)
    Ztr, Zte, pca = build_pca_latent(Xtr, Xte, params.pca_dim, params.seed)
    C = len(data_info["classes"])
    mix = fit_bayes_mixture(Ztr, ytr, C)
    data_info.update(d_latent=int(Ztr.shape[1]), C=C, K=params.K)
    write_json(out / "data_info.json", data_info)
    write_json(
        out / "bayes_params.json",
        dict(
            sigma0_sq=mix.sigma0_sq,
            priors=[float(x) for x in mix.priors.tolist()],
            class_mean_norms=[float(x) for x in mix.class_means.norm(dim=1).tolist()],
        ),
    )

    # Split calibration/validation from training indices.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(params.seed + 333)
    perm = torch.randperm(Ztr.shape[0], generator=gen)
    n_cal = min(params.n_calib, Ztr.shape[0] // 2)
    cal_idx = perm[:n_cal]
    val_idx = perm[n_cal : n_cal + min(max(params.eval_n, 1024), Ztr.shape[0] - n_cal)]
    if val_idx.numel() == 0:
        val_idx = perm[: min(params.eval_n, Ztr.shape[0])]
    Zcal, ycal = Ztr[cal_idx], ytr[cal_idx]
    Zval, yval = Ztr[val_idx], ytr[val_idx]

    print(
        f"Data: train={Ztr.shape}, test={Zte.shape}, C={C}, K={params.K}, sigma0_sq={mix.sigma0_sq:.4g}",
        flush=True,
    )

    # 1) Validate beta_x(t) before any expert training.
    beta_rows = beta_sweep_x(Zte, yte, mix, params, device)
    write_csv(out / "beta_sweep_x.csv", beta_rows)
    print(
        f"beta_emp/theory mean={_mean(beta_rows, 'beta_emp_over_theory'):.4f}; "
        f"class_oracle_to_px_kl mean={_mean(beta_rows, 'class_oracle_to_px_kl'):.4f}",
        flush=True,
    )

    # 2) Train experts with beta_x(t).
    expert, _ = train_experts(Ztr, Zval, mix, params, device, out)

    # 3) Estimate A_{c,k}(t) = E[cost_k | c,t].
    A = compute_risk_table(expert, Zcal, ycal, params, device, out)

    # 4) Train deployable Bayes-risk router.
    router, _ = train_router(expert, Ztr, Zval, mix, A, params, device, out)

    # 5) Final routing diagnostics on held-out test set.
    route_rows = routing_eval_by_t(expert, router, Zte, yte, mix, A, params, device)
    write_csv(out / "routing_by_t.csv", route_rows)
    make_plots(out, beta_rows, route_rows)
    write_summary(out, params, data_info, mix, beta_rows, route_rows)
    save_checkpoint(out, params, pca, mix, expert, router, A, data_info)

    print("\n" + (out / "README_SUMMARY.md").read_text(encoding="utf-8"), flush=True)


def parse_args() -> Params:
    p = argparse.ArgumentParser(description="V7 Bayes-risk router and beta_x(t) validation")
    defaults = Params()
    for k, v in asdict(defaults).items():
        arg = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(arg, action="store_true" if not v else "store_false", default=v)
        elif isinstance(v, int):
            p.add_argument(arg, type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(arg, type=float, default=v)
        else:
            p.add_argument(arg, type=str, default=v)
    ns = vars(p.parse_args())
    return Params(**ns)


if __name__ == "__main__":
    run(parse_args())
