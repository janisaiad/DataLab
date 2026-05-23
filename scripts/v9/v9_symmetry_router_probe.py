#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V9 symmetry-breaking + linear/logistic router probe.

Goal
----
This script is a diagnostic companion for the RF/MCL diffusion experiments.
It tests the local symmetry-breaking theorem:

    symmetric experts stable      iff   beta(t) * lambda_signal(t) < lambda_damp
    symmetric experts unstable    iff   beta(t) * lambda_signal(t) > lambda_damp

with the soft-MCL loss

    L_beta(f_1,...,f_K) = E[-1/beta log(1/K sum_k exp(-beta e_k))]
    e_k = mean_dim ||f_k(x_t) - eps||^2.

It also tests the key routing hypothesis:

    once class/speciation signal appears, a linear multinomial-logistic router
    at the speciation time should be enough, because the risk decision is a
    linear/separable posterior decision for equal-covariance Gaussian mixtures.

Outputs
-------
In --outdir:
  params.json
  phase_by_t.csv
  mcl_by_t_beta.csv
  router_by_t_beta.csv
  README_SUMMARY.md
  plot_phase_curve.png
  plot_speciation_metrics.png
  plot_linear_router.png
  A_ck_heatmap_best.png

Datasets
--------
  --dataset gmm      synthetic Gaussian mixture with exact posterior
  --dataset mnist    MNIST, PCA latent, class posterior learned by linear logistic probe
  --dataset cifar10  CIFAR10, PCA latent, class posterior learned by linear logistic probe

Examples
--------
  python scripts/v9_symmetry_router_probe.py \
    --dataset gmm --C 4 --K 4 --d-latent 64 --n-train 6000 --n-test 3000 \
    --t-grid 0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,3.0 \
    --rho-list 0.5,1.0,2.0 --outdir outputs/v9_symmetry_gmm

  python scripts/v9_symmetry_router_probe.py \
    --dataset mnist --classes 0,1,2,3,4,5,6,7,8,9 --d-latent 128 --K 4 \
    --n-train 12000 --n-test 2048 --t-grid 0.6,1.0,1.4,1.8,2.2,2.6,3.0 \
    --rho-list 0.5,1.0,2.0 --outdir outputs/v9_symmetry_mnist

Notes on scaling
----------------
For e = mean_dim squared error, the second variation around the symmetric
solution is

    Delta L ~= (1/K) sum_k [ D(g_k) - 2 beta Q(g_k) ] / d

where Q is scaled by 1/d. We therefore report

    beta_crit = 0.5 / lambda_signal_scaled

and the instability criterion for a chosen beta is

    beta * lambda_signal_scaled > 0.5.

If you use a 1/2||.||^2 loss instead, the constants change, but the
criterion keeps the same form: beta * lambda_signal > lambda_damp.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    import torchvision.transforms as T
except Exception:
    torchvision = None
    T = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_of(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_classes(dataset: str, s: str, C_fallback: int) -> List[int]:
    if not s:
        return list(range(C_fallback))
    out: List[int] = []
    for item in [x.strip() for x in s.split(",") if x.strip()]:
        if item.isdigit():
            out.append(int(item))
        elif dataset == "cifar10":
            out.append(CIFAR10_CLASSES.index(item))
        else:
            raise ValueError(f"Cannot parse class {item!r} for dataset={dataset}")
    return out


def normalized_entropy_from_counts(counts: torch.Tensor, eps: float = 1e-12) -> float:
    counts = counts.float()
    p = counts / counts.sum().clamp_min(eps)
    k = p.numel()
    if k <= 1:
        return 0.0
    return float((-(p.clamp_min(eps) * p.clamp_min(eps).log()).sum() / math.log(k)).cpu())


def normalized_entropy_probs(probs: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    k = probs.shape[dim]
    if k <= 1:
        return torch.zeros(probs.shape[:-1], device=probs.device)
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim) / math.log(k)


def mi_norm(assign: torch.Tensor, labels: torch.Tensor, K: int, C: int, eps: float = 1e-12) -> float:
    a = assign.detach().cpu().long()
    y = labels.detach().cpu().long()
    n = max(1, int(y.numel()))
    joint = torch.zeros(C, K, dtype=torch.float64)
    for c in range(C):
        for k in range(K):
            joint[c, k] = ((y == c) & (a == k)).sum().item()
    joint /= n
    pc = joint.sum(1, keepdim=True)
    pk = joint.sum(0, keepdim=True)
    den = pc @ pk
    mask = joint > 0
    if not bool(mask.any()):
        return 0.0
    mi = (joint[mask] * (joint[mask] / den[mask].clamp_min(eps)).log()).sum()
    hc = -(pc[pc > 0] * pc[pc > 0].log()).sum()
    hk = -(pk[pk > 0] * pk[pk > 0].log()).sum()
    return float((mi / torch.minimum(hc, hk).clamp_min(eps)).item())


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------------------------------------------------------
# Data loaders and latent preprocessing
# -----------------------------------------------------------------------------


@dataclass
class LoadedData:
    x0_train: torch.Tensor
    y_train: torch.Tensor
    x0_test: torch.Tensor
    y_test: torch.Tensor
    class_names: List[str]
    info: Dict
    gmm_means: Optional[torch.Tensor] = None
    gmm_sigma0: Optional[float] = None


def simplex_means(C: int, d: int, mu: float, device: torch.device) -> torch.Tensor:
    if C == 2:
        m = torch.zeros(C, d, device=device)
        m[0, 0] = -mu * math.sqrt(d)
        m[1, 0] = mu * math.sqrt(d)
        return m
    g = torch.randn(C, d, device=device)
    g = g - g.mean(0, keepdim=True)
    q, _ = torch.linalg.qr(g.T, mode="reduced")
    m = q.T[:C]
    m = m - m.mean(0, keepdim=True)
    m = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12) * (mu * math.sqrt(d))
    return m


def sample_gmm_x0(n: int, means: torch.Tensor, sigma0: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    C, d = means.shape
    labels = torch.randint(0, C, (n,), device=device)
    x0 = means[labels] + sigma0 * torch.randn(n, d, device=device)
    return x0, labels


def collect_balanced_dataset(ds, class_ids: Sequence[int], n: int, seed: int, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
    local = {c: i for i, c in enumerate(class_ids)}
    per = max(1, n // max(1, len(class_ids)))
    cnt = {c: 0 for c in class_ids}
    idx = list(range(len(ds)))
    rng = random.Random(seed + offset)
    rng.shuffle(idx)
    xs, ys = [], []
    for j in idx:
        x, y = ds[j]
        if int(y) not in local:
            continue
        if len(xs) < n or cnt[int(y)] < per:
            xs.append(x.flatten())
            ys.append(local[int(y)])
            cnt[int(y)] += 1
        if len(xs) >= n and all(cnt[c] >= per for c in class_ids):
            break
    if len(xs) == 0:
        raise RuntimeError("No samples collected; check --classes.")
    return torch.stack(xs)[:n].float(), torch.tensor(ys[:n], dtype=torch.long)


def fit_pca(x: torch.Tensor, q: int) -> Tuple[torch.Tensor, torch.Tensor]:
    q = min(q, x.shape[1], max(1, x.shape[0] - 1))
    mean = x.mean(0, keepdim=True)
    xc = (x - mean).cpu()
    # pca_lowrank is much faster than a full SVD on image vectors.
    _, _, v = torch.pca_lowrank(xc, q=q, center=False, niter=4)
    return mean.cpu(), v[:, :q].contiguous().cpu()


def apply_pca(x: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    return (x - mean.to(x.device)) @ components.to(x.device)


def load_data(params, device: torch.device) -> LoadedData:
    if params.dataset == "gmm":
        C = params.C
        means = simplex_means(C, params.d_latent, params.gmm_mu, device)
        xtr, ytr = sample_gmm_x0(params.n_train, means, params.gmm_sigma0, device)
        xte, yte = sample_gmm_x0(params.n_test, means, params.gmm_sigma0, device)
        info = dict(dataset="gmm", C=C, d_latent=params.d_latent, mu=params.gmm_mu, sigma0=params.gmm_sigma0)
        return LoadedData(xtr.cpu(), ytr.cpu(), xte.cpu(), yte.cpu(), [f"c{i}" for i in range(C)], info, means.cpu(), params.gmm_sigma0)

    if torchvision is None or T is None:
        raise ImportError("torchvision is required for MNIST/CIFAR10 runs.")

    if params.dataset == "mnist":
        class_ids = parse_classes("mnist", params.classes, 10)
        transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        tr = torchvision.datasets.MNIST(params.data_root, train=True, download=not params.no_download, transform=transform)
        te = torchvision.datasets.MNIST(params.data_root, train=False, download=not params.no_download, transform=transform)
        raw_tr, ytr = collect_balanced_dataset(tr, class_ids, params.n_train, params.seed, 1)
        raw_te, yte = collect_balanced_dataset(te, class_ids, params.n_test, params.seed, 2)
        names = [str(i) for i in class_ids]
    elif params.dataset == "cifar10":
        class_ids = parse_classes("cifar10", params.classes, 10)
        transform = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
        tr = torchvision.datasets.CIFAR10(params.data_root, train=True, download=not params.no_download, transform=transform)
        te = torchvision.datasets.CIFAR10(params.data_root, train=False, download=not params.no_download, transform=transform)
        raw_tr, ytr = collect_balanced_dataset(tr, class_ids, params.n_train, params.seed, 1)
        raw_te, yte = collect_balanced_dataset(te, class_ids, params.n_test, params.seed, 2)
        names = [CIFAR10_CLASSES[i] for i in class_ids]
    else:
        raise ValueError(params.dataset)

    # Global standardization before PCA.
    raw_mean = raw_tr.mean(0, keepdim=True)
    raw_std = raw_tr.std().clamp_min(1e-6)
    raw_tr = (raw_tr - raw_mean) / raw_std
    raw_te = (raw_te - raw_mean) / raw_std

    pca_mean, pca_components = fit_pca(raw_tr, params.d_latent)
    xtr = apply_pca(raw_tr, pca_mean, pca_components).float()
    xte = apply_pca(raw_te, pca_mean, pca_components).float()

    # Latent standardization so Gaussian channel constants are interpretable.
    lat_mean = xtr.mean(0, keepdim=True)
    lat_std = xtr.std().clamp_min(1e-6)
    xtr = (xtr - lat_mean) / lat_std
    xte = (xte - lat_mean) / lat_std

    info = dict(
        dataset=params.dataset,
        classes=names,
        raw_dim=int(raw_tr.shape[1]),
        d_latent=int(xtr.shape[1]),
        pca_dim=int(xtr.shape[1]),
        n_train=int(xtr.shape[0]),
        n_test=int(xte.shape[0]),
    )
    return LoadedData(xtr.cpu(), ytr.cpu(), xte.cpu(), yte.cpu(), names, info)


# -----------------------------------------------------------------------------
# Diffusion and features
# -----------------------------------------------------------------------------


def diffuse_with_eps(x0: torch.Tensor, t: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Deterministic eps for reproducibility at each t.
    gen = torch.Generator(device=x0.device)
    gen.manual_seed(seed)
    gamma = math.exp(-t)
    eps = torch.randn(x0.shape, device=x0.device, generator=gen)
    xt = gamma * x0 + math.sqrt(max(1.0 - gamma * gamma, 1e-12)) * eps
    return xt, eps


class FeatureMap(nn.Module):
    def __init__(self, d: int, mode: str, rf_dim: int, activation: str, seed: int):
        super().__init__()
        self.mode = mode
        self.activation = activation
        if mode == "linear":
            self.p = d + 1
        elif mode == "rf":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed + 12345)
            self.register_buffer("W", torch.randn(rf_dim, d, generator=gen) / math.sqrt(d))
            self.register_buffer("b", 2 * math.pi * torch.rand(rf_dim, generator=gen))
            self.p = rf_dim + 1
        else:
            raise ValueError(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear":
            return torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
        z = x @ self.W.T.to(x.device)
        if self.activation == "erf":
            h = torch.erf(z)
        elif self.activation == "tanh":
            h = torch.tanh(z)
        elif self.activation == "relu":
            h = F.relu(z)
        elif self.activation == "cos":
            h = torch.cos(z + self.b.to(x.device))
        else:
            raise ValueError(self.activation)
        h = h / math.sqrt(max(1, h.shape[1]))
        return torch.cat([h, torch.ones(x.shape[0], 1, device=x.device)], dim=1)


# -----------------------------------------------------------------------------
# Linear algebra: ridge, whitening, MCL Hessian signal
# -----------------------------------------------------------------------------


@torch.no_grad()
def ridge_fit(phi: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    n, p = phi.shape
    eye = torch.eye(p, device=phi.device)
    A = (phi.T @ phi) / n + ridge * eye
    B = (phi.T @ y) / n
    return torch.linalg.solve(A, B)


@torch.no_grad()
def whiten_features(phi: torch.Tensor, ridge: float) -> torch.Tensor:
    n, p = phi.shape
    C = (phi.T @ phi) / n + ridge * torch.eye(p, device=phi.device)
    ev, V = torch.linalg.eigh(C)
    Winv = V @ torch.diag(ev.clamp_min(ridge).rsqrt()) @ V.T
    return phi @ Winv


@torch.no_grad()
def power_signal_scaled(S: torch.Tensor, R: torch.Tensor, iters: int = 30) -> float:
    """Top eigenvalue of Q/D with Q scaled by 1/d.

    D(B) = E ||S B||^2
    Q(B) = E (R dot S B)^2 / d

    The soft-MCL MSE(mean_dim) threshold is beta * lambda > 0.5.
    """
    if iters <= 0:
        return 0.0
    n, p = S.shape
    d = R.shape[1]
    B = torch.randn(p, d, device=S.device)
    B = B / B.norm().clamp_min(1e-12)
    lam = torch.tensor(0.0, device=S.device)
    sqrt_d = math.sqrt(max(1, d))
    for _ in range(iters):
        delta = S @ B                       # n x d
        a = (R * delta).sum(1, keepdim=True) / sqrt_d
        M = a * R / sqrt_d                  # n x d
        GB = (S.T @ M) / n
        lam = (B * GB).sum() / (B * B).sum().clamp_min(1e-12)
        B = GB / GB.norm().clamp_min(1e-12)
    return max(float(lam.detach().cpu()), 0.0)


@torch.no_grad()
def class_basis_from_clean(x0: torch.Tensor, labels: torch.Tensor, C: int) -> torch.Tensor:
    mats = []
    for c in range(C):
        mask = labels == c
        if mask.any():
            mats.append(x0[mask].mean(0))
    if len(mats) <= 1:
        return torch.empty(x0.shape[1], 0, device=x0.device)
    M = torch.stack(mats)
    M = M - M.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(M, full_matrices=False)
    rank = int((s > 1e-7 * s.max().clamp_min(1e-7)).sum())
    return vh[:rank].T.contiguous()


@torch.no_grad()
def lambda_dir_scaled(S: torch.Tensor, R: torch.Tensor, v: torch.Tensor) -> float:
    """Top feature eigenvalue for perturbations u(x)=a(x) v, scaled by 1/d."""
    n = S.shape[0]
    d = R.shape[1]
    weights = (R @ v).pow(2) / max(1, d)
    H = (S.T * weights[None, :]) @ S / n
    return max(float(torch.linalg.eigvalsh(H)[-1].detach().cpu()), 0.0)


@torch.no_grad()
def calibrate_phase(phi: torch.Tensor, y: torch.Tensor, x0: torch.Tensor, labels: torch.Tensor, C: int, ridge: float, power_iters: int) -> Dict[str, float]:
    A0 = ridge_fit(phi, y, ridge)
    R = y - phi @ A0
    S = whiten_features(phi, ridge)
    lam_free = power_signal_scaled(S, R, power_iters)
    Bclass = class_basis_from_clean(x0, labels, C).to(phi.device)
    class_lams = [lambda_dir_scaled(S, R, Bclass[:, j]) for j in range(Bclass.shape[1])]
    lam_class = max(class_lams or [0.0])
    # transverse probe: random directions orthogonal to class basis
    d = y.shape[1]
    P = Bclass @ Bclass.T if Bclass.numel() else torch.zeros(d, d, device=phi.device)
    vals = []
    for _ in range(12):
        v = torch.randn(d, device=phi.device)
        v = v - P @ v
        if v.norm() > 1e-8:
            vals.append(lambda_dir_scaled(S, R, v / v.norm()))
    lam_trans = max(vals or [0.0])
    def beta_crit(lam: float) -> float:
        return 0.5 / max(lam, 1e-12)
    return dict(
        lambda_free=lam_free,
        lambda_class=lam_class,
        lambda_trans=lam_trans,
        beta_crit_free=beta_crit(lam_free),
        beta_crit_class=beta_crit(lam_class),
        beta_crit_trans=beta_crit(lam_trans),
        residual_mse=float(R.pow(2).mean().detach().cpu()),
        residual_energy_var=float(R.pow(2).mean(1).var(unbiased=False).detach().cpu()),
        class_basis_rank=int(Bclass.shape[1]),
    )


# -----------------------------------------------------------------------------
# Soft-MCL training and evaluation
# -----------------------------------------------------------------------------


class ResidualExperts(nn.Module):
    def __init__(self, K: int, p: int, d: int, init_std: float):
        super().__init__()
        self.G = nn.Parameter(init_std * torch.randn(K, p, d))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bp,kpd->bkd", phi, self.G)


def mcl_loss_from_costs(e: torch.Tensor, beta: float, balance_weight: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    K = e.shape[1]
    if beta <= 1e-12:
        q = torch.full_like(e, 1.0 / K)
        loss = e.mean()
    else:
        logits = -beta * e
        q = torch.softmax(logits, dim=1)
        loss = -((torch.logsumexp(logits, dim=1) - math.log(K)).mean()) / beta
    if balance_weight > 0:
        u = q.mean(0)
        loss = loss + balance_weight * ((u - 1.0 / K) ** 2).sum()
    return loss, q


@torch.no_grad()
def expert_costs(experts: ResidualExperts, phi: torch.Tensor, y: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    pred = f0[:, None, :] + experts(phi)
    return ((pred - y[:, None, :]) ** 2).mean(-1)


def train_mcl_residual_experts(
    phi_train: torch.Tensor,
    y_train: torch.Tensor,
    f0_train: torch.Tensor,
    K: int,
    beta: float,
    steps: int,
    batch_size: int,
    lr: float,
    init_std: float,
    balance_weight: float,
    device: torch.device,
) -> ResidualExperts:
    p, d = phi_train.shape[1], y_train.shape[1]
    experts = ResidualExperts(K, p, d, init_std).to(device)
    opt = torch.optim.AdamW(experts.parameters(), lr=lr)
    n = phi_train.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        pred = f0_train[idx][:, None, :] + experts(phi_train[idx])
        e = ((pred - y_train[idx][:, None, :]) ** 2).mean(-1)
        loss, _ = mcl_loss_from_costs(e, beta, balance_weight=balance_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return experts


@torch.no_grad()
def A_ck_from_costs(e: torch.Tensor, labels: torch.Tensor, C: int) -> torch.Tensor:
    K = e.shape[1]
    A = torch.zeros(C, K, device=e.device)
    global_mean = e.mean(0)
    for c in range(C):
        mask = labels == c
        if mask.any():
            A[c] = e[mask].mean(0)
        else:
            A[c] = global_mean
    return A


@torch.no_grad()
def evaluate_mcl(
    e: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    C: int,
) -> Dict[str, float | List[float]]:
    K = e.shape[1]
    _, q = mcl_loss_from_costs(e, beta)
    oracle = e.argmin(1)
    usage = torch.bincount(oracle, minlength=K).float()
    A = A_ck_from_costs(e, labels, C)
    Ac = A - A.mean(1, keepdim=True)
    svals = torch.linalg.svdvals(Ac.cpu()) if min(Ac.shape) > 0 else torch.zeros(1)
    rank_eps = float(svals.max().item() * 1e-5 + 1e-12)
    out: Dict[str, float | List[float]] = dict(
        teacher_entropy_norm=float(normalized_entropy_probs(q).mean().detach().cpu()),
        oracle_best_mse=float(e.min(1).values.mean().detach().cpu()),
        mean_expert_mse=float(e.mean(1).mean().detach().cpu()),
        soft_oracle_mse=float((q * e).sum(1).mean().detach().cpu()),
        cost_gap_mean=float((e.mean(1) - e.min(1).values).mean().detach().cpu()),
        beta_gap=float((beta * (e.mean(1) - e.min(1).values).mean()).detach().cpu()),
        oracle_usage_entropy=normalized_entropy_from_counts(usage),
        oracle_class_mi_norm=mi_norm(oracle, labels, K, C),
        A_ck_std=float(Ac.std().detach().cpu()),
        A_ck_rank=float((svals > rank_eps).sum().item()),
        A_ck_s1=float(svals[0].item() if svals.numel() else 0.0),
        A_ck_s2=float(svals[1].item() if svals.numel() > 1 else 0.0),
    )
    return out


# -----------------------------------------------------------------------------
# Posterior and logistic routers
# -----------------------------------------------------------------------------


class LinearClassifier(nn.Module):
    def __init__(self, d: int, C: int):
        super().__init__()
        self.fc = nn.Linear(d, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    C: int,
    steps: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    device: torch.device,
) -> Tuple[LinearClassifier, Dict[str, float]]:
    clf = LinearClassifier(x_train.shape[1], C).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    n = x_train.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        loss = F.cross_entropy(clf(x_train[idx]), y_train[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = clf(x_val)
        acc = (logits.argmax(1) == y_val).float().mean().item()
        ce = F.cross_entropy(logits, y_val).item()
    return clf, dict(val_acc=float(acc), val_ce=float(ce))


@torch.no_grad()
def gmm_posterior_xt(xt: torch.Tensor, means: torch.Tensor, sigma0: float, t: float) -> torch.Tensor:
    # x_t = gamma x0 + sqrt(1-gamma^2) eps, x0|c ~ N(mu_c, sigma0^2 I)
    g = math.exp(-t)
    st2 = g * g * sigma0 * sigma0 + max(1.0 - g * g, 1e-12)
    dist = ((xt[:, None, :] - g * means.to(xt.device)[None, :, :]) ** 2).sum(-1)
    return torch.softmax(-0.5 * dist / st2, dim=1)


@torch.no_grad()
def risk_labels_from_posterior(p_c: torch.Tensor, A_ck: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # A_ck[c,k] = expected cost of expert k in class c.
    risk = p_c @ A_ck.to(p_c.device)
    vals, idx = torch.topk(risk, k=min(2, risk.shape[1]), dim=1, largest=False)
    best = idx[:, 0]
    if vals.shape[1] > 1:
        margin = vals[:, 1] - vals[:, 0]
    else:
        margin = torch.zeros(risk.shape[0], device=risk.device)
    return best, margin, risk


@torch.no_grad()
def route_excess(risk: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    return risk[torch.arange(risk.shape[0], device=risk.device), chosen] - risk.min(1).values


def evaluate_linear_router(
    x_train: torch.Tensor,
    risk_label_train: torch.Tensor,
    x_test: torch.Tensor,
    risk_label_test: torch.Tensor,
    risk_test: torch.Tensor,
    true_labels_test: torch.Tensor,
    K: int,
    C: int,
    params,
    device: torch.device,
) -> Dict[str, float]:
    router, _ = train_linear_classifier(
        x_train, risk_label_train, x_test, risk_label_test,
        C=K,
        steps=params.router_logreg_steps,
        lr=params.router_logreg_lr,
        batch_size=params.router_logreg_batch_size,
        weight_decay=params.router_logreg_weight_decay,
        device=device,
    )
    with torch.no_grad():
        logits = router(x_test)
        probs = torch.softmax(logits, 1)
        pred = probs.argmax(1)
        acc = (pred == risk_label_test).float().mean().item()
        excess = route_excess(risk_test, pred)
        counts = torch.bincount(pred, minlength=K).float()
    return dict(
        linear_router_acc_vs_risk=float(acc),
        linear_router_excess_mean=float(excess.mean().item()),
        linear_router_excess_p95=float(torch.quantile(excess, 0.95).item()),
        linear_router_usage_entropy=normalized_entropy_from_counts(counts),
        linear_router_class_mi_norm=mi_norm(pred, true_labels_test, K, C),
        linear_router_mean_conf=float(probs.max(1).values.mean().item()),
    )


# -----------------------------------------------------------------------------
# Main experiment loop
# -----------------------------------------------------------------------------


@dataclass
class Params:
    dataset: str = "gmm"                  # gmm | mnist | cifar10
    data_root: str = "./data"
    classes: str = ""
    C: int = 4
    K: int = 4
    d_latent: int = 64
    n_train: int = 6000
    n_test: int = 3000
    gmm_mu: float = 1.0
    gmm_sigma0: float = 0.5
    t_grid: str = "0.6,1.0,1.4,1.8,2.2,2.6,3.0"
    rho_list: str = "0.5,1.0,2.0"
    feature_mode: str = "linear"          # linear | rf
    rf_dim: int = 512
    rf_activation: str = "erf"
    ridge: float = 1e-5
    power_iters: int = 30
    mcl_steps: int = 2500
    mcl_lr: float = 2e-3
    mcl_batch_size: int = 256
    init_std: float = 1e-3
    balance_weight: float = 0.0
    posterior_logreg_steps: int = 1000
    posterior_logreg_lr: float = 2e-3
    posterior_logreg_batch_size: int = 512
    posterior_logreg_weight_decay: float = 1e-4
    router_logreg_steps: int = 1000
    router_logreg_lr: float = 2e-3
    router_logreg_batch_size: int = 512
    router_logreg_weight_decay: float = 1e-4
    seed: int = 0
    device: str = "auto"
    outdir: str = "./outputs/v9_symmetry_router_probe"
    no_download: bool = False


def run(params: Params) -> None:
    set_seed(params.seed)
    device = device_of(params.device)
    out = ensure_dir(params.outdir)
    t_values = parse_float_list(params.t_grid)
    rho_values = parse_float_list(params.rho_list)

    loaded = load_data(params, device)
    x0_train = loaded.x0_train.to(device)
    y_train = loaded.y_train.to(device)
    x0_test = loaded.x0_test.to(device)
    y_test = loaded.y_test.to(device)
    C = len(loaded.class_names) if params.dataset != "gmm" or params.classes else params.C
    K = params.K
    d = x0_train.shape[1]

    fmap = FeatureMap(d, params.feature_mode, params.rf_dim, params.rf_activation, params.seed).to(device).eval()

    (out / "params.json").write_text(json.dumps(asdict(params), indent=2))
    (out / "data_info.json").write_text(json.dumps(loaded.info, indent=2))

    phase_rows: List[Dict] = []
    mcl_rows: List[Dict] = []
    router_rows: List[Dict] = []
    best_router_row: Optional[Dict] = None
    best_A: Optional[torch.Tensor] = None

    for ti, t in enumerate(t_values):
        print(f"\n=== t={t:.4g} ===", flush=True)
        xt_train, eps_train = diffuse_with_eps(x0_train, t, params.seed + 1000 + ti)
        xt_test, eps_test = diffuse_with_eps(x0_test, t, params.seed + 2000 + ti)
        phi_train = fmap(xt_train)
        phi_test = fmap(xt_test)
        A0 = ridge_fit(phi_train, eps_train, params.ridge)
        f0_train = phi_train @ A0
        f0_test = phi_test @ A0

        phase = calibrate_phase(phi_train, eps_train, x0_train, y_train, C, params.ridge, params.power_iters)
        phase.update(t=t)
        phase_rows.append(phase)
        write_csv(out / "phase_by_t.csv", phase_rows)

        # Class posterior p(c|x_t). For GMM use exact Bayes posterior; for MNIST/CIFAR use a linear logistic probe.
        if params.dataset == "gmm" and loaded.gmm_means is not None:
            p_train = gmm_posterior_xt(xt_train, loaded.gmm_means.to(device), loaded.gmm_sigma0 or params.gmm_sigma0, t)
            p_test = gmm_posterior_xt(xt_test, loaded.gmm_means.to(device), loaded.gmm_sigma0 or params.gmm_sigma0, t)
            posterior_acc = float((p_test.argmax(1) == y_test).float().mean().item())
            posterior_ce = float(F.cross_entropy(p_test.clamp_min(1e-12).log(), y_test).item())
        else:
            post_clf, post_metrics = train_linear_classifier(
                xt_train, y_train, xt_test, y_test,
                C=C,
                steps=params.posterior_logreg_steps,
                lr=params.posterior_logreg_lr,
                batch_size=params.posterior_logreg_batch_size,
                weight_decay=params.posterior_logreg_weight_decay,
                device=device,
            )
            with torch.no_grad():
                p_train = torch.softmax(post_clf(xt_train), 1)
                p_test = torch.softmax(post_clf(xt_test), 1)
            posterior_acc = post_metrics["val_acc"]
            posterior_ce = post_metrics["val_ce"]

        # Use class critical beta as default; if class signal is absent, fall back to free beta.
        beta_base = phase["beta_crit_class"] if math.isfinite(phase["beta_crit_class"]) and phase["lambda_class"] > 1e-12 else phase["beta_crit_free"]

        for rho in rho_values:
            beta = rho * beta_base
            print(f"  rho={rho:.3g}, beta={beta:.6g}, beta*lambda_class={beta * phase['lambda_class']:.4g}", flush=True)
            experts = train_mcl_residual_experts(
                phi_train=phi_train,
                y_train=eps_train,
                f0_train=f0_train,
                K=K,
                beta=beta,
                steps=params.mcl_steps,
                batch_size=params.mcl_batch_size,
                lr=params.mcl_lr,
                init_std=params.init_std,
                balance_weight=params.balance_weight,
                device=device,
            )
            with torch.no_grad():
                e_train = expert_costs(experts, phi_train, eps_train, f0_train)
                e_test = expert_costs(experts, phi_test, eps_test, f0_test)
                A_train = A_ck_from_costs(e_train, y_train, C)
                risk_label_train, risk_margin_train, risk_train = risk_labels_from_posterior(p_train, A_train)
                risk_label_test, risk_margin_test, risk_test = risk_labels_from_posterior(p_test, A_train)
                bayes_excess = route_excess(risk_test, risk_label_test)

            m = evaluate_mcl(e_test, y_test, beta, C)
            row = dict(
                t=t,
                rho=rho,
                beta=beta,
                beta_times_lambda_free=beta * phase["lambda_free"],
                beta_times_lambda_class=beta * phase["lambda_class"],
                beta_times_lambda_trans=beta * phase["lambda_trans"],
                instability_free=float(beta * phase["lambda_free"] > 0.5),
                instability_class=float(beta * phase["lambda_class"] > 0.5),
                instability_trans=float(beta * phase["lambda_trans"] > 0.5),
                posterior_acc=posterior_acc,
                posterior_ce=posterior_ce,
                risk_margin_mean=float(risk_margin_test.mean().item()),
                risk_margin_p95=float(torch.quantile(risk_margin_test, 0.95).item()),
                bayes_risk_excess_mean=float(bayes_excess.mean().item()),
            )
            row.update(m)
            mcl_rows.append(row)
            write_csv(out / "mcl_by_t_beta.csv", mcl_rows)

            rrow = dict(
                t=t,
                rho=rho,
                beta=beta,
                beta_times_lambda_class=beta * phase["lambda_class"],
                posterior_acc=posterior_acc,
                risk_margin_mean=float(risk_margin_test.mean().item()),
                risk_margin_train_mean=float(risk_margin_train.mean().item()),
                risk_label_usage_entropy=normalized_entropy_from_counts(torch.bincount(risk_label_test, minlength=K).float()),
                risk_label_class_mi_norm=mi_norm(risk_label_test, y_test, K, C),
                bayes_risk_excess_mean=float(bayes_excess.mean().item()),
            )
            rrow.update(evaluate_linear_router(
                x_train=xt_train,
                risk_label_train=risk_label_train,
                x_test=xt_test,
                risk_label_test=risk_label_test,
                risk_test=risk_test,
                true_labels_test=y_test,
                K=K,
                C=C,
                params=params,
                device=device,
            ))
            router_rows.append(rrow)
            write_csv(out / "router_by_t_beta.csv", router_rows)

            # Select a representative best row: strong speciation and learnable linear router.
            score = (
                10.0 * rrow["risk_margin_mean"]
                + 1.0 * rrow["linear_router_acc_vs_risk"]
                + 0.2 * row["A_ck_s1"]
                - 0.5 * row["teacher_entropy_norm"]
            )
            if best_router_row is None or score > best_router_row.get("_score", -1e100):
                best_router_row = dict(rrow)
                best_router_row["_score"] = float(score)
                best_A = A_train.detach().cpu()

    write_csv(out / "phase_by_t.csv", phase_rows)
    write_csv(out / "mcl_by_t_beta.csv", mcl_rows)
    write_csv(out / "router_by_t_beta.csv", router_rows)

    make_plots(out, phase_rows, mcl_rows, router_rows, best_A, loaded.class_names)
    write_summary(out, params, loaded.info, phase_rows, mcl_rows, router_rows, best_router_row)
    print(f"\nDone. Summary: {out / 'README_SUMMARY.md'}", flush=True)


# -----------------------------------------------------------------------------
# Plots and summary
# -----------------------------------------------------------------------------


def make_plots(
    out: Path,
    phase_rows: List[Dict],
    mcl_rows: List[Dict],
    router_rows: List[Dict],
    best_A: Optional[torch.Tensor],
    class_names: List[str],
) -> None:
    if plt is None:
        return
    try:
        # Phase curve.
        ts = [r["t"] for r in phase_rows]
        lfree = [r["lambda_free"] for r in phase_rows]
        lclass = [r["lambda_class"] for r in phase_rows]
        ltrans = [r["lambda_trans"] for r in phase_rows]
        plt.figure(figsize=(7, 4))
        plt.plot(ts, lfree, marker="o", label="lambda_free")
        plt.plot(ts, lclass, marker="o", label="lambda_class")
        plt.plot(ts, ltrans, marker="o", label="lambda_trans")
        plt.xlabel("t")
        plt.ylabel("scaled Hessian eigenvalue")
        plt.title("Soft-MCL symmetry-breaking signal")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "plot_phase_curve.png", dpi=180)
        plt.close()

        if mcl_rows:
            # Choose rho closest to 1 for clean plots.
            rhos = sorted(set(float(r["rho"]) for r in mcl_rows))
            rho0 = min(rhos, key=lambda x: abs(x - 1.0))
            rows = [r for r in mcl_rows if abs(float(r["rho"]) - rho0) < 1e-12]
            rows = sorted(rows, key=lambda r: r["t"])
            ts = [r["t"] for r in rows]
            plt.figure(figsize=(7, 4))
            plt.plot(ts, [r["teacher_entropy_norm"] for r in rows], marker="o", label="teacher entropy")
            plt.plot(ts, [r["oracle_class_mi_norm"] for r in rows], marker="o", label="oracle MI(class;expert)")
            plt.plot(ts, [r["A_ck_std"] for r in rows], marker="o", label="std centered A_ck")
            plt.plot(ts, [r["risk_margin_mean"] for r in rows], marker="o", label="risk margin")
            plt.xlabel("t")
            plt.ylabel("diagnostic")
            plt.title(f"Speciation diagnostics, rho={rho0:g}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "plot_speciation_metrics.png", dpi=180)
            plt.close()

        if router_rows:
            rhos = sorted(set(float(r["rho"]) for r in router_rows))
            rho0 = min(rhos, key=lambda x: abs(x - 1.0))
            rows = [r for r in router_rows if abs(float(r["rho"]) - rho0) < 1e-12]
            rows = sorted(rows, key=lambda r: r["t"])
            ts = [r["t"] for r in rows]
            plt.figure(figsize=(7, 4))
            plt.plot(ts, [r["posterior_acc"] for r in rows], marker="o", label="linear class posterior acc")
            plt.plot(ts, [r["linear_router_acc_vs_risk"] for r in rows], marker="o", label="linear router acc vs risk labels")
            plt.plot(ts, [r["linear_router_excess_mean"] for r in rows], marker="o", label="linear router excess risk")
            plt.xlabel("t")
            plt.ylabel("router diagnostic")
            plt.title(f"Linear/logistic router probe, rho={rho0:g}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "plot_linear_router.png", dpi=180)
            plt.close()

        if best_A is not None:
            plt.figure(figsize=(6, 4))
            plt.imshow(best_A.numpy(), aspect="auto")
            plt.colorbar(label="A_ck = E[cost expert k | class c]")
            plt.xlabel("expert k")
            plt.ylabel("class c")
            if len(class_names) == best_A.shape[0]:
                plt.yticks(range(len(class_names)), class_names)
            plt.title("Best observed class-expert risk matrix")
            plt.tight_layout()
            plt.savefig(out / "A_ck_heatmap_best.png", dpi=180)
            plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}", file=sys.stderr)


def mean_of(rows: List[Dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] == r[key]]
    return float(np.mean(vals)) if vals else float("nan")


def max_row(rows: List[Dict], key: str) -> Optional[Dict]:
    rows2 = [r for r in rows if key in r and r[key] == r[key]]
    if not rows2:
        return None
    return max(rows2, key=lambda r: float(r[key]))


def write_summary(
    out: Path,
    params: Params,
    data_info: Dict,
    phase_rows: List[Dict],
    mcl_rows: List[Dict],
    router_rows: List[Dict],
    best_router_row: Optional[Dict],
) -> None:
    best_phase_class = max_row(phase_rows, "lambda_class")
    best_mi = max_row(mcl_rows, "oracle_class_mi_norm")
    best_margin = max_row(mcl_rows, "risk_margin_mean")
    best_router = max_row(router_rows, "linear_router_acc_vs_risk")

    lines: List[str] = []
    lines.append("# V9 symmetry-breaking and linear-router probe")
    lines.append("")
    lines.append("## Setup")
    lines.append("```json")
    lines.append(json.dumps(asdict(params), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Data")
    lines.append("```json")
    lines.append(json.dumps(data_info, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Theory check")
    lines.append("For the loss `mean_dim squared error`, this script uses the local criterion")
    lines.append("")
    lines.append("`beta * lambda_signal_scaled > 0.5`.")
    lines.append("")
    lines.append("The clean theorem form is `beta * lambda_signal > lambda_damp`; the constant depends only on the loss normalization.")
    lines.append("")
    if best_phase_class:
        lines.append("Best class-channel signal:")
        lines.append(f"- t = {best_phase_class['t']}")
        lines.append(f"- lambda_class = {best_phase_class['lambda_class']:.6g}")
        lines.append(f"- beta_crit_class = {best_phase_class['beta_crit_class']:.6g}")
        lines.append(f"- lambda_free = {best_phase_class['lambda_free']:.6g}")
        lines.append(f"- lambda_trans = {best_phase_class['lambda_trans']:.6g}")
    lines.append("")
    lines.append("## Speciation diagnostics")
    lines.append(f"- mean teacher entropy norm: {mean_of(mcl_rows, 'teacher_entropy_norm'):.6g}")
    lines.append(f"- mean beta_gap: {mean_of(mcl_rows, 'beta_gap'):.6g}")
    lines.append(f"- mean risk_margin: {mean_of(mcl_rows, 'risk_margin_mean'):.6g}")
    lines.append(f"- mean A_ck_std: {mean_of(mcl_rows, 'A_ck_std'):.6g}")
    if best_mi:
        lines.append(f"- best oracle_class_mi_norm: {best_mi['oracle_class_mi_norm']:.6g} at t={best_mi['t']}, rho={best_mi['rho']}")
    if best_margin:
        lines.append(f"- best risk_margin_mean: {best_margin['risk_margin_mean']:.6g} at t={best_margin['t']}, rho={best_margin['rho']}")
    lines.append("")
    lines.append("## Linear/logistic router test")
    lines.append("This is the explicit test of the hypothesis: after class/speciation time, a linear multinomial-logistic router should be enough.")
    lines.append("")
    lines.append(f"- mean class posterior acc: {mean_of(router_rows, 'posterior_acc'):.6g}")
    lines.append(f"- mean linear router acc vs risk labels: {mean_of(router_rows, 'linear_router_acc_vs_risk'):.6g}")
    lines.append(f"- mean linear router excess risk: {mean_of(router_rows, 'linear_router_excess_mean'):.6g}")
    if best_router:
        lines.append(f"- best linear router acc: {best_router['linear_router_acc_vs_risk']:.6g} at t={best_router['t']}, rho={best_router['rho']}")
        lines.append(f"  excess_mean={best_router['linear_router_excess_mean']:.6g}, risk_margin={best_router['risk_margin_mean']:.6g}")
    if best_router_row:
        lines.append("")
        lines.append("Best combined route/speciation row:")
        for k, v in best_router_row.items():
            if not k.startswith("_"):
                lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## How to read")
    lines.append("- If `beta_times_lambda_class > 0.5` but `teacher_entropy_norm≈1`, `A_ck_std≈0`, and `risk_margin≈0`, then the theoretical local instability is not realized by the finite optimization run; increase steps/capacity or change initialization/schedule.")
    lines.append("- If `A_ck_std`, `oracle_class_mi_norm`, and `risk_margin` become positive, then symmetry is breaking into class/latent channels.")
    lines.append("- If `linear_router_acc_vs_risk` is high with tiny excess risk at that same t, then the linear/logistic router hypothesis is supported.")
    lines.append("- If random/mixture generation wins while these diagnostics are near zero, the gain is a multi-expert sampling effect, not yet a proven Bayes-risk routing effect.")
    lines.append("")
    lines.append("## Files")
    lines.append("- `phase_by_t.csv`: Hessian/BBP-style signal and critical beta by t")
    lines.append("- `mcl_by_t_beta.csv`: soft-MCL specialization metrics")
    lines.append("- `router_by_t_beta.csv`: posterior and linear/logistic router metrics")
    lines.append("- `plot_phase_curve.png`, `plot_speciation_metrics.png`, `plot_linear_router.png`, `A_ck_heatmap_best.png`")
    (out / "README_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> Params:
    P = argparse.ArgumentParser()
    defaults = asdict(Params())
    for k, v in defaults.items():
        arg = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            P.add_argument(arg, action="store_true" if not v else "store_false")
        elif isinstance(v, int):
            P.add_argument(arg, type=int, default=v)
        elif isinstance(v, float):
            P.add_argument(arg, type=float, default=v)
        else:
            P.add_argument(arg, type=str, default=v)
    return Params(**vars(P.parse_args()))


if __name__ == "__main__":
    run(parse_args())
