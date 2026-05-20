#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V9: Theory-validated Bayes-risk gated MCL diffusion.

Core idea
---------
V6 failed because it tried to make a deployable routeur imitate the sample-wise
MCL teacher q(k | x_t, eps), which uses eps and is unavailable at sampling.
V9 keeps the V8 Bayes-risk idea, but adds explicit theory validation: beta(t) sweeps, MCL speciation probes during training, commit-once routing, and paper-ready plots. It trains experts with the theoretically calibrated beta(t), then deploys a
Bayes-risk router

    k*(x_t,t) = argmin_k sum_c p(c | x_t,t) A_{c,k}(t),

where A_{c,k}(t) = E[ ||f_k(x_t,t)-eps||^2 | class=c,t ].

The script is intentionally self-contained. It can train MNIST/CIFAR10 models,
calibrate the theoretical router, sample, and evaluate with your existing
src.evaluate.evaluate if available.

Recommended quick sanity run
----------------------------
python scripts/run_variant_v9.py \
  --dataset mnist --classes all --image-size 28 --K 4 \
  --outdir outputs/v9_mnist_quick --device cuda \
  --all --baseline-steps 3000 --mcl-steps 6000 --num-samples 256 --sample-steps 50

Recommended 8h-style MNIST run
------------------------------
python scripts/run_variant_v9.py \
  --dataset mnist --classes all --image-size 28 --K 4 \
  --outdir outputs/v9_mnist_gold --device cuda \
  --all --baseline-steps 30000 --mcl-steps 60000 \
  --batch-size 256 --num-samples 2048 --sample-steps 100 \
  --pca-dim-router 64 --beta-rho 1.0 --beta-max 80

CIFAR automobile/horse
----------------------
python scripts/run_variant_v9.py \
  --dataset cifar10 --classes automobile,horse --image-size 32 --K 4 \
  --outdir outputs/v9_cifar_auto_horse --device cuda \
  --all --baseline-steps 50000 --mcl-steps 80000 \
  --batch-size 192 --num-samples 2048 --sample-steps 100 \
  --pca-dim-router 128 --beta-rho 1.0 --beta-max 80

Outputs
-------
  config.json
  baseline_final.pt
  mcl_final.pt
  router_stats.pt
  router_risk_table.pt
  router_calibration_by_t.csv
  samples_*.pt / samples_*.png
  metrics.json
  SUMMARY_v9.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    import torchvision.transforms as T
    from torchvision.utils import make_grid, save_image
except Exception as e:  # pragma: no cover
    torchvision = None
    T = None
    make_grid = None
    save_image = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# =============================================================================
# Utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_of(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def normal_approx_p_value_from_z(z_abs: float) -> float:
    return float(math.erfc(z_abs / math.sqrt(2.0)))


def to01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0, 1)


def normalized_entropy_from_counts(counts: torch.Tensor, eps: float = 1e-12) -> float:
    p = counts.float()
    p = p / p.sum().clamp_min(1)
    k = p.numel()
    return float((-(p.clamp_min(eps) * p.clamp_min(eps).log()).sum() / math.log(k)).cpu())


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def append_csv(path: Path, rows: List[Dict]) -> None:
    """Append rows to a CSV, creating the header if needed."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def mi_norm(assign: torch.Tensor, labels: torch.Tensor, K: int, C: int, eps: float = 1e-12) -> float:
    """Normalized mutual information between expert assignment and class label."""
    a = assign.detach().cpu().long()
    y = labels.detach().cpu().long()
    n = max(1, int(y.numel()))
    joint = torch.zeros(C, K, dtype=torch.float64)
    for c in range(C):
        for k in range(K):
            joint[c, k] = ((y == c) & (a == k)).sum().item()
    joint /= float(n)
    pc = joint.sum(1, keepdim=True)
    pk = joint.sum(0, keepdim=True)
    den = pc @ pk
    m = joint > 0
    if not bool(m.any()):
        return 0.0
    mi = (joint[m] * (joint[m] / den[m].clamp_min(eps)).log()).sum()
    hc = -(pc[pc > 0] * pc[pc > 0].log()).sum()
    hk = -(pk[pk > 0] * pk[pk > 0].log()).sum()
    return float((mi / torch.minimum(hc, hk).clamp_min(eps)).item())


def safe_svdvals(A: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.svdvals(A)
    except Exception:
        return torch.zeros(min(A.shape[-2:]), device=A.device)



# =============================================================================
# Data
# =============================================================================


def parse_classes(dataset: str, classes: str) -> List[int]:
    if classes.strip().lower() in {"all", "*"}:
        return list(range(10))
    names = MNIST_CLASSES if dataset == "mnist" else CIFAR10_CLASSES
    out: List[int] = []
    for raw in [z.strip() for z in classes.split(",") if z.strip()]:
        if raw.isdigit():
            out.append(int(raw))
        else:
            if raw not in names:
                raise ValueError(f"Unknown class {raw!r} for {dataset}; valid names={names}")
            out.append(names.index(raw))
    return out


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, base, class_ids: List[int], max_items: int = 0, seed: int = 0):
        self.base = base
        self.class_ids = list(class_ids)
        self.local = {c: i for i, c in enumerate(self.class_ids)}
        idx: List[int] = []
        for i in range(len(base)):
            y = int(base[i][1])
            if y in self.local:
                idx.append(i)
        rng = random.Random(seed)
        rng.shuffle(idx)
        if max_items and max_items > 0:
            idx = idx[:max_items]
        self.idx = idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        x, y = self.base[self.idx[i]]
        return x, self.local[int(y)]


def make_datasets(params: "Params"):
    if torchvision is None:
        raise ImportError("torchvision is required for this script.")

    ids = parse_classes(params.dataset, params.classes)
    if params.dataset == "mnist":
        tfm = T.Compose([
            T.Resize(params.image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        train_base = torchvision.datasets.MNIST(params.data_root, train=True, download=not params.no_download, transform=tfm)
        test_base = torchvision.datasets.MNIST(params.data_root, train=False, download=not params.no_download, transform=tfm)
        class_names = [MNIST_CLASSES[i] for i in ids]
        channels = 1
    elif params.dataset == "cifar10":
        tfm = T.Compose([
            T.Resize(params.image_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_base = torchvision.datasets.CIFAR10(params.data_root, train=True, download=not params.no_download, transform=tfm)
        test_base = torchvision.datasets.CIFAR10(params.data_root, train=False, download=not params.no_download, transform=tfm)
        class_names = [CIFAR10_CLASSES[i] for i in ids]
        channels = 3
    else:
        raise ValueError(params.dataset)

    train_ds = FilteredDataset(train_base, ids, params.max_train_items, params.seed + 1)
    test_ds = FilteredDataset(test_base, ids, params.max_test_items, params.seed + 2)
    info = {
        "dataset": params.dataset,
        "classes": class_names,
        "class_ids": ids,
        "C": len(ids),
        "channels": channels,
        "image_size": params.image_size,
        "train_n": len(train_ds),
        "test_n": len(test_ds),
        "data_dim": channels * params.image_size * params.image_size,
    }
    return train_ds, test_ds, info


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, seed: int, pin_memory: bool):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


# =============================================================================
# VP diffusion utilities
# =============================================================================


def alpha_sigma(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # t can be scalar or [B]
    alpha = torch.exp(-t)
    sigma2 = (1.0 - torch.exp(-2.0 * t)).clamp_min(1e-12)
    sigma = sigma2.sqrt()
    return alpha, sigma


def expand_like(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    while v.ndim < x.ndim:
        v = v[..., None]
    return v


def diffuse_x0(x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = torch.randn_like(x0)
    a, s = alpha_sigma(t)
    xt = expand_like(a, x0) * x0 + expand_like(s, x0) * eps
    return xt, eps


def sample_t(batch: int, params: "Params", device: torch.device) -> torch.Tensor:
    # uniform in t; simple and stable for the legacy VP convention used in this project
    return torch.rand(batch, device=device) * (params.t_max - params.t_min) + params.t_min


def ddim_step_from_eps(x: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
    a, s = alpha_sigma(t)
    an, sn = alpha_sigma(t_next)
    x0_hat = (x - expand_like(s, x) * eps_pred) / expand_like(a, x).clamp_min(1e-6)
    # Mild clipping is important for stability in small MNIST/CIFAR runs.
    x0_hat = x0_hat.clamp(-1.5, 1.5)
    return expand_like(an, x) * x0_hat + expand_like(sn, x) * eps_pred


# =============================================================================
# Model
# =============================================================================


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


class ResBlock(nn.Module):
    def __init__(self, channels: int, temb_dim: int):
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.temb = nn.Linear(temb_dim, channels)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class EpsBackbone(nn.Module):
    def __init__(self, in_ch: int, width: int = 64, temb_dim: int = 128):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim * 2), nn.SiLU(), nn.Linear(temb_dim * 2, temb_dim)
        )
        self.conv_in = nn.Conv2d(in_ch, width, 3, padding=1)
        self.rb1 = ResBlock(width, temb_dim)
        self.down = nn.Conv2d(width, width, 4, stride=2, padding=1)
        self.rb2 = ResBlock(width, temb_dim)
        self.mid1 = ResBlock(width, temb_dim)
        self.mid2 = ResBlock(width, temb_dim)
        self.up = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.rb3 = ResBlock(width, temb_dim)
        self.rb4 = ResBlock(width, temb_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(sinusoidal_embedding(t, self.temb_dim))
        h0 = self.conv_in(x)
        h1 = self.rb1(h0, temb)
        h2 = self.down(h1)
        h2 = self.rb2(h2, temb)
        h2 = self.mid1(h2, temb)
        h2 = self.mid2(h2, temb)
        h3 = self.up(h2)
        if h3.shape[-2:] != h1.shape[-2:]:
            h3 = F.interpolate(h3, size=h1.shape[-2:], mode="nearest")
        h = h3 + h1
        h = self.rb3(h, temb)
        h = self.rb4(h, temb)
        return h


class BaselineEpsNet(nn.Module):
    def __init__(self, in_ch: int, width: int = 64, temb_dim: int = 128):
        super().__init__()
        self.backbone = EpsBackbone(in_ch, width, temb_dim)
        self.head = nn.Conv2d(width, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x, t))


class MCLEpsNet(nn.Module):
    def __init__(self, in_ch: int, K: int, width: int = 64, temb_dim: int = 128):
        super().__init__()
        self.K = K
        self.in_ch = in_ch
        self.backbone = EpsBackbone(in_ch, width, temb_dim)
        self.head = nn.Conv2d(width, K * in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x, t)
        out = self.head(h)
        b, _, hh, ww = out.shape
        return out.view(b, self.K, self.in_ch, hh, ww)

    @torch.no_grad()
    def init_from_baseline(self, baseline: BaselineEpsNet, noise: float = 1e-3) -> None:
        self.backbone.load_state_dict(baseline.backbone.state_dict())
        w = baseline.head.weight.data
        b = baseline.head.bias.data if baseline.head.bias is not None else None
        for k in range(self.K):
            self.head.weight.data[k * self.in_ch:(k + 1) * self.in_ch].copy_(w)
            self.head.weight.data[k * self.in_ch:(k + 1) * self.in_ch].add_(noise * torch.randn_like(w))
            if b is not None:
                self.head.bias.data[k * self.in_ch:(k + 1) * self.in_ch].copy_(b)
                self.head.bias.data[k * self.in_ch:(k + 1) * self.in_ch].add_(noise * torch.randn_like(b))


# =============================================================================
# Theoretical beta(t) and MCL loss
# =============================================================================


@dataclass
class BetaCalibrator:
    data_dim: int
    v0_image: float
    beta_rho: float
    beta_min: float
    beta_max: float

    def v_t(self, t: torch.Tensor) -> torch.Tensor:
        a, s = alpha_sigma(t)
        return (a * a) * float(self.v0_image) + (s * s)

    def beta_x(self, t: torch.Tensor) -> torch.Tensor:
        # Costs are mean squared errors, so ||.||^2/(2v) = [data_dim/(2v)] * MSE.
        beta = self.beta_rho * float(self.data_dim) / (2.0 * self.v_t(t).clamp_min(1e-8))
        return beta.clamp(float(self.beta_min), float(self.beta_max))


def beta_ramp_factor(step: int, warmup: int, ramp: int) -> float:
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    u = min(1.0, max(0.0, (step - warmup) / float(ramp)))
    return u * u * (3.0 - 2.0 * u)


def mcl_softmin_loss(e: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # e: [B,K], beta: [B]
    K = e.shape[1]
    beta = beta.clamp_min(1e-8)
    logits = -beta[:, None] * e
    q = torch.softmax(logits, dim=1)
    loss_i = -(torch.logsumexp(logits, dim=1) - math.log(K)) / beta
    return loss_i.mean(), q


def mcl_hard_wta_loss(e: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # beta only used for diagnostics/q; gradient is hard WTA.
    idx = e.argmin(dim=1)
    q = F.one_hot(idx, num_classes=e.shape[1]).float()
    return e.gather(1, idx[:, None]).mean(), q


# =============================================================================
# Router statistics: p(c | x_t,t) and Bayes-risk A_{c,k}(t)
# =============================================================================


@dataclass
class RouterStats:
    dataset: str
    classes: List[str]
    C: int
    image_shape: Tuple[int, int, int]
    pca_dim: int
    flat_mean: torch.Tensor
    pca_components: torch.Tensor
    class_mean_z: torch.Tensor
    class_var_z_scalar: torch.Tensor
    class_prior: torch.Tensor
    common_v0_image: float

    def to(self, device: torch.device) -> "RouterStats":
        return RouterStats(
            dataset=self.dataset,
            classes=self.classes,
            C=self.C,
            image_shape=self.image_shape,
            pca_dim=self.pca_dim,
            flat_mean=self.flat_mean.to(device),
            pca_components=self.pca_components.to(device),
            class_mean_z=self.class_mean_z.to(device),
            class_var_z_scalar=self.class_var_z_scalar.to(device),
            class_prior=self.class_prior.to(device),
            common_v0_image=float(self.common_v0_image),
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        return (flat - self.flat_mean) @ self.pca_components.T

    def posterior(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Gaussian class posterior in PCA latent space.
        # z_t | c ~ N(alpha_t mu_c, (alpha_t^2 var_c + sigma_t^2) I)
        z = self.project(x_t)
        a, s = alpha_sigma(t)
        mu_t = a[:, None, None] * self.class_mean_z[None, :, :]
        var_t = (a[:, None] ** 2) * self.class_var_z_scalar[None, :] + (s[:, None] ** 2)
        dist = ((z[:, None, :] - mu_t) ** 2).sum(-1)
        logits = -0.5 * dist / var_t.clamp_min(1e-8)
        logits = logits - 0.5 * float(self.pca_dim) * var_t.clamp_min(1e-8).log()
        logits = logits + self.class_prior.clamp_min(1e-12).log()[None, :]
        return torch.softmax(logits, dim=1)


@dataclass
class RiskTable:
    t_grid: torch.Tensor          # [T]
    A: torch.Tensor               # [T,C,K]
    counts: torch.Tensor          # [T,C]
    beta_grid: torch.Tensor       # [T]
    v_grid: torch.Tensor          # [T]

    def to(self, device: torch.device) -> "RiskTable":
        return RiskTable(
            t_grid=self.t_grid.to(device),
            A=self.A.to(device),
            counts=self.counts.to(device),
            beta_grid=self.beta_grid.to(device),
            v_grid=self.v_grid.to(device),
        )

    def nearest_indices(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        return torch.cdist(t[:, None], self.t_grid[:, None]).argmin(dim=1)


class BayesRiskRouter:
    def __init__(self, stats: RouterStats, risk: RiskTable, soft_temp: float = 1.0):
        self.stats = stats
        self.risk = risk
        self.soft_temp = soft_temp

    @torch.no_grad()
    def posterior(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.stats.posterior(x, t)

    @torch.no_grad()
    def risk_values(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pc = self.posterior(x, t)              # [B,C]
        idx = self.risk.nearest_indices(t)     # [B]
        A = self.risk.A[idx]                   # [B,C,K]
        return torch.einsum("bc,bck->bk", pc, A)

    @torch.no_grad()
    def hard(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        r = self.risk_values(x, t)
        idx = r.argmin(dim=1)
        sorted_r = torch.sort(r, dim=1).values
        margin = sorted_r[:, 1] - sorted_r[:, 0] if r.shape[1] > 1 else torch.zeros_like(idx, dtype=torch.float)
        diag = {
            "risk_margin_mean": float(margin.mean().item()),
            "risk_margin_min": float(margin.min().item()),
            "risk_entropy_usage": normalized_entropy_from_counts(torch.bincount(idx.cpu(), minlength=r.shape[1])),
        }
        return idx, diag

    @torch.no_grad()
    def soft_weights(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r = self.risk_values(x, t)
        temp = max(float(self.soft_temp), 1e-8)
        return torch.softmax(-r / temp, dim=1)


@torch.no_grad()
def build_router_stats(
    params: "Params",
    train_ds,
    info: Dict,
    device: torch.device,
) -> RouterStats:
    # Use a subset for PCA/statistics to keep it cheap.
    n = min(len(train_ds), params.router_stats_items)
    xs: List[torch.Tensor] = []
    ys: List[int] = []
    for i in range(n):
        x, y = train_ds[i]
        xs.append(x)
        ys.append(int(y))
    X = torch.stack(xs).float()
    y = torch.tensor(ys, dtype=torch.long)
    flat = X.flatten(1)
    flat_mean = flat.mean(0, keepdim=True)
    Xc = flat - flat_mean
    q = min(params.pca_dim_router, Xc.shape[0] - 1, Xc.shape[1])
    # pca_lowrank returns V [D,q]
    _, _, V = torch.pca_lowrank(Xc, q=q, center=False, niter=4)
    comps = V[:, :q].T.contiguous()
    Z = Xc @ comps.T

    C = int(info["C"])
    means = []
    vars_scalar = []
    priors = []
    for c in range(C):
        m = y == c
        if not bool(m.any()):
            means.append(torch.zeros(q))
            vars_scalar.append(torch.tensor(1.0))
            priors.append(torch.tensor(1e-6))
            continue
        Zc = Z[m]
        means.append(Zc.mean(0))
        vars_scalar.append(((Zc - Zc.mean(0)) ** 2).mean().clamp_min(1e-6))
        priors.append(m.float().mean().clamp_min(1e-6))

    class_mean_z = torch.stack(means)
    class_var_z_scalar = torch.stack(vars_scalar)
    class_prior = torch.stack(priors)
    class_prior = class_prior / class_prior.sum().clamp_min(1e-12)

    # Image-space v0 for beta(t): E ||x0 - class_mean_image||^2 / d.
    class_means_flat = []
    for c in range(C):
        m = y == c
        class_means_flat.append(flat[m].mean(0) if bool(m.any()) else flat.mean(0))
    class_means_flat = torch.stack(class_means_flat)
    residual = flat - class_means_flat[y]
    v0_image = float(residual.pow(2).mean().item())

    stats = RouterStats(
        dataset=params.dataset,
        classes=info["classes"],
        C=C,
        image_shape=(info["channels"], info["image_size"], info["image_size"]),
        pca_dim=q,
        flat_mean=flat_mean.detach().cpu(),
        pca_components=comps.detach().cpu(),
        class_mean_z=class_mean_z.detach().cpu(),
        class_var_z_scalar=class_var_z_scalar.detach().cpu(),
        class_prior=class_prior.detach().cpu(),
        common_v0_image=v0_image,
    )
    return stats



# =============================================================================
# V9 theory validation: beta_x(t), MCL speciation probes, and plots
# =============================================================================


@torch.no_grad()
def validate_beta_x_temperature(
    params: "Params",
    loader,
    info: Dict,
    device: torch.device,
    outdir: Path,
    stats: RouterStats,
    beta_cal: BetaCalibrator,
) -> List[Dict]:
    """
    Empirically fit the inverse temperature in the class posterior

        p_beta(c|x_t) ∝ pi_c exp[- beta * mean_j (z_tj - alpha_t mu_cj)^2]

    and compare it to the Gaussian-channel theory beta_z(t)=q/(2 v_z(t)).
    This is the practical check behind the claim beta_x(t)=d/(2v_t) when the
    energy is an averaged MSE over d coordinates.
    """
    stats_d = stats.to(device)
    t_grid = torch.linspace(params.t_min, params.t_max, params.beta_validation_t_bins, device=device)
    rows: List[Dict] = []
    C = int(info["C"])
    qdim = int(stats.pca_dim)
    max_batches = int(params.beta_validation_batches)
    mults = torch.exp(torch.linspace(
        math.log(max(params.beta_sweep_min_mult, 1e-6)),
        math.log(max(params.beta_sweep_max_mult, params.beta_sweep_min_mult + 1e-6)),
        int(params.beta_sweep_points),
        device=device,
    ))

    # Accumulate batches once to keep all t comparable.
    cached: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for bi, (x0, y) in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        cached.append((x0.to(device), y.to(device).long()))
    if not cached:
        return rows

    for tval in t_grid:
        ce_by_mult = torch.zeros_like(mults)
        acc_by_mult = torch.zeros_like(mults)
        n_total = 0
        a, sig = alpha_sigma(tval[None])
        # Equal-variance scalar theory in PCA latent space.
        v0_z = stats_d.class_var_z_scalar.mean()
        v_z = (a[0] * a[0]) * v0_z + sig[0] * sig[0]
        beta_theory_z = float(qdim / (2.0 * v_z.clamp_min(1e-8)).item())
        beta_theory_image = float(beta_cal.beta_x(tval[None]).item())
        posterior_ce_vals: List[torch.Tensor] = []
        posterior_acc_vals: List[torch.Tensor] = []
        for x0, y in cached:
            B = x0.shape[0]
            t = torch.full((B,), float(tval.item()), device=device)
            xt, _ = diffuse_x0(x0, t)
            z = stats_d.project(xt)
            mu = a[0] * stats_d.class_mean_z
            # Energy averaged over PCA coordinates, so beta_theory_z=q/(2v_z).
            e = ((z[:, None, :] - mu[None, :, :]) ** 2).mean(-1)
            for mi, mult in enumerate(mults):
                logits = -(beta_theory_z * mult) * e + stats_d.class_prior.clamp_min(1e-12).log()[None, :]
                ce = F.cross_entropy(logits, y, reduction="sum")
                pred = logits.argmax(1)
                ce_by_mult[mi] += ce
                acc_by_mult[mi] += (pred == y).float().sum()
            pc = stats_d.posterior(xt, t)
            posterior_ce_vals.append(F.nll_loss(pc.clamp_min(1e-12).log(), y, reduction="none"))
            posterior_acc_vals.append((pc.argmax(1) == y).float())
            n_total += B
        ce_by_mult = ce_by_mult / max(1, n_total)
        acc_by_mult = acc_by_mult / max(1, n_total)
        best_i = int(torch.argmin(ce_by_mult).item())
        beta_emp = beta_theory_z * float(mults[best_i].item())
        theory_i = int(torch.argmin((mults - 1.0).abs()).item())
        pc_ce = torch.cat(posterior_ce_vals)
        pc_acc = torch.cat(posterior_acc_vals)
        rows.append({
            "t": float(tval.item()),
            "pca_dim": qdim,
            "v_z_theory": float(v_z.item()),
            "beta_z_theory": beta_theory_z,
            "beta_image_theory": beta_theory_image,
            "beta_emp_ce_min": beta_emp,
            "beta_emp_over_beta_z_theory": float(beta_emp / max(beta_theory_z, 1e-12)),
            "ce_at_beta_z_theory": float(ce_by_mult[theory_i].item()),
            "ce_best_beta": float(ce_by_mult[best_i].item()),
            "acc_at_beta_z_theory": float(acc_by_mult[theory_i].item()),
            "acc_best_beta": float(acc_by_mult[best_i].item()),
            "gaussian_posterior_ce": float(pc_ce.mean().item()),
            "gaussian_posterior_acc": float(pc_acc.mean().item()),
            "n": int(n_total),
        })
    write_csv(outdir / "beta_validation_by_t.csv", rows)
    return rows


@torch.no_grad()
def probe_mcl_speciation(
    params: "Params",
    model: MCLEpsNet,
    loader,
    info: Dict,
    device: torch.device,
    step: int,
    outdir: Path,
    stats: RouterStats,
    beta_cal: BetaCalibrator,
) -> List[Dict]:
    """
    Periodic MCL dynamics probe.  This is the practical check for the theory:

      - beta_gap = beta(t) * [mean expert risk - best expert risk]
      - teacher entropy q(k|x_t, eps, t)
      - oracle assignment MI with labels
      - Bayes-risk route excess using A_{c,k}(t) estimated from this probe batch
      - risk-margin / A singular values as signs of expert speciation
    """
    was_training = model.training
    model.eval()
    stats_d = stats.to(device)
    t_grid = torch.linspace(params.t_min, params.t_max, params.probe_t_bins, device=device)
    C = int(info["C"])
    K = int(params.K)
    rows: List[Dict] = []
    max_batches = int(params.probe_batches)

    cached: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for bi, (x0, y) in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        cached.append((x0.to(device), y.to(device).long()))
    if not cached:
        if was_training:
            model.train()
        return rows

    for tval in t_grid:
        sum_e = torch.zeros(C, K, device=device)
        count_c = torch.zeros(C, device=device)
        all_e: List[torch.Tensor] = []
        all_y: List[torch.Tensor] = []
        all_x: List[torch.Tensor] = []
        all_q: List[torch.Tensor] = []
        all_oracle: List[torch.Tensor] = []
        all_pc: List[torch.Tensor] = []
        beta_vals: List[torch.Tensor] = []
        for x0, y in cached:
            B = x0.shape[0]
            t = torch.full((B,), float(tval.item()), device=device)
            xt, eps = diffuse_x0(x0, t)
            pred = model(xt, t)
            e = (pred - eps[:, None]).pow(2).flatten(2).mean(-1)
            beta = beta_cal.beta_x(t)
            q = torch.softmax(-beta[:, None] * e, dim=1)
            for c in range(C):
                m = y == c
                if bool(m.any()):
                    sum_e[c] += e[m].sum(0)
                    count_c[c] += float(m.sum().item())
            all_e.append(e)
            all_y.append(y)
            all_x.append(xt)
            all_q.append(q)
            all_oracle.append(e.argmin(1))
            all_pc.append(stats_d.posterior(xt, t))
            beta_vals.append(beta)
        A = sum_e / count_c[:, None].clamp_min(1.0)
        e_all = torch.cat(all_e, 0)
        y_all = torch.cat(all_y, 0)
        q_all = torch.cat(all_q, 0)
        oracle_all = torch.cat(all_oracle, 0)
        pc_all = torch.cat(all_pc, 0)
        beta_all = torch.cat(beta_vals, 0)
        # Bayes-risk route from probe-estimated A.
        risk_vals = torch.einsum("bc,ck->bk", pc_all, A)
        route = risk_vals.argmin(1)
        sorted_risk = torch.sort(risk_vals, dim=1).values
        risk_margin = sorted_risk[:, 1] - sorted_risk[:, 0] if K > 1 else torch.zeros_like(route, dtype=torch.float)
        route_excess = e_all.gather(1, route[:, None]).squeeze(1) - e_all.min(1).values
        cost_gap_mean = (e_all.mean(1) - e_all.min(1).values).mean()
        top2 = torch.topk(e_all, k=min(2, K), largest=False, dim=1).values
        top2_gap = (top2[:, 1] - top2[:, 0]).mean() if K > 1 else torch.tensor(0.0, device=device)
        usage_teacher = q_all.mean(0)
        usage_oracle = torch.bincount(oracle_all, minlength=K).float().to(device)
        usage_route = torch.bincount(route, minlength=K).float().to(device)
        svals = safe_svdvals(A)
        if K > 1:
            d_terms = []
            for c in range(C):
                vals = A[c]
                dmat = (vals[:, None] - vals[None, :]).abs() + torch.eye(K, device=device) * 1e9
                d_terms.append(dmat.min())
            delta_A = torch.stack(d_terms).mean()
        else:
            delta_A = torch.tensor(0.0, device=device)
        ent_sample = (-(q_all.clamp_min(1e-12) * q_all.clamp_min(1e-12).log()).sum(1) / math.log(K)).mean()
        row = {
            "step": int(step),
            "t": float(tval.item()),
            "beta_x_mean": float(beta_all.mean().item()),
            "teacher_entropy_sample_norm": float(ent_sample.item()),
            "teacher_usage_entropy_norm": normalized_entropy_from_counts(usage_teacher.cpu()),
            "oracle_usage_entropy_norm": normalized_entropy_from_counts(usage_oracle.cpu()),
            "risk_route_usage_entropy_norm": normalized_entropy_from_counts(usage_route.cpu()),
            "oracle_class_mi_norm": mi_norm(oracle_all, y_all, K, C),
            "risk_route_class_mi_norm": mi_norm(route, y_all, K, C),
            "cost_gap_mean": float(cost_gap_mean.item()),
            "top2_cost_gap_mean": float(top2_gap.item()),
            "beta_gap_mean": float((beta_all.mean() * cost_gap_mean).item()),
            "beta_top2_gap_mean": float((beta_all.mean() * top2_gap).item()),
            "route_excess_vs_sample_oracle": float(route_excess.mean().item()),
            "route_excess_p95": float(torch.quantile(route_excess, 0.95).item()),
            "risk_margin_mean": float(risk_margin.mean().item()),
            "risk_margin_p05": float(torch.quantile(risk_margin, 0.05).item()),
            "posterior_acc": float((pc_all.argmax(1) == y_all).float().mean().item()),
            "delta_A_mean_min_interexpert": float(delta_A.item()),
            "A_sval_max": float(svals.max().item()) if svals.numel() else 0.0,
            "A_sval_min": float(svals.min().item()) if svals.numel() else 0.0,
            "A_cond_smax_over_smin": float((svals.max() / svals.min().clamp_min(1e-12)).item()) if svals.numel() else 0.0,
            "n": int(e_all.shape[0]),
        }
        for k in range(K):
            row[f"teacher_usage_{k}"] = float(usage_teacher[k].item())
            row[f"oracle_usage_{k}"] = float((usage_oracle[k] / usage_oracle.sum().clamp_min(1)).item())
            row[f"risk_route_usage_{k}"] = float((usage_route[k] / usage_route.sum().clamp_min(1)).item())
            row[f"A_sval_{k}"] = float(svals[k].item()) if k < int(svals.numel()) else float("nan")
        rows.append(row)
    append_csv(outdir / "mcl_speciation_probe_by_step_t.csv", rows)
    if was_training:
        model.train()
    print(
        f"[probe] step={step:07d} "
        f"entropy={np.mean([r['teacher_entropy_sample_norm'] for r in rows]):.4f} "
        f"beta_gap={np.mean([r['beta_gap_mean'] for r in rows]):.4g} "
        f"risk_margin={np.mean([r['risk_margin_mean'] for r in rows]):.4g}",
        flush=True,
    )
    return rows


def plot_v9_outputs(outdir: Path, metrics: Optional[Dict] = None) -> None:
    """Generate paper-friendly diagnostic figures from CSV/JSON outputs."""
    if plt is None or pd is None:
        return
    outdir = Path(outdir)
    try:
        path = outdir / "beta_validation_by_t.csv"
        if path.exists():
            df = pd.read_csv(path)
            plt.figure(figsize=(7.5, 4.8))
            plt.plot(df["t"], df["beta_emp_over_beta_z_theory"], marker="o", label="empirical/theory")
            plt.axhline(1.0, linestyle="--", linewidth=1.0, label="theory")
            plt.xlabel("diffusion time t")
            plt.ylabel("beta empirical / beta theory")
            plt.title("V9 validation of beta_x(t)=d/(2 v_t)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "plot_v9_beta_validation_ratio.png", dpi=180)
            plt.close()

            plt.figure(figsize=(7.5, 4.8))
            plt.plot(df["t"], df["ce_at_beta_z_theory"], marker="o", label="CE at theory beta")
            plt.plot(df["t"], df["ce_best_beta"], marker="o", label="best CE over beta sweep")
            plt.xlabel("diffusion time t")
            plt.ylabel("cross entropy")
            plt.title("Beta sweep CE: theory vs empirical optimum")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "plot_v9_beta_validation_ce.png", dpi=180)
            plt.close()
    except Exception as e:
        print(f"[plot] beta validation failed: {e}", flush=True)

    try:
        path = outdir / "mcl_speciation_probe_by_step_t.csv"
        if path.exists():
            df = pd.read_csv(path)
            for key in ["teacher_entropy_sample_norm", "beta_gap_mean", "risk_margin_mean", "route_excess_vs_sample_oracle", "oracle_class_mi_norm"]:
                if key not in df.columns:
                    continue
                piv = df.pivot_table(index="step", columns="t", values=key, aggfunc="mean")
                plt.figure(figsize=(8.5, 5.2))
                plt.imshow(piv.values, aspect="auto", origin="lower")
                plt.colorbar(label=key)
                plt.xticks(range(len(piv.columns)), [f"{x:.2f}" for x in piv.columns], rotation=45, ha="right")
                plt.yticks(range(len(piv.index)), [str(int(x)) for x in piv.index])
                plt.xlabel("diffusion time t")
                plt.ylabel("training step")
                plt.title(f"V9 MCL dynamics: {key}")
                plt.tight_layout()
                plt.savefig(outdir / f"plot_v9_heatmap_{key}.png", dpi=180)
                plt.close()

            # Compact line plot of final probe over t.
            last_step = df["step"].max()
            dfl = df[df["step"] == last_step]
            plt.figure(figsize=(8.0, 5.0))
            for key in ["teacher_entropy_sample_norm", "beta_gap_mean", "risk_margin_mean", "route_excess_vs_sample_oracle"]:
                if key in dfl.columns:
                    vals = dfl[key].astype(float).values
                    denom = np.nanmax(np.abs(vals))
                    if denom > 0:
                        vals = vals / denom
                    plt.plot(dfl["t"], vals, marker="o", label=f"{key} (norm.)")
            plt.xlabel("diffusion time t")
            plt.ylabel("normalized value")
            plt.title(f"Final MCL phase diagnostics at step {int(last_step)}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(outdir / "plot_v9_final_phase_diagnostics.png", dpi=180)
            plt.close()
    except Exception as e:
        print(f"[plot] speciation probes failed: {e}", flush=True)

    try:
        path = outdir / "router_calibration_by_t.csv"
        if path.exists():
            df = pd.read_csv(path)
            plt.figure(figsize=(8.0, 5.0))
            for key in ["route_excess_vs_sample_oracle", "risk_margin_mean", "teacher_entropy_norm", "posterior_acc"]:
                if key in df.columns:
                    plt.plot(df["t"], df[key], marker="o", label=key)
            plt.xlabel("diffusion time t")
            plt.title("Bayes-risk router calibration by time")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(outdir / "plot_v9_router_calibration.png", dpi=180)
            plt.close()
    except Exception as e:
        print(f"[plot] router calibration failed: {e}", flush=True)

    try:
        if metrics is None and (outdir / "metrics.json").exists():
            metrics = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
        if metrics:
            names = []
            fids = []
            for k, v in metrics.items():
                if isinstance(v, dict) and "fid" in v:
                    names.append(k)
                    fids.append(float(v["fid"]))
                elif isinstance(v, dict) and "fid_diag_fallback" in v:
                    names.append(k)
                    fids.append(float(v["fid_diag_fallback"]))
            if names:
                plt.figure(figsize=(9.0, 5.0))
                plt.bar(np.arange(len(names)), fids)
                plt.xticks(np.arange(len(names)), names, rotation=35, ha="right")
                plt.ylabel("FID or fallback diagnostic")
                plt.title("V9 generation metrics")
                plt.tight_layout()
                plt.savefig(outdir / "plot_v9_generation_metrics.png", dpi=180)
                plt.close()
    except Exception as e:
        print(f"[plot] metrics bar failed: {e}", flush=True)


def make_v9_report(outdir: Path, params: "Params", metrics: Dict) -> None:
    outdir = Path(outdir)
    lines: List[str] = []
    lines.append("# V9 theory → routing → generation report")
    lines.append("")
    lines.append("## What this run validates")
    lines.append("1. The Bayes/Gaussian inverse-temperature law `beta_x(t)=d/(2 v_t)`, measured by an empirical CE sweep.")
    lines.append("2. MCL expert speciation during training: teacher entropy, beta-gap, A-matrix rank/singular values, route margins.")
    lines.append("3. Deployable Bayes-risk routing, including softmix, hard gate, confident gate, and commit-once routing.")
    lines.append("4. Generation impact relative to baseline, single expert, mixture score, and risk-based routers.")
    lines.append("")
    if (outdir / "metrics.json").exists():
        try:
            m = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            m = metrics
    else:
        m = metrics
    if m:
        lines.append("## Generation metrics")
        lines.append("")
        lines.append("| strategy | FID | diag fallback | precision | recall | fallback frac | commit frac |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, vals in m.items():
            if not isinstance(vals, dict):
                continue
            lines.append(
                f"| {name} | {vals.get('fid', float('nan'))} | {vals.get('fid_diag_fallback', float('nan'))} | "
                f"{vals.get('precision', float('nan'))} | {vals.get('recall', float('nan'))} | "
                f"{vals.get('fallback_fraction', float('nan'))} | {vals.get('commit_fraction', float('nan'))} |"
            )
        lines.append("")
    if (outdir / "beta_validation_by_t.csv").exists() and pd is not None:
        df = pd.read_csv(outdir / "beta_validation_by_t.csv")
        if "beta_emp_over_beta_z_theory" in df:
            lines.append("## Beta validation")
            lines.append("")
            lines.append(f"Mean empirical/theory ratio: `{df['beta_emp_over_beta_z_theory'].mean():.4g}`.")
            lines.append(f"Min/max ratio: `{df['beta_emp_over_beta_z_theory'].min():.4g}` / `{df['beta_emp_over_beta_z_theory'].max():.4g}`.")
            lines.append("")
    if (outdir / "mcl_speciation_probe_by_step_t.csv").exists() and pd is not None:
        df = pd.read_csv(outdir / "mcl_speciation_probe_by_step_t.csv")
        lines.append("## MCL speciation probe")
        lines.append("")
        for key in ["teacher_entropy_sample_norm", "beta_gap_mean", "risk_margin_mean", "route_excess_vs_sample_oracle", "oracle_class_mi_norm"]:
            if key in df:
                lines.append(f"- `{key}` final-step mean: `{df[df['step']==df['step'].max()][key].mean():.6g}`")
        lines.append("")
    lines.append("## Produced plots")
    for png in sorted(outdir.glob("plot_v9_*.png")):
        lines.append(f"- `{png.name}`")
    lines.append("")
    lines.append("## Suggested reading of the run")
    lines.append("- If beta ratio is near 1 but teacher entropy and risk margin stay near 0, the Bayes temperature is right but MCL experts have not speciated.")
    lines.append("- If beta-gap grows and risk margin becomes non-zero, then commit-once routing should become meaningful.")
    lines.append("- If risk_softmix beats hard routing, the model is still in a soft/speciation boundary regime; if commit-once wins, the paper-style one-time routing hypothesis is supported.")
    (outdir / "RESULTS_V9.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Training
# =============================================================================


def save_ckpt(path: Path, model: nn.Module, params: "Params", info: Dict, extra: Optional[Dict] = None) -> None:
    payload = {
        "model": model.state_dict(),
        "params": asdict(params),
        "data_info": info,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_baseline_ckpt(path: Path, in_ch: int, params: "Params", device: torch.device) -> BaselineEpsNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = BaselineEpsNet(in_ch, params.width, params.temb_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model


def load_mcl_ckpt(path: Path, in_ch: int, params: "Params", device: torch.device) -> MCLEpsNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MCLEpsNet(in_ch, params.K, params.width, params.temb_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model


def train_baseline(params: "Params", train_loader, info: Dict, device: torch.device, outdir: Path) -> BaselineEpsNet:
    in_ch = int(info["channels"])
    model = BaselineEpsNet(in_ch, params.width, params.temb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    it = cycle(train_loader)
    rows: List[Dict] = []
    model.train()
    for step in range(1, params.baseline_steps + 1):
        x0, _ = next(it)
        x0 = x0.to(device)
        t = sample_t(x0.shape[0], params, device)
        xt, eps = diffuse_x0(x0, t)
        pred = model(xt, t)
        loss = F.mse_loss(pred, eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
        opt.step()
        if step % params.log_every == 0 or step == 1:
            row = {"step": step, "baseline_loss": float(loss.item())}
            rows.append(row)
            print(f"[baseline] step={step:07d} loss={loss.item():.6f}", flush=True)
    write_csv(outdir / "baseline_train.csv", rows)
    save_ckpt(outdir / "baseline_final.pt", model, params, info)
    return model


def train_mcl(
    params: "Params",
    train_loader,
    info: Dict,
    device: torch.device,
    outdir: Path,
    baseline: Optional[BaselineEpsNet],
    beta_cal: BetaCalibrator,
    stats: Optional[RouterStats] = None,
    probe_loader=None,
) -> MCLEpsNet:
    in_ch = int(info["channels"])
    model = MCLEpsNet(in_ch, params.K, params.width, params.temb_dim).to(device)
    if baseline is not None and params.init_mcl_from_baseline:
        model.init_from_baseline(baseline, noise=params.mcl_init_noise)
        print("[mcl] initialized shared backbone/heads from baseline", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    it = cycle(train_loader)
    rows: List[Dict] = []
    model.train()
    for step in range(1, params.mcl_steps + 1):
        x0, y = next(it)
        x0 = x0.to(device)
        t = sample_t(x0.shape[0], params, device)
        xt, eps = diffuse_x0(x0, t)
        pred = model(xt, t)
        e = (pred - eps[:, None]).pow(2).flatten(2).mean(-1)  # [B,K]
        ramp = beta_ramp_factor(step, params.mcl_warmup_steps, params.mcl_ramp_steps)
        beta = beta_cal.beta_x(t) * ramp
        if params.hard_wta:
            loss, q = mcl_hard_wta_loss(e, beta)
        else:
            # During warmup beta=0 would be singular. Use uniform mean loss.
            if ramp <= 1e-8:
                q = torch.full_like(e, 1.0 / params.K)
                loss = e.mean()
            else:
                loss, q = mcl_softmin_loss(e, beta)
        if params.balance_weight > 0:
            usage = q.mean(0)
            loss = loss + params.balance_weight * ((usage - 1.0 / params.K) ** 2).sum()
        if params.diversity_weight > 0:
            # Encourage experts not to be identical, softly. Negative pairwise variance term.
            mean_pred = pred.mean(1, keepdim=True)
            div = (pred - mean_pred).pow(2).mean()
            loss = loss - params.diversity_weight * div
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
        opt.step()

        if step % params.log_every == 0 or step == 1:
            usage = q.mean(0).detach().cpu()
            ent = float((-(usage.clamp_min(1e-12) * usage.clamp_min(1e-12).log()).sum() / math.log(params.K)).item())
            row = {
                "step": step,
                "loss": float(loss.item()),
                "mcl_e_best": float(e.min(1).values.mean().item()),
                "mcl_e_mean": float(e.mean().item()),
                "beta_mean": float(beta.mean().item()),
                "beta_min": float(beta.min().item()),
                "beta_max": float(beta.max().item()),
                "ramp": float(ramp),
                "teacher_entropy_usage": ent,
            }
            for k in range(params.K):
                row[f"usage_{k}"] = float(usage[k].item())
            rows.append(row)
            print(
                f"[mcl] step={step:07d} loss={loss.item():.6f} "
                f"best={row['mcl_e_best']:.6f} mean={row['mcl_e_mean']:.6f} "
                f"beta={row['beta_mean']:.3f} usage={usage.numpy().round(3).tolist()}",
                flush=True,
            )
        if (
            params.probe_every > 0
            and stats is not None
            and probe_loader is not None
            and (step == 1 or step % params.probe_every == 0 or step == params.mcl_steps)
        ):
            probe_mcl_speciation(params, model, probe_loader, info, device, step, outdir, stats, beta_cal)
    write_csv(outdir / "mcl_train.csv", rows)
    save_ckpt(outdir / "mcl_final.pt", model, params, info, extra={"beta_calibrator": asdict(beta_cal)})
    return model


# =============================================================================
# Router calibration and diagnostics
# =============================================================================


@torch.no_grad()
def calibrate_risk_table(
    params: "Params",
    model: MCLEpsNet,
    loader,
    info: Dict,
    device: torch.device,
    outdir: Path,
    stats: RouterStats,
    beta_cal: BetaCalibrator,
) -> RiskTable:
    model.eval()
    stats_d = stats.to(device)
    t_grid = torch.linspace(params.t_min, params.t_max, params.router_t_bins, device=device)
    C = int(info["C"])
    K = int(params.K)
    sum_e = torch.zeros(params.router_t_bins, C, K, device=device)
    counts = torch.zeros(params.router_t_bins, C, device=device)
    rows: List[Dict] = []

    # First pass: estimate A_{c,k}(t).
    max_batches = params.router_calib_batches
    for bi, (x0, y) in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        x0 = x0.to(device)
        y = y.to(device).long()
        B = x0.shape[0]
        for ti, tval in enumerate(t_grid):
            t = torch.full((B,), float(tval.item()), device=device)
            xt, eps = diffuse_x0(x0, t)
            pred = model(xt, t)
            e = (pred - eps[:, None]).pow(2).flatten(2).mean(-1)
            for c in range(C):
                m = y == c
                if bool(m.any()):
                    sum_e[ti, c] += e[m].sum(0)
                    counts[ti, c] += float(m.sum().item())

    A = sum_e / counts[:, :, None].clamp_min(1.0)
    beta_grid = beta_cal.beta_x(t_grid)
    v_grid = beta_cal.v_t(t_grid)
    risk_table = RiskTable(
        t_grid=t_grid.detach().cpu(),
        A=A.detach().cpu(),
        counts=counts.detach().cpu(),
        beta_grid=beta_grid.detach().cpu(),
        v_grid=v_grid.detach().cpu(),
    )

    # Second pass: offline diagnostics for theoretical router.
    router = BayesRiskRouter(stats_d, risk_table.to(device), soft_temp=params.risk_soft_temp)
    diag_accum: Dict[str, List[float]] = {"route_excess": [], "teacher_entropy": [], "risk_margin": [], "posterior_acc": []}
    usage_teacher = torch.zeros(params.router_t_bins, K, device=device)
    usage_risk = torch.zeros(params.router_t_bins, K, device=device)
    usage_oracle = torch.zeros(params.router_t_bins, K, device=device)

    for ti, tval in enumerate(t_grid):
        n_seen = 0
        route_excess_vals = []
        teacher_entropy_vals = []
        risk_margin_vals = []
        posterior_acc_vals = []
        for bi, (x0, y) in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            x0 = x0.to(device)
            y = y.to(device).long()
            B = x0.shape[0]
            t = torch.full((B,), float(tval.item()), device=device)
            xt, eps = diffuse_x0(x0, t)
            pred = model(xt, t)
            e = (pred - eps[:, None]).pow(2).flatten(2).mean(-1)
            beta = beta_cal.beta_x(t)
            q = torch.softmax(-beta[:, None] * e, dim=1)
            oracle = e.argmin(1)
            risk_vals = router.risk_values(xt, t)
            route = risk_vals.argmin(1)
            pc = router.posterior(xt, t)
            pc_pred = pc.argmax(1)
            sorted_r = torch.sort(risk_vals, dim=1).values
            margin = sorted_r[:, 1] - sorted_r[:, 0] if K > 1 else torch.zeros(B, device=device)

            route_excess_vals.append((e.gather(1, route[:, None]).squeeze(1) - e.min(1).values).detach())
            teacher_entropy_vals.append((-(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(1) / math.log(K)).detach())
            risk_margin_vals.append(margin.detach())
            posterior_acc_vals.append((pc_pred == y).float().detach())

            usage_teacher[ti] += q.sum(0)
            usage_risk[ti] += torch.bincount(route, minlength=K).float()
            usage_oracle[ti] += torch.bincount(oracle, minlength=K).float()
            n_seen += B

        route_excess = torch.cat(route_excess_vals)
        teacher_entropy = torch.cat(teacher_entropy_vals)
        risk_margin = torch.cat(risk_margin_vals)
        posterior_acc = torch.cat(posterior_acc_vals)
        A_t = A[ti]
        delta_terms: List[torch.Tensor] = []
        if K > 1:
            for c in range(C):
                vals = A_t[c]
                dmat = (vals[:, None] - vals[None, :]).abs()
                dmat = dmat + torch.eye(K, device=device) * 1e9
                delta_terms.append(dmat.min())
            delta_a = torch.stack(delta_terms).mean()
        else:
            delta_a = torch.tensor(0.0, device=device)
        svals = torch.linalg.svdvals(A_t)
        risk_usage_t = usage_risk[ti] / usage_risk[ti].sum().clamp_min(1.0)
        row = {
            "t": float(tval.item()),
            "beta_x": float(beta_grid[ti].item()),
            "v_t": float(v_grid[ti].item()),
            "route_excess_vs_sample_oracle": float(route_excess.mean().item()),
            "route_excess_p95": float(torch.quantile(route_excess, 0.95).item()),
            "teacher_entropy_norm": float(teacher_entropy.mean().item()),
            "risk_margin_mean": float(risk_margin.mean().item()),
            "risk_margin_p05": float(torch.quantile(risk_margin, 0.05).item()),
            "posterior_acc": float(posterior_acc.mean().item()),
            "delta_A_mean_min_interexpert": float(delta_a.item()),
            "A_rank_eps1e-8": float((svals > 1e-8).sum().item()),
            "A_sval_max": float(svals.max().item()),
            "A_sval_min": float(svals.min().item()),
            "A_cond_smax_over_smin": float((svals.max() / svals.min().clamp_min(1e-12)).item()),
            "bayes_decision_entropy_t": float(
                (-(risk_usage_t.clamp_min(1e-12) * risk_usage_t.clamp_min(1e-12).log()).sum() / math.log(K)).item()
            ),
            "n": int(n_seen),
        }
        for k in range(K):
            row[f"risk_usage_{k}"] = float((usage_risk[ti, k] / usage_risk[ti].sum().clamp_min(1)).item())
            row[f"oracle_usage_{k}"] = float((usage_oracle[ti, k] / usage_oracle[ti].sum().clamp_min(1)).item())
            row[f"teacher_usage_{k}"] = float((usage_teacher[ti, k] / usage_teacher[ti].sum().clamp_min(1)).item())
            row[f"A_sval_{k}"] = float(svals[k].item()) if k < int(svals.shape[0]) else float("nan")
        rows.append(row)
        print(
            f"[router-calib] t={row['t']:.3f} beta={row['beta_x']:.3f} "
            f"excess={row['route_excess_vs_sample_oracle']:.6g} "
            f"post_acc={row['posterior_acc']:.3f} ent={row['teacher_entropy_norm']:.3f}",
            flush=True,
        )

    write_csv(outdir / "router_calibration_by_t.csv", rows)
    torch.save({
        "router_stats": stats,
        "risk_table": risk_table,
        "rows": rows,
        "params": asdict(params),
        "data_info": info,
    }, outdir / "router_risk_table.pt")

    if plt is not None:
        try:
            ts = [r["t"] for r in rows]
            for key in ["beta_x", "route_excess_vs_sample_oracle", "teacher_entropy_norm", "posterior_acc", "risk_margin_mean"]:
                plt.figure()
                plt.plot(ts, [r[key] for r in rows])
                plt.xlabel("t")
                plt.ylabel(key)
                plt.tight_layout()
                plt.savefig(outdir / f"router_{key}.png", dpi=160)
                plt.close()
        except Exception:
            pass

    return risk_table


# =============================================================================
# Sampling and evaluation
# =============================================================================


@torch.no_grad()
def sample_baseline(
    model: BaselineEpsNet,
    params: "Params",
    info: Dict,
    device: torch.device,
    n: int,
    init_x: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    model.eval()
    ch, h, w = int(info["channels"]), int(info["image_size"]), int(info["image_size"])
    if init_x is None:
        x = torch.randn(n, ch, h, w, device=device)
    else:
        x = init_x.to(device).clone()
    times = torch.linspace(params.t_max, params.t_min, params.sample_steps + 1, device=device)
    for i in range(params.sample_steps):
        t = torch.full((n,), float(times[i].item()), device=device)
        t_next = torch.full((n,), float(times[i + 1].item()), device=device)
        eps = model(x, t)
        x = ddim_step_from_eps(x, eps, t, t_next)
    return x.clamp(-1, 1).cpu(), {}


@torch.no_grad()
def sample_mcl(
    model: MCLEpsNet,
    params: "Params",
    info: Dict,
    device: torch.device,
    n: int,
    strategy: str,
    router: Optional[BayesRiskRouter] = None,
    init_x: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    model.eval()
    ch, h, w = int(info["channels"]), int(info["image_size"]), int(info["image_size"])
    if init_x is None:
        x = torch.randn(n, ch, h, w, device=device)
    else:
        x = init_x.to(device).clone()
    times = torch.linspace(params.t_max, params.t_min, params.sample_steps + 1, device=device)
    K = params.K
    usage = torch.zeros(K, device=device)
    fallback = 0
    total = 0
    margins: List[float] = []
    commit_times = torch.full((n,), float("nan"), device=device)
    committed = torch.full((n,), -1, device=device, dtype=torch.long)
    static_random_idx = torch.randint(0, K, (n,), device=device)

    for i in range(params.sample_steps):
        t = torch.full((n,), float(times[i].item()), device=device)
        t_next = torch.full((n,), float(times[i + 1].item()), device=device)
        pred = model(x, t)  # [B,K,C,H,W]

        if strategy == "single_expert":
            idx = torch.full((n,), params.default_expert, device=device, dtype=torch.long)
            eps = pred[torch.arange(n, device=device), idx]
        elif strategy == "random_expert":
            idx = static_random_idx
            eps = pred[torch.arange(n, device=device), idx]
        elif strategy == "mixture_score":
            idx = torch.full((n,), -1, device=device, dtype=torch.long)
            eps = pred.mean(1)
        elif strategy in {"risk_gated", "risk_confident", "risk_softmix", "risk_commit_once"}:
            if router is None:
                raise ValueError("risk strategies require router")
            risk_vals = router.risk_values(x, t)
            best = risk_vals.argmin(1)
            sorted_r = torch.sort(risk_vals, dim=1).values
            margin = sorted_r[:, 1] - sorted_r[:, 0] if K > 1 else torch.zeros(n, device=device)
            margins.append(float(margin.mean().item()))
            if strategy == "risk_softmix":
                weights = torch.softmax(-risk_vals / max(params.risk_soft_temp, 1e-8), dim=1)
                eps = torch.einsum("bk,bkchw->bchw", weights, pred)
                idx = weights.argmax(1)
            elif strategy == "risk_confident":
                confident = margin >= params.risk_conf_margin
                weights = torch.softmax(-risk_vals / max(params.risk_soft_temp, 1e-8), dim=1)
                soft_eps = torch.einsum("bk,bkchw->bchw", weights, pred)
                hard_eps = pred[torch.arange(n, device=device), best]
                eps = torch.where(confident[:, None, None, None], hard_eps, soft_eps)
                idx = best
                fallback += int((~confident).sum().item())
            elif strategy == "risk_commit_once":
                not_committed = committed < 0
                should_commit = not_committed & ((margin >= params.commit_margin) | (t <= params.commit_force_t))
                committed[should_commit] = best[should_commit]
                commit_times[should_commit] = t[should_commit]
                idx = committed.clone()
                weights = torch.softmax(-risk_vals / max(params.risk_soft_temp, 1e-8), dim=1)
                soft_eps = torch.einsum("bk,bkchw->bchw", weights, pred)
                hard_idx = idx.clamp_min(0)
                hard_eps = pred[torch.arange(n, device=device), hard_idx]
                use_hard = idx >= 0
                if params.commit_soft_before:
                    eps = torch.where(use_hard[:, None, None, None], hard_eps, soft_eps)
                else:
                    default_idx = torch.full((n,), params.default_expert, device=device, dtype=torch.long)
                    eps = pred[torch.arange(n, device=device), torch.where(use_hard, hard_idx, default_idx)]
                    idx = torch.where(use_hard, hard_idx, default_idx)
                fallback += int((~use_hard).sum().item())
            else:
                idx = best
                eps = pred[torch.arange(n, device=device), idx]
        else:
            raise ValueError(strategy)

        valid_idx = idx[idx >= 0]
        if valid_idx.numel() > 0:
            usage += torch.bincount(valid_idx, minlength=K).float()
        total += n
        x = ddim_step_from_eps(x, eps, t, t_next)

    finite_commit_times = commit_times[torch.isfinite(commit_times)]
    diag = {
        "strategy": strategy,
        "num_samples": int(n),
        "num_steps": int(params.sample_steps),
        "fallback_count": int(fallback),
        "fallback_fraction": float(fallback / max(1, total)),
        "gate_usage_entropy": normalized_entropy_from_counts(usage.cpu()) if usage.sum() > 0 else float("nan"),
        "risk_margin_mean_over_steps": float(np.mean(margins)) if margins else float("nan"),
        "commit_fraction": float((committed >= 0).float().mean().item()),
        "commit_time_mean": float(finite_commit_times.mean().item()) if finite_commit_times.numel() else float("nan"),
        "commit_time_min": float(finite_commit_times.min().item()) if finite_commit_times.numel() else float("nan"),
        "commit_time_max": float(finite_commit_times.max().item()) if finite_commit_times.numel() else float("nan"),
    }
    for k in range(K):
        diag[f"usage_frac_{k}"] = float((usage[k] / usage.sum().clamp_min(1)).item()) if usage.sum() > 0 else float("nan")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return x.clamp(-1, 1).cpu(), diag


def save_samples(outdir: Path, name: str, x: torch.Tensor, params: "Params") -> None:
    torch.save(x, outdir / f"samples_{name}.pt")
    if save_image is not None:
        m = min(x.shape[0], 64)
        grid = make_grid(to01(x[:m]), nrow=int(math.sqrt(m)))
        save_image(grid, outdir / f"samples_{name}.png")


def get_real_eval_images(test_ds, n: int) -> torch.Tensor:
    m = min(n, len(test_ds))
    xs = [test_ds[i][0] for i in range(m)]
    return torch.stack(xs)


@torch.no_grad()
def evaluate_generated(real: torch.Tensor, gen: torch.Tensor, device: torch.device) -> Dict[str, float]:
    # Prefer the repo's own evaluator so V9 metrics are comparable to legacy.
    try:
        c = int(real.shape[1])
        if c == 1:
            repo_root = Path.cwd()
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from src.evaluate import evaluate  # type: ignore
            m = evaluate(real, gen, device=str(device), k=5)
            return {k: float(v) for k, v in m.items()}
        if c == 3 and torchvision is not None:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.DEFAULT
            eval_device = torch.device("cpu")
            model = resnet18(weights=weights).to(eval_device).eval()
            feat_extractor = nn.Sequential(*list(model.children())[:-1]).to(eval_device).eval()
            preprocess = weights.transforms()

            def feats(x: torch.Tensor) -> torch.Tensor:
                xin = to01(x.to(eval_device))
                xin = preprocess(xin)
                f = feat_extractor(xin).flatten(1)
                return f.cpu()

            def batched_features(x: torch.Tensor, batch: int = 64) -> torch.Tensor:
                parts: List[torch.Tensor] = []
                for i in range(0, x.shape[0], batch):
                    parts.append(feats(x[i:i + batch]))
                return torch.cat(parts, 0)

            fr = batched_features(real)
            fg = batched_features(gen)
            from scipy import linalg
            import numpy as npl
            fr_np = fr.numpy()
            fg_np = fg.numpy()
            mu_r = fr_np.mean(axis=0)
            mu_g = fg_np.mean(axis=0)
            cov_r = npl.cov(fr_np, rowvar=False)
            cov_g = npl.cov(fg_np, rowvar=False)
            covmean, _ = linalg.sqrtm(cov_r @ cov_g, disp=False)
            if npl.iscomplexobj(covmean):
                covmean = covmean.real
            fid = float(((mu_r - mu_g) ** 2).sum() + npl.trace(cov_r + cov_g - 2.0 * covmean))
            del model, feat_extractor, fr, fg, fr_np, fg_np
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return {
                "fid": fid,
                "precision": float("nan"),
                "recall": float("nan"),
                "evaluator_note": "cifar_resnet18_features_fid_only",
            }
    except Exception as e:
        warning = f"evaluate_failed: {type(e).__name__}: {e}"
    else:
        warning = "unsupported_channels_for_evaluator"
    real_f = real.flatten(1).float()
    gen_f = gen.flatten(1).float()
    mu_r, mu_g = real_f.mean(0), gen_f.mean(0)
    std_r, std_g = real_f.std(0, unbiased=False), gen_f.std(0, unbiased=False)
    fid_diag = (mu_r - mu_g).pow(2).sum() + (std_r - std_g).pow(2).sum()
    return {
        "fid_diag_fallback": float(fid_diag.item()),
        "precision": float("nan"),
        "recall": float("nan"),
        "evaluator_warning": warning,
    }


def paired_strategy_test(
    params: "Params",
    mcl: MCLEpsNet,
    router: BayesRiskRouter,
    info: Dict,
    test_ds,
    device: torch.device,
    outdir: Path,
) -> Optional[Dict[str, float]]:
    if params.paired_batches <= 0:
        return None
    real = get_real_eval_images(test_ds, params.num_samples)
    n = int(real.shape[0])
    ch, h, w = int(info["channels"]), int(info["image_size"]), int(info["image_size"])
    paired_bs = int(params.sample_batch_size)
    if ch == 3:
        paired_bs = min(paired_bs, 8)
    rows: List[Dict[str, float]] = []
    seed0 = int(params.seed + 90000)
    for b in range(params.paired_batches):
        g = torch.Generator(device=device)
        g.manual_seed(seed0 + b)
        mix_parts: List[torch.Tensor] = []
        soft_parts: List[torch.Tensor] = []
        remaining = n
        while remaining > 0:
            nb = min(paired_bs, remaining)
            init_x = torch.randn(nb, ch, h, w, device=device, generator=g)
            mix_chunk, _ = sample_mcl(mcl, params, info, device, nb, "mixture_score", router, init_x=init_x)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            soft_chunk, _ = sample_mcl(mcl, params, info, device, nb, "risk_softmix", router, init_x=init_x)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            mix_parts.append(mix_chunk)
            soft_parts.append(soft_chunk)
            remaining -= nb
        mix = torch.cat(mix_parts, 0)[:n]
        soft = torch.cat(soft_parts, 0)[:n]
        m_mix = evaluate_generated(real[:n], mix, device)
        m_soft = evaluate_generated(real[:n], soft, device)
        if "fid" in m_mix and "fid" in m_soft:
            diff = float(m_mix["fid"] - m_soft["fid"])
            rows.append(
                {
                    "pair_id": float(b),
                    "fid_mixture_score": float(m_mix["fid"]),
                    "fid_risk_softmix": float(m_soft["fid"]),
                    "fid_diff_mix_minus_soft": diff,
                }
            )
    if not rows:
        return None
    write_csv(outdir / "paired_mixture_vs_risk_softmix.csv", rows)
    diffs = np.array([r["fid_diff_mix_minus_soft"] for r in rows], dtype=np.float64)
    n_pairs = int(diffs.size)
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if n_pairs > 1 else 0.0
    sem = float(std_diff / math.sqrt(n_pairs)) if n_pairs > 0 else float("nan")
    ci95_lo = float(mean_diff - 1.96 * sem) if n_pairs > 1 else float("nan")
    ci95_hi = float(mean_diff + 1.96 * sem) if n_pairs > 1 else float("nan")
    z = abs(mean_diff / sem) if n_pairs > 1 and sem > 0 else 0.0
    p_val = normal_approx_p_value_from_z(z) if n_pairs > 1 and sem > 0 else 1.0
    summary = {
        "n_pairs": float(n_pairs),
        "fid_diff_mix_minus_soft_mean": mean_diff,
        "fid_diff_mix_minus_soft_std": std_diff,
        "fid_diff_mix_minus_soft_ci95_lo": ci95_lo,
        "fid_diff_mix_minus_soft_ci95_hi": ci95_hi,
        "fid_diff_mix_minus_soft_pvalue_normal_approx": float(p_val),
    }
    write_json(outdir / "paired_mixture_vs_risk_softmix_summary.json", summary)
    return summary


# =============================================================================
# Summary
# =============================================================================


def make_summary(outdir: Path, params: "Params", info: Dict, beta_cal: BetaCalibrator, metrics: Dict, router_rows: Optional[List[Dict]] = None) -> None:
    lines: List[str] = []
    lines.append("V9 theory-validated Bayes-risk gated MCL diffusion")
    lines.append("")
    lines.append("Config:")
    lines.append(json.dumps(asdict(params), indent=2))
    lines.append("")
    lines.append("Data info:")
    lines.append(json.dumps(info, indent=2))
    lines.append("")
    lines.append("Beta calibrator:")
    for k, v in asdict(beta_cal).items():
        lines.append(f"  {k}: {v}")
    if router_rows:
        lines.append("")
        lines.append("Router calibration aggregate:")
        for key in [
            "route_excess_vs_sample_oracle",
            "teacher_entropy_norm",
            "posterior_acc",
            "risk_margin_mean",
            "delta_A_mean_min_interexpert",
            "bayes_decision_entropy_t",
            "A_cond_smax_over_smin",
        ]:
            vals = [float(r[key]) for r in router_rows if key in r]
            if vals:
                lines.append(f"  {key}: mean={np.mean(vals):.6g}, max={np.max(vals):.6g}, min={np.min(vals):.6g}")
    if metrics:
        lines.append("")
        lines.append("Generation metrics:")
        for name, m in metrics.items():
            lines.append(f"\n[{name}]")
            for k, v in m.items():
                lines.append(f"  {k}: {v}")
    (outdir / "SUMMARY_v9.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Params / main
# =============================================================================


@dataclass
class Params:
    # Data
    dataset: str = "mnist"          # mnist | cifar10
    data_root: str = "./data"
    classes: str = "all"
    image_size: int = 28
    max_train_items: int = 0
    max_test_items: int = 0
    no_download: bool = False
    num_workers: int = 4

    # Model/training
    K: int = 4
    width: int = 64
    temb_dim: int = 128
    batch_size: int = 256
    baseline_steps: int = 30000
    mcl_steps: int = 60000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    init_mcl_from_baseline: bool = True
    mcl_init_noise: float = 1e-3
    hard_wta: bool = False
    balance_weight: float = 0.02
    diversity_weight: float = 0.0
    mcl_warmup_steps: int = 2000
    mcl_ramp_steps: int = 8000
    log_every: int = 500

    # Diffusion
    t_min: float = 0.02
    t_max: float = 3.0

    # Theory beta(t) = rho * d / (2 v_t), clamped.
    beta_rho: float = 1.0
    beta_min: float = 0.0
    beta_max: float = 80.0

    # Router stats/calibration
    pca_dim_router: int = 64
    router_stats_items: int = 12000
    router_t_bins: int = 24
    router_calib_batches: int = 64
    risk_soft_temp: float = 2e-4
    risk_conf_margin: float = 1e-5
    paired_batches: int = 0

    # V9 beta validation and MCL dynamics probes
    beta_validation_t_bins: int = 24
    beta_validation_batches: int = 32
    beta_sweep_min_mult: float = 0.2
    beta_sweep_max_mult: float = 5.0
    beta_sweep_points: int = 41
    probe_every: int = 5000
    probe_batches: int = 8
    probe_t_bins: int = 16

    # Commit-once routing hypothesis: soft before decision, freeze expert after margin threshold.
    commit_margin: float = 1e-5
    commit_soft_before: bool = True
    commit_force_t: float = 0.35

    # Sampling/eval
    num_samples: int = 2048
    sample_steps: int = 100
    default_expert: int = 0
    sample_batch_size: int = 256
    strategies: str = "baseline_heun,single_expert,random_expert,mixture_score,risk_gated,risk_softmix,risk_confident,risk_commit_once"

    # Stages
    all: bool = False
    validate_beta: bool = False
    train_baseline: bool = False
    train_mcl: bool = False
    calibrate_router: bool = False
    sample_eval: bool = False
    baseline_ckpt: str = ""
    mcl_ckpt: str = ""
    router_ckpt: str = ""

    # System
    seed: int = 0
    device: str = "auto"
    outdir: str = "./outputs/v9"


def compute_beta_calibrator(params: Params, stats: RouterStats, info: Dict) -> BetaCalibrator:
    return BetaCalibrator(
        data_dim=int(info["data_dim"]),
        v0_image=float(stats.common_v0_image),
        beta_rho=float(params.beta_rho),
        beta_min=float(params.beta_min),
        beta_max=float(params.beta_max),
    )


def run(params: Params) -> None:
    if params.all:
        params.validate_beta = True
        params.train_baseline = True
        params.train_mcl = True
        params.calibrate_router = True
        params.sample_eval = True

    set_seed(params.seed)
    device = device_of(params.device)
    outdir = ensure_dir(params.outdir)
    train_ds, test_ds, info = make_datasets(params)
    write_json(outdir / "config.json", asdict(params))
    write_json(outdir / "data_info.json", info)

    print(f"[v9] device={device} outdir={outdir}", flush=True)
    print(f"[v9] data={info}", flush=True)

    use_pin = device.type == "cuda"
    train_loader = make_loader(train_ds, params.batch_size, True, params.num_workers, params.seed + 10, pin_memory=use_pin)
    calib_loader = make_loader(train_ds, params.batch_size, False, params.num_workers, params.seed + 20, pin_memory=use_pin)

    # Router stats are needed before MCL because they provide v0_image for beta(t).
    stats_path = outdir / "router_stats.pt"
    if stats_path.exists() and not params.train_baseline and not params.train_mcl:
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)["router_stats"]
    else:
        print("[v9] building PCA/Gaussian router stats", flush=True)
        stats = build_router_stats(params, train_ds, info, device)
        torch.save({"router_stats": stats, "params": asdict(params), "data_info": info}, stats_path)
    beta_cal = compute_beta_calibrator(params, stats, info)
    write_json(outdir / "beta_calibrator.json", asdict(beta_cal))
    print(f"[v9] beta_calibrator={asdict(beta_cal)}", flush=True)

    if params.validate_beta:
        print("[v9] validating beta_x(t) by empirical CE sweep", flush=True)
        validate_beta_x_temperature(params, calib_loader, info, device, outdir, stats, beta_cal)
        plot_v9_outputs(outdir)

    baseline: Optional[BaselineEpsNet] = None
    mcl: Optional[MCLEpsNet] = None
    in_ch = int(info["channels"])

    if params.train_baseline:
        baseline = train_baseline(params, train_loader, info, device, outdir)
    else:
        bpath = Path(params.baseline_ckpt) if params.baseline_ckpt else outdir / "baseline_final.pt"
        if bpath.exists():
            baseline = load_baseline_ckpt(bpath, in_ch, params, device)

    if params.train_mcl:
        mcl = train_mcl(params, train_loader, info, device, outdir, baseline, beta_cal, stats=stats, probe_loader=calib_loader)
    else:
        mpath = Path(params.mcl_ckpt) if params.mcl_ckpt else outdir / "mcl_final.pt"
        if mpath.exists():
            mcl = load_mcl_ckpt(mpath, in_ch, params, device)

    risk_table: Optional[RiskTable] = None
    router_rows: Optional[List[Dict]] = None
    router_table_path = Path(params.router_ckpt) if params.router_ckpt else outdir / "router_risk_table.pt"

    if params.calibrate_router:
        if mcl is None:
            raise RuntimeError("MCL model required for router calibration. Train or pass --mcl-ckpt.")
        risk_table = calibrate_risk_table(params, mcl, calib_loader, info, device, outdir, stats, beta_cal)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            router_rows = pd.read_csv(outdir / "router_calibration_by_t.csv").to_dict("records") if pd is not None else None
        except Exception:
            router_rows = None
    elif router_table_path.exists():
        payload = torch.load(router_table_path, map_location="cpu", weights_only=False)
        risk_table = payload["risk_table"]
        router_rows = payload.get("rows")

    metrics: Dict[str, Dict] = {}
    if params.sample_eval:
        strategies = [s.strip() for s in params.strategies.split(",") if s.strip()]
        real = get_real_eval_images(test_ds, params.num_samples)
        sample_bs = int(params.sample_batch_size)
        if int(info["channels"]) == 3:
            sample_bs = min(sample_bs, 32)

        router: Optional[BayesRiskRouter] = None
        if risk_table is not None:
            router = BayesRiskRouter(stats.to(device), risk_table.to(device), params.risk_soft_temp)

        for name in strategies:
            if name in {"baseline", "baseline_heun", "baseline_euler"}:
                if baseline is None:
                    print(f"[sample] skip {name}: no baseline model", flush=True)
                    continue
                all_samples: List[torch.Tensor] = []
                remaining = params.num_samples
                while remaining > 0:
                    nb = min(sample_bs, remaining)
                    x, diag = sample_baseline(baseline, params, info, device, nb)
                    all_samples.append(x)
                    remaining -= nb
                gen = torch.cat(all_samples, 0)[:params.num_samples]
                save_samples(outdir, name, gen, params)
                m = evaluate_generated(real[:gen.shape[0]], gen, device)
                m.update(diag)
                m["num_samples"] = int(gen.shape[0])
                metrics[name] = m
                print(f"[eval] {name}: {m}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                if mcl is None:
                    print(f"[sample] skip {name}: no MCL model", flush=True)
                    continue
                if name.startswith("risk") and router is None:
                    print(f"[sample] skip {name}: no calibrated router", flush=True)
                    continue
                all_samples = []
                diag_accum: List[Dict[str, float]] = []
                remaining = params.num_samples
                while remaining > 0:
                    nb = min(sample_bs, remaining)
                    x, diag = sample_mcl(mcl, params, info, device, nb, name, router)
                    all_samples.append(x)
                    diag_accum.append(diag)
                    remaining -= nb
                gen = torch.cat(all_samples, 0)[:params.num_samples]
                save_samples(outdir, name, gen, params)
                m = evaluate_generated(real[:gen.shape[0]], gen, device)
                # average per-chunk diagnostics; num_samples is total generated, not last chunk size
                keys = sorted(set().union(*[d.keys() for d in diag_accum]))
                for k in keys:
                    if k == "num_samples":
                        continue
                    vals = [d[k] for d in diag_accum if k in d and isinstance(d[k], (int, float)) and not math.isnan(float(d[k]))]
                    if vals:
                        m[k] = float(np.mean(vals))
                m["num_samples"] = int(gen.shape[0])
                m["sample_batch_size"] = int(sample_bs)
                metrics[name] = m
                print(f"[eval] {name}: {m}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        write_json(outdir / "metrics.json", metrics)
        if mcl is not None and router is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            try:
                paired = paired_strategy_test(params, mcl, router, info, test_ds, device, outdir)
            except RuntimeError as e:
                err = f"{type(e).__name__}: {e}"
                print(f"[paired] skipped: {err}", flush=True)
                metrics["paired_mixture_vs_risk_softmix"] = {"error": err}
            else:
                if paired is not None:
                    metrics["paired_mixture_vs_risk_softmix"] = paired
                    print(f"[paired] mixture_score vs risk_softmix: {paired}", flush=True)
            write_json(outdir / "metrics.json", metrics)

    make_summary(outdir, params, info, beta_cal, metrics, router_rows)
    plot_v9_outputs(outdir, metrics)
    make_v9_report(outdir, params, metrics)
    print(f"[v9] done. Summary: {outdir / 'SUMMARY_v9.txt'}", flush=True)
    print(f"[v9] report: {outdir / 'RESULTS_V9.md'}", flush=True)


def parse_args() -> Params:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    defaults = asdict(Params())
    for k, v in defaults.items():
        arg = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(arg, action="store_true" if not v else "store_false")
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

