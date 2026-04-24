#!/usr/bin/env python3
"""
RF-MCL expert training experiments on a two-class Gaussian mixture (v3).

This script is intentionally *expert-like*: it trains K random-feature denoising
experts with soft/annealed WTA routing, then measures whether the experts really
specialize along the class direction and whether too-cold beta causes dead/glassy
assignments.

It complements the earlier diagnostic-only scripts.  Main outputs:
  - calibration thresholds: beta_class, beta_trans, beta_glass
  - trained variants: uniform, cold, theory_anneal, fixed_good, optional beta grid
  - metrics over training: test loss, class MI of winner routing, usage entropy,
    effective assignment fraction, alignment of expert differences with m
  - plots + CSVs + SUMMARY.txt

Conventions:
  - fixed diffusion time t
  - noise predictor epsilon_A(x_t) = A phi(x_t) / sqrt(p)
  - routing energy E_{nk}=||epsilon_k(x_t)-epsilon||^2, not divided by d
  - optimization loss is E/d for stable gradients
  - beta is inverse WTA temperature for the unnormalized energy E

Quick run:
  python rf_mcl_gmm_expert_training_v3.py --quick --outdir ./gmm_v3

Serious run:
  python rf_mcl_gmm_expert_training_v3.py --d 64 --p 256 --n-train 6000 \
      --n-test 3000 --steps 4000 --outdir ./gmm_v3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# ----------------------------- config -----------------------------

@dataclass
class Params:
    d: int = 64
    p: int = 256
    K: int = 4
    mu: float = 1.0
    sigma0: float = 0.5
    t: float = 2.05
    n_train: int = 6000
    n_test: int = 3000
    batch_size: int = 256
    steps: int = 4000
    lr: float = 3e-3
    weight_decay: float = 0.0
    init_std: float = 1e-3
    seed: int = 0
    calib_steps: int = 1200
    warmup_steps: int = 800
    ramp_steps: int = 1000
    eval_every: int = 200
    power_iters: int = 30
    ridge: float = 1e-5
    beta_target_mult: float = 3.0
    beta_glass_safety: float = 0.45
    cold_mult_glass: float = 1.3
    fixed_good_mult: float = 1.2
    device: str = "auto"
    outdir: str = "rf_mcl_gmm_expert_v3"
    beta_grid: bool = False


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- data -----------------------------

def make_gmm_data(n: int, d: int, mu: float, sigma0: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced two-class GMM. Returns x0 [n,d], y01 [n], m_vec [d]."""
    n0 = n // 2
    n1 = n - n0
    y = torch.cat([torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)], dim=0)
    signs = 2.0 * y.float() - 1.0
    perm = torch.randperm(n)
    y = y[perm]
    signs = signs[perm]
    m_vec = torch.full((d,), mu, dtype=torch.float32)
    x = signs[:, None] * m_vec[None, :] + sigma0 * torch.randn(n, d)
    return x.to(device), y.to(device), m_vec.to(device)


def sample_batch(x0: torch.Tensor, y: torch.Tensor, batch_size: int, t: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, d = x0.shape
    idx = torch.randint(0, n, (batch_size,), device=x0.device)
    xb = x0[idx]
    yb = y[idx]
    eps = torch.randn_like(xb)
    a = math.exp(-t)
    Delta = 1.0 - math.exp(-2.0 * t)
    xt = a * xb + math.sqrt(Delta) * eps
    return xt, eps, yb, idx


# ----------------------------- RF helpers -----------------------------

def make_W(p: int, d: int, device: torch.device) -> torch.Tensor:
    return torch.randn(p, d, device=device)


def features(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    d = x.shape[1]
    return torch.tanh(x @ W.t() / math.sqrt(d))


def expert_preds(A: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """A [K,d,p], phi [B,p] -> pred [B,K,d]."""
    p = phi.shape[1]
    return torch.einsum("kdp,bp->bkd", A, phi) / math.sqrt(p)


def single_pred(A: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """A [d,p], phi [B,p] -> pred [B,d]."""
    p = phi.shape[1]
    return phi @ A.t() / math.sqrt(p)


def soft_weights_from_energy(energy: torch.Tensor, beta: float, hard: bool = False) -> torch.Tensor:
    """energy [B,K], beta on unnormalized energy."""
    if hard:
        idx = energy.argmin(dim=1)
        return F.one_hot(idx, num_classes=energy.shape[1]).float()
    if beta <= 0:
        return torch.full_like(energy, 1.0 / energy.shape[1])
    # center for numerical stability
    z = -beta * (energy - energy.min(dim=1, keepdim=True).values)
    return torch.softmax(z, dim=1)


# ----------------------------- training single RF for calibration -----------------------------

def train_single_rf(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    W: torch.Tensor,
    params: Params,
    device: torch.device,
    log_prefix: str = "calib",
) -> torch.Tensor:
    d, p = params.d, params.p
    A = torch.zeros(d, p, device=device, requires_grad=True)
    opt = torch.optim.Adam([A], lr=params.lr, weight_decay=params.weight_decay)
    for step in range(params.calib_steps):
        xt, eps, _, _ = sample_batch(x_train, y_train, params.batch_size, params.t)
        phi = features(xt, W)
        pred = single_pred(A, phi)
        loss = ((pred - eps) ** 2).sum(dim=1).mean() / d
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return A.detach()


# ----------------------------- channel diagnostics -----------------------------

@torch.no_grad()
def eval_phi_residual(
    A_single: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    W: torch.Tensor,
    params: Params,
    n_eval: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # one large sampled noisy batch from x_eval
    n = x_eval.shape[0]
    idx = torch.randint(0, n, (n_eval,), device=x_eval.device)
    xb = x_eval[idx]
    yb = y_eval[idx]
    eps = torch.randn_like(xb)
    a = math.exp(-params.t)
    Delta = 1.0 - math.exp(-2 * params.t)
    xt = a * xb + math.sqrt(Delta) * eps
    phi = features(xt, W)
    pred = single_pred(A_single, phi)
    q = pred - eps
    return phi, q, yb, xt


def _power_generalized_class(phi: torch.Tensor, scalar_q: torch.Tensor, ridge: float, iters: int) -> float:
    """lambda_max of C^{-1} M, M=E[scalar_q^2 phi phi^T]."""
    device = phi.device
    n, p = phi.shape
    C = (phi.t() @ phi) / n + ridge * torch.eye(p, device=device)
    wphi = phi * (scalar_q[:, None] ** 2)
    M = (phi.t() @ wphi) / n
    v = torch.randn(p, device=device)
    v = v / (torch.sqrt(v @ C @ v) + 1e-12)
    lam = torch.tensor(0.0, device=device)
    for _ in range(iters):
        Mv = M @ v
        v_new = torch.linalg.solve(C, Mv)
        norm = torch.sqrt(v_new @ C @ v_new) + 1e-12
        v = v_new / norm
        lam = (v @ M @ v) / (v @ C @ v + 1e-12)
    return float(lam.detach().cpu())


def _power_generalized_free(phi: torch.Tensor, q: torch.Tensor, ridge: float, iters: int) -> Tuple[float, float, torch.Tensor]:
    """Generalized top lambda over H [d,p]. Denominator Tr(H C H^T)."""
    device = phi.device
    n, p = phi.shape
    d = q.shape[1]
    C = (phi.t() @ phi) / n + ridge * torch.eye(p, device=device)
    H = torch.randn(d, p, device=device)
    # normalize in C metric
    denom = torch.trace(H @ C @ H.t()) + 1e-12
    H = H / torch.sqrt(denom)
    lam = torch.tensor(0.0, device=device)
    for _ in range(iters):
        # s_n = q_n^T H phi_n
        HPhi = phi @ H.t()  # [n,d]
        s = (q * HPhi).sum(dim=1)  # [n]
        R = (q.t() @ (phi * s[:, None])) / n  # [d,p]
        # generalized update H <- R C^{-1}
        H_new = torch.linalg.solve(C, R.t()).t()
        denom = torch.trace(H_new @ C @ H_new.t()) + 1e-12
        H = H_new / torch.sqrt(denom)
        HPhi = phi @ H.t()
        num = ((q * HPhi).sum(dim=1) ** 2).mean()
        den = torch.trace(H @ C @ H.t()) + 1e-12
        lam = num / den
    return float(lam.detach().cpu()), 0.0, H.detach()


def estimate_channels(
    A_single: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    W: torch.Tensor,
    m_vec: torch.Tensor,
    params: Params,
    n_eval: int = 2048,
) -> Dict[str, float]:
    phi, q, _, _ = eval_phi_residual(A_single, x_eval, y_eval, W, params, n_eval)
    mhat = m_vec / (m_vec.norm() + 1e-12)
    q_class = q @ mhat
    lam_class = _power_generalized_class(phi, q_class, params.ridge, params.power_iters)
    # max over a few random transverse directions
    lam_trans = 0.0
    for _ in range(4):
        u = torch.randn_like(mhat)
        u = u - (u @ mhat) * mhat
        u = u / (u.norm() + 1e-12)
        lam_trans = max(lam_trans, _power_generalized_class(phi, q @ u, params.ridge, max(8, params.power_iters // 2)))
    lam_free, _, H = _power_generalized_free(phi, q, params.ridge, params.power_iters)
    align = float(((mhat @ H).norm() / (H.norm() + 1e-12)).detach().cpu())
    return {
        "lambda_free": lam_free,
        "lambda_class": lam_class,
        "lambda_trans": lam_trans,
        "beta_free": 1.0 / (2.0 * lam_free + 1e-12),
        "beta_class": 1.0 / (2.0 * lam_class + 1e-12),
        "beta_trans": 1.0 / (2.0 * lam_trans + 1e-12),
        "free_align_m": align,
    }


def estimate_beta_glass_gmm(params: Params, n_effective: Optional[int] = None) -> Dict[str, float]:
    # same-class exact chi-square GMM formula at tau=infty, energy E=r ||a_i-a_j||^2.
    # alpha = log(n/2)/d, y solves y - 1 - log y = 2 alpha.
    n = n_effective or params.n_train
    alpha = math.log(max(n // 2, 2)) / params.d
    r_t = math.exp(-2 * params.t) / (1.0 - math.exp(-2 * params.t))
    # solve y in (0,1)
    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        val = mid - 1.0 - math.log(mid)
        if val > 2.0 * alpha:
            lo = mid
        else:
            hi = mid
    y_alpha = 0.5 * (lo + hi)
    beta_exact = (1.0 / y_alpha - 1.0) / (4.0 * r_t * params.sigma0 ** 2 + 1e-30)
    v_gauss = 8.0 * (r_t ** 2) * (params.sigma0 ** 4)
    beta_gauss = math.sqrt(2.0 * alpha / (v_gauss + 1e-30))
    return {"alpha": alpha, "r_t": r_t, "y_alpha": y_alpha, "beta_glass_exact": beta_exact, "beta_glass_gauss": beta_gauss}


# ----------------------------- schedules -----------------------------

class BetaSchedule:
    def __init__(self, name: str, params: Params, beta_target: float, beta_glass: float, beta_class: float):
        self.name = name
        self.params = params
        self.beta_target = float(beta_target)
        self.beta_glass = float(beta_glass)
        self.beta_class = float(beta_class)
        if name == "uniform":
            self.hard = False
        elif name == "hard_cold":
            self.hard = False
        else:
            self.hard = False

    def beta(self, step: int) -> float:
        p = self.params
        if self.name == "uniform":
            return 0.0
        if self.name == "hard_cold":
            return max(1.5 * self.beta_target, p.cold_mult_glass * self.beta_glass)
        if self.name == "fixed_good":
            return self.beta_target
        if self.name == "fixed_class":
            return p.fixed_good_mult * self.beta_class
        if self.name == "theory_anneal":
            if step < p.warmup_steps:
                return 0.0
            u = min(1.0, max(0.0, (step - p.warmup_steps) / max(1, p.ramp_steps)))
            # smooth ramp from a very small beta to target
            return self.beta_target * (0.5 - 0.5 * math.cos(math.pi * u))
        if self.name.startswith("grid_"):
            return float(self.name.split("_")[1])
        return self.beta_target


# ----------------------------- expert training -----------------------------

@torch.no_grad()
def expert_metrics(
    A: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    W: torch.Tensor,
    m_vec: torch.Tensor,
    params: Params,
    beta_eval: float,
    n_eval: int = 2048,
) -> Dict[str, float]:
    device = x.device
    d = params.d
    K = params.K
    n = x.shape[0]
    idx = torch.randint(0, n, (min(n_eval, n),), device=device)
    xb = x[idx]
    yb = y[idx]
    eps = torch.randn_like(xb)
    a = math.exp(-params.t)
    Delta = 1.0 - math.exp(-2 * params.t)
    xt = a * xb + math.sqrt(Delta) * eps
    phi = features(xt, W)
    pred = expert_preds(A, phi)
    energy = ((pred - eps[:, None, :]) ** 2).sum(dim=2)  # [B,K]
    winner = energy.argmin(dim=1)
    best_mse = energy.min(dim=1).values.mean() / d
    mean_mse = energy.mean() / d
    w_soft = soft_weights_from_energy(energy, beta_eval, hard=False)
    soft_loss = (w_soft * energy).sum(dim=1).mean() / d
    # usage hard winners
    usage = torch.bincount(winner, minlength=K).float()
    usage = usage / usage.sum().clamp_min(1.0)
    usage_entropy = -(usage * (usage + 1e-12).log()).sum() / math.log(K)
    # class MI between hard winner and y
    conf = torch.zeros(K, 2, device=device)
    for k in range(K):
        for c in range(2):
            conf[k, c] = ((winner == k) & (yb == c)).float().sum()
    P = conf / conf.sum().clamp_min(1.0)
    pk = P.sum(dim=1, keepdim=True)
    pc = P.sum(dim=0, keepdim=True)
    mi = (P * ((P + 1e-12) / (pk @ pc + 1e-12)).log()).sum()
    Hy = -(pc * (pc + 1e-12).log()).sum()
    mi_norm = mi / (Hy + 1e-12)
    # purity weighted by usage: 1 means each active expert pure, 0 means balanced class per expert
    purity = 0.0
    for k in range(K):
        if conf[k].sum() > 0:
            p1 = conf[k, 1] / conf[k].sum()
            purity += float(usage[k].cpu()) * float(abs(2.0 * p1 - 1.0).cpu())
    # effective soft assignment fraction per expert over eval set
    eff_fracs = []
    for k in range(K):
        wk = w_soft[:, k]
        if wk.sum() <= 1e-12:
            eff_fracs.append(0.0)
        else:
            pk_s = wk / wk.sum()
            eff = torch.exp(-(pk_s * (pk_s + 1e-12).log()).sum()) / wk.numel()
            eff_fracs.append(float(eff.cpu()))
    eff_frac_min = min(eff_fracs)
    eff_frac_mean = sum(eff_fracs) / len(eff_fracs)
    # expert parameter diversity + alignment of expert deviations with class direction
    A_center = A.mean(dim=0, keepdim=True)
    Dlt = A - A_center
    mhat = m_vec / (m_vec.norm() + 1e-12)
    row_proj = torch.einsum("d,kdp->kp", mhat, Dlt)
    align_expert = row_proj.norm() / (Dlt.norm() + 1e-12)
    diversity = Dlt.norm() / (A.norm() + 1e-12)
    return {
        "best_mse": float(best_mse.cpu()),
        "mean_mse": float(mean_mse.cpu()),
        "soft_mse": float(soft_loss.cpu()),
        "usage_entropy": float(usage_entropy.cpu()),
        "class_mi_norm": float(mi_norm.cpu()),
        "class_purity": float(purity),
        "eff_frac_min": float(eff_frac_min),
        "eff_frac_mean": float(eff_frac_mean),
        "expert_align_m": float(align_expert.cpu()),
        "expert_diversity": float(diversity.cpu()),
    }


def train_experts_variant(
    name: str,
    schedule: BetaSchedule,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    W: torch.Tensor,
    m_vec: torch.Tensor,
    params: Params,
    outdir: Path,
    device: torch.device,
) -> List[Dict[str, float]]:
    d, p, K = params.d, params.p, params.K
    A = params.init_std * torch.randn(K, d, p, device=device)
    A.requires_grad_(True)
    opt = torch.optim.Adam([A], lr=params.lr, weight_decay=params.weight_decay)
    rows: List[Dict[str, float]] = []
    for step in range(params.steps + 1):
        if step % params.eval_every == 0 or step == params.steps:
            beta_now = schedule.beta(step)
            met_test = expert_metrics(A.detach(), x_test, y_test, W, m_vec, params, beta_now, n_eval=min(2048, params.n_test))
            met_train = expert_metrics(A.detach(), x_train, y_train, W, m_vec, params, beta_now, n_eval=min(2048, params.n_train))
            row = {"variant": name, "step": step, "beta": beta_now}
            row.update({f"test_{k}": v for k, v in met_test.items()})
            row.update({f"train_{k}": v for k, v in met_train.items()})
            rows.append(row)
        if step == params.steps:
            break
        beta_now = schedule.beta(step)
        xt, eps, _, _ = sample_batch(x_train, y_train, params.batch_size, params.t)
        phi = features(xt, W)
        pred = expert_preds(A, phi)
        energy = ((pred - eps[:, None, :]) ** 2).sum(dim=2)
        with torch.no_grad():
            weights = soft_weights_from_energy(energy.detach(), beta_now, hard=False)
        loss = (weights * energy).sum(dim=1).mean() / d
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([A], 10.0)
        opt.step()
    # save final expert tensor only if not huge
    torch.save({"A": A.detach().cpu(), "W": W.detach().cpu(), "params": asdict(params)}, outdir / f"{name}_experts.pt")
    return rows


# ----------------------------- plotting / CSV -----------------------------

def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_variants(rows: List[Dict[str, float]], outdir: Path) -> None:
    if not HAS_PLT or not rows:
        return
    variants = sorted(set(r["variant"] for r in rows))
    metrics = [
        ("test_best_mse", "test best MSE"),
        ("test_class_mi_norm", "winner-class normalized MI"),
        ("test_usage_entropy", "usage entropy"),
        ("test_eff_frac_min", "min effective assignment fraction"),
        ("test_expert_align_m", "expert deviation alignment with m"),
        ("beta", "inverse WTA beta"),
    ]
    for key, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for v in variants:
            rr = [r for r in rows if r["variant"] == v]
            rr = sorted(rr, key=lambda z: z["step"])
            xs = [r["step"] for r in rr]
            ys = [r.get(key, float("nan")) for r in rr]
            ax.plot(xs, ys, marker="o", label=v)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        if key in ["test_best_mse", "test_eff_frac_min"]:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"metric_{key}.png", dpi=160)
        plt.close(fig)


# ----------------------------- main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--d", type=int, default=Params.d)
    parser.add_argument("--p", type=int, default=Params.p)
    parser.add_argument("--K", type=int, default=Params.K)
    parser.add_argument("--mu", type=float, default=Params.mu)
    parser.add_argument("--sigma0", type=float, default=Params.sigma0)
    parser.add_argument("--t", type=float, default=Params.t)
    parser.add_argument("--n-train", type=int, default=Params.n_train)
    parser.add_argument("--n-test", type=int, default=Params.n_test)
    parser.add_argument("--batch-size", type=int, default=Params.batch_size)
    parser.add_argument("--steps", type=int, default=Params.steps)
    parser.add_argument("--lr", type=float, default=Params.lr)
    parser.add_argument("--calib-steps", type=int, default=Params.calib_steps)
    parser.add_argument("--warmup-steps", type=int, default=Params.warmup_steps)
    parser.add_argument("--ramp-steps", type=int, default=Params.ramp_steps)
    parser.add_argument("--eval-every", type=int, default=Params.eval_every)
    parser.add_argument("--power-iters", type=int, default=Params.power_iters)
    parser.add_argument("--device", type=str, default=Params.device)
    parser.add_argument("--outdir", type=str, default=Params.outdir)
    parser.add_argument("--beta-grid", action="store_true")
    args = parser.parse_args()

    p = Params(**{k: v for k, v in vars(args).items() if k in Params.__dataclass_fields__})
    if args.quick:
        p.d = min(p.d, 24)
        p.p = min(p.p, 48)
        p.n_train = min(p.n_train, 700)
        p.n_test = min(p.n_test, 400)
        p.batch_size = min(p.batch_size, 96)
        p.steps = min(p.steps, 300)
        p.calib_steps = min(p.calib_steps, 80)
        p.warmup_steps = min(p.warmup_steps, 60)
        p.ramp_steps = min(p.ramp_steps, 80)
        p.eval_every = min(p.eval_every, 60)
        p.power_iters = min(p.power_iters, 6)
    p.beta_grid = bool(args.beta_grid)

    set_seed(p.seed)
    device = get_device(p.device)
    outdir = Path(p.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, m_vec = make_gmm_data(p.n_train, p.d, p.mu, p.sigma0, device)
    x_test, y_test, _ = make_gmm_data(p.n_test, p.d, p.mu, p.sigma0, device)
    W = make_W(p.p, p.d, device)

    print("[1/4] Calibrating single RF...")
    A_single = train_single_rf(x_train, y_train, W, p, device)
    print("[2/4] Estimating channels...")
    channels = estimate_channels(A_single, x_test, y_test, W, m_vec, p, n_eval=min(2048, p.n_test))
    glass = estimate_beta_glass_gmm(p)
    beta_class = channels["beta_class"]
    beta_glass = glass["beta_glass_exact"]
    # choose target: above class threshold, below glass threshold
    beta_target = min(p.beta_target_mult * beta_class, p.beta_glass_safety * beta_glass)
    # avoid target lower than class by too much; if no window, still use class multiplier and warn
    no_window = beta_target <= 1.05 * beta_class
    if no_window:
        beta_target = p.beta_target_mult * beta_class

    calib = {"channels": channels, "glass": glass, "beta_target": beta_target, "no_window_warning": no_window}
    with open(outdir / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)

    variants = ["uniform", "hard_cold", "fixed_class", "fixed_good", "theory_anneal"]
    if p.beta_grid:
        grid = sorted(set([0.25 * beta_class, 0.75 * beta_class, 1.25 * beta_class, beta_target, 0.8 * beta_glass, 1.2 * beta_glass]))
        variants += [f"grid_{b:.6g}" for b in grid if b > 0]

    print("[3/4] Training expert variants...")
    all_rows: List[Dict[str, float]] = []
    for v in variants:
        print(f"  variant={v}")
        sched = BetaSchedule(v, p, beta_target=beta_target, beta_glass=beta_glass, beta_class=beta_class)
        rows = train_experts_variant(v, sched, x_train, y_train, x_test, y_test, W, m_vec, p, outdir, device)
        all_rows.extend(rows)
        write_csv(outdir / f"metrics_{v}.csv", rows)

    write_csv(outdir / "metrics_all.csv", all_rows)
    plot_variants(all_rows, outdir)

    print("[4/4] Writing summary...")
    with open(outdir / "SUMMARY.txt", "w") as f:
        f.write("RF-MCL GMM expert training v3\n")
        f.write(json.dumps(asdict(p), indent=2) + "\n\n")
        f.write("Calibration channels:\n")
        for k, v in channels.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nGlass calibration:\n")
        for k, v in glass.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nChosen beta_target: {beta_target}\n")
        if no_window:
            f.write("WARNING: class/glass safe window was narrow or absent with these params.\n")
        f.write("\nFinal metrics by variant:\n")
        for v in variants:
            rr = [r for r in all_rows if r["variant"] == v]
            if rr:
                last = sorted(rr, key=lambda z: z["step"])[-1]
                f.write(f"\n[{v}]\n")
                for key in ["beta", "test_best_mse", "test_class_mi_norm", "test_class_purity", "test_usage_entropy", "test_eff_frac_min", "test_expert_align_m", "test_expert_diversity"]:
                    f.write(f"  {key}: {last.get(key)}\n")
    print(f"Done. Results in {outdir}")


if __name__ == "__main__":
    main()
