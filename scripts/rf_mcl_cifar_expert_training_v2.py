#!/usr/bin/env python3
"""
RF-MCL expert training on CIFAR-10 two classes (v2, actual experts).

Unlike the diagnostic-only CIFAR script, this one trains K random-feature denoising
experts with WTA/annealed-WTA routing and checks if a beta chosen from the channel
and empirical-glass diagnostics actually produces useful, class-aligned expert
specialization.

Outputs:
  - calibration.json: beta_class, beta_trans, beta_free, beta_glass_empirical
  - metrics_*.csv and metrics_all.csv
  - plots for loss, class MI, usage entropy, effective assignment fraction, beta
  - final expert tensors for each variant

Examples:
  python rf_mcl_cifar_expert_training_v2.py --quick --outdir ./cifar_v2_quick

  python rf_mcl_cifar_expert_training_v2.py --classes automobile horse \
      --d-mode pca --pca-dim 512 --p 512 --steps 3000 --outdir ./cifar_v2

  python rf_mcl_cifar_expert_training_v2.py --classes automobile horse \
      --d-mode full --p 512 --n-train 8000 --n-test 2000 --outdir ./cifar_v2_full
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

CIFAR_LABELS = {
    "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
    "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9,
}

@dataclass
class Params:
    data_root: str = "./data"
    classes: Tuple[str, str] = ("automobile", "horse")
    d_mode: str = "pca"  # pca|full|rp
    pca_dim: int = 512
    rp_dim: int = 512
    p: int = 512
    K: int = 4
    t: float = 1.5
    n_train: int = 6000
    n_test: int = 2000
    batch_size: int = 192
    steps: int = 3000
    lr: float = 2e-3
    weight_decay: float = 0.0
    init_std: float = 1e-3
    seed: int = 0
    calib_steps: int = 900
    warmup_steps: int = 600
    ramp_steps: int = 900
    eval_every: int = 200
    power_iters: int = 20
    ridge: float = 1e-5
    beta_target_mult: float = 3.0
    beta_glass_safety: float = 0.45
    cold_mult_glass: float = 1.2
    device: str = "auto"
    outdir: str = "rf_mcl_cifar_expert_v2"
    no_download: bool = False
    beta_grid: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

# ----------------------------- CIFAR data -----------------------------

def _load_cifar_split(root: str, train: bool, download: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
    except Exception as e:
        raise RuntimeError("torchvision is required for CIFAR loading. Install torchvision or use a prepared data root.") from e
    ds = CIFAR10(root=root, train=train, download=download, transform=transforms.ToTensor())
    xs, ys = [], []
    for x, y in ds:
        xs.append((x * 2.0 - 1.0).reshape(-1))
        ys.append(y)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def _balanced_filter(x: torch.Tensor, y: torch.Tensor, labels: Tuple[int, int], n_total: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    per = n_total // 2
    xs, ys = [], []
    for c, lab in enumerate(labels):
        idx = torch.where(y == lab)[0]
        idx = idx[torch.randperm(len(idx), generator=g)[:per]]
        xs.append(x[idx]); ys.append(torch.full((len(idx),), c, dtype=torch.long))
    X = torch.cat(xs, dim=0); Y = torch.cat(ys, dim=0)
    perm = torch.randperm(len(Y), generator=g)
    return X[perm], Y[perm]


def prepare_cifar(params: Params, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    labels = (CIFAR_LABELS[params.classes[0]], CIFAR_LABELS[params.classes[1]])
    download = not params.no_download
    xtr_raw, ytr_raw = _load_cifar_split(params.data_root, True, download)
    xte_raw, yte_raw = _load_cifar_split(params.data_root, False, download)
    xtr, ytr = _balanced_filter(xtr_raw, ytr_raw, labels, params.n_train, params.seed)
    xte, yte = _balanced_filter(xte_raw, yte_raw, labels, params.n_test, params.seed + 1)

    # Center and scale globally first.
    mean = xtr.mean(dim=0, keepdim=True)
    xtr = xtr - mean; xte = xte - mean
    global_std = xtr.std().clamp_min(1e-6)
    xtr = xtr / global_std; xte = xte / global_std

    info: Dict[str, object] = {"classes": params.classes, "raw_dim": int(xtr.shape[1]), "global_std": float(global_std)}

    if params.d_mode == "pca":
        q = min(params.pca_dim, xtr.shape[0] - 1, xtr.shape[1])
        # CPU low-rank PCA is often more memory-stable for CIFAR.
        Xcpu = xtr.cpu()
        try:
            U, S, V = torch.pca_lowrank(Xcpu, q=q, center=False, niter=4)
            P = V[:, :q]
        except Exception:
            # fallback exact SVD
            _, _, Vh = torch.linalg.svd(Xcpu, full_matrices=False)
            P = Vh[:q].t().contiguous()
        xtr = xtr @ P
        xte = xte @ P
        std = xtr.std(dim=0, keepdim=True).clamp_min(1e-5)
        xtr = xtr / std; xte = xte / std
        info.update({"d_mode": "pca", "d": q})
    elif params.d_mode == "rp":
        q = min(params.rp_dim, xtr.shape[1])
        R = torch.randn(xtr.shape[1], q) / math.sqrt(q)
        xtr = xtr @ R; xte = xte @ R
        std = xtr.std(dim=0, keepdim=True).clamp_min(1e-5)
        xtr = xtr / std; xte = xte / std
        info.update({"d_mode": "rp", "d": q})
    elif params.d_mode == "full":
        info.update({"d_mode": "full", "d": int(xtr.shape[1])})
    else:
        raise ValueError("--d-mode must be pca, rp, or full")

    # class direction in processed space
    m0 = xtr[ytr == 0].mean(dim=0)
    m1 = xtr[ytr == 1].mean(dim=0)
    m_vec = (m1 - m0)
    return xtr.to(device).float(), ytr.to(device), xte.to(device).float(), yte.to(device), m_vec.to(device).float(), info

# ----------------------------- RF helpers -----------------------------

def make_W(p: int, d: int, device: torch.device) -> torch.Tensor:
    return torch.randn(p, d, device=device)


def features(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x @ W.t() / math.sqrt(x.shape[1]))


def single_pred(A: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return phi @ A.t() / math.sqrt(phi.shape[1])


def expert_preds(A: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return torch.einsum("kdp,bp->bkd", A, phi) / math.sqrt(phi.shape[1])


def sample_batch(x0: torch.Tensor, y: torch.Tensor, batch_size: int, t: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x0.shape[0]
    idx = torch.randint(0, n, (batch_size,), device=x0.device)
    xb = x0[idx]; yb = y[idx]
    eps = torch.randn_like(xb)
    a = math.exp(-t); Delta = 1.0 - math.exp(-2.0 * t)
    xt = a * xb + math.sqrt(Delta) * eps
    return xt, eps, yb, idx


def soft_weights(energy: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0:
        return torch.full_like(energy, 1.0 / energy.shape[1])
    z = -beta * (energy - energy.min(dim=1, keepdim=True).values)
    return torch.softmax(z, dim=1)

# ----------------------------- calibration -----------------------------

def train_single_rf(x_train: torch.Tensor, y_train: torch.Tensor, W: torch.Tensor, params: Params, d: int, device: torch.device) -> torch.Tensor:
    A = torch.zeros(d, params.p, device=device, requires_grad=True)
    opt = torch.optim.Adam([A], lr=params.lr, weight_decay=params.weight_decay)
    for _ in range(params.calib_steps):
        xt, eps, _, _ = sample_batch(x_train, y_train, params.batch_size, params.t)
        phi = features(xt, W)
        pred = single_pred(A, phi)
        loss = ((pred - eps) ** 2).sum(dim=1).mean() / d
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return A.detach()


@torch.no_grad()
def eval_phi_residual(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, W: torch.Tensor, t: float, n_eval: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    idx = torch.randint(0, n, (min(n_eval, n),), device=x.device)
    xb = x[idx]
    eps = torch.randn_like(xb)
    a = math.exp(-t); Delta = 1.0 - math.exp(-2.0 * t)
    xt = a * xb + math.sqrt(Delta) * eps
    phi = features(xt, W)
    q = single_pred(A, phi) - eps
    return phi, q


def _lambda_class(phi: torch.Tensor, scalar_q: torch.Tensor, ridge: float, iters: int) -> float:
    n, p = phi.shape; device = phi.device
    C = phi.t() @ phi / n + ridge * torch.eye(p, device=device)
    M = phi.t() @ (phi * (scalar_q[:, None] ** 2)) / n
    v = torch.randn(p, device=device); v = v / (torch.sqrt(v @ C @ v) + 1e-12)
    lam = torch.tensor(0.0, device=device)
    for _ in range(iters):
        v = torch.linalg.solve(C, M @ v)
        v = v / (torch.sqrt(v @ C @ v) + 1e-12)
        lam = (v @ M @ v) / (v @ C @ v + 1e-12)
    return float(lam.detach().cpu())


def _lambda_free(phi: torch.Tensor, q: torch.Tensor, ridge: float, iters: int) -> Tuple[float, torch.Tensor]:
    n, p = phi.shape; d = q.shape[1]; device = phi.device
    C = phi.t() @ phi / n + ridge * torch.eye(p, device=device)
    H = torch.randn(d, p, device=device)
    H = H / torch.sqrt(torch.trace(H @ C @ H.t()) + 1e-12)
    lam = torch.tensor(0.0, device=device)
    for _ in range(iters):
        s = (q * (phi @ H.t())).sum(dim=1)
        R = q.t() @ (phi * s[:, None]) / n
        H = torch.linalg.solve(C, R.t()).t()
        H = H / torch.sqrt(torch.trace(H @ C @ H.t()) + 1e-12)
        s = (q * (phi @ H.t())).sum(dim=1)
        lam = (s ** 2).mean() / (torch.trace(H @ C @ H.t()) + 1e-12)
    return float(lam.detach().cpu()), H.detach()


def estimate_channels(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, W: torch.Tensor, m_vec: torch.Tensor, params: Params, d: int) -> Dict[str, float]:
    phi, q = eval_phi_residual(A, x, y, W, params.t, min(2048, x.shape[0]))
    mhat = m_vec / (m_vec.norm() + 1e-12)
    lam_class = _lambda_class(phi, q @ mhat, params.ridge, params.power_iters)
    lam_trans = 0.0
    for _ in range(4):
        u = torch.randn_like(mhat); u = u - (u @ mhat) * mhat; u = u / (u.norm() + 1e-12)
        lam_trans = max(lam_trans, _lambda_class(phi, q @ u, params.ridge, max(6, params.power_iters // 2)))
    lam_free, H = _lambda_free(phi, q, params.ridge, params.power_iters)
    align = float(((mhat @ H).norm() / (H.norm() + 1e-12)).detach().cpu())
    return {
        "lambda_free": lam_free, "lambda_class": lam_class, "lambda_trans": lam_trans,
        "beta_free": 1.0 / (2 * lam_free + 1e-12),
        "beta_class": 1.0 / (2 * lam_class + 1e-12),
        "beta_trans": 1.0 / (2 * lam_trans + 1e-12),
        "free_align_m": align,
    }


@torch.no_grad()
def estimate_beta_glass_empirical(x: torch.Tensor, y: torch.Tensor, t: float, n_pairs: int = 50000) -> Dict[str, float]:
    # Empirical REM approximation from same-class pair energies E=r_t ||x_i-x_j||^2.
    device = x.device; d = x.shape[1]
    r_t = math.exp(-2.0 * t) / (1.0 - math.exp(-2.0 * t))
    Es = []
    per_class_counts = []
    for c in [0, 1]:
        idx = torch.where(y == c)[0]
        per_class_counts.append(int(len(idx)))
        if len(idx) < 2: continue
        m = n_pairs // 2
        a = idx[torch.randint(0, len(idx), (m,), device=device)]
        b = idx[torch.randint(0, len(idx), (m,), device=device)]
        diff = x[a] - x[b]
        E = r_t * (diff ** 2).sum(dim=1)
        Es.append(E)
    Eall = torch.cat(Es)
    v = float(Eall.var(unbiased=True).cpu()) / d
    alpha = math.log(max(min(per_class_counts), 2)) / d
    beta = math.sqrt(max(0.0, 2.0 * alpha / (v + 1e-30)))
    return {"alpha": alpha, "r_t": r_t, "v_emp": v, "beta_glass_emp": beta, "E_mean_per_d": float(Eall.mean().cpu()) / d}

# ----------------------------- expert training -----------------------------

class BetaSchedule:
    def __init__(self, name: str, params: Params, beta_target: float, beta_class: float, beta_glass: float):
        self.name = name; self.params = params; self.beta_target = beta_target; self.beta_class = beta_class; self.beta_glass = beta_glass
    def beta(self, step: int) -> float:
        p = self.params
        if self.name == "uniform": return 0.0
        if self.name == "hard_cold": return max(1.5 * self.beta_target, p.cold_mult_glass * self.beta_glass)
        if self.name == "fixed_class": return 1.2 * self.beta_class
        if self.name == "fixed_good": return self.beta_target
        if self.name == "theory_anneal":
            if step < p.warmup_steps: return 0.0
            u = min(1.0, max(0.0, (step - p.warmup_steps) / max(1, p.ramp_steps)))
            return self.beta_target * (0.5 - 0.5 * math.cos(math.pi * u))
        if self.name.startswith("grid_"): return float(self.name.split("_")[1])
        return self.beta_target


@torch.no_grad()
def expert_metrics(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, W: torch.Tensor, m_vec: torch.Tensor, t: float, beta: float, n_eval: int = 1024) -> Dict[str, float]:
    d = x.shape[1]; K = A.shape[0]
    idx = torch.randint(0, x.shape[0], (min(n_eval, x.shape[0]),), device=x.device)
    xb = x[idx]; yb = y[idx]
    eps = torch.randn_like(xb)
    a = math.exp(-t); Delta = 1.0 - math.exp(-2.0 * t)
    xt = a * xb + math.sqrt(Delta) * eps
    phi = features(xt, W)
    pred = expert_preds(A, phi)
    energy = ((pred - eps[:, None, :]) ** 2).sum(dim=2)
    winner = energy.argmin(dim=1)
    w = soft_weights(energy, beta)
    best_mse = energy.min(dim=1).values.mean() / d
    soft_mse = (w * energy).sum(dim=1).mean() / d
    usage = torch.bincount(winner, minlength=K).float(); usage = usage / usage.sum().clamp_min(1)
    usage_entropy = -(usage * (usage + 1e-12).log()).sum() / math.log(K)
    conf = torch.zeros(K, 2, device=x.device)
    for k in range(K):
        for c in [0, 1]: conf[k, c] = ((winner == k) & (yb == c)).float().sum()
    P = conf / conf.sum().clamp_min(1); pk = P.sum(1, keepdim=True); pc = P.sum(0, keepdim=True)
    mi = (P * ((P + 1e-12) / (pk @ pc + 1e-12)).log()).sum()
    Hy = -(pc * (pc + 1e-12).log()).sum()
    mi_norm = mi / (Hy + 1e-12)
    purity = 0.0
    for k in range(K):
        if conf[k].sum() > 0:
            p1 = conf[k, 1] / conf[k].sum()
            purity += float(usage[k].cpu()) * float(abs(2.0 * p1 - 1.0).cpu())
    effs = []
    for k in range(K):
        wk = w[:, k]
        pk_s = wk / wk.sum().clamp_min(1e-12)
        effs.append(float((torch.exp(-(pk_s * (pk_s + 1e-12).log()).sum()) / len(wk)).cpu()))
    A0 = A - A.mean(dim=0, keepdim=True)
    mhat = m_vec / (m_vec.norm() + 1e-12)
    align = torch.einsum("d,kdp->kp", mhat, A0).norm() / (A0.norm() + 1e-12)
    diversity = A0.norm() / (A.norm() + 1e-12)
    return {
        "best_mse": float(best_mse.cpu()), "soft_mse": float(soft_mse.cpu()),
        "usage_entropy": float(usage_entropy.cpu()), "class_mi_norm": float(mi_norm.cpu()),
        "class_purity": float(purity), "eff_frac_min": min(effs), "eff_frac_mean": sum(effs)/len(effs),
        "expert_align_m": float(align.cpu()), "expert_diversity": float(diversity.cpu()),
    }


def train_variant(name: str, sched: BetaSchedule, xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor, yte: torch.Tensor, W: torch.Tensor, m_vec: torch.Tensor, params: Params, outdir: Path, device: torch.device) -> List[Dict[str, float]]:
    d = xtr.shape[1]
    A = params.init_std * torch.randn(params.K, d, params.p, device=device)
    A.requires_grad_(True)
    opt = torch.optim.Adam([A], lr=params.lr, weight_decay=params.weight_decay)
    rows: List[Dict[str, float]] = []
    for step in range(params.steps + 1):
        if step % params.eval_every == 0 or step == params.steps:
            beta_now = sched.beta(step)
            met = expert_metrics(A.detach(), xte, yte, W, m_vec, params.t, beta_now, n_eval=min(1024, xte.shape[0]))
            row = {"variant": name, "step": step, "beta": beta_now}; row.update({f"test_{k}": v for k, v in met.items()})
            rows.append(row)
        if step == params.steps: break
        beta_now = sched.beta(step)
        xt, eps, _, _ = sample_batch(xtr, ytr, params.batch_size, params.t)
        phi = features(xt, W)
        pred = expert_preds(A, phi)
        energy = ((pred - eps[:, None, :]) ** 2).sum(dim=2)
        with torch.no_grad(): w = soft_weights(energy.detach(), beta_now)
        loss = (w * energy).sum(dim=1).mean() / d
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([A], 10.0); opt.step()
    torch.save({"A": A.detach().cpu(), "W": W.detach().cpu(), "params": asdict(params)}, outdir / f"{name}_experts.pt")
    return rows

# ----------------------------- output -----------------------------

def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows: return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def plot(rows: List[Dict[str, float]], outdir: Path) -> None:
    if not HAS_PLT or not rows: return
    variants = sorted(set(r["variant"] for r in rows))
    metrics = [
        ("test_best_mse", "test best MSE"), ("test_class_mi_norm", "winner-class normalized MI"),
        ("test_usage_entropy", "usage entropy"), ("test_eff_frac_min", "min effective assignment fraction"),
        ("test_expert_align_m", "expert deviation alignment with class mean"), ("beta", "inverse WTA beta"),
    ]
    for key, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(8,5))
        for v in variants:
            rr = sorted([r for r in rows if r["variant"] == v], key=lambda z: z["step"])
            ax.plot([r["step"] for r in rr], [r.get(key, float("nan")) for r in rr], marker="o", label=v)
        ax.set_xlabel("training step"); ax.set_ylabel(ylabel); ax.grid(True, alpha=.3); ax.legend()
        if key in ["test_best_mse", "test_eff_frac_min"]: ax.set_yscale("log")
        fig.tight_layout(); fig.savefig(outdir / f"metric_{key}.png", dpi=160); plt.close(fig)

# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data-root", type=str, default=Params.data_root)
    ap.add_argument("--classes", nargs=2, default=list(Params.classes))
    ap.add_argument("--d-mode", choices=["pca", "rp", "full"], default=Params.d_mode)
    ap.add_argument("--pca-dim", type=int, default=Params.pca_dim)
    ap.add_argument("--rp-dim", type=int, default=Params.rp_dim)
    ap.add_argument("--p", type=int, default=Params.p)
    ap.add_argument("--K", type=int, default=Params.K)
    ap.add_argument("--t", type=float, default=Params.t)
    ap.add_argument("--n-train", type=int, default=Params.n_train)
    ap.add_argument("--n-test", type=int, default=Params.n_test)
    ap.add_argument("--batch-size", type=int, default=Params.batch_size)
    ap.add_argument("--steps", type=int, default=Params.steps)
    ap.add_argument("--lr", type=float, default=Params.lr)
    ap.add_argument("--calib-steps", type=int, default=Params.calib_steps)
    ap.add_argument("--warmup-steps", type=int, default=Params.warmup_steps)
    ap.add_argument("--ramp-steps", type=int, default=Params.ramp_steps)
    ap.add_argument("--eval-every", type=int, default=Params.eval_every)
    ap.add_argument("--power-iters", type=int, default=Params.power_iters)
    ap.add_argument("--device", type=str, default=Params.device)
    ap.add_argument("--outdir", type=str, default=Params.outdir)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--beta-grid", action="store_true")
    args = ap.parse_args()
    p = Params(**{k: v for k, v in vars(args).items() if k in Params.__dataclass_fields__})
    p.classes = tuple(args.classes)
    if args.quick:
        p.d_mode = "rp"; p.rp_dim = min(p.rp_dim, 128); p.p = min(p.p, 128)
        p.n_train = min(p.n_train, 1200); p.n_test = min(p.n_test, 600); p.batch_size = min(p.batch_size, 96)
        p.steps = min(p.steps, 500); p.calib_steps = min(p.calib_steps, 150); p.warmup_steps = min(p.warmup_steps, 100)
        p.ramp_steps = min(p.ramp_steps, 150); p.eval_every = min(p.eval_every, 100); p.power_iters = min(p.power_iters, 8)
    p.beta_grid = bool(args.beta_grid)

    set_seed(p.seed); device = get_device(p.device); outdir = Path(p.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print("[1/5] Loading CIFAR...")
    xtr, ytr, xte, yte, m_vec, info = prepare_cifar(p, device)
    d = xtr.shape[1]
    W = make_W(p.p, d, device)
    print(f"processed dimension d={d}, train={len(ytr)}, test={len(yte)}")
    print("[2/5] Calibrating single RF...")
    A_single = train_single_rf(xtr, ytr, W, p, d, device)
    print("[3/5] Estimating beta channels/glass...")
    channels = estimate_channels(A_single, xte, yte, W, m_vec, p, d)
    glass = estimate_beta_glass_empirical(xtr, ytr, p.t, n_pairs=30000 if not args.quick else 6000)
    beta_class = channels["beta_class"]; beta_glass = glass["beta_glass_emp"]
    beta_target = min(p.beta_target_mult * beta_class, p.beta_glass_safety * beta_glass)
    no_window = beta_target <= 1.05 * beta_class
    if no_window: beta_target = p.beta_target_mult * beta_class
    calib = {"data_info": info, "channels": channels, "glass": glass, "beta_target": beta_target, "no_window_warning": no_window}
    with open(outdir / "calibration.json", "w") as f: json.dump(calib, f, indent=2)

    variants = ["uniform", "hard_cold", "fixed_class", "fixed_good", "theory_anneal"]
    if p.beta_grid:
        grid = sorted(set([.5*beta_class, 1.2*beta_class, beta_target, .7*beta_glass, 1.2*beta_glass]))
        variants += [f"grid_{b:.6g}" for b in grid if b > 0]

    print("[4/5] Training expert variants...")
    rows_all: List[Dict[str, float]] = []
    for v in variants:
        print(f"  variant={v}")
        sched = BetaSchedule(v, p, beta_target, beta_class, beta_glass)
        rows = train_variant(v, sched, xtr, ytr, xte, yte, W, m_vec, p, outdir, device)
        rows_all.extend(rows); write_csv(outdir / f"metrics_{v}.csv", rows)
    write_csv(outdir / "metrics_all.csv", rows_all); plot(rows_all, outdir)

    print("[5/5] Summary...")
    with open(outdir / "SUMMARY.txt", "w") as f:
        f.write("RF-MCL CIFAR expert training v2\n")
        f.write(json.dumps(asdict(p), indent=2) + "\n\n")
        f.write("Data info:\n" + json.dumps(info, indent=2) + "\n\n")
        f.write("Channels:\n")
        for k, v in channels.items(): f.write(f"  {k}: {v}\n")
        f.write("\nGlass:\n")
        for k, v in glass.items(): f.write(f"  {k}: {v}\n")
        f.write(f"\nChosen beta_target: {beta_target}\n")
        if no_window: f.write("WARNING: class/glass window narrow or absent; target uses class multiplier.\n")
        f.write("\nFinal metrics by variant:\n")
        for v in variants:
            rr = sorted([r for r in rows_all if r["variant"] == v], key=lambda z: z["step"])
            if rr:
                last = rr[-1]; f.write(f"\n[{v}]\n")
                for key in ["beta", "test_best_mse", "test_class_mi_norm", "test_class_purity", "test_usage_entropy", "test_eff_frac_min", "test_expert_align_m", "test_expert_diversity"]:
                    f.write(f"  {key}: {last.get(key)}\n")
    print(f"Done. Results in {outdir}")

if __name__ == "__main__":
    main()
