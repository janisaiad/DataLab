#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pandas as pd
except Exception:
    pd = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_of(s: str):
    return torch.device("cuda" if s == "auto" and torch.cuda.is_available() else ("cpu" if s == "auto" else s))


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def norm_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    k = probs.numel()
    p = probs.clamp_min(eps)
    return float((-(p * p.log()).sum() / math.log(k)).cpu())


def mi_norm(assign: torch.Tensor, labels: torch.Tensor, k: int, c: int, eps: float = 1e-12) -> float:
    a = assign.detach().cpu().long()
    y = labels.detach().cpu().long()
    n = max(1, y.numel())
    joint = torch.zeros(c, k, dtype=torch.float64)
    for ci in range(c):
        for ki in range(k):
            joint[ci, ki] = ((y == ci) & (a == ki)).sum().item()
    joint /= n
    pc = joint.sum(1, keepdim=True)
    pk = joint.sum(0, keepdim=True)
    den = pc @ pk
    m = joint > 0
    mi = (joint[m] * (joint[m] / den[m]).log()).sum()
    hc = -(pc[pc > 0] * pc[pc > 0].log()).sum()
    hk = -(pk[pk > 0] * pk[pk > 0].log()).sum()
    return float((mi / torch.minimum(hc, hk).clamp_min(eps)).item())


def best_perm(score: np.ndarray):
    c, k = score.shape
    best = None
    val = -1e100
    for p in itertools.permutations(range(k), min(c, k)):
        s = sum(score[ci, p[ci]] for ci in range(len(p)))
        if s > val:
            val = s
            best = list(p)
    return best if best is not None else list(range(min(c, k)))


def simplex_means(c: int, d: int, mu: float, device):
    if c == 2:
        m = torch.zeros(c, d, device=device)
        m[0, 0] = -mu * math.sqrt(d)
        m[1, 0] = mu * math.sqrt(d)
        return m
    g = torch.randn(c, d, device=device)
    g = g - g.mean(0, keepdim=True)
    q, _ = torch.linalg.qr(g.T, mode="reduced")
    m = q.T[:c]
    m = m - m.mean(0, keepdim=True)
    m = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12) * (mu * math.sqrt(d))
    return m


def sample_gmm(n: int, means: torch.Tensor, sigma0: float, t: float, device):
    c, _ = means.shape
    labels = torch.randint(0, c, (n,), device=device)
    x0 = means[labels] + sigma0 * torch.randn(n, means.shape[1], device=device)
    gamma = math.exp(-t)
    eps = torch.randn_like(x0)
    xt = gamma * x0 + math.sqrt(max(1.0 - gamma * gamma, 1e-12)) * eps
    return xt, eps, labels


def bayes_post(xt: torch.Tensor, means: torch.Tensor, sigma0: float, t: float):
    g = math.exp(-t)
    st2 = g * g * sigma0 * sigma0 + max(1 - g * g, 1e-12)
    dist = ((xt[:, None, :] - g * means[None, :, :]) ** 2).sum(-1)
    return torch.softmax(-0.5 * dist / st2, dim=1)


class RandomFeatures(nn.Module):
    def __init__(self, d: int, p: int, activation: str = "erf", seed: int = 0):
        super().__init__()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 12345)
        self.register_buffer("W", torch.randn(p, d, generator=gen) / math.sqrt(d))
        self.register_buffer("b", 2 * math.pi * torch.rand(p, generator=gen))
        self.p = p
        self.activation = activation

    def forward(self, x):
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
        return h / math.sqrt(self.p)


class RFExperts(nn.Module):
    def __init__(self, p: int, d: int, k: int, init_std: float):
        super().__init__()
        self.A = nn.Parameter(init_std * torch.randn(k, p, d))

    def forward(self, phi):
        return torch.einsum("bp,kpd->bkd", phi, self.A)


class Router(nn.Module):
    def __init__(self, p: int, k: int):
        super().__init__()
        self.fc = nn.Linear(p, k)

    def forward(self, phi):
        return self.fc(phi)


def mcl_loss(e, beta: float):
    k = e.shape[1]
    if beta <= 1e-12:
        q = torch.full_like(e, 1.0 / k)
        loss = e.mean()
    else:
        logits = -beta * e
        q = torch.softmax(logits, 1)
        loss = -(torch.logsumexp(logits, 1) - math.log(k)).mean() / beta
    return loss, q


@torch.no_grad()
def ridge_fit(phi, y, ridge: float):
    n, p = phi.shape
    c = (phi.T @ phi) / n + ridge * torch.eye(p, device=phi.device)
    b = (phi.T @ y) / n
    return torch.linalg.solve(c, b)


@torch.no_grad()
def whiten(phi, ridge: float):
    n, p = phi.shape
    c = (phi.T @ phi) / n + ridge * torch.eye(p, device=phi.device)
    ev, v = torch.linalg.eigh(c)
    return phi @ (v @ torch.diag(ev.clamp_min(ridge).rsqrt()) @ v.T)


@torch.no_grad()
def power_free(s, r, iters: int):
    if iters <= 0:
        return 0.0
    n, p = s.shape
    d = r.shape[1]
    b = torch.randn(p, d, device=s.device)
    b = b / b.norm().clamp_min(1e-12)
    lam = torch.tensor(0.0, device=s.device)
    for _ in range(iters):
        delta = s @ b
        a = (r * delta).sum(1, keepdim=True) / math.sqrt(d)
        m = a * r / math.sqrt(d)
        gb = (s.T @ m) / n
        lam = (b * gb).sum()
        b = gb / gb.norm().clamp_min(1e-12)
    return max(float(lam.cpu()), 0.0)


@torch.no_grad()
def lambda_dir(s, r, v):
    n = s.shape[0]
    a = (r @ v).pow(2)
    h = (s.T * a[None, :]) @ s / n
    return max(float(torch.linalg.eigvalsh(h)[-1].cpu()), 0.0)


@torch.no_grad()
def class_basis(means: torch.Tensor):
    m = means - means.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(m, full_matrices=False)
    rank = int((s > 1e-8 * s.max().clamp_min(1e-8)).sum())
    return vh[:rank].T.contiguous()


@torch.no_grad()
def calibrate(rf, x, y, labels, means, ridge: float, power_iters: int):
    phi = rf(x)
    a0 = ridge_fit(phi, y, ridge)
    r = y - phi @ a0
    s = whiten(phi, ridge)
    lf = power_free(s, r, power_iters)
    b = class_basis(means).to(x.device)
    lc = max([lambda_dir(s, r, b[:, j]) for j in range(b.shape[1])] or [0.0])
    d = y.shape[1]
    pmat = b @ b.T if b.numel() else torch.zeros(d, d, device=x.device)
    vals = []
    for _ in range(12):
        v = torch.randn(d, device=x.device)
        v = v - pmat @ v
        if v.norm() > 1e-8:
            vals.append(lambda_dir(s, r, v / v.norm()))
    lt = max(vals or [0.0])
    beta = lambda l: 0.5 / max(l, 1e-12)
    e = r.pow(2).mean(1)
    v_emp = float(e.var(unbiased=False).cpu())
    alpha = math.log(max(2, x.shape[0])) / max(1, d)
    bg = math.sqrt(2 * alpha / max(v_emp, 1e-12))
    return dict(
        lambda_free=lf,
        lambda_class=lc,
        lambda_trans=lt,
        beta_free=beta(lf),
        beta_class=beta(lc),
        beta_trans=beta(lt),
        beta_glass_emp=bg,
        alpha_log_n_over_d=alpha,
        residual_mse=float(e.mean().cpu()),
        v_emp=v_emp,
    )


def sched(step: int, variant: str, beta_final: float, warm: int, ramp: int):
    if variant == "uniform":
        return 0.0
    if variant.startswith("fixed") or variant.startswith("grid") or variant == "hard_cold":
        return beta_final
    if step < warm:
        return 0.0
    u = min(1.0, max(0.0, (step - warm) / max(1, ramp)))
    u = u * u * (3 - 2 * u)
    return beta_final * u


@dataclass
class Params:
    d: int = 64
    p: int = 256
    C: int = 4
    K: int = 4
    mu: float = 1.0
    sigma0: float = 0.5
    t: float = 2.05
    n_train: int = 6000
    n_test: int = 3000
    n_calib: int = 2500
    batch_size: int = 256
    steps: int = 4000
    lr: float = 3e-3
    init_std: float = 1e-3
    activation: str = "erf"
    seed: int = 0
    warmup_steps: int = 800
    ramp_steps: int = 1000
    eval_every: int = 200
    power_iters: int = 30
    ridge: float = 1e-5
    router_steps: int = 1200
    router_lr: float = 2e-3
    router_batch_size: int = 256
    beta_grid: bool = False
    quick: bool = False
    device: str = "auto"
    outdir: str = "./gmm_v5_router"


def train_variant(name: str, beta_final: float, params: Params, rf, data, device):
    ex = RFExperts(params.p, params.d, params.K, params.init_std).to(device)
    opt = torch.optim.AdamW(ex.parameters(), lr=params.lr)
    rows = []
    xt, eps_t, ltr = data["x_train"], data["y_train"], data["labels_train"]
    xv, eps_v, lte = data["x_test"], data["y_test"], data["labels_test"]
    phi_train = data["phi_train"]
    phi_test = data["phi_test"]
    f0_train = data["f0_train"]
    f0_test = data["f0_test"]
    for step in range(params.steps + 1):
        if step % params.eval_every == 0 or step == params.steps:
            b = sched(step, name, beta_final, params.warmup_steps, params.ramp_steps)
            with torch.no_grad():
                pred_test = f0_test[:, None, :] + ex(phi_test)
                e_test = ((pred_test - eps_v[:, None, :]) ** 2).mean(-1)
                _, q_test = mcl_loss(e_test, b)
                row = {
                    "variant": name,
                    "step": step,
                    "beta": b,
                    "test_oracle_best_mse": float(e_test.min(1).values.mean().item()),
                    "test_soft_oracle_mse": float((q_test * e_test).sum(1).mean().item()),
                    "test_mean_expert_mse": float(e_test.mean(1).mean().item()),
                    "test_teacher_entropy_norm": float((-(q_test.clamp_min(1e-12) * q_test.clamp_min(1e-12).log()).sum(1).mean() / math.log(params.K)).item()),
                    "test_oracle_class_mi_norm": mi_norm(e_test.argmin(1), lte, params.K, params.C),
                }
                rows.append(row)
        if step == params.steps:
            break
        idx = torch.randint(0, xt.shape[0], (params.batch_size,), device=device)
        phi = phi_train[idx]
        pred = f0_train[idx][:, None, :] + ex(phi)
        e = ((pred - eps_t[idx][:, None, :]) ** 2).mean(-1)
        b = sched(step, name, beta_final, params.warmup_steps, params.ramp_steps)
        loss, _ = mcl_loss(e, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return ex, rows


@torch.no_grad()
def teacher_q(experts, phi, y, f0, beta: float):
    pred = f0[:, None, :] + experts(phi)
    e = ((pred - y[:, None, :]) ** 2).mean(-1)
    _, q = mcl_loss(e, beta)
    return q


def train_router(experts, phi, y, f0, beta: float, params: Params, device):
    router = Router(params.p, params.K).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=params.router_lr)
    experts.eval()
    for _ in range(params.router_steps):
        idx = torch.randint(0, phi.shape[0], (params.router_batch_size,), device=device)
        with torch.no_grad():
            pred = f0[idx][:, None, :] + experts(phi[idx])
            e = ((pred - y[idx][:, None, :]) ** 2).mean(-1)
            _, q = mcl_loss(e, beta)
        loss = -(q * torch.log_softmax(router(phi[idx]), 1)).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return router


@torch.no_grad()
def eval_router(experts, router, phi, y, labels, f0, beta: float, c: int):
    pred = f0[:, None, :] + experts(phi)
    e = ((pred - y[:, None, :]) ** 2).mean(-1)
    _, q = mcl_loss(e, beta)
    rq = torch.softmax(router(phi), 1)
    mix = torch.einsum("bk,bkd->bd", rq, pred)
    u = torch.bincount(rq.argmax(1), minlength=pred.shape[1]).float()
    u = u / u.sum().clamp_min(1)
    return dict(
        oracle_best_mse=float(e.min(1).values.mean().item()),
        soft_oracle_mse=float((q * e).sum(1).mean().item()),
        mean_expert_mse=float(e.mean(1).mean().item()),
        router_mix_mse=float(((mix - y) ** 2).mean(1).mean().item()),
        router_soft_mse=float((rq * e).sum(1).mean().item()),
        oracle_class_mi_norm=mi_norm(e.argmin(1), labels, pred.shape[1], c),
        router_class_mi_norm=mi_norm(rq.argmax(1), labels, pred.shape[1], c),
        oracle_usage_entropy=norm_entropy(torch.bincount(e.argmin(1), minlength=pred.shape[1]).float().div(max(1, y.shape[0]))),
        router_usage_entropy=norm_entropy(u),
        teacher_entropy_norm=float((-(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(1).mean() / math.log(pred.shape[1])).item()),
        router_vs_teacher_ce=float((-(q.clamp_min(1e-12) * rq.clamp_min(1e-12).log()).sum(1).mean()).item()),
        router_vs_teacher_kl=float((q.clamp_min(1e-12) * (q.clamp_min(1e-12).log() - rq.clamp_min(1e-12).log())).sum(1).mean().item()),
    )


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run(params: Params):
    if params.quick:
        params.n_train = min(params.n_train, 2500)
        params.n_test = min(params.n_test, 1200)
        params.n_calib = min(params.n_calib, 1200)
        params.steps = min(params.steps, 1500)
        params.eval_every = min(params.eval_every, 150)
        params.router_steps = min(params.router_steps, 600)
    set_seed(params.seed)
    device = device_of(params.device)
    out = ensure_dir(params.outdir)
    means = simplex_means(params.C, params.d, params.mu, device)
    xtr, eps_tr, ltr = sample_gmm(params.n_train, means, params.sigma0, params.t, device)
    xte, eps_te, lte = sample_gmm(params.n_test, means, params.sigma0, params.t, device)
    xca, eps_ca, lca = sample_gmm(params.n_calib, means, params.sigma0, params.t, device)
    rf = RandomFeatures(params.d, params.p, params.activation, params.seed).to(device).eval()
    phi_tr = rf(xtr)
    phi_te = rf(xte)
    a0 = ridge_fit(phi_tr, eps_tr, params.ridge)
    f0_tr = phi_tr @ a0
    f0_te = phi_te @ a0
    cal = calibrate(rf, xca, eps_ca, lca, means, params.ridge, params.power_iters)
    base_grid = [0.2, 0.4, 0.7, 1.1]
    bc = cal["beta_class"]
    bg = cal["beta_glass_emp"]
    variants = [
        ("uniform", 0.0),
        ("fixed_class", max(0.2, 1.2 * bc)),
        ("fixed_good", min(max(0.2, 3.0 * bc), 0.45 * bg)),
        ("hard_cold", 1.2 * bg),
        ("theory_anneal", min(max(0.2, 3.0 * bc), 0.45 * bg)),
    ]
    if params.beta_grid:
        for b in base_grid:
            variants.append((f"grid_{b:.3f}", b))
    data = dict(x_train=xtr, y_train=eps_tr, labels_train=ltr, x_test=xte, y_test=eps_te, labels_test=lte, phi_train=phi_tr, phi_test=phi_te, f0_train=f0_tr, f0_test=f0_te)
    all_rows = []
    final_rows = []
    for name, bfin in variants:
        print(f"=== {name} beta={bfin:.6g} ===", flush=True)
        ex, rows = train_variant(name, bfin, params, rf, data, device)
        all_rows.extend(rows)
        beval = bfin if name != "uniform" else max(0.2, bc)
        router = train_router(ex, phi_tr, eps_tr, f0_tr, beval, params, device)
        final = eval_router(ex, router, phi_te, eps_te, lte, f0_te, beval, params.C)
        final.update(variant=name, beta_eval=beval)
        final_rows.append(final)
        torch.save(dict(experts=ex.state_dict(), router=router.state_dict(), rf=rf.state_dict(), ridge_head=a0, params=asdict(params), calibration=cal, variant=name, beta_eval=beval), out / f"checkpoint_{name}.pt")
        write_csv(out / "metrics_all.csv", all_rows)
        write_csv(out / "router_final_metrics.csv", final_rows)
    if plt is not None and pd is not None:
        df = pd.DataFrame(all_rows)
        if not df.empty:
            for y in ["test_oracle_best_mse", "test_oracle_class_mi_norm", "test_teacher_entropy_norm"]:
                if y in df:
                    plt.figure()
                    for v, s in df.groupby("variant"):
                        plt.plot(s["step"], s[y], label=v)
                    plt.xlabel("step")
                    plt.ylabel(y)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(out / f"{y}.png", dpi=180)
                    plt.close()
    lines = ["RF-MCL GMM v5 router (target=eps, residualized)", json.dumps(asdict(params), indent=2), "\nCalibration:"]
    lines.extend([f"  {k}: {v}" for k, v in cal.items()])
    lines.append("\nFinal router metrics:")
    for r in final_rows:
        lines.append(f"\n[{r['variant']}] beta_eval={r['beta_eval']}")
        for k in ["oracle_best_mse", "soft_oracle_mse", "mean_expert_mse", "router_mix_mse", "router_soft_mse", "oracle_class_mi_norm", "router_class_mi_norm", "teacher_entropy_norm", "router_vs_teacher_ce", "router_vs_teacher_kl"]:
            lines.append(f"  {k}: {r[k]}")
        if r["teacher_entropy_norm"] > 0.95:
            lines.append("  teacher_status: near_uniform (router likely uninformative)")
        elif r["teacher_entropy_norm"] < 0.05:
            lines.append("  teacher_status: near_collapse")
    (out / "SUMMARY.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def parse():
    p = argparse.ArgumentParser()
    for k, v in asdict(Params()).items():
        arg = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(arg, action="store_true" if not v else "store_false")
        elif isinstance(v, int):
            p.add_argument(arg, type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(arg, type=float, default=v)
        else:
            p.add_argument(arg, type=str, default=v)
    return Params(**vars(p.parse_args()))


if __name__ == "__main__":
    run(parse())

