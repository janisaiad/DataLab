#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

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
    import pandas as pd
except Exception:
    pd = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


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


def parse_classes(classes: str) -> List[int]:
    out = []
    for c in [z.strip() for z in classes.split(",") if z.strip()]:
        out.append(int(c) if c.isdigit() else CIFAR10_CLASSES.index(c))
    return out


def load_cifar(data_root: str, class_ids: List[int], n_train: int, n_test: int, no_download: bool, seed: int):
    if torchvision is None:
        raise ImportError("torchvision is required for CIFAR.")
    tr = torchvision.datasets.CIFAR10(data_root, train=True, download=not no_download, transform=T.ToTensor())
    te = torchvision.datasets.CIFAR10(data_root, train=False, download=not no_download, transform=T.ToTensor())

    def collect(ds, n, offset):
        local = {c: i for i, c in enumerate(class_ids)}
        per = max(1, n // len(class_ids))
        cnt = {c: 0 for c in class_ids}
        idx = list(range(len(ds)))
        rng = random.Random(seed + offset)
        rng.shuffle(idx)
        xs = []
        ys = []
        for j in idx:
            x, y = ds[j]
            if y not in local:
                continue
            if len(xs) < n or cnt[y] < per:
                xs.append(x.flatten())
                ys.append(local[y])
                cnt[y] += 1
            if len(xs) >= n and all(cnt[c] >= per for c in class_ids):
                break
        x = torch.stack(xs)[:n]
        y = torch.tensor(ys[:n], dtype=torch.long)
        return x, y

    xtr, ytr = collect(tr, n_train, 1)
    xte, yte = collect(te, n_test, 2)
    mean = xtr.mean(0, keepdim=True)
    std = xtr.std().clamp_min(1e-6)
    xtr = (xtr - mean) / std
    xte = (xte - mean) / std
    info = dict(classes=[CIFAR10_CLASSES[i] for i in class_ids], raw_dim=int(xtr.shape[1]), global_std=float(std), train_n=int(xtr.shape[0]), test_n=int(xte.shape[0]))
    return xtr, ytr, xte, yte, info


@dataclass
class FeatureMap:
    mode: str
    mean: Optional[torch.Tensor] = None
    pca: Optional[torch.Tensor] = None
    rp: Optional[torch.Tensor] = None

    def transform(self, x):
        if self.mode == "full":
            return x
        if self.mode == "pca":
            return (x - self.mean.to(x.device)) @ self.pca.to(x.device).T
        if self.mode == "rp":
            return x @ self.rp.to(x.device).T
        raise ValueError(self.mode)


def build_feature_map(x: torch.Tensor, mode: str, pca_dim: int, rp_dim: int, seed: int):
    if mode == "full":
        return FeatureMap("full")
    if mode == "pca":
        q = min(pca_dim, x.shape[1], x.shape[0] - 1)
        mean = x.mean(0, keepdim=True)
        xc = (x - mean).cpu()
        _, _, v = torch.pca_lowrank(xc, q=q, center=False, niter=4)
        return FeatureMap("pca", mean.cpu(), v[:, :q].T.contiguous().cpu(), None)
    if mode == "rp":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 999)
        r = torch.randn(rp_dim, x.shape[1], generator=gen) / math.sqrt(rp_dim)
        return FeatureMap("rp", None, None, r)
    raise ValueError(mode)


def diffuse_with_target(x0: torch.Tensor, t: float):
    gamma = math.exp(-t)
    eps = torch.randn_like(x0)
    xt = gamma * x0 + math.sqrt(max(1.0 - gamma * gamma, 1e-12)) * eps
    return xt, eps


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
def class_basis(target, labels, c: int):
    mats = []
    for ci in range(c):
        m = labels == ci
        if m.sum():
            mats.append(target[m].mean(0))
    m = torch.stack(mats)
    m = m - m.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(m, full_matrices=False)
    rank = int((s > 1e-8 * s.max().clamp_min(1e-8)).sum())
    return vh[:rank].T.contiguous()


@torch.no_grad()
def calibrate(rf, x, y, labels, c: int, ridge: float, power_iters: int, class_target=None):
    phi = rf(x)
    a0 = ridge_fit(phi, y, ridge)
    r = y - phi @ a0
    s = whiten(phi, ridge)
    lf = power_free(s, r, power_iters)
    target_for_basis = y if class_target is None else class_target
    b = class_basis(target_for_basis, labels, c).to(x.device)
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
    data_root: str = "./data"
    classes: str = "automobile,horse"
    d_mode: str = "pca"
    pca_dim: int = 512
    rp_dim: int = 512
    p: int = 512
    K: int = 4
    t: float = 1.5
    n_train: int = 8000
    n_test: int = 2500
    n_calib: int = 2048
    batch_size: int = 192
    steps: int = 3000
    lr: float = 2e-3
    init_std: float = 1e-3
    activation: str = "erf"
    seed: int = 0
    warmup_steps: int = 600
    ramp_steps: int = 900
    eval_every: int = 200
    power_iters: int = 30
    ridge: float = 1e-5
    router_steps: int = 1000
    router_lr: float = 2e-3
    router_batch_size: int = 192
    beta_grid: bool = False
    quick: bool = False
    no_download: bool = False
    device: str = "auto"
    outdir: str = "./cifar_v5_router"


def train_variant(name: str, beta_final: float, params: Params, rf, data, device, c: int):
    ex = RFExperts(params.p, data["y_train"].shape[1], params.K, params.init_std).to(device)
    opt = torch.optim.AdamW(ex.parameters(), lr=params.lr)
    rows = []
    phi_train, phi_test = data["phi_train"], data["phi_test"]
    f0_train, f0_test = data["f0_train"], data["f0_test"]
    ytr, yte = data["y_train"], data["y_test"]
    ltr, lte = data["labels_train"], data["labels_test"]
    for step in range(params.steps + 1):
        if step % params.eval_every == 0 or step == params.steps:
            b = sched(step, name, beta_final, params.warmup_steps, params.ramp_steps)
            with torch.no_grad():
                pred_test = f0_test[:, None, :] + ex(phi_test)
                e_test = ((pred_test - yte[:, None, :]) ** 2).mean(-1)
                _, q_test = mcl_loss(e_test, b)
                row = {
                    "variant": name,
                    "step": step,
                    "beta": b,
                    "test_oracle_best_mse": float(e_test.min(1).values.mean().item()),
                    "test_soft_oracle_mse": float((q_test * e_test).sum(1).mean().item()),
                    "test_mean_expert_mse": float(e_test.mean(1).mean().item()),
                    "test_cost_gap_mean": float((e_test.mean(1) - e_test.min(1).values).mean().item()),
                    "test_beta_gap": float((b * (e_test.mean(1) - e_test.min(1).values)).mean().item()),
                    "test_teacher_entropy_norm": float((-(q_test.clamp_min(1e-12) * q_test.clamp_min(1e-12).log()).sum(1).mean() / math.log(params.K)).item()),
                    "test_oracle_class_mi_norm": mi_norm(e_test.argmin(1), lte, params.K, c),
                }
                rows.append(row)
        if step == params.steps:
            break
        idx = torch.randint(0, phi_train.shape[0], (params.batch_size,), device=device)
        pred = f0_train[idx][:, None, :] + ex(phi_train[idx])
        e = ((pred - ytr[idx][:, None, :]) ** 2).mean(-1)
        b = sched(step, name, beta_final, params.warmup_steps, params.ramp_steps)
        loss, _ = mcl_loss(e, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return ex, rows


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
def eval_router(experts, router, phi, y, labels, f0, beta: float, c: int, k: int):
    pred = f0[:, None, :] + experts(phi)
    e = ((pred - y[:, None, :]) ** 2).mean(-1)
    _, q = mcl_loss(e, beta)
    rq = torch.softmax(router(phi), 1)
    mix = torch.einsum("bk,bkd->bd", rq, pred)
    return dict(
        oracle_best_mse=float(e.min(1).values.mean().item()),
        soft_oracle_mse=float((q * e).sum(1).mean().item()),
        mean_expert_mse=float(e.mean(1).mean().item()),
        cost_gap_mean=float((e.mean(1) - e.min(1).values).mean().item()),
        beta_gap=float((beta * (e.mean(1) - e.min(1).values)).mean().item()),
        router_mix_mse=float(((mix - y) ** 2).mean(1).mean().item()),
        router_soft_mse=float((rq * e).sum(1).mean().item()),
        oracle_class_mi_norm=mi_norm(e.argmin(1), labels, k, c),
        router_class_mi_norm=mi_norm(rq.argmax(1), labels, k, c),
        oracle_usage_entropy=norm_entropy(torch.bincount(e.argmin(1), minlength=k).float().div(max(1, y.shape[0]))),
        router_usage_entropy=norm_entropy(torch.bincount(rq.argmax(1), minlength=k).float().div(max(1, y.shape[0]))),
        teacher_entropy_norm=float((-(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(1).mean() / math.log(k)).item()),
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
        params.n_train = min(params.n_train, 3000)
        params.n_test = min(params.n_test, 1000)
        params.n_calib = min(params.n_calib, 1000)
        params.steps = min(params.steps, 1200)
        params.eval_every = min(params.eval_every, 150)
        params.router_steps = min(params.router_steps, 500)
        params.p = min(params.p, 256)
        params.pca_dim = min(params.pca_dim, 256)
    set_seed(params.seed)
    device = device_of(params.device)
    out = ensure_dir(params.outdir)
    ids = parse_classes(params.classes)
    c = len(ids)
    xtr_raw, ytr, xte_raw, yte, info = load_cifar(params.data_root, ids, params.n_train, params.n_test, params.no_download, params.seed)
    fmap = build_feature_map(xtr_raw, params.d_mode, params.pca_dim, params.rp_dim, params.seed)
    x0 = fmap.transform(xtr_raw).float()
    xt0 = fmap.transform(xte_raw).float()
    lat_mean = x0.mean(0, keepdim=True)
    lat_std = x0.std().clamp_min(1e-6)
    x0 = (x0 - lat_mean) / lat_std
    xt0 = (xt0 - lat_mean) / lat_std
    xtr_t, eps_tr = diffuse_with_target(x0, params.t)
    xte_t, eps_te = diffuse_with_target(xt0, params.t)
    xtr = xtr_t.to(device)
    ytr_eps = eps_tr.to(device)
    ltr = ytr.to(device)
    xte = xte_t.to(device)
    yte_eps = eps_te.to(device)
    lte = yte.to(device)
    d = ytr_eps.shape[1]
    rf = RandomFeatures(d, params.p, params.activation, params.seed).to(device).eval()
    phi_tr = rf(xtr)
    phi_te = rf(xte)
    a0 = ridge_fit(phi_tr, ytr_eps, params.ridge)
    f0_tr = phi_tr @ a0
    f0_te = phi_te @ a0
    idx = torch.randperm(xtr.shape[0], device=device)[: min(params.n_calib, xtr.shape[0])]
    idx_cpu = idx.cpu()
    cal = calibrate(
        rf,
        xtr[idx],
        ytr_eps[idx],
        ltr[idx],
        c,
        params.ridge,
        params.power_iters,
        class_target=x0[idx_cpu].to(device),
    )
    (out / "params.json").write_text(json.dumps(asdict(params), indent=2))
    info.update(d_mode=params.d_mode, d_latent=int(d), p=params.p, K=params.K, t=params.t, target="eps")
    (out / "data_info.json").write_text(json.dumps(info, indent=2))
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
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
        for b in [0.2, 0.4, 0.7, 1.1]:
            variants.append((f"grid_{b:.3f}", b))
    all_rows = []
    final_rows = []
    for name, bfin in variants:
        print(f"=== {name} beta={bfin:.6g} ===", flush=True)
        ex, rows = train_variant(name, bfin, params, rf, dict(phi_train=phi_tr, phi_test=phi_te, f0_train=f0_tr, f0_test=f0_te, y_train=ytr_eps, y_test=yte_eps, labels_train=ltr, labels_test=lte), device, c)
        all_rows.extend(rows)
        beval = bfin if name != "uniform" else max(0.2, bc)
        router = train_router(ex, phi_tr, ytr_eps, f0_tr, beval, params, device)
        final = eval_router(ex, router, phi_te, yte_eps, lte, f0_te, beval, c, params.K)
        final.update(variant=name, beta_eval=beval)
        final_rows.append(final)
        torch.save(dict(experts=ex.state_dict(), router=router.state_dict(), rf=rf.state_dict(), ridge_head=a0, params=asdict(params), data_info=info, calibration=cal, variant=name, beta_eval=beval), out / f"checkpoint_{name}.pt")
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
    lines = ["RF-MCL CIFAR v5 router (target=eps, residualized)", json.dumps(asdict(params), indent=2), "\nData info:", json.dumps(info, indent=2), "\nCalibration:"]
    lines.extend([f"  {k}: {v}" for k, v in cal.items()])
    lines.append("\nFinal router metrics:")
    for r in final_rows:
        lines.append(f"\n[{r['variant']}] beta_eval={r['beta_eval']}")
        for k in ["oracle_best_mse", "soft_oracle_mse", "mean_expert_mse", "cost_gap_mean", "beta_gap", "router_mix_mse", "router_soft_mse", "oracle_class_mi_norm", "router_class_mi_norm", "teacher_entropy_norm", "router_vs_teacher_ce", "router_vs_teacher_kl"]:
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

