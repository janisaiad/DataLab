#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_mcl_phase_diagram_experiments.py

Numerical checks for the three calculations discussed in the MCL/diffusion/RF/GMM toy theory:

A. RF population GMM, fixed diffusion time t, tau=infinity
   - Compute beta_spec(t) from the transverse Hessian of annealed WTA/MCL.
   - Verify that the top unstable mode is aligned with the GMM class direction m.

B. RF population GMM, finite training time tau
   - Use the closed-form RF gradient-flow solution
       A_tau/sqrt(p) = -(1/sqrt(Delta_t)) V^T U^{-1}(I-exp(-gamma U))
     with gamma = 2 Delta_t tau / psi_p.
   - Compute beta_spec(t,tau) and show how the spectral filter controls when experts split.

C. RF empirical GMM, finite n
   - Build the empirical REM partition function
       Z_beta = sum_mu exp[- beta E_mu(t,tau)]
     with exact GMM atom costs E_{mu nu}(t)=e^{-2t}/Delta_t ||a_mu-a_nu||^2.
   - Use an RF spectral training filter to define an effective learned empirical variance
       v(t,tau) = v_inf(t) R(t,tau),
       R(t,tau)=mean_lambda (1-exp(-2 Delta_t lambda tau / psi_p))^2.
   - Check that Gibbs entropy collapses as beta/beta_glass crosses 1.

The script outputs CSV tables and PNG figures.

Example:
    python rf_mcl_phase_diagram_experiments.py --quick --outdir ./rf_mcl_results

For better accuracy:
    python rf_mcl_phase_diagram_experiments.py --d 64 --p 192 --n-train 12000 --n-eval 5000 --power-iters 35

Notes:
    * beta is the inverse temperature of the MCL/WTA expert assignment, not the diffusion noise.
    * losses are unnormalized ||.||^2; if you use loss/d in your implementation, replace beta by beta*d.
    * The RF/GEP formulas are intended as high-dimensional approximations; finite d,p,n plots are sanity checks.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


# -----------------------------
# Basic utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def logsumexp(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def stable_sech2(x: np.ndarray) -> np.ndarray:
    # sech^2(x) = 1/cosh^2(x), stable enough after clipping.
    x_clip = np.clip(x, -40.0, 40.0)
    c = np.cosh(x_clip)
    return 1.0 / (c * c)


def solve_y_alpha(alpha: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Solve y - 1 - log y = 2 alpha for y in (0,1)."""
    lo, hi = 1e-14, 1.0 - 1e-14
    target = 2.0 * alpha
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = mid - 1.0 - math.log(mid)
        # val decreases from +inf at 0 to 0 at 1.
        if val > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def sym_eigh(A: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    return vals, vecs


def inv_sqrt_from_eigh(vals: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def filter_matrix_from_eigh(vals: np.ndarray, vecs: np.ndarray, gamma: float, ridge: float = 1e-9) -> np.ndarray:
    """U^{-1}(I-exp(-gamma U)) from eigendecomposition U=P diag(vals) P^T."""
    coeff = (1.0 - np.exp(-gamma * vals)) / (vals + ridge)
    return (vecs * coeff) @ vecs.T


def inverse_matrix_from_eigh(vals: np.ndarray, vecs: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    coeff = 1.0 / (vals + ridge)
    return (vecs * coeff) @ vecs.T


# -----------------------------
# GMM + RF model
# -----------------------------

@dataclass
class ProblemConfig:
    d: int = 48
    p: int = 128
    mu: float = 1.0
    sigma0: float = 0.4
    seed: int = 0
    activation: str = "tanh"


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_class_direction(d: int, mu: float) -> np.ndarray:
    # ||m||^2 = d mu^2.
    return mu * np.ones(d, dtype=np.float64)


def sample_gmm(n: int, m: np.ndarray, sigma0: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    d = m.shape[0]
    y = rng.choice(np.array([-1.0, 1.0]), size=n)
    X0 = y[:, None] * m[None, :] + sigma0 * rng.normal(size=(n, d))
    return X0, y


def diffuse(X0: np.ndarray, t: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    a = math.exp(-t)
    Delta = 1.0 - math.exp(-2.0 * t)
    sqrtDelta = math.sqrt(max(Delta, 1e-15))
    Xi = rng.normal(size=X0.shape)
    Xt = a * X0 + sqrtDelta * Xi
    Gamma = Delta  # Will be overwritten in callers if sigma0 needed; here Xt already generated.
    return Xt, Xi, a, Delta, Gamma


def activation_fn(Z: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return np.tanh(Z)
    if name == "erf":
        # Smooth bounded odd activation; numpy has no vectorized erf in base in some installations.
        return np.vectorize(math.erf)(Z / math.sqrt(2.0))
    if name == "relu":
        return np.maximum(Z, 0.0)
    if name == "linear":
        return Z
    raise ValueError(f"unknown activation {name}")


def activation_prime_moments(Gamma: float, activation: str, n_quad: int = 200000, seed: int = 123) -> Tuple[float, float]:
    """Monte Carlo estimates of q0=E sigma(sqrt(Gamma)Z)^2 and a1=E sigma'(sqrt(Gamma)Z)."""
    rng = make_rng(seed)
    z = rng.normal(size=n_quad)
    u = math.sqrt(Gamma) * z
    if activation == "tanh":
        sig = np.tanh(u)
        sigp = 1.0 - sig * sig
    elif activation == "relu":
        sig = np.maximum(u, 0.0)
        sigp = (u > 0.0).astype(np.float64)
    elif activation == "linear":
        sig = u
        sigp = np.ones_like(u)
    elif activation == "erf":
        sig = np.vectorize(math.erf)(u / math.sqrt(2.0))
        sigp = math.sqrt(2.0 / math.pi) * np.exp(-0.5 * u * u)
    else:
        raise ValueError(activation)
    return float(np.mean(sig * sig)), float(np.mean(sigp))


class RandomFeatureMap:
    def __init__(self, d: int, p: int, activation: str, rng: np.random.Generator):
        self.d = d
        self.p = p
        self.activation = activation
        self.W = rng.normal(size=(p, d))

    def phi(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.W.T / math.sqrt(self.d)
        return activation_fn(Z, self.activation)


@dataclass
class RFPopulationFit:
    U: np.ndarray
    V: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    U_inv_sqrt: np.ndarray
    U_inv: np.ndarray


def fit_population_rf(Phi: np.ndarray, Xi: np.ndarray, ridge: float = 1e-8) -> RFPopulationFit:
    # U = E phi phi^T, V = E phi xi^T.
    n = Phi.shape[0]
    U = (Phi.T @ Phi) / n
    V = (Phi.T @ Xi) / n
    vals, vecs = sym_eigh(U, eps=ridge)
    U_inv_sqrt = inv_sqrt_from_eigh(vals, vecs)
    U_inv = inverse_matrix_from_eigh(vals, vecs, ridge=ridge)
    return RFPopulationFit(U=U, V=V, eigvals=vals, eigvecs=vecs, U_inv_sqrt=U_inv_sqrt, U_inv=U_inv)


def predict_noise_from_fit(Phi: np.ndarray, fit: RFPopulationFit, Delta: float, tau: float | None, psi_p: float) -> np.ndarray:
    """Return f_tau(x)=V^T U^{-1}(I-exp(-gamma U))phi.

    tau=None means tau=infinity.
    """
    if tau is None or math.isinf(tau):
        F = fit.U_inv
    else:
        gamma = 2.0 * Delta * tau / psi_p
        F = filter_matrix_from_eigh(fit.eigvals, fit.eigvecs, gamma)
    # Phi @ F @ V has shape (N,d). This is E[xi|x] approximation.
    return Phi @ F @ fit.V


def f_star_gmm(Xt: np.ndarray, m: np.ndarray, t: float, Delta: float, Gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    a = math.exp(-t)
    S = a * (Xt @ m) / Gamma
    f = math.sqrt(Delta) / Gamma * (Xt - a * np.tanh(S)[:, None] * m[None, :])
    return f, S


# -----------------------------
# Hessian top eigenvalue by power method
# -----------------------------

def whiten_features(Phi: np.ndarray, U_inv_sqrt: np.ndarray) -> np.ndarray:
    return Phi @ U_inv_sqrt


def top_hessian_eigen_power(
    Psi: np.ndarray,
    Q: np.ndarray,
    m: np.ndarray,
    n_iter: int = 30,
    seed: int = 0,
) -> Tuple[float, float, np.ndarray]:
    """Top eigenvalue of H -> E[q q^T H psi psi^T] in whitened feature coordinates.

    The Rayleigh quotient is E[(q^T H psi)^2] / ||H||_F^2.
    Returns (lambda, output_alignment_with_m, H_top).
    """
    rng = make_rng(seed)
    N, p = Psi.shape
    d = Q.shape[1]
    H = rng.normal(size=(d, p))
    H /= np.linalg.norm(H) + 1e-30

    for _ in range(n_iter):
        # s_i = q_i^T H psi_i.
        QH = Q @ H  # (N,p)
        s = np.sum(QH * Psi, axis=1)  # (N,)
        T = (Q.T @ (s[:, None] * Psi)) / N  # (d,p)
        norm_T = np.linalg.norm(T)
        if norm_T < 1e-30:
            break
        H = T / norm_T

    QH = Q @ H
    s = np.sum(QH * Psi, axis=1)
    lam = float(np.mean(s * s) / (np.linalg.norm(H) ** 2 + 1e-30))
    mhat = m / (np.linalg.norm(m) + 1e-30)
    align = float(np.linalg.norm(mhat @ H) / (np.linalg.norm(H) + 1e-30))
    return lam, align, H


def class_channel_lambda(
    Phi: np.ndarray,
    Xt: np.ndarray,
    f_tau: np.ndarray,
    f_star: np.ndarray,
    S: np.ndarray,
    fit: RFPopulationFit,
    m: np.ndarray,
    t: float,
    Delta: float,
    Gamma: float,
    sigma0: float,
) -> Tuple[float, float, float, float]:
    """Class-channel formula for lambda_parallel and beta_spec.

    lambda_parallel = iso + (Delta/Gamma) kappa lambda_RF_tau.
    """
    a = math.exp(-t)
    mnorm2 = float(np.dot(m, m))
    kappa = a * a * mnorm2 / Gamma
    iso = a * a * sigma0 * sigma0 / Gamma
    mhat = m / math.sqrt(mnorm2)
    b_m = (f_star - f_tau) @ mhat
    if kappa < 1e-12:
        w = stable_sech2(S)
    else:
        w = stable_sech2(S) + (Gamma / (Delta * kappa + 1e-30)) * b_m * b_m
    M = (Phi.T @ (w[:, None] * Phi)) / Phi.shape[0]
    G = fit.U_inv_sqrt @ M @ fit.U_inv_sqrt
    G = 0.5 * (G + G.T)
    eigs = np.linalg.eigvalsh(G)
    lambda_rf = float(np.max(eigs))
    lambda_parallel = iso + (Delta / Gamma) * kappa * lambda_rf
    beta = 1.0 / (2.0 * lambda_parallel + 1e-30)
    return lambda_parallel, beta, lambda_rf, float(np.mean(w))


# -----------------------------
# Experiments A/B
# -----------------------------

def run_population_experiment_for_t(
    cfg: ProblemConfig,
    rf: RandomFeatureMap,
    t: float,
    tau: float | None,
    n_train: int,
    n_eval: int,
    power_iters: int,
    seed: int,
) -> Dict[str, float]:
    rng_train = make_rng(seed + 1000 + int(1000 * t) + (0 if tau is None else int(17 * tau)))
    rng_eval = make_rng(seed + 2000 + int(1000 * t) + (0 if tau is None else int(19 * tau)))
    m = make_class_direction(cfg.d, cfg.mu)
    a = math.exp(-t)
    Delta = 1.0 - math.exp(-2.0 * t)
    Gamma = Delta + a * a * cfg.sigma0 * cfg.sigma0
    psi_p = cfg.p / cfg.d
    kappa = a * a * float(np.dot(m, m)) / Gamma

    X0_tr, _ = sample_gmm(n_train, m, cfg.sigma0, rng_train)
    Xt_tr, Xi_tr, _, _, _ = diffuse(X0_tr, t, rng_train)
    Phi_tr = rf.phi(Xt_tr)
    fit = fit_population_rf(Phi_tr, Xi_tr)

    X0_ev, _ = sample_gmm(n_eval, m, cfg.sigma0, rng_eval)
    Xt_ev, Xi_ev, _, _, _ = diffuse(X0_ev, t, rng_eval)
    Phi_ev = rf.phi(Xt_ev)
    f_tau = predict_noise_from_fit(Phi_ev, fit, Delta, tau=tau, psi_p=psi_p)
    Q = Xi_ev - f_tau
    fstar, S = f_star_gmm(Xt_ev, m, t, Delta, Gamma)
    Psi_ev = whiten_features(Phi_ev, fit.U_inv_sqrt)

    lam_power, align, _ = top_hessian_eigen_power(Psi_ev, Q, m, n_iter=power_iters, seed=seed + 333)
    beta_power = 1.0 / (2.0 * lam_power + 1e-30)

    lam_class, beta_class, lambda_rf, mean_w = class_channel_lambda(
        Phi_ev, Xt_ev, f_tau, fstar, S, fit, m, t, Delta, Gamma, cfg.sigma0
    )
    q0, a1 = activation_prime_moments(Gamma, cfg.activation, n_quad=50000, seed=seed + 444)
    lambda_c_U = q0 + psi_p * Gamma * a1 * a1 * (1.0 + kappa)
    eps_c = 0.0 if tau is None or math.isinf(tau) else math.exp(-2.0 * Delta * lambda_c_U * tau / psi_p)

    return {
        "t": t,
        "tau": math.inf if tau is None else tau,
        "Delta": Delta,
        "Gamma": Gamma,
        "kappa": kappa,
        "lambda_power": lam_power,
        "beta_power": beta_power,
        "output_alignment_m": align,
        "lambda_class": lam_class,
        "beta_class": beta_class,
        "lambda_rf": lambda_rf,
        "mean_w": mean_w,
        "q0": q0,
        "a1": a1,
        "lambda_c_U": lambda_c_U,
        "eps_c": eps_c,
    }


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _get_pyplot():
    import importlib.util
    if importlib.util.find_spec("matplotlib") is None:
        print("  [plot skipped] matplotlib unavailable")
        return None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"  [plot skipped] matplotlib import failed: {exc}")
        return None


def plot_A(rows: List[Dict[str, float]], outdir: str) -> None:
    plt = _get_pyplot()
    if plt is None:
        return
    t = np.array([r["t"] for r in rows])
    beta_power = np.array([r["beta_power"] for r in rows])
    beta_class = np.array([r["beta_class"] for r in rows])
    align = np.array([r["output_alignment_m"] for r in rows])
    kappa = np.array([r["kappa"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
    ax1.plot(t, beta_power, marker="o", label=r"$\beta_{spec}$ power Hessian")
    ax1.plot(t, beta_class, marker="s", linestyle="--", label=r"class-channel formula")
    ax1.set_xlabel(r"diffusion time $t$")
    ax1.set_ylabel(r"inverse WTA temperature $\beta_{spec}$")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(t, align, marker="^", color="black", alpha=0.65, label="output alignment with m")
    ax2.set_ylabel(r"alignment with class direction $m$")
    ax2.set_ylim(0.0, 1.05)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "A_beta_spec_population_tau_inf.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(t, kappa, marker="o")
    ax.axhline(1.0, linestyle="--", color="black", alpha=0.5)
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel(r"class SNR $\kappa_t=e^{-2t}\|m\|^2/\Gamma_t$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "A_class_snr_kappa.png"), dpi=180)
    plt.close(fig)


def plot_B(rows: List[Dict[str, float]], outdir: str) -> None:
    plt = _get_pyplot()
    if plt is None:
        return
    tau = np.array([r["tau"] for r in rows])
    beta_power = np.array([r["beta_power"] for r in rows])
    beta_class = np.array([r["beta_class"] for r in rows])
    align = np.array([r["output_alignment_m"] for r in rows])
    eps = np.array([r["eps_c"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
    ax1.plot(tau, beta_power, marker="o", label=r"$\beta_{spec}(t,\tau)$ power Hessian")
    ax1.plot(tau, beta_class, marker="s", linestyle="--", label="class-channel formula")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"training time $\tau$")
    ax1.set_ylabel(r"inverse WTA temperature $\beta_{spec}$")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(tau, align, marker="^", color="black", alpha=0.65, label="alignment with m")
    ax2.set_ylabel("top-mode alignment with m")
    ax2.set_ylim(0.0, 1.05)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "B_beta_spec_finite_tau.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(tau, eps, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"training time $\tau$")
    ax.set_ylabel(r"unlearned class-mode filter $\epsilon_c(\tau)$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "B_unlearned_filter_eps_c.png"), dpi=180)
    plt.close(fig)


# -----------------------------
# Experiment C: empirical REM glass
# -----------------------------

def exact_same_class_costs(points: np.ndarray, t: float, anchor_index: int) -> np.ndarray:
    Delta = 1.0 - math.exp(-2.0 * t)
    r_t = math.exp(-2.0 * t) / Delta
    a0 = points[anchor_index]
    D = points - a0[None, :]
    costs = r_t * np.sum(D * D, axis=1)
    mask = np.ones(points.shape[0], dtype=bool)
    mask[anchor_index] = False
    return costs[mask]


def gibbs_entropy_and_free_energy(costs: np.ndarray, beta: float, d: int) -> Tuple[float, float, float]:
    # Returns entropy density H/d, free energy Phi=-logZ/(beta d), mean energy density.
    if beta <= 0:
        H = math.log(len(costs))
        return H / d, float(np.mean(costs)) / d, float(np.mean(costs)) / d
    logits = -beta * costs
    lz = float(logsumexp(logits))
    w = np.exp(logits - lz)
    meanE = float(np.sum(w * costs))
    H = lz + beta * meanE
    Phi = -lz / (beta * d)
    return H / d, Phi, meanE / d


def run_glass_experiment(
    cfg: ProblemConfig,
    rf: RandomFeatureMap,
    outdir: str,
    t_glass: float,
    n_emp: int,
    tau_grid: np.ndarray,
    beta_scaled_grid: np.ndarray,
    n_anchors: int,
    seed: int,
) -> List[Dict[str, float]]:
    rng = make_rng(seed + 7000)
    m = make_class_direction(cfg.d, cfg.mu)
    X0, y = sample_gmm(n_emp, m, cfg.sigma0, rng)
    same = X0[y > 0]
    if same.shape[0] < 10:
        raise RuntimeError("not enough same-class points")
    n_same = same.shape[0] - 1
    alpha = math.log(n_same) / cfg.d
    Delta = 1.0 - math.exp(-2.0 * t_glass)
    r_t = math.exp(-2.0 * t_glass) / Delta
    psi_p = cfg.p / cfg.d

    # RF spectral distribution for the effective training filter R(t,tau).
    # We use noised empirical samples to estimate a finite-p RF covariance spectrum.
    Xt, _, _, _, _ = diffuse(X0, t_glass, rng)
    Phi = rf.phi(Xt)
    U = (Phi.T @ Phi) / Phi.shape[0]
    eigvals = np.linalg.eigvalsh(0.5 * (U + U.T))
    eigvals = np.maximum(eigvals, 1e-12)

    y_alpha = solve_y_alpha(alpha)
    v_inf = 8.0 * (r_t ** 2) * (cfg.sigma0 ** 4)
    beta_inf_gauss = math.sqrt(2.0 * alpha / v_inf)
    beta_inf_exact = (1.0 / y_alpha - 1.0) / (4.0 * r_t * cfg.sigma0 * cfg.sigma0)

    anchor_indices = rng.choice(np.arange(same.shape[0]), size=min(n_anchors, same.shape[0]), replace=False)
    raw_costs_list = [exact_same_class_costs(same, t_glass, int(idx)) for idx in anchor_indices]

    rows: List[Dict[str, float]] = []
    for tau in tau_grid:
        filt = 1.0 - np.exp(-2.0 * Delta * eigvals * tau / psi_p)
        R_eff = float(np.mean(filt * filt))
        rho = math.sqrt(max(R_eff, 0.0))
        if rho < 1e-12:
            beta_g_gauss = math.inf
            beta_g_exact = math.inf
        else:
            beta_g_gauss = math.sqrt(2.0 * alpha / (v_inf * R_eff))
            beta_g_exact = (1.0 / y_alpha - 1.0) / (4.0 * rho * r_t * cfg.sigma0 * cfg.sigma0)

        for z in beta_scaled_grid:
            if not math.isfinite(beta_g_gauss):
                continue
            beta = z * beta_g_gauss
            H_vals, Phi_vals, E_vals = [], [], []
            for raw_costs in raw_costs_list:
                # Training at finite tau scales the learned empirical energy amplitude by rho.
                costs_tau = rho * raw_costs
                H, Phi_val, Emean = gibbs_entropy_and_free_energy(costs_tau, beta, cfg.d)
                H_vals.append(H)
                Phi_vals.append(Phi_val)
                E_vals.append(Emean)
            rows.append({
                "t": t_glass,
                "tau": float(tau),
                "R_eff": R_eff,
                "rho": rho,
                "alpha": alpha,
                "beta_scaled": float(z),
                "beta": beta,
                "beta_glass_gauss": beta_g_gauss,
                "beta_glass_exact": beta_g_exact,
                "beta_inf_gauss": beta_inf_gauss,
                "beta_inf_exact": beta_inf_exact,
                "entropy_density": float(np.mean(H_vals)),
                "free_energy": float(np.mean(Phi_vals)),
                "mean_energy_density": float(np.mean(E_vals)),
                "theory_entropy_density_liquid": max(alpha * (1.0 - z * z), 0.0),
            })

    write_csv(os.path.join(outdir, "C_empirical_REM_glass.csv"), rows)

    plt = _get_pyplot()
    if plt is None:
        return rows

    # Plot beta_glass vs tau.
    unique_tau = np.array(sorted(set(r["tau"] for r in rows)))
    beta_gs = []
    rhos = []
    for tau in unique_tau:
        subset = [r for r in rows if r["tau"] == tau]
        beta_gs.append(subset[0]["beta_glass_gauss"])
        rhos.append(subset[0]["rho"])
    fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
    ax1.plot(unique_tau, beta_gs, marker="o", label=r"$\beta_{glass}(t,\tau)$ REM")
    ax1.axhline(beta_inf_gauss, linestyle="--", color="black", alpha=0.5, label=r"$\tau=\infty$ gaussian REM")
    ax1.axhline(beta_inf_exact, linestyle=":", color="black", alpha=0.5, label=r"$\tau=\infty$ exact GMM")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"training time $\tau$")
    ax1.set_ylabel(r"inverse WTA glass temperature")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(unique_tau, rhos, marker="s", color="tab:orange", alpha=0.75, label=r"$\rho=\sqrt{R}$")
    ax2.set_ylabel(r"learned empirical amplitude $\rho(t,\tau)$")
    ax2.set_ylim(0.0, 1.05)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "C_beta_glass_vs_tau.png"), dpi=180)
    plt.close(fig)

    # Entropy collapse plot.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    # Plot only a few tau values to keep the figure readable.
    tau_plot = unique_tau[:: max(1, len(unique_tau) // 4)]
    if unique_tau[-1] not in tau_plot:
        tau_plot = np.append(tau_plot, unique_tau[-1])
    for tau in tau_plot:
        subset = [r for r in rows if r["tau"] == tau]
        subset = sorted(subset, key=lambda r: r["beta_scaled"])
        ax.plot([r["beta_scaled"] for r in subset], [r["entropy_density"] for r in subset], marker="o", label=fr"$\tau={tau:.2g}$")
    zgrid = np.linspace(min(beta_scaled_grid), max(beta_scaled_grid), 200)
    ax.plot(zgrid, [max(alpha * (1 - z*z), 0.0) for z in zgrid], color="black", linestyle="--", label="REM entropy theory")
    ax.axvline(1.0, color="black", linestyle=":", alpha=0.6)
    ax.set_xlabel(r"scaled inverse temperature $\beta/\beta_{glass}$")
    ax.set_ylabel(r"Gibbs entropy density $H_\beta/d$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "C_entropy_collapse_beta_scaled.png"), dpi=180)
    plt.close(fig)

    return rows


# -----------------------------
# Main CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Numerical checks for RF-MCL beta_spec and beta_glass calculations.")
    p.add_argument("--outdir", type=str, default="/mnt/data/rf_mcl_results", help="Output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--p", type=int, default=128)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--sigma0", type=float, default=0.4)
    p.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "linear", "erf"])
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-eval", type=int, default=2500)
    p.add_argument("--power-iters", type=int, default=25)
    p.add_argument("--quick", action="store_true", help="Use smaller dimensions/samples for a fast smoke test")
    p.add_argument("--t-min", type=float, default=1.1)
    p.add_argument("--t-max", type=float, default=3.0)
    p.add_argument("--num-t", type=int, default=7)
    p.add_argument("--t-finite", type=float, default=None, help="Diffusion time for finite-tau experiment B; default ~0.5 log d")
    p.add_argument("--n-emp", type=int, default=4000)
    p.add_argument("--t-glass", type=float, default=0.85)
    p.add_argument("--n-anchors", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.d = 32
        args.p = 80
        args.n_train = 2500
        args.n_eval = 1200
        args.power_iters = 15
        args.num_t = 5
        args.n_emp = 1500
        args.n_anchors = 8

    ensure_dir(args.outdir)
    cfg = ProblemConfig(d=args.d, p=args.p, mu=args.mu, sigma0=args.sigma0, seed=args.seed, activation=args.activation)
    rng = make_rng(args.seed)
    rf = RandomFeatureMap(cfg.d, cfg.p, cfg.activation, rng)

    # ---------------- A: tau=infinity vs t ----------------
    t_grid = np.linspace(args.t_min, args.t_max, args.num_t)
    rows_A: List[Dict[str, float]] = []
    print("\n[A] Population GMM, tau=infinity: computing beta_spec(t)")
    for t in t_grid:
        row = run_population_experiment_for_t(
            cfg, rf, float(t), tau=None, n_train=args.n_train, n_eval=args.n_eval,
            power_iters=args.power_iters, seed=args.seed,
        )
        rows_A.append(row)
        print(
            f"  t={t:.3f} kappa={row['kappa']:.3g} "
            f"beta_power={row['beta_power']:.3g} beta_class={row['beta_class']:.3g} "
            f"align_m={row['output_alignment_m']:.3f}"
        )
    write_csv(os.path.join(args.outdir, "A_population_tau_inf.csv"), rows_A)
    plot_A(rows_A, args.outdir)

    # ---------------- B: finite tau at one t ----------------
    t_finite = args.t_finite
    if t_finite is None:
        # Around the Biroli speciation window kappa~1 when sigma0 is not too large.
        t_finite = 0.5 * math.log(cfg.d * cfg.mu * cfg.mu)
    tau_grid_B = np.logspace(-3, 2.0, 9)
    rows_B: List[Dict[str, float]] = []
    print(f"\n[B] Population GMM, finite tau at t={t_finite:.3f}")
    for tau in tau_grid_B:
        row = run_population_experiment_for_t(
            cfg, rf, float(t_finite), tau=float(tau), n_train=args.n_train, n_eval=args.n_eval,
            power_iters=args.power_iters, seed=args.seed + 10,
        )
        rows_B.append(row)
        print(
            f"  tau={tau:.3g} eps_c={row['eps_c']:.3g} "
            f"beta_power={row['beta_power']:.3g} beta_class={row['beta_class']:.3g} "
            f"align_m={row['output_alignment_m']:.3f}"
        )
    write_csv(os.path.join(args.outdir, "B_population_finite_tau.csv"), rows_B)
    plot_B(rows_B, args.outdir)

    # ---------------- C: empirical REM glass ----------------
    tau_grid_C = np.logspace(-3, 2.0, 10)
    beta_scaled = np.linspace(0.2, 2.0, 15)
    print(f"\n[C] Empirical GMM REM/glass at t={args.t_glass:.3f}, n_emp={args.n_emp}")
    rows_C = run_glass_experiment(
        cfg, rf, args.outdir, t_glass=args.t_glass, n_emp=args.n_emp,
        tau_grid=tau_grid_C, beta_scaled_grid=beta_scaled,
        n_anchors=args.n_anchors, seed=args.seed,
    )
    # Print one row per tau.
    seen = set()
    for r in rows_C:
        tau = r["tau"]
        if tau in seen:
            continue
        seen.add(tau)
        print(
            f"  tau={tau:.3g} rho={r['rho']:.3g} "
            f"beta_glass={r['beta_glass_gauss']:.3g} alpha={r['alpha']:.3g}"
        )

    # Save run metadata.
    meta_path = os.path.join(args.outdir, "README_run.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("RF-MCL phase diagram numerical experiment\n")
        f.write("=========================================\n\n")
        f.write(f"d={cfg.d}, p={cfg.p}, psi_p={cfg.p/cfg.d:.4g}, mu={cfg.mu}, sigma0={cfg.sigma0}\n")
        f.write(f"activation={cfg.activation}, seed={cfg.seed}\n")
        f.write(f"n_train={args.n_train}, n_eval={args.n_eval}, power_iters={args.power_iters}\n")
        f.write("\nOutputs:\n")
        f.write("  A_population_tau_inf.csv / A_beta_spec_population_tau_inf.png\n")
        f.write("  B_population_finite_tau.csv / B_beta_spec_finite_tau.png\n")
        f.write("  C_empirical_REM_glass.csv / C_beta_glass_vs_tau.png / C_entropy_collapse_beta_scaled.png\n")
        f.write("\nInterpretation:\n")
        f.write("  A checks beta_spec=1/(2 lambda_max) and top-mode alignment with the GMM class direction.\n")
        f.write("  B checks the finite-training spectral filter in beta_spec(t,tau).\n")
        f.write("  C checks beta_glass(t,tau)=sqrt(2 alpha / v(t,tau)) with v(t,tau)=v_inf R(t,tau).\n")
    print(f"\nDone. Results written to: {args.outdir}")


if __name__ == "__main__":
    main()
