#!/usr/bin/env python3
"""
Corrected numerical checks for RF-MCL specialization and empirical glass transitions.

This is a second, corrected experimental script.  It separates three channels that were
mixed in the first version:

  A. population GMM, tau = infinity:
     - free Hessian top eigenmode;
     - class-constrained Hessian mode h(x)=m g(x);
     - transverse/non-class Hessian mode h(x) \\perp m;
     - top-mode alignment with the class direction m.

  B. population GMM, finite training tau:
     - same three channels, but with the RF finite-time residual
       q_tau = xi - V^T U^{-1}(I-exp(-2 Delta_t U tau / psi_p)) phi(x_t).
     - this distinguishes generic expert splitting from true class speciation.

  C. empirical GMM, finite n:
     - exact atom-to-atom WTA cost
       E_{mu nu}(t) = exp(-2t)/Delta_t * ||a_mu-a_nu||^2;
     - exact chi-square REM entropy curve for same-class costs;
     - dynamic RF empirical-mode amplitude rho(tau), which makes beta_glass(t,tau)
       decrease during training;
     - fixed-beta plot showing that glass appears late in training.

The script writes CSV files and PNG figures if matplotlib is available.

Example quick run:
    python rf_mcl_phase_diagram_corrected.py --quick --outdir ./rf_mcl_corrected

More serious run:
    python rf_mcl_phase_diagram_corrected.py --d 64 --p 512 --n-train 8000 --n-eval 3000 \
        --n-emp 2048 --power-iters 35 --outdir ./rf_mcl_corrected

Notes on normalization:
    beta is the inverse WTA expert temperature multiplying the unnormalized squared
    DSM/WTA loss. If your implementation uses loss/d, rescale beta by d.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


def stable_beta_from_lambda(lam: float) -> float:
    if lam <= 0 or not np.isfinite(lam):
        return float("inf")
    return 1.0 / (2.0 * lam)


# -----------------------------------------------------------------------------
# GMM + RF primitives
# -----------------------------------------------------------------------------


@dataclass
class Params:
    d: int = 48
    p: int = 384
    mu: float = 1.0
    sigma0: float = 0.5
    seed: int = 0
    ridge: float = 1e-6

    @property
    def psi_p(self) -> float:
        return self.p / self.d


def make_class_direction(d: int, mu: float) -> np.ndarray:
    # ||m||^2 = d mu^2
    return mu * np.ones(d, dtype=np.float64)


def sample_clean_gmm(rng: np.random.Generator, n: int, m: np.ndarray, sigma0: float) -> Tuple[np.ndarray, np.ndarray]:
    d = m.shape[0]
    y = rng.choice(np.array([-1.0, 1.0]), size=n)
    z = rng.normal(size=(n, d))
    x0 = y[:, None] * m[None, :] + sigma0 * z
    return x0, y


def sample_noisy_gmm(
    rng: np.random.Generator, n: int, m: np.ndarray, sigma0: float, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = m.shape[0]
    x0, y = sample_clean_gmm(rng, n, m, sigma0)
    delta = 1.0 - math.exp(-2.0 * t)
    xi = rng.normal(size=(n, d))
    xt = math.exp(-t) * x0 + math.sqrt(delta) * xi
    return xt, xi, x0, y


def rf_features(W: np.ndarray, X: np.ndarray, activation: str = "tanh") -> np.ndarray:
    H = X @ W.T / math.sqrt(W.shape[1])
    if activation == "tanh":
        return np.tanh(H)
    if activation == "relu":
        return np.maximum(H, 0.0)
    if activation == "erf":
        # cheap smooth odd activation without scipy
        return np.tanh(1.2 * H)
    raise ValueError(f"unknown activation: {activation}")


def fit_rf_population(
    rng: np.random.Generator,
    par: Params,
    W: np.ndarray,
    t: float,
    n_train: int,
    activation: str,
) -> Dict[str, np.ndarray]:
    xt, xi, _, _ = sample_noisy_gmm(rng, n_train, make_class_direction(par.d, par.mu), par.sigma0, t)
    Phi = rf_features(W, xt, activation=activation)
    U = (Phi.T @ Phi) / n_train
    U = 0.5 * (U + U.T)
    # Add tiny ridge for numerical stability. This corresponds to an infinitesimal RF ridge.
    U_reg = U + par.ridge * np.eye(par.p)
    V = (Phi.T @ xi) / n_train  # p x d
    evals, Q = np.linalg.eigh(U_reg)
    evals = np.maximum(evals, par.ridge)
    return {"U": U_reg, "V": V, "evals": evals, "Q": Q}


def predict_noise_from_rf(Phi: np.ndarray, fit: Dict[str, np.ndarray], t: float, tau: float, psi_p: float) -> np.ndarray:
    """Return f_tau(x)=V^T U^{-1}(I-exp(-gamma U)) phi(x), row-wise.

    Phi: n x p
    V: p x d
    """
    evals = fit["evals"]
    Q = fit["Q"]
    V = fit["V"]
    delta = 1.0 - math.exp(-2.0 * t)
    if math.isinf(tau):
        filt_over_lam = 1.0 / evals
    else:
        gamma = 2.0 * delta * tau / psi_p
        filt_over_lam = (1.0 - np.exp(-gamma * evals)) / evals
    # Phi @ Q @ diag(filt/lam) @ Q.T @ V
    Z = Phi @ Q
    Z *= filt_over_lam[None, :]
    return Z @ Q.T @ V


def whiten_features(Phi: np.ndarray, fit: Dict[str, np.ndarray]) -> np.ndarray:
    evals = fit["evals"]
    Q = fit["Q"]
    Z = Phi @ Q
    Z *= (1.0 / np.sqrt(evals))[None, :]
    return Z


# -----------------------------------------------------------------------------
# Hessian channel estimators
# -----------------------------------------------------------------------------


def power_free_hessian(
    Qres: np.ndarray,
    Psi: np.ndarray,
    rng: np.random.Generator,
    n_iter: int = 30,
    mhat: Optional[np.ndarray] = None,
    project_output: Optional[str] = None,
) -> Tuple[float, float]:
    """Power iteration on T(C)=E[q q^T C psi psi^T].

    In whitened coordinates denominator is ||C||_F^2.

    project_output:
        None        : free output direction
        "class"    : project C to span(mhat)
        "transverse": project C to mhat^perp
    Returns (largest eigenvalue estimate, output alignment with mhat).
    """
    n, d = Qres.shape
    p = Psi.shape[1]
    C = rng.normal(size=(d, p)) / math.sqrt(d * p)

    def project(C_: np.ndarray) -> np.ndarray:
        if project_output is None:
            return C_
        if mhat is None:
            raise ValueError("mhat required for projected power method")
        row = mhat @ C_
        if project_output == "class":
            return np.outer(mhat, row)
        if project_output == "transverse":
            return C_ - np.outer(mhat, row)
        raise ValueError(project_output)

    C = project(C)
    C /= np.linalg.norm(C) + 1e-300

    for _ in range(n_iter):
        CPsi = Qres @ C  # n x p? no, Qres(n,d) @ C(d,p)=n x p
        s = np.sum(CPsi * Psi, axis=1)
        T = Qres.T @ (Psi * s[:, None]) / n
        T = project(T)
        norm = np.linalg.norm(T)
        if norm < 1e-300 or not np.isfinite(norm):
            break
        C = T / norm

    CPsi = Qres @ C
    s = np.sum(CPsi * Psi, axis=1)
    lam = float(np.mean(s ** 2) / (np.linalg.norm(C) ** 2 + 1e-300))
    if mhat is None:
        align = float("nan")
    else:
        align = float(np.linalg.norm(mhat @ C) ** 2 / (np.linalg.norm(C) ** 2 + 1e-300))
    return lam, align


def class_channel_hessian(Qres: np.ndarray, Psi: np.ndarray, mhat: np.ndarray) -> Tuple[float, np.ndarray]:
    """Exact class-constrained channel h(x)=mhat * (v^T psi).

    The generalized/whitened operator is M=E[(mhat^T q)^2 psi psi^T].
    """
    q_m = Qres @ mhat
    M = Psi.T @ (Psi * (q_m ** 2)[:, None]) / Psi.shape[0]
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    return float(vals[-1]), vecs[:, -1]


def fixed_direction_channel(Qres: np.ndarray, Psi: np.ndarray, u: np.ndarray) -> float:
    q_u = Qres @ u
    M = Psi.T @ (Psi * (q_u ** 2)[:, None]) / Psi.shape[0]
    M = 0.5 * (M + M.T)
    vals = np.linalg.eigvalsh(M)
    return float(vals[-1])


def random_transverse_direction(rng: np.random.Generator, d: int, mhat: np.ndarray) -> np.ndarray:
    u = rng.normal(size=d)
    u -= mhat * float(mhat @ u)
    u /= np.linalg.norm(u) + 1e-300
    return u


def compute_hessian_channels(
    rng: np.random.Generator,
    par: Params,
    W: np.ndarray,
    fit: Dict[str, np.ndarray],
    t: float,
    tau: float,
    n_eval: int,
    activation: str,
    power_iters: int,
) -> Dict[str, float]:
    m = make_class_direction(par.d, par.mu)
    mhat = m / np.linalg.norm(m)
    xt, xi, _, _ = sample_noisy_gmm(rng, n_eval, m, par.sigma0, t)
    Phi = rf_features(W, xt, activation=activation)
    f_tau = predict_noise_from_rf(Phi, fit, t, tau, par.psi_p)
    Qres = xi - f_tau
    Psi = whiten_features(Phi, fit)

    lam_free, align = power_free_hessian(Qres, Psi, rng, power_iters, mhat=mhat, project_output=None)
    lam_trans, align_trans = power_free_hessian(Qres, Psi, rng, power_iters, mhat=mhat, project_output="transverse")
    lam_class, _ = class_channel_hessian(Qres, Psi, mhat)
    u_perp = random_transverse_direction(rng, par.d, mhat)
    lam_one_perp = fixed_direction_channel(Qres, Psi, u_perp)

    a = math.exp(-t)
    delta = 1.0 - math.exp(-2.0 * t)
    gamma_t = delta + (a ** 2) * (par.sigma0 ** 2)
    kappa = (a ** 2) * (np.linalg.norm(m) ** 2) / gamma_t

    return {
        "t": float(t),
        "tau": float(tau) if not math.isinf(tau) else float("inf"),
        "kappa": float(kappa),
        "lambda_free": lam_free,
        "lambda_class": lam_class,
        "lambda_transverse": lam_trans,
        "lambda_one_perp": lam_one_perp,
        "beta_free": stable_beta_from_lambda(lam_free),
        "beta_class": stable_beta_from_lambda(lam_class),
        "beta_transverse": stable_beta_from_lambda(lam_trans),
        "beta_one_perp": stable_beta_from_lambda(lam_one_perp),
        "free_alignment_m": align,
        "trans_alignment_m": align_trans,
    }


# -----------------------------------------------------------------------------
# Glass / exact GMM cost theory
# -----------------------------------------------------------------------------


def y_alpha_exact(alpha: float) -> float:
    """Solve y - 1 - log y = 2 alpha on y in (0,1)."""
    lo, hi = 1e-14, 1.0 - 1e-14
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = mid - 1.0 - math.log(mid)
        if val > 2.0 * alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rate_chi_square_same_class(e: float, r_eff: float, sigma0: float) -> float:
    if r_eff <= 0 or e <= 0:
        return float("inf")
    y = e / (2.0 * r_eff * sigma0 ** 2)
    return 0.5 * (y - 1.0 - math.log(y))


def beta_glass_exact(alpha: float, r_eff: float, sigma0: float) -> float:
    if r_eff <= 0:
        return float("inf")
    y = y_alpha_exact(alpha)
    return (1.0 / y - 1.0) / (4.0 * r_eff * sigma0 ** 2)


def beta_glass_gaussian(alpha: float, r_eff: float, sigma0: float) -> float:
    # v/d = 8 r_eff^2 sigma0^4
    v = 8.0 * (r_eff ** 2) * sigma0 ** 4
    if v <= 0:
        return float("inf")
    return math.sqrt(2.0 * alpha / v)


def exact_entropy_density(beta: float, alpha: float, r_eff: float, sigma0: float) -> float:
    if r_eff <= 0 or beta <= 0:
        return alpha
    e_beta = 2.0 * r_eff * sigma0 ** 2 / (1.0 + 4.0 * beta * r_eff * sigma0 ** 2)
    I = rate_chi_square_same_class(e_beta, r_eff, sigma0)
    return max(alpha - I, 0.0)


def gibbs_entropy_density(E: np.ndarray, beta: float, d: int) -> float:
    logw = -beta * E
    logZ = logsumexp(logw)
    p = np.exp(logw - logZ)
    mean_E = float(np.sum(p * E))
    H = logZ + beta * mean_E
    return H / d


def rf_empirical_amplitude(tau: float, t: float, psi_n: float) -> float:
    """Simple RF empirical-bulk learning amplitude.

    In the overparameterized RF theory, the slow empirical/memorization bulk has
    lambda_emp ~ psi_p/psi_n, hence the filter exponent is
        -2 Delta_t lambda_emp tau / psi_p = -2 Delta_t tau / psi_n.
    """
    delta = 1.0 - math.exp(-2.0 * t)
    return 1.0 - math.exp(-2.0 * delta * tau / psi_n)


def empirical_same_class_costs(
    rng: np.random.Generator,
    d: int,
    n_emp: int,
    mu: float,
    sigma0: float,
    t: float,
) -> Tuple[np.ndarray, int]:
    m = make_class_direction(d, mu)
    # Force balanced classes for cleaner same-class costs.
    n_half = n_emp // 2
    z_plus = rng.normal(size=(n_half, d))
    a_plus = m[None, :] + sigma0 * z_plus
    anchor = a_plus[0]
    others = a_plus[1:]
    r_t = math.exp(-2.0 * t) / (1.0 - math.exp(-2.0 * t))
    E = r_t * np.sum((others - anchor[None, :]) ** 2, axis=1)
    return E, others.shape[0]


# -----------------------------------------------------------------------------
# Experiment runners
# -----------------------------------------------------------------------------


def run_A_tau_infty(
    outdir: str,
    par: Params,
    activation: str,
    n_train: int,
    n_eval: int,
    power_iters: int,
    t_grid: np.ndarray,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(par.seed + 101)
    W = rng.normal(size=(par.p, par.d))
    rows: List[Dict[str, float]] = []
    for t in t_grid:
        fit = fit_rf_population(rng, par, W, float(t), n_train, activation)
        row = compute_hessian_channels(rng, par, W, fit, float(t), math.inf, n_eval, activation, power_iters)
        rows.append(row)
        print(
            f"[A] t={t:.3g} kappa={row['kappa']:.3g} "
            f"beta_free={row['beta_free']:.3g} beta_class={row['beta_class']:.3g} "
            f"beta_trans={row['beta_transverse']:.3g} align={row['free_alignment_m']:.3g}"
        )
    save_csv(os.path.join(outdir, "A_tau_infty_channels.csv"), rows)

    if HAS_MPL:
        ts = np.array([r["t"] for r in rows])
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(ts, [r["beta_free"] for r in rows], "o-", label=r"free $\beta_{split}$")
        ax1.plot(ts, [r["beta_class"] for r in rows], "s--", label=r"class $\beta_{class}$")
        ax1.plot(ts, [r["beta_transverse"] for r in rows], "^--", label=r"transverse $\beta_{trans}$")
        ax1.set_yscale("log")
        ax1.set_xlabel("diffusion time t")
        ax1.set_ylabel("inverse WTA temperature threshold")
        ax1.grid(True, alpha=0.35)
        ax2 = ax1.twinx()
        ax2.plot(ts, [r["free_alignment_m"] for r in rows], "k.-", label="free-mode alignment with m")
        ax2.plot(ts, [r["kappa"] for r in rows], color="gray", linestyle=":", label=r"class SNR $\kappa_t$")
        ax2.set_yscale("log")
        ax2.set_ylabel(r"alignment / $\kappa_t$")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "A_tau_infty_corrected_channels.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([r["kappa"] for r in rows], [r["lambda_free"] for r in rows], "o-", label="free")
        ax.plot([r["kappa"] for r in rows], [r["lambda_class"] for r in rows], "s--", label="class")
        ax.plot([r["kappa"] for r in rows], [r["lambda_transverse"] for r in rows], "^--", label="transverse")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(r"class SNR $\kappa_t=e^{-2t}\|m\|^2/\Gamma_t$")
        ax.set_ylabel(r"Hessian eigenvalue $\lambda$")
        ax.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "A_lambda_vs_kappa_channels.png"), dpi=180)
        plt.close(fig)

    return rows


def run_B_finite_tau(
    outdir: str,
    par: Params,
    activation: str,
    n_train: int,
    n_eval: int,
    power_iters: int,
    t_fixed: float,
    tau_grid: np.ndarray,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(par.seed + 202)
    W = rng.normal(size=(par.p, par.d))
    fit = fit_rf_population(rng, par, W, t_fixed, n_train, activation)
    rows: List[Dict[str, float]] = []
    for tau in tau_grid:
        row = compute_hessian_channels(rng, par, W, fit, t_fixed, float(tau), n_eval, activation, power_iters)
        rows.append(row)
        dominant = max(
            [(row["lambda_class"], "class"), (row["lambda_transverse"], "trans"), (row["lambda_free"], "free")],
            key=lambda x: x[0],
        )[1]
        print(
            f"[B] tau={tau:.3g} beta_free={row['beta_free']:.3g} "
            f"beta_class={row['beta_class']:.3g} beta_trans={row['beta_transverse']:.3g} "
            f"align={row['free_alignment_m']:.3g} dominant={dominant}"
        )
    save_csv(os.path.join(outdir, "B_finite_tau_channels.csv"), rows)

    if HAS_MPL:
        taus = np.array([r["tau"] for r in rows])
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(taus, [r["beta_free"] for r in rows], "o-", label=r"free $\beta_{split}$")
        ax1.plot(taus, [r["beta_class"] for r in rows], "s--", label=r"class $\beta_{class}$")
        ax1.plot(taus, [r["beta_transverse"] for r in rows], "^--", label=r"transverse $\beta_{trans}$")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(r"training time $\tau$")
        ax1.set_ylabel("inverse WTA temperature threshold")
        ax1.grid(True, alpha=0.35)
        ax2 = ax1.twinx()
        ax2.plot(taus, [r["free_alignment_m"] for r in rows], "k.-", label="free-mode alignment with m")
        ax2.set_ylabel("alignment with m")
        ax2.set_ylim(-0.02, 1.02)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "B_finite_tau_corrected_channels.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(taus, [r["lambda_free"] for r in rows], "o-", label="free")
        ax.plot(taus, [r["lambda_class"] for r in rows], "s--", label="class")
        ax.plot(taus, [r["lambda_transverse"] for r in rows], "^--", label="transverse")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"training time $\tau$")
        ax.set_ylabel(r"Hessian eigenvalue $\lambda$")
        ax.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "B_lambda_finite_tau_channels.png"), dpi=180)
        plt.close(fig)

    return rows


def run_C_empirical_glass(
    outdir: str,
    par: Params,
    n_emp: int,
    t_glass: float,
    tau_grid: np.ndarray,
    beta_fixed_factor: float,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(par.seed + 303)
    E_base, n_same = empirical_same_class_costs(rng, par.d, n_emp, par.mu, par.sigma0, t_glass)
    alpha = math.log(n_same) / par.d
    delta = 1.0 - math.exp(-2.0 * t_glass)
    r_t = math.exp(-2.0 * t_glass) / delta
    psi_n = n_emp / par.d
    y_a = y_alpha_exact(alpha)
    beta_inf_exact = beta_glass_exact(alpha, r_t, par.sigma0)
    beta_inf_gauss = beta_glass_gaussian(alpha, r_t, par.sigma0)
    beta_fixed = beta_fixed_factor * beta_inf_exact

    rows: List[Dict[str, float]] = []
    for tau in tau_grid:
        rho = rf_empirical_amplitude(float(tau), t_glass, psi_n)
        r_eff = rho * r_t
        beta_g_exact = beta_glass_exact(alpha, r_eff, par.sigma0)
        beta_g_gauss = beta_glass_gaussian(alpha, r_eff, par.sigma0)
        E_eff = rho * E_base
        H_emp = gibbs_entropy_density(E_eff, beta_fixed, par.d)
        H_theory = exact_entropy_density(beta_fixed, alpha, r_eff, par.sigma0)
        rows.append({
            "tau": float(tau),
            "rho": float(rho),
            "alpha_same": float(alpha),
            "r_t": float(r_t),
            "beta_fixed": float(beta_fixed),
            "beta_glass_exact": float(beta_g_exact),
            "beta_glass_gaussian": float(beta_g_gauss),
            "entropy_empirical_fixed_beta": float(H_emp),
            "entropy_exact_fixed_beta": float(H_theory),
            "y_alpha": float(y_a),
        })
        print(
            f"[C] tau={tau:.3g} rho={rho:.3g} beta_g_exact={beta_g_exact:.3g} "
            f"H_emp/d={H_emp:.3g} H_exact/d={H_theory:.3g}"
        )
    save_csv(os.path.join(outdir, "C_glass_fixed_beta_vs_tau.csv"), rows)

    # Scaled-beta entropy curves for several tau values; this validates exact chi-square REM curve.
    scaled_rows: List[Dict[str, float]] = []
    z_grid = np.linspace(0.1, 2.0, 30)
    # choose a few representative tau values with non-negligible rho
    idxs = np.linspace(0, len(tau_grid) - 1, min(6, len(tau_grid))).astype(int)
    for idx in idxs:
        tau = float(tau_grid[idx])
        rho = rf_empirical_amplitude(tau, t_glass, psi_n)
        if rho <= 1e-12:
            continue
        r_eff = rho * r_t
        beta_g = beta_glass_exact(alpha, r_eff, par.sigma0)
        E_eff = rho * E_base
        for z in z_grid:
            beta = z * beta_g
            H_emp = gibbs_entropy_density(E_eff, beta, par.d)
            H_ex = exact_entropy_density(beta, alpha, r_eff, par.sigma0)
            scaled_rows.append({
                "tau": tau,
                "rho": rho,
                "scaled_beta": float(z),
                "beta": float(beta),
                "entropy_empirical": float(H_emp),
                "entropy_exact": float(H_ex),
            })
    save_csv(os.path.join(outdir, "C_glass_scaled_beta_entropy.csv"), scaled_rows)

    if HAS_MPL:
        taus = np.array([r["tau"] for r in rows])
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(taus, [r["entropy_empirical_fixed_beta"] for r in rows], "o-", label="empirical Gibbs entropy")
        ax1.plot(taus, [r["entropy_exact_fixed_beta"] for r in rows], "k--", label="exact chi-square theory")
        ax1.set_xscale("log")
        ax1.set_xlabel(r"training time $\tau$")
        ax1.set_ylabel(r"entropy density $H_\beta/d$ at fixed $\beta$")
        ax1.grid(True, alpha=0.35)
        ax2 = ax1.twinx()
        ax2.plot(taus, [r["beta_glass_exact"] for r in rows], color="tab:red", marker="s", label=r"$\beta_{glass}^{exact}(\tau)$")
        ax2.axhline(beta_fixed, color="tab:red", linestyle=":", label=r"fixed $\beta$")
        ax2.set_yscale("log")
        ax2.set_ylabel("inverse WTA temperature")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "C_fixed_beta_entropy_late_glass.png"), dpi=180)
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(taus, [r["beta_glass_exact"] for r in rows], "o-", label=r"exact $\beta_{glass}(\tau)$")
        ax1.plot(taus, [r["beta_glass_gaussian"] for r in rows], "s--", label=r"Gaussian REM approximation")
        ax1.axhline(beta_inf_exact, color="gray", linestyle=":", label=r"$\tau=\infty$ exact")
        ax1.axhline(beta_inf_gauss, color="gray", linestyle="--", label=r"$\tau=\infty$ Gaussian")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(r"training time $\tau$")
        ax1.set_ylabel(r"inverse WTA glass temperature")
        ax1.grid(True, alpha=0.35)
        ax2 = ax1.twinx()
        ax2.plot(taus, [r["rho"] for r in rows], color="tab:orange", marker="^", label=r"learned amplitude $\rho(\tau)$")
        ax2.set_ylabel(r"learned empirical amplitude $\rho(\tau)$")
        ax2.set_ylim(-0.02, 1.02)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "C_beta_glass_exact_vs_tau.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot scaled entropy curves by tau.
        by_tau: Dict[float, List[Dict[str, float]]] = {}
        for r in scaled_rows:
            by_tau.setdefault(r["tau"], []).append(r)
        for tau, rr in by_tau.items():
            rr = sorted(rr, key=lambda x: x["scaled_beta"])
            ax.plot([x["scaled_beta"] for x in rr], [x["entropy_empirical"] for x in rr], marker="o", linewidth=1, label=fr"$\tau={tau:.2g}$")
        # exact theory curve, independent of tau after exact scaling
        z_plot = np.linspace(0.1, 2.0, 200)
        r_eff = r_t  # use rho=1, scaling removes it
        beta_g = beta_glass_exact(alpha, r_eff, par.sigma0)
        H_curve = [exact_entropy_density(z * beta_g, alpha, r_eff, par.sigma0) for z in z_plot]
        ax.plot(z_plot, H_curve, "k--", linewidth=2, label="exact chi-square theory")
        ax.axvline(1.0, color="gray", linestyle=":")
        ax.set_xlabel(r"scaled inverse temperature $\beta/\beta_{glass}^{exact}(\tau)$")
        ax.set_ylabel(r"Gibbs entropy density $H_\beta/d$")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "C_scaled_beta_exact_entropy.png"), dpi=180)
        plt.close(fig)

    print(
        f"[C summary] alpha_same={alpha:.4g}, beta_inf_exact={beta_inf_exact:.4g}, "
        f"beta_inf_gauss={beta_inf_gauss:.4g}, beta_fixed={beta_fixed:.4g}"
    )
    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Corrected RF-MCL phase-diagram numerical experiments")
    ap.add_argument("--outdir", type=str, default="/mnt/data/rf_mcl_corrected_results")
    ap.add_argument("--quick", action="store_true", help="Use small dimensions for a quick smoke test")
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--p", type=int, default=384)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "erf"])
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-eval", type=int, default=2500)
    ap.add_argument("--n-emp", type=int, default=2048)
    ap.add_argument("--power-iters", type=int, default=30)
    ap.add_argument("--t-min", type=float, default=1.15)
    ap.add_argument("--t-max", type=float, default=3.00)
    ap.add_argument("--t-points", type=int, default=7)
    ap.add_argument("--t-fixed", type=float, default=2.05, help="Diffusion time for finite-tau experiment B")
    ap.add_argument("--t-glass", type=float, default=1.25, help="Diffusion time for empirical glass experiment C")
    ap.add_argument("--tau-min", type=float, default=1e-3)
    ap.add_argument("--tau-max", type=float, default=1e2)
    ap.add_argument("--tau-points", type=int, default=9)
    ap.add_argument("--beta-fixed-factor", type=float, default=1.35, help="fixed beta = factor * beta_glass_exact(tau=infty)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.d = 24
        args.p = 192
        args.n_train = 1600
        args.n_eval = 900
        args.n_emp = 512
        args.power_iters = 18
        args.t_points = 5
        args.tau_points = 7
    ensure_dir(args.outdir)

    par = Params(d=args.d, p=args.p, mu=args.mu, sigma0=args.sigma0, seed=args.seed, ridge=args.ridge)
    print("=== Corrected RF-MCL experiments ===")
    print(par)
    print(f"outdir={args.outdir}")
    print(f"matplotlib={'yes' if HAS_MPL else 'no'}")

    t_grid = np.linspace(args.t_min, args.t_max, args.t_points)
    tau_grid = np.logspace(math.log10(args.tau_min), math.log10(args.tau_max), args.tau_points)

    rows_A = run_A_tau_infty(
        args.outdir, par, args.activation, args.n_train, args.n_eval, args.power_iters, t_grid
    )
    rows_B = run_B_finite_tau(
        args.outdir, par, args.activation, args.n_train, args.n_eval, args.power_iters, args.t_fixed, tau_grid
    )
    rows_C = run_C_empirical_glass(
        args.outdir, par, args.n_emp, args.t_glass, tau_grid, args.beta_fixed_factor
    )

    # Simple text summary for quick inspection.
    summary_path = os.path.join(args.outdir, "SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("Corrected RF-MCL numerical experiments\n")
        f.write(str(par) + "\n\n")
        f.write("A: tau=infty channels\n")
        for r in rows_A:
            f.write(
                f"t={r['t']:.4g}, kappa={r['kappa']:.4g}, beta_free={r['beta_free']:.4g}, "
                f"beta_class={r['beta_class']:.4g}, beta_trans={r['beta_transverse']:.4g}, "
                f"align={r['free_alignment_m']:.4g}\n"
            )
        f.write("\nB: finite tau channels\n")
        for r in rows_B:
            f.write(
                f"tau={r['tau']:.4g}, beta_free={r['beta_free']:.4g}, beta_class={r['beta_class']:.4g}, "
                f"beta_trans={r['beta_transverse']:.4g}, align={r['free_alignment_m']:.4g}\n"
            )
        f.write("\nC: glass fixed beta\n")
        for r in rows_C:
            f.write(
                f"tau={r['tau']:.4g}, rho={r['rho']:.4g}, beta_g_exact={r['beta_glass_exact']:.4g}, "
                f"H_emp={r['entropy_empirical_fixed_beta']:.4g}, H_exact={r['entropy_exact_fixed_beta']:.4g}\n"
            )
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
