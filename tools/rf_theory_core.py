from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class TheoryConfig:
    sigma0: float = 0.7
    mu: float = 1.2
    psi_p: float = 64.0
    alpha: float = 0.08
    t_min: float = 0.05
    t_max: float = 2.2
    n_t: int = 80
    tau_min: float = 0.0
    tau_max: float = 12000.0
    n_tau: int = 90
    n_mc: int = 60000
    random_seed: int = 42


def _safe_div(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (y + eps)


def _lambert_w0_newton(x: np.ndarray, n_iter: int = 50) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    w = np.where(x < 1.0, x, np.log1p(x))
    for _ in range(n_iter):
        ew = np.exp(w)
        f = w * ew - x
        denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0)
        w = w - _safe_div(f, denom)
    return w


def y_alpha(alpha: float) -> float:
    c = 2.0 * alpha + 1.0
    x = -np.exp(-c)
    return float(-_lambert_w0_newton(np.array([x]))[0])


def build_grid(cfg: TheoryConfig) -> tuple[np.ndarray, np.ndarray]:
    t_grid = np.linspace(cfg.t_min, cfg.t_max, cfg.n_t)
    tau_grid = np.linspace(cfg.tau_min, cfg.tau_max, cfg.n_tau)
    return t_grid, tau_grid


def base_scalars(t: np.ndarray, cfg: TheoryConfig) -> dict[str, np.ndarray]:
    delta_t = 1.0 - np.exp(-2.0 * t)
    a = np.exp(-t)
    gamma_t = delta_t + (a**2) * (cfg.sigma0**2)
    kappa_t = (a**2) * (cfg.mu**2) / np.maximum(gamma_t, 1e-12)
    return {
        "delta_t": delta_t,
        "a": a,
        "gamma_t": gamma_t,
        "kappa_t": kappa_t,
    }


def monte_carlo_chi(
    cfg: TheoryConfig,
    t: np.ndarray,
    tau: np.ndarray,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    y = rng.choice(np.array([-1.0, 1.0]), size=cfg.n_mc)
    z = rng.normal(size=cfg.n_mc)
    z_act = rng.normal(size=cfg.n_mc)
    out = {}
    delta_t = 1.0 - np.exp(-2.0 * t)
    a = np.exp(-t)
    gamma_t = delta_t + (a**2) * (cfg.sigma0**2)
    kappa_t = (a**2) * (cfg.mu**2) / np.maximum(gamma_t, 1e-12)
    gamma_sqrt = np.sqrt(np.maximum(gamma_t, 1e-12))
    q0 = np.empty_like(t)
    a1 = np.empty_like(t)
    chi0 = np.empty_like(t)
    chi2 = np.empty_like(t)
    for i in range(t.size):
        u = gamma_sqrt[i] * z_act
        sig = np.tanh(u)
        q0[i] = np.mean(sig**2)
        a1[i] = np.mean(1.0 - np.tanh(u) ** 2)
        s = np.sqrt(np.maximum(kappa_t[i], 1e-12)) * y + z
        S = np.sqrt(np.maximum(kappa_t[i], 1e-12)) * s
        sech2 = 1.0 / np.cosh(S) ** 2
        chi0[i] = np.mean(sech2)
        chi2[i] = np.mean((s**2) * sech2)
    out["q0"] = q0
    out["a1"] = a1
    out["chi0"] = chi0
    out["chi2"] = chi2
    out["y_samples"] = y
    out["z_samples"] = z
    return out


def compute_beta_spec_surface(
    cfg: TheoryConfig,
    t_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    s = base_scalars(t_grid, cfg)
    mc = monte_carlo_chi(cfg, t_grid, tau_grid)
    delta_t = s["delta_t"]
    a = s["a"]
    gamma_t = s["gamma_t"]
    kappa_t = s["kappa_t"]
    q0 = mc["q0"]
    a1 = mc["a1"]
    chi0 = mc["chi0"]
    chi2 = mc["chi2"]
    lambda_c_u = q0 + cfg.psi_p * gamma_t * (a1**2) * (1.0 + kappa_t)
    tau_col = tau_grid[None, :]
    t_col = t_grid[:, None]
    gamma_col = gamma_t[:, None]
    delta_col = delta_t[:, None]
    kappa_col = kappa_t[:, None]
    q0_col = q0[:, None]
    a1_col = a1[:, None]
    chi0_col = chi0[:, None]
    chi2_col = chi2[:, None]
    lambda_c_col = lambda_c_u[:, None]
    eps_c = np.exp(-2.0 * delta_col * tau_col * lambda_c_col / cfg.psi_p)
    chi0_tau = chi0_col + 0.35 * (eps_c**2)
    chi2_tau = chi2_col + 0.45 * (eps_c**2) * (1.0 + kappa_col)
    num = q0_col * chi0_tau + cfg.psi_p * gamma_col * (a1_col**2) * chi2_tau
    den = q0_col + cfg.psi_p * gamma_col * (a1_col**2) * (1.0 + kappa_col)
    lambda_spike = _safe_div(num, den)
    lambda_bulk = chi0_tau
    lambda_rf = np.maximum(lambda_bulk, lambda_spike)
    lambda_parallel = (a[:, None] ** 2) * (cfg.sigma0**2) / np.maximum(gamma_col, 1e-12)
    lambda_parallel = (
        lambda_parallel + _safe_div(delta_col, gamma_col) * kappa_col * lambda_rf
    )
    lambda_perp = (a[:, None] ** 2) * (cfg.sigma0**2) / np.maximum(gamma_col, 1e-12)
    lambda_perp = np.broadcast_to(lambda_perp, lambda_parallel.shape)
    beta_spec = 0.5 * _safe_div(1.0, lambda_parallel)
    return {
        "t_grid": t_grid,
        "tau_grid": tau_grid,
        "delta_t": delta_t,
        "a": a,
        "gamma_t": gamma_t,
        "kappa_t": kappa_t,
        "q0": q0,
        "a1": a1,
        "chi0": chi0,
        "chi2": chi2,
        "lambda_c_u": lambda_c_u,
        "eps_c": eps_c,
        "chi0_tau": chi0_tau,
        "chi2_tau": chi2_tau,
        "lambda_bulk": lambda_bulk,
        "lambda_spike": lambda_spike,
        "lambda_rf": lambda_rf,
        "lambda_parallel": lambda_parallel,
        "lambda_perp": lambda_perp,
        "beta_spec": beta_spec,
    }


def compute_beta_glass_surface(
    cfg: TheoryConfig,
    t_grid: np.ndarray,
    tau_grid: np.ndarray,
    lambda_emp_ratio: float = 0.08,
) -> dict[str, np.ndarray]:
    s = base_scalars(t_grid, cfg)
    delta_t = s["delta_t"]
    rt = np.exp(-2.0 * t_grid) / np.maximum(delta_t, 1e-12)
    y_a = y_alpha(cfg.alpha)
    tau_col = tau_grid[None, :]
    delta_col = delta_t[:, None]
    lambda_emp = lambda_emp_ratio * cfg.psi_p
    g_tau = 1.0 - np.exp(-2.0 * delta_col * lambda_emp * tau_col / cfg.psi_p)
    r_tau = g_tau**2
    v_inf = 8.0 * (rt[:, None] ** 2) * (cfg.sigma0**4)
    v_tau = v_inf * r_tau
    beta_glass_approx = _safe_div(2.0 * cfg.alpha, np.sqrt(np.maximum(v_tau, 1e-24)))
    beta_glass_inf_approx = _safe_div(
        2.0 * cfg.alpha, np.sqrt(np.maximum(v_inf, 1e-24))
    )
    beta_glass_exact = _safe_div(
        (1.0 / y_a) - 1.0,
        4.0 * np.maximum(r_tau * rt[:, None] * (cfg.sigma0**2), 1e-24),
    )
    beta_glass_inf_exact = _safe_div(
        (1.0 / y_a) - 1.0, 4.0 * np.maximum(rt[:, None] * (cfg.sigma0**2), 1e-24)
    )
    tau_mem = _safe_div(cfg.psi_p, 2.0 * delta_t * lambda_emp)
    return {
        "t_grid": t_grid,
        "tau_grid": tau_grid,
        "delta_t": delta_t,
        "rt": rt,
        "lambda_emp": np.array([lambda_emp]),
        "R_tau": r_tau,
        "v_inf": v_inf,
        "v_tau": v_tau,
        "beta_glass_approx": beta_glass_approx,
        "beta_glass_inf_approx": beta_glass_inf_approx,
        "beta_glass_exact": beta_glass_exact,
        "beta_glass_inf_exact": beta_glass_inf_exact,
        "tau_mem": tau_mem,
        "y_alpha": np.array([y_a]),
    }


def save_summary_json(path: Path, cfg: TheoryConfig, payload: dict[str, float]) -> None:
    obj = {
        "config": asdict(cfg),
        "summary": payload,
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
