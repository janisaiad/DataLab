from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rf_theory_core import (
    TheoryConfig,
    build_grid,
    compute_beta_glass_surface,
    save_summary_json,
)


def save_csv_surface(
    path: Path, t_grid: np.ndarray, tau_grid: np.ndarray, z: np.ndarray, name: str
) -> None:
    tau_mesh, t_mesh = np.meshgrid(tau_grid, t_grid)
    stacked = np.column_stack([t_mesh.ravel(), tau_mesh.ravel(), z.ravel()])
    header = f"t,tau,{name}"
    np.savetxt(path, stacked, delimiter=",", header=header, comments="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/theory_rf")
    parser.add_argument("--sigma0", type=float, default=0.7)
    parser.add_argument("--mu", type=float, default=1.2)
    parser.add_argument("--psi_p", type=float, default=64.0)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--n_t", type=int, default=80)
    parser.add_argument("--n_tau", type=int, default=90)
    parser.add_argument("--tau_max", type=float, default=12000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_emp_ratio", type=float, default=0.08)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TheoryConfig(
        sigma0=args.sigma0,
        mu=args.mu,
        psi_p=args.psi_p,
        alpha=args.alpha,
        n_t=args.n_t,
        n_tau=args.n_tau,
        tau_max=args.tau_max,
        random_seed=args.seed,
    )
    t_grid, tau_grid = build_grid(cfg)
    res = compute_beta_glass_surface(
        cfg, t_grid, tau_grid, lambda_emp_ratio=args.lambda_emp_ratio
    )

    np.savez_compressed(out_dir / "step_c_beta_glass_arrays.npz", **res)
    save_csv_surface(
        out_dir / "step_c_beta_glass_approx_surface.csv",
        t_grid,
        tau_grid,
        res["beta_glass_approx"],
        "beta_glass_approx",
    )
    save_csv_surface(
        out_dir / "step_c_beta_glass_exact_surface.csv",
        t_grid,
        tau_grid,
        res["beta_glass_exact"],
        "beta_glass_exact",
    )
    save_csv_surface(
        out_dir / "step_c_R_tau_surface.csv", t_grid, tau_grid, res["R_tau"], "R_tau"
    )
    save_csv_surface(
        out_dir / "step_c_v_tau_surface.csv", t_grid, tau_grid, res["v_tau"], "v_tau"
    )

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    im1 = ax1.imshow(
        res["beta_glass_approx"],
        aspect="auto",
        origin="lower",
        extent=[tau_grid[0], tau_grid[-1], t_grid[0], t_grid[-1]],
        cmap="cividis",
    )
    ax1.set_xlabel("tau")
    ax1.set_ylabel("t")
    ax1.set_title("beta_glass approx(t, tau)")
    fig1.colorbar(im1, ax=ax1, label="beta_glass approx")
    fig1.tight_layout()
    fig1.savefig(out_dir / "step_c_beta_glass_approx_heatmap.png", dpi=170)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    im2 = ax2.imshow(
        res["beta_glass_exact"],
        aspect="auto",
        origin="lower",
        extent=[tau_grid[0], tau_grid[-1], t_grid[0], t_grid[-1]],
        cmap="plasma",
    )
    ax2.set_xlabel("tau")
    ax2.set_ylabel("t")
    ax2.set_title("beta_glass exact(t, tau)")
    fig2.colorbar(im2, ax=ax2, label="beta_glass exact")
    fig2.tight_layout()
    fig2.savefig(out_dir / "step_c_beta_glass_exact_heatmap.png", dpi=170)
    plt.close(fig2)

    t_idx = [0, len(t_grid) // 3, 2 * len(t_grid) // 3, -1]
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    for idx in t_idx:
        ax3.plot(tau_grid, res["R_tau"][idx], label=f"t={t_grid[idx]:.2f}")
    ax3.set_xlabel("tau")
    ax3.set_ylabel("R(t, tau)")
    ax3.set_title("Spectral activation R(t, tau)")
    ax3.legend()
    ax3.grid(alpha=0.25)
    fig3.tight_layout()
    fig3.savefig(out_dir / "step_c_R_tau_slices.png", dpi=170)
    plt.close(fig3)

    summary = {
        "beta_glass_approx_min": float(np.min(res["beta_glass_approx"])),
        "beta_glass_approx_max": float(np.max(res["beta_glass_approx"])),
        "beta_glass_exact_min": float(np.min(res["beta_glass_exact"])),
        "beta_glass_exact_max": float(np.max(res["beta_glass_exact"])),
        "tau_mem_min": float(np.min(res["tau_mem"])),
        "tau_mem_max": float(np.max(res["tau_mem"])),
    }
    save_summary_json(out_dir / "step_c_beta_glass_summary.json", cfg, summary)


if __name__ == "__main__":
    main()
