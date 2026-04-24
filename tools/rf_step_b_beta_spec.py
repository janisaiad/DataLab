from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rf_theory_core import (
    TheoryConfig,
    build_grid,
    compute_beta_spec_surface,
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
    parser.add_argument("--n_mc", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
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
        n_mc=args.n_mc,
        random_seed=args.seed,
    )
    t_grid, tau_grid = build_grid(cfg)
    res = compute_beta_spec_surface(cfg, t_grid, tau_grid)

    np.savez_compressed(out_dir / "step_b_beta_spec_arrays.npz", **res)
    save_csv_surface(
        out_dir / "step_b_beta_spec_surface.csv",
        t_grid,
        tau_grid,
        res["beta_spec"],
        "beta_spec",
    )
    save_csv_surface(
        out_dir / "step_b_lambda_rf_surface.csv",
        t_grid,
        tau_grid,
        res["lambda_rf"],
        "lambda_rf",
    )
    save_csv_surface(
        out_dir / "step_b_lambda_parallel_surface.csv",
        t_grid,
        tau_grid,
        res["lambda_parallel"],
        "lambda_parallel",
    )
    save_csv_surface(
        out_dir / "step_b_lambda_perp_surface.csv",
        t_grid,
        tau_grid,
        res["lambda_perp"],
        "lambda_perp",
    )

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    im1 = ax1.imshow(
        res["beta_spec"],
        aspect="auto",
        origin="lower",
        extent=[tau_grid[0], tau_grid[-1], t_grid[0], t_grid[-1]],
        cmap="viridis",
    )
    ax1.set_xlabel("tau")
    ax1.set_ylabel("t")
    ax1.set_title("beta_spec(t, tau)")
    fig1.colorbar(im1, ax=ax1, label="beta_spec")
    fig1.tight_layout()
    fig1.savefig(out_dir / "step_b_beta_spec_heatmap.png", dpi=170)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    im2 = ax2.imshow(
        res["lambda_rf"],
        aspect="auto",
        origin="lower",
        extent=[tau_grid[0], tau_grid[-1], t_grid[0], t_grid[-1]],
        cmap="magma",
    )
    ax2.set_xlabel("tau")
    ax2.set_ylabel("t")
    ax2.set_title("lambda_RF(t, tau)")
    fig2.colorbar(im2, ax=ax2, label="lambda_RF")
    fig2.tight_layout()
    fig2.savefig(out_dir / "step_b_lambda_rf_heatmap.png", dpi=170)
    plt.close(fig2)

    tau_idx = [0, len(tau_grid) // 4, len(tau_grid) // 2, -1]
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    for idx in tau_idx:
        ax3.plot(t_grid, res["beta_spec"][:, idx], label=f"tau={tau_grid[idx]:.0f}")
    ax3.set_xlabel("t")
    ax3.set_ylabel("beta_spec")
    ax3.set_title("beta_spec(t) for selected tau")
    ax3.legend()
    ax3.grid(alpha=0.25)
    fig3.tight_layout()
    fig3.savefig(out_dir / "step_b_beta_spec_slices.png", dpi=170)
    plt.close(fig3)

    summary = {
        "beta_spec_min": float(np.min(res["beta_spec"])),
        "beta_spec_max": float(np.max(res["beta_spec"])),
        "lambda_rf_min": float(np.min(res["lambda_rf"])),
        "lambda_rf_max": float(np.max(res["lambda_rf"])),
        "class_alignment_fraction": float(
            np.mean(res["lambda_parallel"] > res["lambda_perp"])
        ),
    }
    save_summary_json(out_dir / "step_b_beta_spec_summary.json", cfg, summary)


if __name__ == "__main__":
    main()
