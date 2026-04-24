from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


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
    parser.add_argument("--lambda_emp_ratio", type=float, default=0.08)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--out_dir",
        str(out_dir),
        "--sigma0",
        str(args.sigma0),
        "--mu",
        str(args.mu),
        "--psi_p",
        str(args.psi_p),
        "--alpha",
        str(args.alpha),
        "--n_t",
        str(args.n_t),
        "--n_tau",
        str(args.n_tau),
        "--tau_max",
        str(args.tau_max),
        "--seed",
        str(args.seed),
    ]

    run_cmd(
        [
            "python",
            "tools/rf_step_b_beta_spec.py",
            *common,
            "--n_mc",
            str(args.n_mc),
        ],
        root,
    )
    run_cmd(
        [
            "python",
            "tools/rf_step_c_beta_glass.py",
            *common,
            "--lambda_emp_ratio",
            str(args.lambda_emp_ratio),
        ],
        root,
    )

    spec = np.load(out_dir / "step_b_beta_spec_arrays.npz")
    glass = np.load(out_dir / "step_c_beta_glass_arrays.npz")
    t_grid = spec["t_grid"]
    tau_grid = spec["tau_grid"]
    beta_spec = spec["beta_spec"]
    beta_glass = glass["beta_glass_exact"]
    gap = beta_glass - beta_spec

    np.savez_compressed(
        out_dir / "step_d_combined_phase_window.npz",
        t_grid=t_grid,
        tau_grid=tau_grid,
        beta_spec=beta_spec,
        beta_glass=beta_glass,
        beta_gap=gap,
    )

    tau_mesh, t_mesh = np.meshgrid(tau_grid, t_grid)
    stacked = np.column_stack([t_mesh.ravel(), tau_mesh.ravel(), gap.ravel()])
    np.savetxt(
        out_dir / "step_d_beta_gap_surface.csv",
        stacked,
        delimiter=",",
        header="t,tau,beta_glass_minus_beta_spec",
        comments="",
    )

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    im1 = ax1.imshow(
        gap,
        aspect="auto",
        origin="lower",
        extent=[tau_grid[0], tau_grid[-1], t_grid[0], t_grid[-1]],
        cmap="RdYlGn",
    )
    ax1.set_xlabel("tau")
    ax1.set_ylabel("t")
    ax1.set_title("Phase window: beta_glass - beta_spec")
    fig1.colorbar(im1, ax=ax1, label="gap")
    fig1.tight_layout()
    fig1.savefig(out_dir / "step_d_phase_window_heatmap.png", dpi=170)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    idx = [0, len(t_grid) // 3, 2 * len(t_grid) // 3, -1]
    for i in idx:
        ax2.plot(
            tau_grid, beta_spec[i], label=f"beta_spec t={t_grid[i]:.2f}", linestyle="-"
        )
        ax2.plot(
            tau_grid,
            beta_glass[i],
            label=f"beta_glass t={t_grid[i]:.2f}",
            linestyle="--",
        )
    ax2.set_xlabel("tau")
    ax2.set_ylabel("beta")
    ax2.set_title("beta_spec vs beta_glass slices")
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(out_dir / "step_d_beta_slices.png", dpi=170)
    plt.close(fig2)


if __name__ == "__main__":
    main()
