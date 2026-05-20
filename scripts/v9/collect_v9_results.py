#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect a V9 suite root into CSV/Markdown/plots."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect(root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        metrics = load_json(d / "metrics.json") or {}
        config = load_json(d / "config.json") or {}
        beta_mean = beta_min = beta_max = float("nan")
        if (d / "beta_validation_by_t.csv").exists():
            b = pd.read_csv(d / "beta_validation_by_t.csv")
            if "beta_emp_over_beta_z_theory" in b:
                beta_mean = float(b["beta_emp_over_beta_z_theory"].mean())
                beta_min = float(b["beta_emp_over_beta_z_theory"].min())
                beta_max = float(b["beta_emp_over_beta_z_theory"].max())
        probe_entropy = probe_margin = probe_betagap = probe_excess = float("nan")
        if (d / "mcl_speciation_probe_by_step_t.csv").exists():
            p = pd.read_csv(d / "mcl_speciation_probe_by_step_t.csv")
            last = p[p["step"] == p["step"].max()]
            for col, target in [
                ("teacher_entropy_sample_norm", "entropy"),
                ("risk_margin_mean", "margin"),
                ("beta_gap_mean", "betagap"),
                ("route_excess_vs_sample_oracle", "excess"),
            ]:
                if col in last:
                    val = float(last[col].mean())
                    if target == "entropy": probe_entropy = val
                    if target == "margin": probe_margin = val
                    if target == "betagap": probe_betagap = val
                    if target == "excess": probe_excess = val
        for strategy, m in metrics.items():
            if not isinstance(m, dict):
                continue
            fid = m.get("fid", m.get("fid_diag_fallback", float("nan")))
            rows.append({
                "run": d.name,
                "dataset": config.get("dataset", ""),
                "classes": config.get("classes", ""),
                "K": config.get("K", float("nan")),
                "beta_rho": config.get("beta_rho", float("nan")),
                "strategy": strategy,
                "fid_or_diag": fid,
                "precision": m.get("precision", float("nan")),
                "recall": m.get("recall", float("nan")),
                "fallback_fraction": m.get("fallback_fraction", float("nan")),
                "commit_fraction": m.get("commit_fraction", float("nan")),
                "beta_ratio_mean": beta_mean,
                "beta_ratio_min": beta_min,
                "beta_ratio_max": beta_max,
                "final_probe_teacher_entropy": probe_entropy,
                "final_probe_risk_margin": probe_margin,
                "final_probe_beta_gap": probe_betagap,
                "final_probe_route_excess": probe_excess,
            })
    return pd.DataFrame(rows)


def plots(root: Path, df: pd.DataFrame):
    if df.empty:
        return
    # Best strategy per run.
    best = df.dropna(subset=["fid_or_diag"]).sort_values("fid_or_diag").groupby("run").head(1)
    plt.figure(figsize=(11, 5.5))
    x = np.arange(len(best))
    plt.bar(x, best["fid_or_diag"].astype(float))
    plt.xticks(x, [f"{r}\n{strat}" for r, strat in zip(best["run"], best["strategy"])], rotation=35, ha="right")
    plt.ylabel("best FID / diagnostic")
    plt.title("V9 suite: best strategy per run")
    plt.tight_layout()
    plt.savefig(root / "plot_v9_suite_best_fid.png", dpi=180)
    plt.close()

    # Strategy bars for each run.
    for run, g in df.groupby("run"):
        g = g.dropna(subset=["fid_or_diag"])
        if g.empty:
            continue
        plt.figure(figsize=(9, 5))
        x = np.arange(len(g))
        plt.bar(x, g["fid_or_diag"].astype(float))
        plt.xticks(x, g["strategy"], rotation=35, ha="right")
        plt.ylabel("FID / diagnostic")
        plt.title(f"V9 strategies: {run}")
        plt.tight_layout()
        plt.savefig(root / f"plot_v9_suite_strategies_{run}.png", dpi=180)
        plt.close()

    # Theory summary scatter.
    one = df.groupby("run").head(1)
    plt.figure(figsize=(8, 5))
    plt.scatter(one["beta_ratio_mean"], one["final_probe_risk_margin"])
    for _, r in one.iterrows():
        plt.annotate(str(r["run"]), (r["beta_ratio_mean"], r["final_probe_risk_margin"]), fontsize=7)
    plt.axvline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("mean beta empirical/theory")
    plt.ylabel("final risk margin")
    plt.title("V9 suite: temperature validation vs routing margin")
    plt.tight_layout()
    plt.savefig(root / "plot_v9_suite_beta_vs_margin.png", dpi=180)
    plt.close()


def markdown(root: Path, df: pd.DataFrame):
    lines = ["# V9 suite aggregate report", ""]
    if df.empty:
        lines.append("No completed runs found.")
    else:
        lines.append("## Best strategy per run")
        lines.append("")
        best = df.dropna(subset=["fid_or_diag"]).sort_values("fid_or_diag").groupby("run").head(1)
        lines.append("| run | best strategy | fid/diag | beta ratio mean | final risk margin | final beta gap |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in best.iterrows():
            lines.append(f"| {r['run']} | {r['strategy']} | {r['fid_or_diag']:.6g} | {r['beta_ratio_mean']:.4g} | {r['final_probe_risk_margin']:.4g} | {r['final_probe_beta_gap']:.4g} |")
        lines.append("")
        lines.append("## All strategy rows")
        lines.append("")
        lines.append(df.to_markdown(index=False))
    lines.append("")
    lines.append("## Plots")
    for p in sorted(root.glob("plot_v9_suite*.png")):
        lines.append(f"- `{p.name}`")
    (root / "V9_SUITE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    df = collect(args.root)
    df.to_csv(args.root / "V9_SUITE_SUMMARY.csv", index=False)
    plots(args.root, df)
    markdown(args.root, df)
    print(args.root / "V9_SUITE_SUMMARY.csv")
    print(args.root / "V9_SUITE_REPORT.md")


if __name__ == "__main__":
    main()
