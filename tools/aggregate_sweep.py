"""Aggregate K-sweep and batch-sweep results into LaTeX tables and figures."""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


# Matches <variant>[_K<int>][_B<int>] (subdir name under outputs/).
TAG_RE = re.compile(
    r"^(?P<variant>hard_wta|annealed_wta|relaxed_wta|resilient_mcl)"
    r"(?:_K(?P<K>\d+))?(?:_B(?P<B>\d+))?$"
)


def discover_runs(root):
    """Return list of dicts: {variant, K, B, metrics_path, log_path}.

    Walks ``<root>/outputs/<tag>/metrics.json``; ``<tag>`` is e.g.
    ``annealed_wta_K6`` or ``hard_wta``. K defaults to 4 and B to 512 when
    omitted from the tag.
    """
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "outputs", "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        m = TAG_RE.match(name)
        if not m:
            continue
        variant = m.group("variant")
        K = int(m.group("K")) if m.group("K") else 4
        B = int(m.group("B")) if m.group("B") else 512

        metrics_path = os.path.join(d, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        log_path = os.path.join(root, "checkpoints", name, f"mcl_K{K}_log.json")
        if not os.path.exists(log_path):
            log_path = None

        runs.append({
            "variant": variant, "K": K, "B": B,
            "tag": name, "metrics_path": metrics_path, "log_path": log_path,
        })
    return runs


def load_metrics(path):
    """Load metrics JSON from disk.

    Args:
        path: Path to a metrics JSON file.

    Returns:
        Parsed dictionary of metrics.
    """
    with open(path) as f:
        return json.load(f)


def load_usage(log_path, n_avg=10):
    """Average expert-usage over the last n_avg epochs."""
    if log_path is None or not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        log = json.load(f)
    usage = log.get("expert_usage", [])
    if not usage:
        return None
    last = usage[-n_avg:]
    K = len(last[0])
    avg = [sum(row[k] for row in last) / len(last) for k in range(K)]
    final_loss = log["loss"][-n_avg:]
    return {
        "usage": avg,
        "K": K,
        "starved": sum(1 for u in avg if u < 0.05),
        "final_loss": sum(final_loss) / len(final_loss),
    }


STRATEGIES = ["baseline_euler", "baseline_heun", "single_expert",
              "random_expert", "best_expert", "mixture_score", "gated"]
STRAT_DISPLAY = {
    "baseline_euler": "Baseline (Euler)",
    "baseline_heun":  "Baseline (Heun)",
    "single_expert":  "Single Expert",
    "random_expert":  "Random per-step",
    "best_expert":    "Best per-step",
    "mixture_score":  "Mixture Score",
    "gated":          "Learned Gating",
}


def fmt(x, prec=2):
    """Format metric values for table output."""
    if x is None:
        return "---"
    return f"{x:.{prec}f}"


def ksweep_table(runs, variant, batch_size=512):
    """LaTeX table: rows = strategies, columns = K values."""
    rows = [r for r in runs
            if r["variant"] == variant and r["B"] == batch_size]
    rows.sort(key=lambda r: r["K"])
    if not rows:
        return f"% No runs found for variant={variant}, B={batch_size}\n"

    Ks = [r["K"] for r in rows]
    metrics_by_K = {r["K"]: load_metrics(r["metrics_path"]) for r in rows}

    label = variant.replace("_", "\\_")
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{$K$-sweep on \\texttt{{{label}}} (batch=$ {batch_size}$, "
               f"200 epochs, $N=2{{,}}048$). FID $\\downarrow$. Baseline rows are shared "
               f"(single-model diffusion, no MCL).}}")
    out.append(f"\\label{{tab:ksweep_{variant}}}")
    out.append("\\small")
    col_spec = "l" + "c" * len(Ks)
    out.append(f"\\begin{{tabular}}{{{col_spec}}}")
    out.append("\\toprule")
    header = "\\textbf{Strategy} & " + " & ".join(f"$K={k}$" for k in Ks) + " \\\\"
    out.append(header)
    out.append("\\midrule")
    for strat in STRATEGIES:
        cells = [STRAT_DISPLAY[strat]]
        vals = [metrics_by_K[k].get(strat, {}).get("fid") for k in Ks]
        # Bold the best (min) FID per row, ignoring missing cells.
        valid = [v for v in vals if v is not None]
        best = min(valid) if valid else None
        for v in vals:
            if v is None:
                cells.append("---")
            elif best is not None and abs(v - best) < 1e-6:
                cells.append(f"\\textbf{{{fmt(v)}}}")
            else:
                cells.append(fmt(v))
        out.append(" & ".join(cells) + " \\\\")
        if strat == "baseline_heun":
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def batchsweep_table(runs, variant="annealed_wta", K=4):
    """LaTeX table: rows = strategies, columns = batch sizes."""
    rows = [r for r in runs
            if r["variant"] == variant and r["K"] == K]
    rows.sort(key=lambda r: r["B"])
    if not rows:
        return f"% No runs found for variant={variant}, K={K}\n"

    Bs = [r["B"] for r in rows]
    metrics_by_B = {r["B"]: load_metrics(r["metrics_path"]) for r in rows}

    label = variant.replace("_", "\\_")
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{Batch-size sweep on \\texttt{{{label}}} ($K={K}$, 200 epochs, "
               f"$N=2{{,}}048$). The baseline is shared across all batch sizes "
               f"(trained once at $B=512$); only MCL re-trains per cell. "
               f"FID $\\downarrow$.}}")
    out.append(f"\\label{{tab:batchsweep_{variant}}}")
    out.append("\\small")
    col_spec = "l" + "c" * len(Bs)
    out.append(f"\\begin{{tabular}}{{{col_spec}}}")
    out.append("\\toprule")
    header = "\\textbf{Strategy} & " + " & ".join(f"$B={b}$" for b in Bs) + " \\\\"
    out.append(header)
    out.append("\\midrule")
    for strat in STRATEGIES:
        cells = [STRAT_DISPLAY[strat]]
        vals = [metrics_by_B[b].get(strat, {}).get("fid") for b in Bs]
        valid = [v for v in vals if v is not None]
        best = min(valid) if valid else None
        for v in vals:
            if v is None:
                cells.append("---")
            elif best is not None and abs(v - best) < 1e-6:
                cells.append(f"\\textbf{{{fmt(v)}}}")
            else:
                cells.append(fmt(v))
        out.append(" & ".join(cells) + " \\\\")
        if strat == "baseline_heun":
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def collapse_table(runs):
    """LaTeX table: per (variant, K), show #starved experts and entropy of usage."""
    import math

    rows = []
    for r in runs:
        if r["B"] != 512:
            continue
        u = load_usage(r["log_path"])
        if u is None:
            continue
        # Effective K = exp(entropy) of the usage distribution.
        usage = [max(x, 1e-8) for x in u["usage"]]
        H = -sum(x * math.log(x) for x in usage)
        keff = math.exp(H)
        rows.append({
            "variant": r["variant"], "K": r["K"], "starved": u["starved"],
            "K_eff": keff, "loss": u["final_loss"],
            "max_use": max(u["usage"]), "min_use": min(u["usage"]),
        })
    rows.sort(key=lambda r: (r["variant"], r["K"]))
    if not rows:
        return "% No usage logs found.\n"

    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\caption{Expert usage analysis at training end (averaged over the last "
               "10 epochs). $K_{\\text{eff}} = \\exp(H(\\text{usage}))$ is the "
               "effective number of active experts (information-theoretic). "
               "$K_{\\text{eff}} \\ll K$ signals collapse.}")
    out.append("\\label{tab:expert_collapse}")
    out.append("\\small")
    out.append("\\begin{tabular}{llcccc}")
    out.append("\\toprule")
    out.append("\\textbf{Variant} & $K$ & \\textbf{Starved} ($<5\\%$) & "
               "$K_{\\text{eff}}$ & \\textbf{Max usage} & \\textbf{Min usage} \\\\")
    out.append("\\midrule")
    cur_var = None
    for r in rows:
        var = r["variant"].replace("_", "\\_")
        if r["variant"] != cur_var:
            if cur_var is not None:
                out.append("\\midrule")
            cur_var = r["variant"]
            var_cell = f"\\texttt{{{var}}}"
        else:
            var_cell = ""
        out.append(f"{var_cell} & {r['K']} & {r['starved']}/{r['K']} & "
                   f"{r['K_eff']:.2f} & {r['max_use']:.3f} & {r['min_use']:.3f} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def make_figures(runs, out_dir):
    """Create sweep figures and save them to disk.

    Args:
        runs: List of discovered run metadata dictionaries.
        out_dir: Output directory for generated figures.

    Returns:
        List of saved figure paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available - skipping figures.")
        return []

    saved = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, color, marker in [("annealed_wta", "#1f77b4", "o"),
                                    ("resilient_mcl", "#d62728", "s")]:
        rows = sorted([r for r in runs
                       if r["variant"] == variant and r["B"] == 512],
                      key=lambda r: r["K"])
        if not rows:
            continue
        Ks = [r["K"] for r in rows]
        for strat, ls, alpha in [("gated", "-", 1.0), ("single_expert", "--", 0.6)]:
            fids = [load_metrics(r["metrics_path"]).get(strat, {}).get("fid") for r in rows]
            fids = [f for f in fids if f is not None]
            if len(fids) != len(Ks):
                continue
            ax.plot(Ks, fids, color=color, marker=marker, linestyle=ls, alpha=alpha,
                    label=f"{variant.replace('_', ' ')} - {strat.replace('_', ' ')}")
    any_run = next(iter(runs), None)
    if any_run is not None:
        m = load_metrics(any_run["metrics_path"])
        bl = m.get("baseline_heun", {}).get("fid")
        if bl is not None:
            ax.axhline(bl, color="gray", linestyle=":", alpha=0.7,
                       label=f"Baseline Heun ({bl:.1f})")
    ax.set_xlabel("Number of experts $K$")
    ax.set_ylabel("FID $\\downarrow$")
    ax.set_yscale("log")
    ax.set_title("FID vs $K$ - annealed vs resilient WTA")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(out_dir, "ksweep_fig.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    rows = sorted([r for r in runs
                   if r["variant"] == "annealed_wta" and r["K"] == 4],
                  key=lambda r: r["B"])
    if len(rows) >= 2:
        Bs = [r["B"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for strat, color, marker in [("gated", "#2ca02c", "o"),
                                      ("single_expert", "#9467bd", "s"),
                                      ("random_expert", "#ff7f0e", "^")]:
            fids = [load_metrics(r["metrics_path"]).get(strat, {}).get("fid") for r in rows]
            fids = [f for f in fids if f is not None]
            if len(fids) != len(Bs):
                continue
            ax.plot(Bs, fids, marker=marker, color=color,
                    label=strat.replace("_", " "))
        ax.set_xscale("log", base=2)
        ax.set_xticks(Bs)
        ax.set_xticklabels(Bs)
        ax.set_xlabel("Batch size $B$")
        ax.set_ylabel("FID $\\downarrow$")
        ax.set_title("FID vs batch size - annealed WTA, $K=4$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(out_dir, "batchsweep_fig.pdf")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    return saved


def text_summary(runs):
    """Build a readable text summary of discovered runs.

    Args:
        runs: List of discovered run metadata dictionaries.

    Returns:
        Multiline string summary.
    """
    lines = []
    lines.append(f"{len(runs)} runs discovered.")
    lines.append("")
    by_variant = defaultdict(list)
    for r in runs:
        by_variant[r["variant"]].append(r)
    for v, rs in sorted(by_variant.items()):
        rs = sorted(rs, key=lambda r: (r["K"], r["B"]))
        lines.append(f"=== {v} ({len(rs)} runs) ===")
        lines.append(f"  {'K':>3} {'B':>5}  {'gated FID':>10}  "
                     f"{'single FID':>11}  {'baseline_heun':>13}")
        for r in rs:
            m = load_metrics(r["metrics_path"])
            g = m.get("gated", {}).get("fid")
            s = m.get("single_expert", {}).get("fid")
            b = m.get("baseline_heun", {}).get("fid")
            lines.append(f"  {r['K']:>3} {r['B']:>5}  {fmt(g):>10}  "
                         f"{fmt(s):>11}  {fmt(b):>13}")
        lines.append("")
    return "\n".join(lines)


def main():
    """Discover runs and generate summary artefacts."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".", help="Project root (default: cwd)")
    p.add_argument("--out-dir", default="tools/sweep_artefacts",
                   help="Output directory for tables/figures")
    args = p.parse_args()

    runs = discover_runs(args.root)
    if not runs:
        print(f"No runs found under {args.root}/outputs/*/metrics.json")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    by_variant_K = defaultdict(set)
    for r in runs:
        if r["B"] == 512:
            by_variant_K[r["variant"]].add(r["K"])

    ksweep_combined = []
    for v, Ks in sorted(by_variant_K.items()):
        if len(Ks) >= 2:
            ksweep_combined.append(ksweep_table(runs, v))
    if ksweep_combined:
        with open(os.path.join(args.out_dir, "ksweep_table.tex"), "w") as f:
            f.write("\n".join(ksweep_combined))

    Bs = sorted({r["B"] for r in runs
                 if r["variant"] == "annealed_wta" and r["K"] == 4})
    if len(Bs) >= 2:
        with open(os.path.join(args.out_dir, "batchsweep_table.tex"), "w") as f:
            f.write(batchsweep_table(runs))

    with open(os.path.join(args.out_dir, "expert_collapse_table.tex"), "w") as f:
        f.write(collapse_table(runs))

    saved = make_figures(runs, args.out_dir)

    summary = text_summary(runs)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    print(summary)
    print(f"\nArtefacts written to {args.out_dir}/:")
    for fname in sorted(os.listdir(args.out_dir)):
        print(f"  {fname}")


if __name__ == "__main__":
    main()
