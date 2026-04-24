"""Generate the outputs/extra figures referenced by the report.

Reads per-run metrics from ``<BASE>/outputs/<tag>/metrics.json`` (e.g.
``annealed_wta_K6``, ``hard_wta``, ``annealed_wta_B128``). The "reference"
run (annealed WTA, K=4, B=512) has tag ``annealed_wta_K4``.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde, wasserstein_distance

BASE = Path("/home/ayoudeba1/IASD/DataLab")
OUTPUTS = BASE / "outputs"
OUT = OUTPUTS / "extra"
OUT.mkdir(exist_ok=True, parents=True)

REF = "annealed_wta_K4"  # K=4 annealed B=512 reference run

CONFIGS = {
    REF: "K=4 annealed B=512",
    "hard_wta": "Hard WTA K=4",
    "relaxed_wta": "Relaxed WTA K=4",
    "resilient_mcl": "Resilient K=4",
    "annealed_wta_K2": "Ann K=2",
    "annealed_wta_K3": "Ann K=3",
    "annealed_wta_K6": "Ann K=6",
    "annealed_wta_K8": "Ann K=8",
    "resilient_mcl_K2": "Res K=2",
    "resilient_mcl_K3": "Res K=3",
    "resilient_mcl_K6": "Res K=6",
    "resilient_mcl_K8": "Res K=8",
    "annealed_wta_B128": "Ann B=128",
    "annealed_wta_B256": "Ann B=256",
    "annealed_wta_B1024": "Ann B=1024",
}


def load_metrics(tag):
    """Load metrics for one output tag.

    Args:
        tag: Run tag under `outputs/`.

    Returns:
        Parsed metrics dictionary, or `None` if unavailable.
    """
    p = OUTPUTS / tag / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


STRATS = ["single_expert", "random_expert", "best_expert", "mixture_score", "gated"]
STRAT_LABELS = {
    "single_expert": "Single",
    "random_expert": "Random",
    "best_expert": "Best-norm",
    "mixture_score": "Mixture",
    "gated": "Gated",
}
STRAT_COLORS = {
    "single_expert": "#888888",
    "random_expert": "#1f77b4",
    "best_expert": "#2ca02c",
    "mixture_score": "#9467bd",
    "gated": "#d62728",
}


def plot_fid_vs_K():
    """Plot FID trends versus the number of experts."""
    Ks = [2, 3, 4, 6, 8]
    ann_dirs = [
        "annealed_wta_K2",
        "annealed_wta_K3",
        REF,
        "annealed_wta_K6",
        "annealed_wta_K8",
    ]
    res_dirs = [
        "resilient_mcl_K2",
        "resilient_mcl_K3",
        "resilient_mcl",
        "resilient_mcl_K6",
        "resilient_mcl_K8",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, dirs, title in [
        (axes[0], ann_dirs, "Annealed WTA"),
        (axes[1], res_dirs, "Resilient MCL"),
    ]:
        for s in STRATS:
            ys = []
            for d in dirs:
                m = load_metrics(d)
                ys.append(m[s]["fid"] if m else np.nan)
            ax.plot(
                Ks, ys, marker="o", lw=1.7, label=STRAT_LABELS[s], color=STRAT_COLORS[s]
            )
        base = load_metrics(REF)["baseline_heun"]["fid"]
        ax.axhline(base, ls="--", color="black", alpha=0.6, label="Baseline (Heun)")
        ax.set_yscale("log")
        ax.set_xlabel("Number of experts $K$")
        ax.set_title(title)
        ax.grid(which="both", ls=":", alpha=0.4)
        ax.set_xticks(Ks)
    axes[0].set_ylabel("FID (log)")
    axes[1].legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fid_vs_K.png", dpi=150)
    plt.close(fig)


def plot_usage_histograms():
    """Plot usage histograms from reference values in the report."""
    # Values are copied from report text and not recomputed in this script.
    usages = {
        "K=4, B=512": np.array([0.40, 0.31, 0.28, 0.01]),
        "K=6, B=512": np.array([0.26, 0.10, 0.26, 0.22, 0.16, 0.00]),
        "K=8, B=512": np.array([0.19, 0.00, 0.14, 0.25, 0.00, 0.00, 0.23, 0.19]),
        "K=4, B=128": np.array([0.49, 0.00, 0.00, 0.51]),
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.5))
    for ax, (name, u) in zip(axes.flat, usages.items()):
        K = len(u)
        ax.bar(range(K), u, color="#3b7dd8")
        ax.axhline(
            1.0 / K, ls="--", color="black", alpha=0.6, label=f"$1/K = {1 / K:.2f}$"
        )
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"E{i}" for i in range(K)])
        ax.set_ylim(0, max(0.6, u.max() * 1.15))
        ax.set_title(f"Annealed, {name}")
        ax.set_ylabel("Training usage")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "usage_histograms.png", dpi=150)
    plt.close(fig)


def plot_wasserstein_K6():
    """Plot pairwise Wasserstein distances for the K=6 expert grids."""
    K = 6
    pix = []
    rng = np.random.default_rng(0)
    for k in range(K):
        p = OUTPUTS / "annealed_wta_K6" / f"expert_{k}_grid.png"
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        flat = arr.ravel()
        idx = rng.choice(flat.size, size=4000, replace=False)
        pix.append(flat[idx])
    W = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            W[i, j] = wasserstein_distance(pix[i], pix[j])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(W, cmap="viridis")
    for i in range(K):
        for j in range(K):
            ax.text(
                j,
                i,
                f"{W[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if W[i, j] < W.max() * 0.55 else "black",
                fontsize=8,
            )
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"E{i}" for i in range(K)])
    ax.set_yticklabels([f"E{i}" for i in range(K)])
    ax.set_title(
        "Pairwise $W_1$ between per-expert pixel distributions\n"
        "(Annealed $K=6$, $B=512$)"
    )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "wasserstein_K6.png", dpi=150)
    plt.close(fig)


def plot_fid_vs_B():
    """Plot FID trends versus batch size for annealed WTA."""
    Bs = [128, 256, 512, 1024]
    dirs = ["annealed_wta_B128", "annealed_wta_B256", REF, "annealed_wta_B1024"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for s in STRATS:
        ys = [load_metrics(d)[s]["fid"] for d in dirs]
        ax.plot(
            Bs, ys, marker="o", lw=1.7, label=STRAT_LABELS[s], color=STRAT_COLORS[s]
        )
    base = load_metrics(REF)["baseline_heun"]["fid"]
    ax.axhline(base, ls="--", color="black", alpha=0.6, label="Baseline (Heun)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mini-batch size $B$ (log)")
    ax.set_ylabel("FID (log)")
    ax.set_xticks(Bs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("Annealed WTA, $K=4$: FID vs.\\ batch size")
    ax.grid(which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fid_vs_B.png", dpi=150)
    plt.close(fig)


def plot_pixel_kde_B128():
    """Plot per-expert pixel intensity KDE curves for one configuration."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = np.linspace(0, 1, 256)
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    rng = np.random.default_rng(0)
    for k in range(4):
        p = OUTPUTS / "annealed_wta_B128" / f"expert_{k}_grid.png"
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        flat = arr.ravel()
        sample = rng.choice(flat, size=min(20000, flat.size), replace=False)
        kde = gaussian_kde(sample, bw_method=0.02)
        ax.plot(xs, kde(xs), lw=1.8, color=colors[k], label=f"Expert {k}")
    ax.set_yscale("log")
    ax.set_xlabel("Pixel intensity (normalised)")
    ax.set_ylabel("Density (log)")
    ax.set_title("Per-expert pixel-intensity KDE\n(Annealed $K=4$, $B=128$)")
    ax.legend(fontsize=9)
    ax.grid(which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "pixel_kde_B128.png", dpi=150)
    plt.close(fig)


def plot_pr_scatter():
    """Plot precision-recall points across available configurations."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for s in STRATS:
        xs, ys = [], []
        for d in CONFIGS:
            m = load_metrics(d)
            if m is None:
                continue
            rec = m[s]
            if rec["fid"] > 1000 or rec["precision"] == 0:
                continue
            xs.append(rec["recall"])
            ys.append(rec["precision"])
        ax.scatter(
            xs,
            ys,
            color=STRAT_COLORS[s],
            s=55,
            alpha=0.85,
            label=STRAT_LABELS[s],
            edgecolors="white",
            linewidths=0.7,
        )
    m = load_metrics(REF)
    ax.scatter(
        [m["baseline_heun"]["recall"]],
        [m["baseline_heun"]["precision"]],
        color="black",
        marker="*",
        s=200,
        label="Baseline (Heun)",
        zorder=5,
    )
    ax.set_xlabel("Recall (higher is better)")
    ax.set_ylabel("Precision (higher is better)")
    ax.set_title("Precision vs.\\ Recall across non-collapsed configurations")
    ax.grid(ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "precision_recall_scatter.png", dpi=150)
    plt.close(fig)


def plot_single_to_gated_ratio():
    """Plot the FID ratio between single-expert and gated routing."""
    configs = [
        ("Hard WTA K=4", "hard_wta", "unit"),
        ("Relaxed WTA K=4", "relaxed_wta", "unit"),
        ("Resilient K=4", "resilient_mcl", "unit"),
        ("Annealed K=3", "annealed_wta_K3", "soft"),
        ("Annealed K=4", REF, "soft"),
        ("Annealed B=128", "annealed_wta_B128", "soft"),
        ("Annealed K=6", "annealed_wta_K6", "hero"),
        ("Annealed B=1024", "annealed_wta_B1024", "hero"),
        ("Resilient K=8", "resilient_mcl_K8", "hero"),
    ]
    labels, ratios, tags = [], [], []
    for name, d, tag in configs:
        m = load_metrics(d)
        r = m["single_expert"]["fid"] / max(m["gated"]["fid"], 1e-6)
        labels.append(name)
        ratios.append(r)
        tags.append(tag)
    color_map = {"unit": "#aaaaaa", "soft": "#ff8c33", "hero": "#d62728"}
    colors = [color_map[t] for t in tags]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    order = np.argsort(ratios)
    ax.barh(
        [labels[i] for i in order],
        [ratios[i] for i in order],
        color=[colors[i] for i in order],
    )
    ax.axvline(1.0, color="black", ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Single-expert / Gated FID ratio (log)")
    ax.set_title("FID ratio of Single-expert over Gated routing, per configuration")
    for patch, i in zip(ax.patches, order):
        w = patch.get_width()
        ax.text(
            w * 1.05,
            patch.get_y() + patch.get_height() / 2,
            f"{w:.1f}$\\times$",
            va="center",
            fontsize=8,
        )
    ax.grid(axis="x", which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "single_to_gated_ratio.png", dpi=150)
    plt.close(fig)


def plot_gated_vs_alive():
    """Plot gated FID against the number of active experts."""
    # Alive counts come from reported usage fractions in project analyses.
    points = [
        ("Hard WTA K=4", "hard_wta", 1, "collapse"),
        ("Relaxed WTA K=4", "relaxed_wta", 3, "soft"),
        ("Resilient K=2", "resilient_mcl_K2", 1, "collapse"),
        ("Resilient K=3", "resilient_mcl_K3", 1, "collapse"),
        ("Resilient K=4", "resilient_mcl", 1, "collapse"),
        ("Resilient K=8", "resilient_mcl_K8", 1, "collapse"),
        ("Annealed K=2", "annealed_wta_K2", 1, "collapse"),
        ("Annealed K=3", "annealed_wta_K3", 3, "soft"),
        ("Annealed K=4", REF, 3, "soft"),
        ("Annealed K=6", "annealed_wta_K6", 5, "soft"),
        ("Annealed K=8", "annealed_wta_K8", 5, "soft"),
        ("Annealed B=128", "annealed_wta_B128", 2, "soft"),
        ("Annealed B=256", "annealed_wta_B256", 2, "soft"),
        ("Annealed B=1024", "annealed_wta_B1024", 3, "soft"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    color_map = {"collapse": "#d62728", "soft": "#2ca02c"}
    for name, d, alive, tag in points:
        m = load_metrics(d)
        if m is None:
            continue
        fid = m["gated"]["fid"]
        ax.scatter(
            alive,
            fid,
            color=color_map[tag],
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.7,
        )
        ax.annotate(
            name, (alive, fid), xytext=(5, 4), textcoords="offset points", fontsize=7
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of alive experts (usage $\\geq 0.01$)")
    ax.set_ylabel("Gated FID (log)")
    ax.set_title("Gated FID vs.\\ alive-expert count across all studied configurations")
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d62728",
            markersize=9,
            label="Effectively single-expert",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2ca02c",
            markersize=9,
            label="Genuinely multi-expert",
        ),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="upper right")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "gated_vs_alive.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_fid_vs_K()
    plot_usage_histograms()
    plot_wasserstein_K6()
    plot_fid_vs_B()
    plot_pixel_kde_B128()
    plot_pr_scatter()
    plot_single_to_gated_ratio()
    plot_gated_vs_alive()
    print("Extra figures written to", OUT)
