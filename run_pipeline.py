"""End-to-end pipeline: train, sample, evaluate, analyse."""

import subprocess, sys, os, json, time, torch

PYTHON = sys.executable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

K = 4
MCL_VARIANT = "annealed_wta"
TAG = f"{MCL_VARIANT}_K{K}"
SHARED_CKPT = os.path.join(BASE_DIR, "checkpoints", "shared")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", TAG)
OUT_DIR = os.path.join(BASE_DIR, "outputs", TAG)
for d in [SHARED_CKPT, CKPT_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

ARCH = {
    "base_ch": 32,
    "ch_mult": "1 2",
    "num_res_blocks": 2,
    "time_dim": 128,
    "dropout": 0.1,
    "sigma_min": 0.01,
    "sigma_max": 80.0,
}
BATCH = 256
BASELINE_EPOCHS = 200
MCL_EPOCHS = 200
SAVE_EVERY = 20
GATING_EPOCHS = 25

def run(desc, cmd):
    """Run a shell command for one pipeline step.

    Args:
        desc: Human-readable step description.
        cmd: Shell command executed from `BASE_DIR`.
    """
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    elapsed = time.time() - t0
    print(f"  [{desc}] done in {elapsed/60:.1f} min  (exit {r.returncode})")
    if r.returncode != 0:
        print(f"  ** FAILED - stopping pipeline **")
        sys.exit(1)

arch_flags = " ".join(f"--{k} {v}" for k, v in ARCH.items())

run(f"1/7  Train baseline ({BASELINE_EPOCHS} epochs)",
    f"{PYTHON} -m src.train --mode baseline --epochs {BASELINE_EPOCHS} --batch_size {BATCH} "
    f"--lr 3e-4 --device {DEVICE} --out_dir {SHARED_CKPT} --save_every {SAVE_EVERY} "
    f"--seed 42 {arch_flags}")

run(f"2/7  Train MCL K={K} ({MCL_EPOCHS} epochs, {MCL_VARIANT})",
    f"{PYTHON} -m src.train --mode mcl --K {K} --mcl_variant {MCL_VARIANT} "
    f"--epochs {MCL_EPOCHS} --batch_size {BATCH} "
    f"--lr 3e-4 --device {DEVICE} --out_dir {CKPT_DIR} --save_every {SAVE_EVERY} "
    f"--seed 42 {arch_flags}")

run("3/7  Train gating network",
    f"{PYTHON} -m src.gating --mcl_ckpt {CKPT_DIR}/mcl_K{K}_final.pt "
    f"--epochs {GATING_EPOCHS} --batch_size 512 --collect_batches 80 "
    f"--device {DEVICE} --out_dir {CKPT_DIR}")

N_EVAL = 2048
strategies = [
    ("baseline",       f"--checkpoint {SHARED_CKPT}/baseline_final.pt --mode baseline"),
    ("single_expert",  f"--checkpoint {CKPT_DIR}/mcl_K{K}_final.pt --mode mcl --strategy single_expert --expert_id 0"),
    ("random_expert",  f"--checkpoint {CKPT_DIR}/mcl_K{K}_final.pt --mode mcl --strategy random_expert"),
    ("best_expert",    f"--checkpoint {CKPT_DIR}/mcl_K{K}_final.pt --mode mcl --strategy best_expert"),
    ("mixture_score",  f"--checkpoint {CKPT_DIR}/mcl_K{K}_final.pt --mode mcl --strategy mixture_score"),
    ("gated",          f"--checkpoint {CKPT_DIR}/mcl_K{K}_final.pt --mode mcl --strategy gated "
                       f"--gating_ckpt {CKPT_DIR}/gating_K{K}.pt"),
]

STEPS = 200
for name, flags in strategies:
    run(f"4/7  Sample [{name}]",
        f"{PYTHON} -m src.sample {flags} --num_samples {N_EVAL} "
        f"--num_steps {STEPS} --solver euler --seed 0 --device {DEVICE} "
        f"--out_dir {OUT_DIR}")

run("4/7  Sample [baseline_heun]",
    f"{PYTHON} -m src.sample --checkpoint {SHARED_CKPT}/baseline_final.pt "
    f"--mode baseline --num_samples {N_EVAL} --num_steps {STEPS} --solver heun "
    f"--seed 0 --device {DEVICE} --out_dir {OUT_DIR}")

for eid in range(K):
    run(f"4/7  Sample [expert_{eid}]",
        f"{PYTHON} -m src.sample --checkpoint {CKPT_DIR}/mcl_K{K}_final.pt "
        f"--mode mcl --strategy single_expert --expert_id {eid} "
        f"--num_samples 64 --num_steps {STEPS} --solver euler --seed 0 "
        f"--device {DEVICE} --out_dir {OUT_DIR}")

all_metrics = {}
sample_files = {
    "baseline_euler":  f"{OUT_DIR}/baseline_euler_n{N_EVAL}.pt",
    "baseline_heun":   f"{OUT_DIR}/baseline_heun_n{N_EVAL}.pt",
    "single_expert":   f"{OUT_DIR}/mcl_K{K}_single_expert_e0_euler_n{N_EVAL}.pt",
    "random_expert":   f"{OUT_DIR}/mcl_K{K}_random_expert_euler_n{N_EVAL}.pt",
    "best_expert":     f"{OUT_DIR}/mcl_K{K}_best_expert_euler_n{N_EVAL}.pt",
    "mixture_score":   f"{OUT_DIR}/mcl_K{K}_mixture_score_euler_n{N_EVAL}.pt",
    "gated":           f"{OUT_DIR}/mcl_K{K}_gated_euler_n{N_EVAL}.pt",
}

for name, pt_path in sample_files.items():
    if not os.path.exists(pt_path):
        print(f"  [SKIP] {pt_path} not found")
        continue
    run(f"5/7  Evaluate [{name}]",
        f"{PYTHON} -m src.evaluate --samples_pt {pt_path} "
        f"--num_samples {N_EVAL} --k 5 --device {DEVICE}")

print("\n\nCollecting metrics programmatically ...")
sys.path.insert(0, BASE_DIR)
from src.evaluate import evaluate
from src.utils import get_mnist_loaders
_, test_loader = get_mnist_loaders(batch_size=512)
real_images = torch.cat([x for x, _ in test_loader])[:N_EVAL]

for name, pt_path in sample_files.items():
    if not os.path.exists(pt_path):
        continue
    gen = torch.load(pt_path, weights_only=True)[:N_EVAL]
    m = evaluate(real_images, gen, device=DEVICE, k=5)
    all_metrics[name] = m
    print(f"  {name:20s}  FID={m['fid']:8.2f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nMetrics saved -> {OUT_DIR}/metrics.json")

run("6/7  Analysis plots",
    f"{PYTHON} -m src.analyze --mcl_ckpt {CKPT_DIR}/mcl_K{K}_final.pt "
    f"--baseline_ckpt {SHARED_CKPT}/baseline_final.pt "
    f"--out_dir {OUT_DIR}/analysis --num_batches 40 --seed 42 --device {DEVICE}")

print("\n7/7  Plotting training curves ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

bl_log = os.path.join(SHARED_CKPT, "baseline_log.json")
if os.path.exists(bl_log):
    with open(bl_log) as f:
        bl = json.load(f)
    axes[0].plot(range(1, len(bl["loss"]) + 1), bl["loss"], "b-")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Baseline Training Loss")
    axes[0].grid(True, alpha=0.3)

mcl_log = os.path.join(CKPT_DIR, f"mcl_K{K}_log.json")
if os.path.exists(mcl_log):
    with open(mcl_log) as f:
        ml = json.load(f)
    axes[1].plot(range(1, len(ml["loss"]) + 1), ml["loss"], "r-")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss (winner)")
    axes[1].set_title(f"MCL K={K} Training Loss")
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

if os.path.exists(mcl_log):
    with open(mcl_log) as f:
        ml = json.load(f)
    if "expert_usage" in ml and ml["expert_usage"]:
        import numpy as np
        usage = np.array(ml["expert_usage"])
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in range(usage.shape[1]):
            ax.plot(range(1, len(usage) + 1), usage[:, k], label=f"Expert {k}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Usage fraction")
        ax.set_title("Expert Usage During Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "expert_usage_training.png"), dpi=150, bbox_inches="tight")
        plt.close()

if all_metrics:
    names = list(all_metrics.keys())
    fids = [all_metrics[n]["fid"] for n in names]
    precs = [all_metrics[n]["precision"] for n in names]
    recs = [all_metrics[n]["recall"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = range(len(names))

    axes[0].bar(x, fids, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("FID (lower is better)")
    axes[0].set_title("FID Comparison")

    axes[1].bar(x, precs, color="seagreen")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision Comparison")

    axes[2].bar(x, recs, color="coral")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Recall")
    axes[2].set_title("Recall Comparison")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETE")
print(f"  Checkpoints:  {CKPT_DIR}/")
print(f"  Outputs:      {OUT_DIR}/")
print(f"  Analysis:     {OUT_DIR}/analysis/")
print(f"{'='*60}")
