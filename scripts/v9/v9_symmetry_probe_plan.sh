#!/usr/bin/env bash
# Plan propre symmetry-breaking (séparé des runs diffusion V9/V10).
set -uo pipefail
cd "$(dirname "$0")/../.."
SCRIPT=scripts/v9/v9_symmetry_router_probe.py
LOG=outputs/v9_symmetry_plan/master_run.log
mkdir -p outputs/v9_symmetry_plan
exec > >(tee -a "$LOG") 2>&1

echo "[plan] $(date -Is) start — NOT diffusion FID; theory probe only"

run_one() {
  local tag="$1"
  shift
  echo "[plan] $(date -Is) BEGIN $tag"
  if python "$SCRIPT" "$@"; then
    echo "[plan] $(date -Is) OK $tag"
  else
    echo "[plan] $(date -Is) FAIL $tag"
  fi
}

python -m py_compile "$SCRIPT"

# 1) GMM — posterior exact
run_one gmm \
  --dataset gmm --C 4 --K 4 --d-latent 64 \
  --n-train 6000 --n-test 3000 \
  --t-grid 0.4,0.7,1.0,1.3,1.6,1.9,2.2,2.5,2.8,3.1 \
  --rho-list 0.5,1.0,2.0 --mcl-steps 3000 --power-iters 40 \
  --device cuda --outdir outputs/v9_symmetry_plan/gmm

# 2) MNIST PCA64
run_one mnist_pca64 \
  --dataset mnist --classes 0,1,2,3,4,5,6,7,8,9 \
  --d-latent 64 --K 4 --n-train 12000 --n-test 2048 \
  --t-grid 0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,3.0 \
  --rho-list 0.5,1.0,2.0 --mcl-steps 3000 \
  --posterior-logreg-steps 1500 --router-logreg-steps 1500 \
  --device cuda --outdir outputs/v9_symmetry_plan/mnist_pca64

# 3) MNIST PCA128
run_one mnist_pca128 \
  --dataset mnist --classes 0,1,2,3,4,5,6,7,8,9 \
  --d-latent 128 --K 4 --n-train 12000 --n-test 2048 \
  --t-grid 0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,3.0 \
  --rho-list 0.5,1.0,2.0 --mcl-steps 3000 \
  --posterior-logreg-steps 1500 --router-logreg-steps 1500 \
  --device cuda --outdir outputs/v9_symmetry_plan/mnist_pca128

# 4) CIFAR auto/horse
run_one cifar_auto_horse \
  --dataset cifar10 --classes automobile,horse \
  --d-latent 128 --K 2 --n-train 8000 --n-test 2000 \
  --t-grid 0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,3.0 \
  --rho-list 0.5,1.0,2.0 --mcl-steps 3000 \
  --posterior-logreg-steps 1500 --router-logreg-steps 1500 \
  --device cuda --outdir outputs/v9_symmetry_plan/cifar_auto_horse

python3 <<'PY'
from pathlib import Path
lines = ["# Rapport plan symmetry probe (V9 theory)", "", "Séparé des runs diffusion `outputs/v9_suite/`, `v9_spec_*`, `v10_night/`.", ""]
for name in ["gmm", "mnist_pca64", "mnist_pca128", "cifar_auto_horse"]:
    d = Path("outputs/v9_symmetry_plan") / name
    lines.append(f"## {name}")
    r = d / "README_SUMMARY.md"
    lines.append(r.read_text(encoding="utf-8") if r.exists() else "*(manquant)*")
    lines.append("")
Path("outputs/v9_symmetry_plan/REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print("Wrote outputs/v9_symmetry_plan/REPORT.md")
PY

echo "[plan] $(date -Is) finished"
