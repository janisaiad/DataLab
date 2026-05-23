#!/usr/bin/env bash
# V11 CIFAR-10 multiclass grande échelle: 10 classes, K=10 experts (K=C).
# Même protocole que MNIST scale (auto t* puis t* prior fixe). Plusieurs jours OK.
set -uo pipefail
cd "$(dirname "$0")/../.."
SCRIPT=scripts/v11/run_variant_v11_speciation_commit.py
LOG=outputs/v11_cifar10_scale/master_run.log
mkdir -p outputs/v11_cifar10_scale
exec > >(tee -a "$LOG") 2>&1

echo "[v11-cifar10] $(date -Is) start K=10 C=10"
python -m py_compile "$SCRIPT"

# CIFAR-10 full 10 classes, one expert per class
CIFAR_COMMON=(
  --dataset cifar10 --classes all --image-size 32 --K 10
  --device cuda --all
  --baseline-steps 60000 --mcl-steps 120000 --batch-size 128
  --mcl-warmup-steps 3000 --mcl-ramp-steps 12000
  --num-samples 2048 --sample-steps 100 --sample-batch-size 32
  --pca-dim-router 128 --beta-rho 1.0 --beta-max 80
  --router-stats-items 15000
  --spec-delta 0.20 --spec-window-frac 0.85
  --balance-weight 0.05 --diversity-weight 0.01
  --orthogonal-output-weight 0.002 --a-variance-weight 0.005
  --probe-every 10000 --probe-batches 8 --probe-t-bins 16
  --router-calib-batches 96 --beta-validation-batches 48
  --linear-router-batches 96 --linear-router-steps 250
  --linear-router-target risk
  --paired-batches 4
  --risk-soft-temp 2e-4 --risk-conf-margin 1e-5
  --commit-margin 1e-5
)

run_one() {
  local tag="$1"
  local outdir="$2"
  shift 2
  echo "[v11-cifar10] $(date -Is) BEGIN $tag -> $outdir"
  if python "$SCRIPT" --outdir "$outdir" "${CIFAR_COMMON[@]}" "$@"; then
    echo "[v11-cifar10] $(date -Is) OK $tag"
  else
    echo "[v11-cifar10] $(date -Is) FAIL $tag"
  fi
}

# 1) t_star auto (first_reverse_crossing_beta_lambda)
run_one cifar10_auto_tstar outputs/v11_cifar10_scale/cifar10_specwin_K10_auto \
  --spec-t-star -1 --spec-t-star-prior 0.5

# 2) t_star fixe mid-noise (prior CIFAR typique; ajuster si probe le suggère)
run_one cifar10_fixed_t05 outputs/v11_cifar10_scale/cifar10_specwin_K10_t05 \
  --spec-t-star 0.5 --spec-t-star-prior 0.5

echo "[v11-cifar10] $(date -Is) finished"
