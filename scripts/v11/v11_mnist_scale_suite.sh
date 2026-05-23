#!/usr/bin/env bash
# V11 MNIST at scale: auto t* then probe-informed t*=0.6
set -uo pipefail
cd "$(dirname "$0")/../.."
SCRIPT=scripts/v11/run_variant_v11_speciation_commit.py
LOG=outputs/v11_scale/master_run.log
mkdir -p outputs/v11_scale
exec > >(tee -a "$LOG") 2>&1

echo "[v11-scale] $(date -Is) start"
python -m py_compile "$SCRIPT"

COMMON=(
  --dataset mnist --classes all --image-size 28 --K 4
  --device cuda --all
  --baseline-steps 30000 --mcl-steps 60000 --batch-size 256
  --mcl-warmup-steps 2000 --mcl-ramp-steps 8000
  --num-samples 2048 --sample-steps 100 --sample-batch-size 256
  --pca-dim-router 64 --beta-rho 1.0 --beta-max 80
  --spec-delta 0.20 --spec-window-frac 0.85
  --balance-weight 0.05 --diversity-weight 0.01
  --orthogonal-output-weight 0.002 --a-variance-weight 0.005
  --probe-every 5000 --probe-batches 8 --probe-t-bins 16
  --router-calib-batches 64 --beta-validation-batches 32
  --linear-router-batches 64 --linear-router-steps 200
  --linear-router-target risk
  --paired-batches 4
  --risk-soft-temp 2e-4 --risk-conf-margin 1e-5
  --commit-margin 1e-5
)

run_one() {
  local tag="$1"
  local outdir="$2"
  shift 2
  echo "[v11-scale] $(date -Is) BEGIN $tag -> $outdir"
  if python "$SCRIPT" --outdir "$outdir" "${COMMON[@]}" "$@"; then
    echo "[v11-scale] $(date -Is) OK $tag"
  else
    echo "[v11-scale] $(date -Is) FAIL $tag"
  fi
}

# 1) Auto-estimated t_star from probes (spec-t-star -1)
run_one mnist_auto_tstar outputs/v11_scale/mnist_specwin_K4_auto \
  --spec-t-star -1 --spec-t-star-prior 0.6

# 2) Forced t_star=0.6 (MNIST symmetry probe best regime)
run_one mnist_fixed_t06 outputs/v11_scale/mnist_specwin_K4_t06 \
  --spec-t-star 0.6 --spec-t-star-prior 0.6

echo "[v11-scale] $(date -Is) finished"
