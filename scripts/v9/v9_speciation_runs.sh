#!/usr/bin/env bash
# Speciation follow-up: hard-WTA then diversity (no gold retrain).
set -euo pipefail
cd "$(dirname "$0")/../.."
SCRIPT=scripts/v9/run_variant_v9.py
DEVICE=${DEVICE:-cuda}
BASELINE=outputs/v9_suite/mnist_gold_K4/baseline_final.pt
LOG=outputs/v9_speciation_runs.log

if [[ ! -f "$BASELINE" ]]; then
  echo "missing baseline: $BASELINE" | tee -a "$LOG"
  exit 1
fi

COMMON=(
  --dataset mnist --classes all --image-size 28 --K 4
  --device "$DEVICE"
  --train-mcl --calibrate-router --sample-eval --validate-beta
  --baseline-ckpt "$BASELINE"
  --mcl-steps 60000 --batch-size 256
  --mcl-warmup-steps 2000 --mcl-ramp-steps 8000
  --probe-every 10000 --probe-batches 8 --probe-t-bins 16
  --balance-weight 0.0
  --mcl-init-noise 0.03
  --beta-max 800
  --beta-sweep-min-mult 0.02 --beta-sweep-max-mult 2.0 --beta-sweep-points 61
  --num-samples 2048 --sample-steps 100 --sample-batch-size 256
  --pca-dim-router 64 --router-stats-items 12000 --router-calib-batches 64
  --beta-validation-batches 32
  --risk-soft-temp 2e-4 --risk-conf-margin 1e-5
  --commit-margin 1e-5 --commit-force-t 0.35
  --paired-batches 2
)

run_one() {
  local tag="$1"
  shift
  local outdir="$1"
  shift
  echo "[spec] $(date -Is) START $tag -> $outdir" | tee -a "$LOG"
  python "$SCRIPT" --outdir "$outdir" "$@" "${COMMON[@]}" 2>&1 | tee -a "$LOG"
  echo "[spec] $(date -Is) DONE $tag" | tee -a "$LOG"
}

echo "[spec] $(date -Is) pipeline start" | tee "$LOG"

run_one "hard-WTA" outputs/v9_spec_hardwta_nobalance --hard-wta
run_one "diversity" outputs/v9_spec_diversity --diversity-weight 0.01

echo "[spec] $(date -Is) pipeline finished" | tee -a "$LOG"
