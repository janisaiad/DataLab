#!/usr/bin/env bash
# Full-night V10 symmetry/router suite. Waits for V9 jobs, then runs sequentially.
set -uo pipefail
cd "$(dirname "$0")/../.."
SCRIPT=scripts/v10/run_variant_v10_symmetry_router.py
DEVICE=${DEVICE:-cuda}
ROOT=outputs/v10_night
LOG=outputs/v10_night/master_run.log
mkdir -p "$ROOT" outputs/v10_night

exec > >(tee -a "$LOG") 2>&1

echo "[v10-night] $(date -Is) start"

wait_for_gpu() {
  local max_wait_sec=$((12 * 3600))
  local t0=$SECONDS
  while true; do
    if pgrep -f "run_variant_v9.py" >/dev/null 2>&1; then
      echo "[v10-night] $(date -Is) waiting: v9 still running"
    elif pgrep -f "v9_speciation_runs.sh" >/dev/null 2>&1; then
      echo "[v10-night] $(date -Is) waiting: v9_speciation shell"
    else
      local used
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
      if [[ "${used:-0}" -lt 8000 ]]; then
        echo "[v10-night] $(date -Is) GPU free (${used} MiB used)"
        break
      fi
      echo "[v10-night] $(date -Is) waiting: GPU mem ${used} MiB"
    fi
    if (( SECONDS - t0 > max_wait_sec )); then
      echo "[v10-night] WARN: wait timeout, starting anyway"
      break
    fi
    sleep 90
  done
}

run_one() {
  local name="$1"
  shift
  local outdir="$1"
  shift
  echo "[v10-night] $(date -Is) BEGIN $name -> $outdir"
  if python "$SCRIPT" --outdir "$outdir" --device "$DEVICE" "$@"; then
    echo "[v10-night] $(date -Is) OK   $name"
  else
    echo "[v10-night] $(date -Is) FAIL $name (exit $?)"
  fi
}

wait_for_gpu

# 0) Smoke — pipeline check
run_one "quick" "$ROOT/mnist_quick_K4" \
  --all --baseline-steps 3000 --mcl-steps 6000 --batch-size 256 \
  --num-samples 256 --sample-steps 50 --sample-batch-size 128 \
  --probe-every 1000 --probe-batches 4 --probe-t-bins 12 \
  --router-calib-batches 16 --beta-validation-batches 16 \
  --linear-router-batches 16 --linear-router-steps 100 \
  --dataset mnist --classes all --image-size 28 --K 4

# 1) Gold MNIST — full train + all default V10 strategies
run_one "gold" "$ROOT/mnist_gold_K4" \
  --all --baseline-steps 30000 --mcl-steps 60000 --batch-size 256 \
  --mcl-warmup-steps 2000 --mcl-ramp-steps 8000 \
  --num-samples 2048 --sample-steps 100 --sample-batch-size 256 \
  --pca-dim-router 64 --beta-rho 1.0 --beta-max 80 \
  --router-stats-items 12000 --router-calib-batches 64 \
  --beta-validation-batches 32 --probe-every 5000 --probe-batches 8 --probe-t-bins 16 \
  --linear-router-batches 64 --linear-router-steps 200 \
  --paired-batches 4 \
  --dataset mnist --classes all --image-size 28 --K 4

# 2) Router-only from V9 gold ckpt (fast comparison of router families)
V9G=outputs/v9_suite/mnist_gold_K4
if [[ -f "$V9G/baseline_final.pt" && -f "$V9G/mcl_final.pt" ]]; then
  run_one "router_from_v9_gold" "$ROOT/router_from_v9_gold_K4" \
    --dataset mnist --classes all --image-size 28 --K 4 \
    --baseline-ckpt "$V9G/baseline_final.pt" \
    --mcl-ckpt "$V9G/mcl_final.pt" \
    --validate-beta --calibrate-router --train-linear-router --sample-eval \
    --num-samples 2048 --sample-steps 100 --sample-batch-size 256 \
    --pca-dim-router 64 --router-calib-batches 64 --linear-router-batches 64 \
    --linear-router-steps 200 --probe-every 0 \
    --strategies baseline_heun,random_expert,mixture_score,risk_gated,risk_softmix,risk_commit_once,risk_commit_tstar,posterior_map_gated,posterior_map_softmix,posterior_map_commit_once,geodesic_map_gated,geodesic_map_softmix,geodesic_map_commit_once,linear_gated,linear_softmix,linear_commit_once
else
  echo "[v10-night] skip router_from_v9_gold: missing ckpt"
fi

# 3) Beta-rho ablation (subset of strategies for speed)
for rho in 0.5 1.0 2.0 4.0; do
  run_one "beta_rho_${rho}" "$ROOT/mnist_beta_rho_${rho}" \
    --all --baseline-steps 12000 --mcl-steps 24000 --batch-size 256 \
    --mcl-warmup-steps 1000 --mcl-ramp-steps 5000 \
    --num-samples 1024 --sample-steps 80 --sample-batch-size 256 \
    --beta-rho "$rho" --beta-max 80 \
    --probe-every 3000 --probe-batches 8 --probe-t-bins 16 \
    --router-calib-batches 32 --linear-router-batches 32 --linear-router-steps 150 \
    --paired-batches 2 \
    --strategies baseline_heun,random_expert,mixture_score,risk_gated,risk_softmix,risk_commit_tstar,posterior_map_commit_once,geodesic_map_commit_once,linear_gated,linear_commit_once \
    --dataset mnist --classes all --image-size 28 --K 4
done

# 4) Hard-WTA speciation (symmetry-breaking stress test)
run_one "hard_wta" "$ROOT/mnist_hard_wta_K4" \
  --all --baseline-steps 12000 --mcl-steps 60000 --batch-size 256 \
  --hard-wta --balance-weight 0.0 --mcl-init-noise 0.03 --beta-max 800 \
  --mcl-warmup-steps 2000 --mcl-ramp-steps 8000 \
  --num-samples 2048 --sample-steps 100 --sample-batch-size 256 \
  --probe-every 10000 --router-calib-batches 64 --linear-router-batches 64 \
  --linear-router-steps 200 \
  --strategies baseline_heun,random_expert,mixture_score,risk_gated,risk_commit_tstar,posterior_map_commit_once,linear_gated,linear_commit_once \
  --dataset mnist --classes all --image-size 28 --K 4

echo "[v10-night] $(date -Is) finished"
