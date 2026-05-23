#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SCRIPT=scripts/v10/run_variant_v10_symmetry_router.py

# Quick sanity
python "$SCRIPT" \
  --dataset mnist --classes all --image-size 28 --K 4 \
  --outdir outputs/v10_mnist_quick --device cuda \
  --all --baseline-steps 3000 --mcl-steps 6000 \
  --num-samples 256 --sample-steps 50 \
  --probe-every 1000 --router-calib-batches 16 \
  --linear-router-batches 16 --linear-router-steps 100

# Beta-rho/speciation sweep around the theoretically calibrated temperature
for rho in 0.5 1.0 2.0 4.0; do
  python "$SCRIPT" \
    --dataset mnist --classes all --image-size 28 --K 4 \
    --outdir outputs/v10_mnist_beta_rho_${rho} --device cuda \
    --all --baseline-steps 12000 --mcl-steps 24000 \
    --batch-size 256 --num-samples 512 --sample-steps 80 \
    --beta-rho ${rho} --probe-every 3000 --router-calib-batches 32 \
    --linear-router-batches 32 --linear-router-steps 150 \
    --strategies baseline_heun,random_expert,mixture_score,risk_gated,risk_commit_tstar,posterior_map_commit_once,geodesic_map_commit_once,linear_gated,linear_commit_once
done
