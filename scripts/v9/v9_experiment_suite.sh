#!/usr/bin/env bash
set -euo pipefail

# V9 experiment suite: theory -> MCL dynamics -> Bayes-risk routing -> generation.
# Edit DEVICE/cuda index if needed.
DEVICE=${DEVICE:-cuda}
ROOT=${ROOT:-outputs/v9_suite}
SCRIPT=${SCRIPT:-scripts/run_variant_v9.py}

mkdir -p "$ROOT"

# 0) Cheap pipeline smoke: checks beta validation, speciation probes, router calibration, sampling.
python "$SCRIPT" \
  --dataset mnist --classes all --image-size 28 --K 4 \
  --outdir "$ROOT/mnist_smoke_K4" --device "$DEVICE" --all \
  --baseline-steps 3000 --mcl-steps 6000 --batch-size 256 \
  --mcl-warmup-steps 500 --mcl-ramp-steps 2000 \
  --num-samples 256 --sample-steps 50 --sample-batch-size 128 \
  --pca-dim-router 64 --router-stats-items 8000 \
  --router-calib-batches 16 --beta-validation-batches 16 \
  --probe-every 1500 --probe-batches 4 --probe-t-bins 12 \
  --paired-batches 2

# 1) Gold MNIST reference: directly comparable to your V8 gold numbers.
python "$SCRIPT" \
  --dataset mnist --classes all --image-size 28 --K 4 \
  --outdir "$ROOT/mnist_gold_K4" --device "$DEVICE" --all \
  --baseline-steps 30000 --mcl-steps 60000 --batch-size 256 \
  --mcl-warmup-steps 2000 --mcl-ramp-steps 8000 \
  --num-samples 2048 --sample-steps 100 --sample-batch-size 256 \
  --pca-dim-router 64 --router-stats-items 12000 \
  --router-calib-batches 64 --beta-validation-batches 32 \
  --probe-every 10000 --probe-batches 8 --probe-t-bins 16 \
  --risk-soft-temp 2e-4 --risk-conf-margin 1e-5 \
  --commit-margin 1e-5 --commit-force-t 0.35 \
  --paired-batches 4

# 2) Temperature ablation around the derived beta_x(t): subcritical / theory / supercritical.
for RHO in 0.5 1.0 2.0; do
  python "$SCRIPT" \
    --dataset mnist --classes all --image-size 28 --K 4 \
    --outdir "$ROOT/mnist_beta_rho_${RHO}" --device "$DEVICE" --all \
    --baseline-steps 12000 --mcl-steps 24000 --batch-size 256 \
    --mcl-warmup-steps 1000 --mcl-ramp-steps 5000 \
    --num-samples 1024 --sample-steps 80 --sample-batch-size 256 \
    --pca-dim-router 64 --router-stats-items 12000 \
    --router-calib-batches 48 --beta-validation-batches 32 \
    --probe-every 6000 --probe-batches 8 --probe-t-bins 16 \
    --beta-rho "$RHO" --paired-batches 2
 done

# 3) Expert-count ablation: does speciation/routing improve or fragment as K changes?
for K in 2 4 8; do
  python "$SCRIPT" \
    --dataset mnist --classes all --image-size 28 --K "$K" \
    --outdir "$ROOT/mnist_K${K}" --device "$DEVICE" --all \
    --baseline-steps 12000 --mcl-steps 24000 --batch-size 256 \
    --mcl-warmup-steps 1000 --mcl-ramp-steps 5000 \
    --num-samples 1024 --sample-steps 80 --sample-batch-size 256 \
    --pca-dim-router 64 --router-stats-items 12000 \
    --router-calib-batches 48 --beta-validation-batches 32 \
    --probe-every 6000 --probe-batches 8 --probe-t-bins 16 \
    --paired-batches 2
 done

# 4) CIFAR auto/horse: same theory with 3-channel eval fallback/resnet feature FID if available.
python "$SCRIPT" \
  --dataset cifar10 --classes automobile,horse --image-size 32 --K 4 \
  --outdir "$ROOT/cifar_auto_horse_K4" --device "$DEVICE" --all \
  --baseline-steps 50000 --mcl-steps 80000 --batch-size 192 \
  --mcl-warmup-steps 3000 --mcl-ramp-steps 10000 \
  --num-samples 2048 --sample-steps 100 --sample-batch-size 32 \
  --pca-dim-router 128 --router-stats-items 10000 \
  --router-calib-batches 64 --beta-validation-batches 32 \
  --probe-every 16000 --probe-batches 8 --probe-t-bins 16 \
  --paired-batches 2
