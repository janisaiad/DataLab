# V10 symmetry-breaking / router experiments

This bundle modifies the V9 script into `run_variant_v10_symmetry_router.py`.

## What V10 adds

1. **Symmetry-breaking diagnostics**: keeps the V9 `beta_gap`, teacher entropy, A-matrix singular values, route margins, and route excess probes.
2. **One-shot routing at speciation time**: adds `risk_commit_tstar`, in addition to the previous threshold-based `risk_commit_once`.
3. **Posterior-map routers**: `posterior_map_gated`, `posterior_map_softmix`, `posterior_map_commit_once` route by class/posterior-to-expert map.
4. **Geodesic proxy routers**: `geodesic_map_gated`, `geodesic_map_softmix`, `geodesic_map_commit_once` use a time-interpolated Euclidean -> local diagonal Mahalanobis posterior.
5. **Learned linear/logistic routers**: `linear_gated`, `linear_softmix`, `linear_commit_once` train a per-time logistic expert router in PCA coordinates.

The key scientific test is:

- If MCL has not broken symmetry, `A_{c,k}(t)` is almost degenerate, route margins are near zero, and all routers should behave similarly.
- If symmetry breaking occurs, `A_{c,k}(t)` becomes non-degenerate, the risk/posterior/logistic routers should start to differ, and commit-once near the inferred `t_star` becomes meaningful.

## Quick MNIST test

```bash
python scripts/run_variant_v10_symmetry_router.py \
  --dataset mnist --classes all --image-size 28 --K 4 \
  --outdir outputs/v10_mnist_quick --device cuda \
  --all --baseline-steps 3000 --mcl-steps 6000 \
  --num-samples 256 --sample-steps 50 \
  --probe-every 1000 --router-calib-batches 16 \
  --linear-router-batches 16 --linear-router-steps 100
```

## Stronger MNIST run

```bash
python scripts/run_variant_v10_symmetry_router.py \
  --dataset mnist --classes all --image-size 28 --K 4 \
  --outdir outputs/v10_mnist_gold --device cuda \
  --all --baseline-steps 30000 --mcl-steps 60000 \
  --batch-size 256 --num-samples 2048 --sample-steps 100 \
  --pca-dim-router 64 --beta-rho 1.0 --beta-max 80 \
  --probe-every 5000 --router-calib-batches 64 \
  --linear-router-batches 64 --linear-router-steps 200 \
  --paired-batches 4
```

## Focused router-only run from existing checkpoints

```bash
python scripts/run_variant_v10_symmetry_router.py \
  --dataset mnist --classes all --image-size 28 --K 4 \
  --outdir outputs/v10_router_from_ckpt --device cuda \
  --baseline-ckpt outputs/v9_suite/mnist_gold_K4/baseline_final.pt \
  --mcl-ckpt outputs/v9_suite/mnist_gold_K4/mcl_final.pt \
  --calibrate-router --train-linear-router --sample-eval \
  --num-samples 2048 --sample-steps 100 \
  --strategies baseline_heun,mixture_score,risk_gated,risk_commit_tstar,posterior_map_commit_once,geodesic_map_commit_once,linear_gated,linear_commit_once
```

## Outputs to inspect

- `THEORY_SYMMETRY_BREAKING_V10.md`
- `mcl_speciation_probe_by_step_t.csv`
- `router_calibration_by_t.csv`
- `linear_router_train_by_t.csv`
- `metrics.json`
- `RESULTS_V10.md`
- `plot_v9_heatmap_*`, `plot_v9_generation_metrics.png`
