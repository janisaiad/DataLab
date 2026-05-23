# V9 symmetry-breaking and linear-router probe

## Setup
```json
{
  "dataset": "mnist",
  "data_root": "./data",
  "classes": "0,1,2,3,4,5,6,7,8,9",
  "C": 4,
  "K": 4,
  "d_latent": 128,
  "n_train": 12000,
  "n_test": 2048,
  "gmm_mu": 1.0,
  "gmm_sigma0": 0.5,
  "t_grid": "0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,3.0",
  "rho_list": "0.5,1.0,2.0",
  "feature_mode": "linear",
  "rf_dim": 512,
  "rf_activation": "erf",
  "ridge": 1e-05,
  "power_iters": 30,
  "mcl_steps": 3000,
  "mcl_lr": 0.002,
  "mcl_batch_size": 256,
  "init_std": 0.001,
  "balance_weight": 0.0,
  "posterior_logreg_steps": 1500,
  "posterior_logreg_lr": 0.002,
  "posterior_logreg_batch_size": 512,
  "posterior_logreg_weight_decay": 0.0001,
  "router_logreg_steps": 1500,
  "router_logreg_lr": 0.002,
  "router_logreg_batch_size": 512,
  "router_logreg_weight_decay": 0.0001,
  "seed": 0,
  "device": "cuda",
  "outdir": "outputs/v9_symmetry_plan/mnist_pca128",
  "no_download": false
}
```

## Data
```json
{
  "dataset": "mnist",
  "classes": [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
  ],
  "raw_dim": 784,
  "d_latent": 128,
  "pca_dim": 128,
  "n_train": 12000,
  "n_test": 2048
}
```

## Theory check
For the loss `mean_dim squared error`, this script uses the local criterion

`beta * lambda_signal_scaled > 0.5`.

The clean theorem form is `beta * lambda_signal > lambda_damp`; the constant depends only on the loss normalization.

Best class-channel signal:
- t = 0.5
- lambda_class = 0.00966087
- beta_crit_class = 51.7552
- lambda_free = 0.0287075
- lambda_trans = 0.0027034

## Speciation diagnostics
- mean teacher entropy norm: 0.718537
- mean beta_gap: 2.42503
- mean risk_margin: 0.0111533
- mean A_ck_std: 0.0148817
- best oracle_class_mi_norm: 0.624152 at t=0.5, rho=2.0
- best risk_margin_mean: 0.0798894 at t=0.5, rho=1.0

## Linear/logistic router test
This is the explicit test of the hypothesis: after class/speciation time, a linear multinomial-logistic router should be enough.

- mean class posterior acc: 0.380425
- mean linear router acc vs risk labels: 0.837059
- mean linear router excess risk: 0.00100615
- best linear router acc: 1 at t=3.0, rho=1.0
  excess_mean=0, risk_margin=9.16052e-08

Best combined route/speciation row:
- t: 0.5
- rho: 2.0
- beta: 103.51031818952592
- beta_times_lambda_class: 1.0
- posterior_acc: 0.783203125
- risk_margin_mean: 0.07732892036437988
- risk_margin_train_mean: 0.07577895373106003
- risk_label_usage_entropy: 0.9801253080368042
- risk_label_class_mi_norm: 0.6098249051576438
- bayes_risk_excess_mean: 0.0
- linear_router_acc_vs_risk: 0.86767578125
- linear_router_excess_mean: 0.006348105613142252
- linear_router_excess_p95: 0.04073234274983406
- linear_router_usage_entropy: 0.9795297384262085
- linear_router_class_mi_norm: 0.5365226109527951
- linear_router_mean_conf: 0.85355544090271

## How to read
- If `beta_times_lambda_class > 0.5` but `teacher_entropy_norm≈1`, `A_ck_std≈0`, and `risk_margin≈0`, then the theoretical local instability is not realized by the finite optimization run; increase steps/capacity or change initialization/schedule.
- If `A_ck_std`, `oracle_class_mi_norm`, and `risk_margin` become positive, then symmetry is breaking into class/latent channels.
- If `linear_router_acc_vs_risk` is high with tiny excess risk at that same t, then the linear/logistic router hypothesis is supported.
- If random/mixture generation wins while these diagnostics are near zero, the gain is a multi-expert sampling effect, not yet a proven Bayes-risk routing effect.

## Files
- `phase_by_t.csv`: Hessian/BBP-style signal and critical beta by t
- `mcl_by_t_beta.csv`: soft-MCL specialization metrics
- `router_by_t_beta.csv`: posterior and linear/logistic router metrics
- `plot_phase_curve.png`, `plot_speciation_metrics.png`, `plot_linear_router.png`, `A_ck_heatmap_best.png`