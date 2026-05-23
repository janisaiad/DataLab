# V9 symmetry-breaking and linear-router probe

## Setup
```json
{
  "dataset": "cifar10",
  "data_root": "./data",
  "classes": "automobile,horse",
  "C": 4,
  "K": 2,
  "d_latent": 128,
  "n_train": 8000,
  "n_test": 2000,
  "gmm_mu": 1.0,
  "gmm_sigma0": 0.5,
  "t_grid": "0.6,1.0,1.4,1.8,2.2,2.6,3.0",
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
  "posterior_logreg_steps": 1000,
  "posterior_logreg_lr": 0.002,
  "posterior_logreg_batch_size": 512,
  "posterior_logreg_weight_decay": 0.0001,
  "router_logreg_steps": 1000,
  "router_logreg_lr": 0.002,
  "router_logreg_batch_size": 512,
  "router_logreg_weight_decay": 0.0001,
  "seed": 0,
  "device": "cuda",
  "outdir": "outputs/v9_symmetry_cifar_auto_horse",
  "no_download": false
}
```

## Data
```json
{
  "dataset": "cifar10",
  "classes": [
    "automobile",
    "horse"
  ],
  "raw_dim": 3072,
  "d_latent": 128,
  "pca_dim": 128,
  "n_train": 8000,
  "n_test": 2000
}
```

## Theory check
For the loss `mean_dim squared error`, this script uses the local criterion

`beta * lambda_signal_scaled > 0.5`.

The clean theorem form is `beta * lambda_signal > lambda_damp`; the constant depends only on the loss normalization.

Best class-channel signal:
- t = 0.6
- lambda_class = 0.00805427
- beta_crit_class = 62.0789
- lambda_free = 0.0161232
- lambda_trans = 0.00242188

## Speciation diagnostics
- mean teacher entropy norm: 0.675323
- mean beta_gap: 1.03209
- mean risk_margin: 0.000889132
- mean A_ck_std: 0.000760952
- best oracle_class_mi_norm: 0.0527317 at t=0.6, rho=2.0
- best risk_margin_mean: 0.008055 at t=0.6, rho=2.0

## Linear/logistic router test
This is the explicit test of the hypothesis: after class/speciation time, a linear multinomial-logistic router should be enough.

- mean class posterior acc: 0.596214
- mean linear router acc vs risk labels: 0.985905
- mean linear router excess risk: 8.38654e-07
- best linear router acc: 1 at t=1.0, rho=0.5
  excess_mean=0, risk_margin=7.4245e-09

Best combined route/speciation row:
- t: 0.6
- rho: 2.0
- beta: 124.15772423604983
- beta_times_lambda_class: 1.0
- posterior_acc: 0.7255000472068787
- risk_margin_mean: 0.0080550042912364
- risk_margin_train_mean: 0.008022177033126354
- risk_label_usage_entropy: 0.9989025592803955
- risk_label_class_mi_norm: 0.15828601801241102
- bayes_risk_excess_mean: 0.0
- linear_router_acc_vs_risk: 0.9820000529289246
- linear_router_excess_mean: 8.77936963661341e-06
- linear_router_excess_p95: 0.0
- linear_router_usage_entropy: 0.9990122318267822
- linear_router_class_mi_norm: 0.15255533270415333
- linear_router_mean_conf: 0.9188032150268555

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