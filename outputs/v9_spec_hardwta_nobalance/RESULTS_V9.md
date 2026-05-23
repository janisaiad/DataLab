# V9 theory → routing → generation report

## What this run validates
1. The Bayes/Gaussian inverse-temperature law `beta_x(t)=d/(2 v_t)`, measured by an empirical CE sweep.
2. MCL expert speciation during training: teacher entropy, beta-gap, A-matrix rank/singular values, route margins.
3. Deployable Bayes-risk routing, including softmix, hard gate, confident gate, and commit-once routing.
4. Generation impact relative to baseline, single expert, mixture score, and risk-based routers.

## Generation metrics

| strategy | FID | diag fallback | precision | recall | fallback frac | commit frac |
|---|---:|---:|---:|---:|---:|---:|
| baseline_heun | 56.02374565205295 | nan | 0.5810546875 | 0.77001953125 | nan | nan |
| mixture_score | 958.5641916201455 | nan | 0.19873046875 | 0.013671875 | 0.0 | 0.0 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan |
| random_expert | 828.4313648970357 | nan | 0.1748046875 | 0.83056640625 | 0.0 | 0.0 |
| risk_commit_once | 50.123421886929194 | nan | 0.64892578125 | 0.73046875 | 0.0 | 1.0 |
| risk_confident | 48.46772243372549 | nan | 0.62841796875 | 0.75 | 0.0 | 0.0 |
| risk_gated | 45.51393510889713 | nan | 0.65380859375 | 0.77392578125 | 0.0 | 0.0 |
| risk_softmix | 47.007404626009674 | nan | 0.634765625 | 0.77001953125 | 0.0 | 0.0 |
| single_expert | 1252.2794942149426 | nan | 0.0390625 | 0.0 | 0.0 | 0.0 |

## Beta validation

Mean empirical/theory ratio: `0.1994`.
Min/max ratio: `0.09283` / `0.3991`.

## MCL speciation probe

- `teacher_entropy_sample_norm` final-step mean: `1.76646e-10`
- `beta_gap_mean` final-step mean: `42.693`
- `risk_margin_mean` final-step mean: `0.0914907`
- `route_excess_vs_sample_oracle` final-step mean: `0`
- `oracle_class_mi_norm` final-step mean: `0`

## Produced plots
- `plot_v9_beta_validation_ce.png`
- `plot_v9_beta_validation_ratio.png`
- `plot_v9_final_phase_diagnostics.png`
- `plot_v9_generation_metrics.png`
- `plot_v9_heatmap_beta_gap_mean.png`
- `plot_v9_heatmap_oracle_class_mi_norm.png`
- `plot_v9_heatmap_risk_margin_mean.png`
- `plot_v9_heatmap_route_excess_vs_sample_oracle.png`
- `plot_v9_heatmap_teacher_entropy_sample_norm.png`
- `plot_v9_router_calibration.png`

## Suggested reading of the run
- If beta ratio is near 1 but teacher entropy and risk margin stay near 0, the Bayes temperature is right but MCL experts have not speciated.
- If beta-gap grows and risk margin becomes non-zero, then commit-once routing should become meaningful.
- If risk_softmix beats hard routing, the model is still in a soft/speciation boundary regime; if commit-once wins, the paper-style one-time routing hypothesis is supported.