# V9 theory → routing → generation report

## What this run validates
1. The Bayes/Gaussian inverse-temperature law `beta_x(t)=d/(2 v_t)`, measured by an empirical CE sweep.
2. MCL expert speciation during training: teacher entropy, beta-gap, A-matrix rank/singular values, route margins.
3. Deployable Bayes-risk routing, including softmix, hard gate, confident gate, and commit-once routing.
4. Generation impact relative to baseline, single expert, mixture score, and risk-based routers.

## Generation metrics

| strategy | FID | diag fallback | precision | recall | fallback frac | commit frac |
|---|---:|---:|---:|---:|---:|---:|
| baseline_heun | 73.15226674585293 | nan | 0.556640625 | 0.740234375 | nan | nan |
| mixture_score | 61.13196953764173 | nan | 0.62109375 | 0.7958984375 | 0.0 | 0.0 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan |
| random_expert | 63.62725776266613 | nan | 0.623046875 | 0.787109375 | 0.0 | 0.0 |
| risk_commit_once | 59.23601224603614 | nan | 0.64453125 | 0.7890625 | 0.9 | 1.0 |
| risk_confident | 62.65539687875588 | nan | 0.6328125 | 0.80859375 | 1.0 | 0.0 |
| risk_gated | 58.043868535700796 | nan | 0.626953125 | 0.7705078125 | 0.0 | 0.0 |
| risk_softmix | 62.13834622494667 | nan | 0.634765625 | 0.794921875 | 0.0 | 0.0 |
| single_expert | 61.83847748797499 | nan | 0.6376953125 | 0.75390625 | 0.0 | 0.0 |

## Beta validation

Mean empirical/theory ratio: `0.2484`.
Min/max ratio: `0.2` / `0.4126`.

## MCL speciation probe

- `teacher_entropy_sample_norm` final-step mean: `1`
- `beta_gap_mean` final-step mean: `5.97529e-08`
- `risk_margin_mean` final-step mean: `5.08841e-10`
- `route_excess_vs_sample_oracle` final-step mean: `6.88595e-10`
- `oracle_class_mi_norm` final-step mean: `0.0168509`

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