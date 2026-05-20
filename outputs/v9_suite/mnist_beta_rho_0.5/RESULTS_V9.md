# V9 theory → routing → generation report

## What this run validates
1. The Bayes/Gaussian inverse-temperature law `beta_x(t)=d/(2 v_t)`, measured by an empirical CE sweep.
2. MCL expert speciation during training: teacher entropy, beta-gap, A-matrix rank/singular values, route margins.
3. Deployable Bayes-risk routing, including softmix, hard gate, confident gate, and commit-once routing.
4. Generation impact relative to baseline, single expert, mixture score, and risk-based routers.

## Generation metrics

| strategy | FID | diag fallback | precision | recall | fallback frac | commit frac |
|---|---:|---:|---:|---:|---:|---:|
| baseline_heun | 71.36248355085968 | nan | 0.5537109375 | 0.7958984375 | nan | nan |
| mixture_score | 45.1988155556637 | nan | 0.638671875 | 0.8125 | 0.0 | 0.0 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan |
| random_expert | 43.23124889526969 | nan | 0.6240234375 | 0.810546875 | 0.0 | 0.0 |
| risk_commit_once | 43.38652548746781 | nan | 0.6650390625 | 0.80078125 | 0.9 | 1.0 |
| risk_confident | 45.87755192809837 | nan | 0.6416015625 | 0.783203125 | 1.0 | 0.0 |
| risk_gated | 45.60997756033366 | nan | 0.6416015625 | 0.7890625 | 0.0 | 0.0 |
| risk_softmix | 44.96453053739582 | nan | 0.634765625 | 0.791015625 | 0.0 | 0.0 |
| single_expert | 43.37464314040204 | nan | 0.62890625 | 0.783203125 | 0.0 | 0.0 |

## Beta validation

Mean empirical/theory ratio: `0.2484`.
Min/max ratio: `0.2` / `0.4126`.

## MCL speciation probe

- `teacher_entropy_sample_norm` final-step mean: `1`
- `beta_gap_mean` final-step mean: `7.79372e-08`
- `risk_margin_mean` final-step mean: `2.62405e-10`
- `route_excess_vs_sample_oracle` final-step mean: `9.29247e-10`
- `oracle_class_mi_norm` final-step mean: `0.0101619`

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