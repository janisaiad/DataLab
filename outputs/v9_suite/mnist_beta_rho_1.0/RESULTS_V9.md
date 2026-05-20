# V9 theory → routing → generation report

## What this run validates
1. The Bayes/Gaussian inverse-temperature law `beta_x(t)=d/(2 v_t)`, measured by an empirical CE sweep.
2. MCL expert speciation during training: teacher entropy, beta-gap, A-matrix rank/singular values, route margins.
3. Deployable Bayes-risk routing, including softmix, hard gate, confident gate, and commit-once routing.
4. Generation impact relative to baseline, single expert, mixture score, and risk-based routers.

## Generation metrics

| strategy | FID | diag fallback | precision | recall | fallback frac | commit frac |
|---|---:|---:|---:|---:|---:|---:|
| baseline_heun | 75.4975624722596 | nan | 0.5595703125 | 0.7734375 | nan | nan |
| mixture_score | 46.96875441462777 | nan | 0.638671875 | 0.79296875 | 0.0 | 0.0 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan |
| random_expert | 43.71264231075078 | nan | 0.642578125 | 0.818359375 | 0.0 | 0.0 |
| risk_commit_once | 42.730484442700636 | nan | 0.68359375 | 0.76171875 | 0.9 | 1.0 |
| risk_confident | 44.356112103081784 | nan | 0.6513671875 | 0.796875 | 1.0 | 0.0 |
| risk_gated | 44.718705023909436 | nan | 0.6279296875 | 0.78125 | 0.0 | 0.0 |
| risk_softmix | 44.20413140053444 | nan | 0.6494140625 | 0.77734375 | 0.0 | 0.0 |
| single_expert | 43.89387754481204 | nan | 0.6328125 | 0.7919921875 | 0.0 | 0.0 |

## Beta validation

Mean empirical/theory ratio: `0.2484`.
Min/max ratio: `0.2` / `0.4126`.

## MCL speciation probe

- `teacher_entropy_sample_norm` final-step mean: `1`
- `beta_gap_mean` final-step mean: `8.66244e-08`
- `risk_margin_mean` final-step mean: `3.92173e-10`
- `route_excess_vs_sample_oracle` final-step mean: `9.73563e-10`
- `oracle_class_mi_norm` final-step mean: `0.0147496`

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