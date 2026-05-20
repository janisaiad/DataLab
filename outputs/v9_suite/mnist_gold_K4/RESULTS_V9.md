# V9 theory → routing → generation report

## What this run validates
1. The Bayes/Gaussian inverse-temperature law `beta_x(t)=d/(2 v_t)`, measured by an empirical CE sweep.
2. MCL expert speciation during training: teacher entropy, beta-gap, A-matrix rank/singular values, route margins.
3. Deployable Bayes-risk routing, including softmix, hard gate, confident gate, and commit-once routing.
4. Generation impact relative to baseline, single expert, mixture score, and risk-based routers.

## Generation metrics

| strategy | FID | diag fallback | precision | recall | fallback frac | commit frac |
|---|---:|---:|---:|---:|---:|---:|
| baseline_heun | 57.65846198062344 | nan | 0.56494140625 | 0.759765625 | nan | nan |
| mixture_score | 30.054389787767455 | nan | 0.63671875 | 0.77099609375 | 0.0 | 0.0 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan |
| random_expert | 27.7654146702691 | nan | 0.63623046875 | 0.76220703125 | 0.0 | 0.0 |
| risk_commit_once | 33.08907279266639 | nan | 0.658203125 | 0.77587890625 | 0.89 | 1.0 |
| risk_confident | 29.511006584868472 | nan | 0.67138671875 | 0.740234375 | 1.0 | 0.0 |
| risk_gated | 30.180834080634053 | nan | 0.62841796875 | 0.75 | 0.0 | 0.0 |
| risk_softmix | 31.4131401541055 | nan | 0.6611328125 | 0.7587890625 | 0.0 | 0.0 |
| single_expert | 34.444879639505444 | nan | 0.64013671875 | 0.75830078125 | 0.0 | 0.0 |

## Beta validation

Mean empirical/theory ratio: `0.2484`.
Min/max ratio: `0.2` / `0.4126`.

## MCL speciation probe

- `teacher_entropy_sample_norm` final-step mean: `1`
- `beta_gap_mean` final-step mean: `7.52402e-08`
- `risk_margin_mean` final-step mean: `2.45199e-10`
- `route_excess_vs_sample_oracle` final-step mean: `8.79029e-10`
- `oracle_class_mi_norm` final-step mean: `0.011486`

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