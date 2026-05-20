# V9 theory → routing → generation report

## What this run validates
1. The Bayes/Gaussian inverse-temperature law `beta_x(t)=d/(2 v_t)`, measured by an empirical CE sweep.
2. MCL expert speciation during training: teacher entropy, beta-gap, A-matrix rank/singular values, route margins.
3. Deployable Bayes-risk routing, including softmix, hard gate, confident gate, and commit-once routing.
4. Generation impact relative to baseline, single expert, mixture score, and risk-based routers.

## Generation metrics

| strategy | FID | diag fallback | precision | recall | fallback frac | commit frac |
|---|---:|---:|---:|---:|---:|---:|
| baseline_heun | 134.53510860051364 | nan | 0.53515625 | 0.796875 | nan | nan |
| mixture_score | 107.23407005762866 | nan | 0.63671875 | 0.8515625 | 0.0 | 0.0 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan |
| random_expert | 107.1979644928864 | nan | 0.625 | 0.84765625 | 0.0 | 0.0 |
| risk_commit_once | 86.53103044440176 | nan | 0.67578125 | 0.87890625 | 0.9 | 1.0 |
| risk_confident | 107.88830955022787 | nan | 0.6171875 | 0.84375 | 1.0 | 0.0 |
| risk_gated | 92.62367657414036 | nan | 0.6015625 | 0.89453125 | 0.0 | 0.0 |
| risk_softmix | 106.01885652474328 | nan | 0.6484375 | 0.81640625 | 0.0 | 0.0 |
| single_expert | 108.13298752986854 | nan | 0.59375 | 0.76171875 | 0.0 | 0.0 |

## Beta validation

Mean empirical/theory ratio: `0.2462`.
Min/max ratio: `0.2` / `0.4126`.

## MCL speciation probe

- `teacher_entropy_sample_norm` final-step mean: `1`
- `beta_gap_mean` final-step mean: `0.000190021`
- `risk_margin_mean` final-step mean: `4.04231e-07`
- `route_excess_vs_sample_oracle` final-step mean: `1.96744e-06`
- `oracle_class_mi_norm` final-step mean: `0.0494985`

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