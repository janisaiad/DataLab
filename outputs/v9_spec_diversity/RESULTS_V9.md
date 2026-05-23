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
| mixture_score | 47.38358405348849 | nan | 0.65869140625 | 0.736328125 | 0.0 | 0.0 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan |
| random_expert | 45.24864277111713 | nan | 0.646484375 | 0.73388671875 | 0.0 | 0.0 |
| risk_commit_once | 63.94192853665735 | nan | 0.6298828125 | 0.7626953125 | 0.274609375 | 1.0 |
| risk_confident | 49.843872143955885 | nan | 0.6474609375 | 0.7265625 | 0.2847314453125 | 0.0 |
| risk_gated | 47.72616962764388 | nan | 0.666015625 | 0.7587890625 | 0.0 | 0.0 |
| risk_softmix | 45.30995505727624 | nan | 0.650390625 | 0.765625 | 0.0 | 0.0 |
| single_expert | 58.510931982976324 | nan | 0.6416015625 | 0.751953125 | 0.0 | 0.0 |

## Beta validation

Mean empirical/theory ratio: `0.1994`.
Min/max ratio: `0.09283` / `0.3991`.

## MCL speciation probe

- `teacher_entropy_sample_norm` final-step mean: `0.887434`
- `beta_gap_mean` final-step mean: `0.618641`
- `risk_margin_mean` final-step mean: `0.000102899`
- `route_excess_vs_sample_oracle` final-step mean: `0.000868119`
- `oracle_class_mi_norm` final-step mean: `0.0420565`

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