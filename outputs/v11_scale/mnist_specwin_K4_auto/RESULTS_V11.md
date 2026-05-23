# V11 speciation-window / commit-once report

## Intended test

V11 tests the claim that routing should be activated once, near the symmetry-breaking/speciation time:

\[\hat k=\arg\min_k \sum_c p_\theta(c\mid x_{t_\star})A_{c,k}(t_\star).\]

Before `t_star`, use either the baseline score or the MCL mixture score; after `t_star`, keep the chosen expert fixed.

## Inferred/forced t_star

```json
{
  "lambda_between_top": 12.778247833251953,
  "pca_dim": 64,
  "search_max": 1.0,
  "search_min": 0.35,
  "source": "first_reverse_crossing_beta_lambda",
  "t_star": 0.8121519088745117,
  "threshold": 1.0,
  "v0_z": 2.1721529960632324
}
```

## Generation metrics

| strategy | FID | diag fallback | precision | recall | commit frac | t_star | usage entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_heun | 54.88066665537547 | nan | 0.57421875 | 0.7646484375 | nan | nan | nan |
| random_expert | 42.94099969710103 | nan | 0.658203125 | 0.75830078125 | 0.0 | nan | 0.995641864836216 |
| mixture_score | 45.18923608223764 | nan | 0.642578125 | 0.77294921875 | 0.0 | nan | nan |
| risk_commit_tstar | 44.4933447112421 | nan | 0.63818359375 | 0.734375 | 1.0 | nan | 0.3616176377981901 |
| linear_commit_once | 43.18291864924142 | nan | 0.662109375 | 0.75732421875 | 1.0 | nan | 0.3216996565461159 |
| baseline_then_risk_commit_tstar | 54.136930748756306 | nan | 0.59423828125 | 0.71923828125 | 1.0 | 0.8121519088745117 | 0.34973839297890663 |
| mixture_then_risk_commit_tstar | 47.35288772251321 | nan | 0.65087890625 | 0.7265625 | 1.0 | 0.8121519088745117 | 0.38156455382704735 |
| baseline_then_linear_commit_tstar | 57.91523427769607 | nan | 0.6357421875 | 0.7724609375 | 1.0 | 0.8121519088745117 | 0.2857063151896 |
| mixture_then_linear_commit_tstar | 48.61505227916141 | nan | 0.66162109375 | 0.7373046875 | 1.0 | 0.8121519088745117 | 0.2926885839551687 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan | nan |

## Final speciation probe

- `teacher_entropy_sample_norm`: mean=0.999998, max=1
- `beta_gap_mean`: mean=0.00184337, max=0.00990522
- `risk_margin_mean`: mean=4.72186e-06, max=5.04694e-05
- `oracle_class_mi_norm`: mean=0.0453214, max=0.121388
- `delta_A_mean_min_interexpert`: mean=1.56279e-06, max=1.51169e-05

## Reading rule

A positive V11 result is not merely lower FID. It should also show non-flat `A_{c,k}`, non-zero risk margin, non-collapsed usage, and commit-once strategies that are competitive with random/mixture baselines.