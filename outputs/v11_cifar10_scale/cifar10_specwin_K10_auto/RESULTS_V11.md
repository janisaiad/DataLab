# V11 speciation-window / commit-once report

## Intended test

V11 tests the claim that routing should be activated once, near the symmetry-breaking/speciation time:

\[\hat k=\arg\min_k \sum_c p_\theta(c\mid x_{t_\star})A_{c,k}(t_\star).\]

Before `t_star`, use either the baseline score or the MCL mixture score; after `t_star`, keep the chosen expert fixed.

## Inferred/forced t_star

```json
{
  "lambda_between_top": 34.56774139404297,
  "pca_dim": 128,
  "search_max": 1.0,
  "search_min": 0.35,
  "source": "first_reverse_crossing_beta_lambda",
  "t_star": 0.9630379676818848,
  "threshold": 1.0,
  "v0_z": 5.053045749664307
}
```

## Generation metrics

| strategy | FID | diag fallback | precision | recall | commit frac | t_star | usage entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_heun | 409.15101233351413 | nan | nan | nan | nan | nan | nan |
| random_expert | 420.8490164511953 | nan | nan | nan | 0.0 | nan | 0.9302407968789339 |
| mixture_score | 420.2798583961117 | nan | nan | nan | 0.0 | nan | nan |
| risk_commit_tstar | 420.9384497629415 | nan | nan | nan | 1.0 | nan | 0.23105878802016377 |
| linear_commit_once | 417.5334406751434 | nan | nan | nan | 1.0 | nan | 0.18439670279622078 |
| baseline_then_risk_commit_tstar | 417.42435790981426 | nan | nan | nan | 1.0 | 0.9630379676818848 | 0.21515195141546428 |
| mixture_then_risk_commit_tstar | 418.27587112663883 | nan | nan | nan | 1.0 | 0.9630379676818848 | 0.23970839497633278 |
| baseline_then_linear_commit_tstar | 414.53448945818366 | nan | nan | nan | 1.0 | 0.9630379676818848 | 0.17258628527633846 |
| mixture_then_linear_commit_tstar | 421.48357388122474 | nan | nan | nan | 1.0 | 0.9630379676818848 | 0.1800388873089105 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan | nan |

## Final speciation probe

- `teacher_entropy_sample_norm`: mean=0.999999, max=1
- `beta_gap_mean`: mean=0.00178371, max=0.0136808
- `risk_margin_mean`: mean=1.72409e-06, max=1.28883e-05
- `oracle_class_mi_norm`: mean=0.0341929, max=0.0578532
- `delta_A_mean_min_interexpert`: mean=7.64192e-08, max=7.38353e-07

## Reading rule

A positive V11 result is not merely lower FID. It should also show non-flat `A_{c,k}`, non-zero risk margin, non-collapsed usage, and commit-once strategies that are competitive with random/mixture baselines.