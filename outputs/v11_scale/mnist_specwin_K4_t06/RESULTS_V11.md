# V11 speciation-window / commit-once report

## Intended test

V11 tests the claim that routing should be activated once, near the symmetry-breaking/speciation time:

\[\hat k=\arg\min_k \sum_c p_\theta(c\mid x_{t_\star})A_{c,k}(t_\star).\]

Before `t_star`, use either the baseline score or the MCL mixture score; after `t_star`, keep the chosen expert fixed.

## Inferred/forced t_star

```json
{
  "source": "manual",
  "t_star": 0.6,
  "threshold": 1.0
}
```

## Generation metrics

| strategy | FID | diag fallback | precision | recall | commit frac | t_star | usage entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_heun | 55.029489844287795 | nan | 0.56982421875 | 0.75390625 | nan | nan | nan |
| random_expert | 48.90212018120958 | nan | 0.6630859375 | 0.75732421875 | 0.0 | nan | 0.9978479444980621 |
| mixture_score | 48.573384663423006 | nan | 0.65380859375 | 0.759765625 | 0.0 | nan | nan |
| risk_commit_tstar | 46.76392291729622 | nan | 0.65234375 | 0.73095703125 | 1.0 | nan | 0.15467693842947483 |
| linear_commit_once | 48.13933493228804 | nan | 0.64111328125 | 0.76171875 | 1.0 | nan | 0.06442166352644563 |
| baseline_then_risk_commit_tstar | 56.725604636803666 | nan | 0.609375 | 0.76220703125 | 1.0 | 0.6 | 0.1441477993503213 |
| mixture_then_risk_commit_tstar | 50.00619794307036 | nan | 0.63134765625 | 0.75634765625 | 1.0 | 0.6 | 0.18230408150702715 |
| baseline_then_linear_commit_tstar | 55.00117382595645 | nan | 0.609375 | 0.771484375 | 1.0 | 0.6 | 0.049686632592254795 |
| mixture_then_linear_commit_tstar | 49.92005680731153 | nan | 0.654296875 | 0.7607421875 | 1.0 | 0.6 | 0.06577943614684045 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan | nan |

## Final speciation probe

- `teacher_entropy_sample_norm`: mean=0.999957, max=0.999998
- `beta_gap_mean`: mean=0.0089054, max=0.0398807
- `risk_margin_mean`: mean=2.33483e-05, max=0.000105796
- `oracle_class_mi_norm`: mean=0.0627857, max=0.236662
- `delta_A_mean_min_interexpert`: mean=9.65559e-06, max=7.48698e-05

## Reading rule

A positive V11 result is not merely lower FID. It should also show non-flat `A_{c,k}`, non-zero risk margin, non-collapsed usage, and commit-once strategies that are competitive with random/mixture baselines.