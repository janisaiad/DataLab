# V11 speciation-window / commit-once report

## Intended test

V11 tests the claim that routing should be activated once, near the symmetry-breaking/speciation time:

\[\hat k=\arg\min_k \sum_c p_\theta(c\mid x_{t_\star})A_{c,k}(t_\star).\]

Before `t_star`, use either the baseline score or the MCL mixture score; after `t_star`, keep the chosen expert fixed.

## Inferred/forced t_star

```json
{
  "source": "manual",
  "t_star": 0.5,
  "threshold": 1.0
}
```

## Generation metrics

| strategy | FID | diag fallback | precision | recall | commit frac | t_star | usage entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_heun | 407.0967355528213 | nan | nan | nan | nan | nan | nan |
| random_expert | 418.01167858616475 | nan | nan | nan | 0.0 | nan | 0.9295949619263411 |
| mixture_score | 418.9907797080448 | nan | nan | nan | 0.0 | nan | nan |
| risk_commit_tstar | 419.31141302202866 | nan | nan | nan | 1.0 | nan | 0.03461399913152482 |
| linear_commit_once | 404.10454518109935 | nan | nan | nan | 1.0 | nan | 0.006605520940647032 |
| baseline_then_risk_commit_tstar | 416.2859800021753 | nan | nan | nan | 1.0 | 0.5 | 0.025505951674924705 |
| mixture_then_risk_commit_tstar | 419.93926764666463 | nan | nan | nan | 1.0 | 0.5 | 0.041819953869199775 |
| baseline_then_linear_commit_tstar | 413.9773319697147 | nan | nan | nan | 1.0 | 0.5 | 0.0028309376077058695 |
| mixture_then_linear_commit_tstar | 417.8161711202085 | nan | nan | nan | 1.0 | 0.5 | 0.004718229274176451 |
| paired_mixture_vs_risk_softmix | nan | nan | nan | nan | nan | nan | nan |

## Final speciation probe

- `teacher_entropy_sample_norm`: mean=0.999752, max=0.999991
- `beta_gap_mean`: mean=0.0294636, max=0.161333
- `risk_margin_mean`: mean=3.95046e-05, max=0.000166522
- `oracle_class_mi_norm`: mean=0.0344829, max=0.0576931
- `delta_A_mean_min_interexpert`: mean=3.73127e-06, max=1.89267e-05

## Reading rule

A positive V11 result is not merely lower FID. It should also show non-flat `A_{c,k}`, non-zero risk margin, non-collapsed usage, and commit-once strategies that are competitive with random/mixture baselines.