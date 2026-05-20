# V9: validation théorie → MCL dynamics → routing → génération

## But
V9 est construite pour vérifier explicitement les points théoriques discutés :

1. `beta_x(t)=d/(2 v_t)` par sweep empirique de température.
2. Spéciation MCL pendant l'entraînement : `beta_gap`, entropie du teacher, marge de risque, rang/spectre de `A_{c,k}(t)`.
3. Routeur déployable Bayes-risk : `argmin_k sum_c p(c|x_t,t) A_{c,k}(t)`.
4. Hypothèse paper-style : soft/shared avant spéciation, puis commit expert une seule fois (`risk_commit_once`).
5. Comparaison génération : `baseline_heun`, `single_expert`, `random_expert`, `mixture_score`, `risk_gated`, `risk_softmix`, `risk_confident`, `risk_commit_once`.

## Fichiers produits par run

- `beta_validation_by_t.csv`
- `mcl_speciation_probe_by_step_t.csv`
- `router_calibration_by_t.csv`
- `metrics.json`
- `SUMMARY_v9.txt`
- `RESULTS_V9.md`
- `plot_v9_*.png`

## Commande gold MNIST minimale

```bash
python scripts/run_variant_v9.py \
  --dataset mnist --classes all --image-size 28 --K 4 \
  --outdir outputs/v9_mnist_gold --device cuda --all \
  --baseline-steps 30000 --mcl-steps 60000 --batch-size 256 \
  --mcl-warmup-steps 2000 --mcl-ramp-steps 8000 \
  --num-samples 2048 --sample-steps 100 --sample-batch-size 256 \
  --pca-dim-router 64 --router-stats-items 12000 \
  --router-calib-batches 64 --beta-validation-batches 32 \
  --probe-every 10000 --probe-batches 8 --probe-t-bins 16 \
  --risk-soft-temp 2e-4 --risk-conf-margin 1e-5 \
  --commit-margin 1e-5 --commit-force-t 0.35 \
  --paired-batches 4
```

## Lecture attendue

- Si `beta_emp_over_beta_z_theory ≈ 1`, la forme bayésienne de la température est validée.
- Si `teacher_entropy_sample_norm ≈ 1` et `risk_margin_mean ≈ 0`, le routage dur n'a rien de fiable à décider.
- Si `beta_gap_mean` augmente et que `risk_margin_mean` devient non nul, la spéciation MCL est active.
- Si `risk_softmix` gagne, on est encore dans un régime frontière/soft.
- Si `risk_commit_once` gagne ou rejoint `risk_softmix`, l'hypothèse diffusion classique jusqu'à spéciation puis commit expert est soutenue.
