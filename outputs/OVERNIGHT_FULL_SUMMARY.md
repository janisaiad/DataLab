# Résumé complet — runs nuit (V9 suite + speciation + V10)

Date de synthèse : 2026-05-20

## 1. Verdict scientifique global

**MCL améliore la génération vs baseline Heun**, mais sur MNIST gold soft-MCL le **routeur Bayes-risk n’explique pas le gain** : `random_expert` (FID 27.77) bat `risk_softmix` (31.41) alors que `teacher_entropy ≈ 1`, `risk_margin ≈ 10⁻¹⁰`, `A_{c,k}` quasi plat.

**Negative result utile** : le bottleneck est la **stabilité / brisure de symétrie** des experts, pas la formule du routeur $k^*=\arg\min_k\sum_c p(c|x)A_{c,k}$.

**Validation β(t)** : ratio `beta_emp/beta_theory ≈ 0.25` (pas ≈ 1) — diagnostic isotrope vs PCA anisotrope probable.

**Eval FID** : gold et spec runs utilisent **2048** images générées (`samples_*.pt` vérifiés). Le champ `num_samples: 256` dans l’ancien `metrics.json` était un **bug de logging** (corrigé dans `run_variant_v9.py`).

---

## 2. Suite V9 (`outputs/v9_suite/`)

| Run | Statut | Baseline FID | Meilleur FID | Stratégie gagnante |
|-----|--------|--------------|--------------|-------------------|
| mnist_smoke_K4 | OK | 134.5 | 86.5 | risk_commit_once |
| **mnist_gold_K4** | OK | **57.66** | **27.77** | **random_expert** |
| mnist_beta_rho_0.5 | OK | 71.4 | 43.2 | random_expert |
| mnist_beta_rho_1.0 | OK | 75.5 | 42.7 | risk_commit_once |
| mnist_beta_rho_2.0 | OK | 77.2 | 43.6 | risk_confident |
| mnist_K2 | OK | 73.2 | 58.0 | risk_gated |
| mnist_K4 | OK | 72.8 | 45.2 | risk_commit_once |
| mnist_K8 | **non fini** | — | — | — |
| cifar_auto_horse | **non fini** | — | — | — |

### MNIST gold — détail stratégies (2048 samples)

| Stratégie | FID |
|-----------|-----|
| baseline_heun | 57.66 |
| random_expert | **27.77** |
| risk_confident | 29.51 |
| mixture_score | 30.05 |
| risk_gated | 30.18 |
| risk_softmix | 31.41 |
| risk_commit_once | 33.09 |
| single_expert | 34.44 |

**Probes finales gold** : `teacher_entropy=1`, `beta_gap≈0`, `risk_margin≈10⁻¹⁰` → pas de spéciation routeable.

**Paired** mix vs softmix : p ≈ 0.32 (non significatif).

---

## 3. Runs spéciation V9 (`outputs/v9_spec_*`)

### hard-WTA (`v9_spec_hardwta_nobalance`) — 60k, hard-WTA, β_max=800

| Stratégie | FID |
|-----------|-----|
| risk_gated | **45.51** |
| risk_confident | 48.47 |
| risk_commit_once | 50.12 |
| baseline | 56.02 |
| mixture / random / single | **pathologiques** (828–1252) |

**Probes** : `teacher_entropy≈0`, `beta_gap≈43`, `risk_margin≈0.09` — forte marge de *risque* mais **collapse** `usage=[0,1,0,0]` (1 expert). Pas spéciation 4-way saine ; sampling multi-expert dégénéré.

### diversity (`v9_spec_diversity`) — soft-MCL + diversity_weight=0.01

| Stratégie | FID |
|-----------|-----|
| risk_softmix | **45.31** |
| random_expert | 45.25 |
| risk_gated | 47.73 |
| baseline | 56.02 |

**Probes** : `teacher_entropy≈0.89`, `beta_gap≈0.62`, `risk_margin≈10⁻⁴` — **meilleur compromis** vers teacher non uniforme, mais marge routage encore faible.

---

## 4. Suite V10 nuit (`outputs/v10_night/`) — **ÉCHEC**

Tous les jobs ont crashé immédiatement :

```text
NameError: class_var_z_diag is not defined
```

dans `build_router_stats` (ligne ~753). **Aucun FID V10**, **aucun** `RESULTS_V10.md`. Bug corrigé localement (`vars_diag` rempli) — **à relancer**.

---

## 5. Probe symmetry-breaking (`v9_symmetry_router_probe.py`)

**Pas encore lancé** au moment de ce résumé. Pipeline prévu :

- `outputs/v9_symmetry_gmm`
- `outputs/v9_symmetry_mnist`
- `outputs/v9_symmetry_cifar_auto_horse`
- rapport agrégé `outputs/V9_SYMMETRY_PROBE_REPORT.md`

Critère théorique testé : $\beta(t)\lambda_{\mathrm{signal}}(t) \lessgtr \frac12$ vs métriques $A_{c,k}$, marge de risque, routeur linéaire/logistique.

---

## 6. Messages pour le papier (prudents)

1. Multi-experts MCL ↓ FID vs baseline : **oui** (gold : 58 → 28–34 selon stratégie).
2. Routeur Bayes-risk cause du gain : **non** sur gold soft-MCL (random meilleur, $A$ plat).
3. $\beta_x(t)=d/(2v_t)$ tel qu’implémenté : **non validé** (ratio ~0.25).
4. Spéciation spontanée depuis baseline : **non** ; hard-WTA → collapse ; diversity → entropie teacher ~0.89 mais marge risque faible.
5. Commit-once / speciation time : **pas validé** sur gold ; hard-WTA donne marge risque élevée mais experts non interchangeables de façon utile.
6. Routeur linéaire suffisant **après** spéciation : **à trancher** via `v9_symmetry_router_probe` (en cours).

---

## 7. Fichiers clés

- V9 gold : `outputs/v9_suite/mnist_gold_K4/RESULTS_V9.md`, `metrics.json`
- Spec : `outputs/v9_spec_hardwta_nobalance/`, `outputs/v9_spec_diversity/`
- V10 log : `outputs/v10_night/master_run.log`
- Théorie routeurs : `refs/routers.tex` / `routers.pdf`
