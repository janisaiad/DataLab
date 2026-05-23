# Séparation des conclusions : diffusion V9/V10 vs symmetry probe

## A. Runs diffusion (nuit) — `run_variant_v9.py` / V10

**Question** : MCL + routeurs améliorent-ils la **génération** (FID) ?

**Verdict** (MNIST gold, 2048 samples) :

| Hypothèse | Résultat |
|-----------|----------|
| MCL bat baseline | **Oui** (57.7 → ~28–34) |
| Routeur Bayes-risk = cause du gain | **Non** (`random_expert` 27.8 > `risk_softmix` 31.4) |
| Spéciation soft-MCL | **Non** (`teacher_entropy≈1`, `risk_margin≈10⁻¹⁰`, `A` plat) |
| β empirique / théorie | **Non** (~0.25, pas ≈1) |
| Hard-WTA | Brisure **mauvaise** (collapse 1 expert, FID pathologiques sur mixture/random) |
| Diversity | **Meilleur compromis** (entropy ~0.89, β_gap ~0.6, FID ~45) |

**Message papier** : le bottleneck est la **brisure de symétrie des experts**, pas le routeur.

**Dossiers** : `outputs/v9_suite/`, `outputs/v9_spec_*`, `outputs/v10_night/` (V10 crashé bug `class_var_z_diag`, corrigé).

---

## B. Probe théorie — `v9_symmetry_router_probe.py`

**Question** : la solution symétrique devient-elle instable quand $\beta\lambda_{\mathrm{signal}}>\frac12$, et un routeur linéaire suffit-il **après** spéciation ?

**Pas de FID.** Métriques : `phase_by_t.csv`, `mcl_by_t_beta.csv`, `router_by_t_beta.csv`, `linear_router_acc_vs_risk`, etc.

### B1. Premier lancement rapide (suite courte, paramètres README)

Dossiers : `outputs/v9_symmetry_gmm`, `outputs/v9_symmetry_mnist`, `outputs/v9_symmetry_cifar_auto_horse`

→ Résultats **indicatifs** (GMM OK, MNIST partiel @ bas $t$, CIFAR faible MI). **Ne pas confondre** avec le plan propre ci-dessous.

### B2. Plan propre (grilles / PCA / logreg renforcés)

Lancé via `scripts/v9/v9_symmetry_probe_plan.sh` → `outputs/v9_symmetry_plan/`

| Run | Outdir |
|-----|--------|
| GMM, power-iters 40, t fin | `gmm/` |
| MNIST PCA64 | `mnist_pca64/` |
| MNIST PCA128 | `mnist_pca128/` |
| CIFAR auto/horse | `cifar_auto_horse/` |

Rapport : `outputs/v9_symmetry_plan/REPORT.md`

---

## C. Ce qu’il ne faut pas mélanger

- Un bon **probe GMM** ne prouve pas que la **diffusion full** a spécié.
- Un mauvais **FID random > risk_softmix** ne invalide pas le **seuil local** $\beta\lambda>\frac12$ en modèle simplifié.
- Le probe demande : « y a-t-il quelque chose à router ? » — la diffusion demande : « est-ce que ça améliore les images ? »
