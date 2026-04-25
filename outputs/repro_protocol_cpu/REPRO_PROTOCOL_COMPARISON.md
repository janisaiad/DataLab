# Reproduction du protocole et comparaison

## 1) Protocole reproduit (legacy)
- Checkpoints utilisés: `checkpoints/shared/baseline_final.pt`, `checkpoints/annealed_wta_K4/mcl_K4_final.pt`, `checkpoints/annealed_wta_K4/gating_K4.pt`
- Sampler utilisé: `src.sample.generate_baseline` et `src.sample.generate_mcl` (legacy, inchangé)
- Stratégies testées: `baseline_heun`, `single_expert`, `best_expert`, `mixture_score`, `gated`
- Run de reproduction: CPU compact (`n=256`, `steps=50`)

## 2) Résultats reproduction legacy (CPU compact)
- `baseline_heun`: FID=151.199, P=0.5312, R=0.9180
- `single_expert`: FID=370.121, P=0.7578, R=0.7656
- `best_expert`: FID=315.225, P=0.4961, R=0.8828
- `mixture_score`: FID=161.394, P=0.3984, R=0.8359
- `gated`: FID=116.989, P=0.6602, R=0.8203

## 3) Résultats legacy historiques (full run K4)
- `baseline_heun`: FID=111.439, P=0.4341, R=0.8325
- `single_expert`: FID=339.118, P=0.6270, R=0.9014
- `best_expert`: FID=287.860, P=0.3804, R=0.8042
- `mixture_score`: FID=113.398, P=0.3408, R=0.8140
- `gated`: FID=88.250, P=0.6440, R=0.8364

## 4) Résultats V6.1 smoke
- `baseline_heun`: FID=357.316, P=0.7969, R=0.8906
- `single_expert`: FID=3583.072, P=0.0625, R=0.0000
- `best_expert`: FID=3521.605, P=0.0781, R=0.0000
- `mixture_score`: FID=3438.479, P=0.0469, R=0.0000
- `gated`: FID=3583.072, P=0.0625, R=0.0000
- `gated_confident`: FID=3580.900, P=0.0625, R=0.0000
- `legacy_single_expert_e0`: FID=3583.072, P=0.0625, R=0.0000

## 5) Lecture
- La reproduction legacy (même sampler/conventions) reste cohérente avec les résultats historiques: `gated` et/ou `mixture_score` ne s'effondrent pas comme en V6 smoke.
- En V6.1 smoke, toutes les stratégies MCL sont catastrophiques (FID ~3400-3580), donc le problème est en amont du routeur (régime d'entraînement/protocole/schedule), pas seulement la tête de gating.
- Le test de parité `legacy_single_expert_e0` vs `v6 single_expert` dans le run V6.1 smoke est ~identique, donc le bug principal n'est pas un simple mismatch legacy/v6 sur ce cas; c'est surtout le setup du run smoke (K=2, B=64, steps=10, labels faibles).

## 6) Fichiers produits
- `outputs/repro_protocol_cpu/metrics_repro_cpu_n256.json`
- `outputs/repro_protocol_cpu/*.pt` et `*.png`
- `outputs/annealed_wta_K2_B64_v6_1_smoke/SUMMARY_v6.txt`