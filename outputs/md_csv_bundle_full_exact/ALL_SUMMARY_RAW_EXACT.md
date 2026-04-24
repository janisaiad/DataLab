# ALL SUMMARY RAW (exact content)

Total SUMMARY files: 6

## 1. `outputs/rf_mcl_corrected/SUMMARY.txt`

```text
Corrected RF-MCL numerical experiments
Params(d=64, p=512, mu=1.0, sigma0=0.5, seed=0, ridge=1e-06)

A: tau=infty channels
t=1.15, kappa=6.938, beta_free=0.09087, beta_class=0.1248, beta_trans=0.2166, align=0.6509
t=1.458, kappa=3.61, beta_free=0.1252, beta_class=0.1471, beta_trans=0.21, align=0.8109
t=1.767, kappa=1.911, beta_free=0.1247, beta_class=0.1475, beta_trans=0.2535, align=0.8336
t=2.075, kappa=1.021, beta_free=0.1355, beta_class=0.1747, beta_trans=0.2963, align=0.7615
t=2.383, kappa=0.5481, beta_free=0.2331, beta_class=0.3093, beta_trans=0.3359, align=0.7299
t=2.692, kappa=0.2949, beta_free=0.3243, beta_class=0.5871, beta_trans=0.3604, align=0.1977
t=3, kappa=0.1589, beta_free=0.2643, beta_class=0.6936, beta_trans=0.2696, align=0.03024

B: finite tau channels
tau=0.001, beta_free=0.00609, beta_class=0.09424, beta_trans=0.006138, align=0.008194
tau=0.004217, beta_free=0.006149, beta_class=0.1136, beta_trans=0.006204, align=0.009283
tau=0.01778, beta_free=0.006271, beta_class=0.08889, beta_trans=0.006324, align=0.0088
tau=0.07499, beta_free=0.006801, beta_class=0.1086, beta_trans=0.006854, align=0.007944
tau=0.3162, beta_free=0.009397, beta_class=0.1579, beta_trans=0.009457, align=0.006562
tau=1.334, beta_free=0.03088, beta_class=0.1929, beta_trans=0.0312, align=0.01109
tau=5.623, beta_free=0.1703, beta_class=0.2318, beta_trans=0.2291, align=0.4018
tau=23.71, beta_free=0.1985, beta_class=0.2219, beta_trans=0.3197, align=0.8367
tau=100, beta_free=0.1641, beta_class=0.2443, beta_trans=0.2561, align=0.4421

C: glass fixed beta
tau=0.001, rho=5.737e-05, beta_g_exact=2.132e+05, H_emp=0.1083, H_exact=0.1083
tau=0.004217, rho=0.0002419, beta_g_exact=5.056e+04, H_emp=0.1083, H_exact=0.1083
tau=0.01778, rho=0.00102, beta_g_exact=1.2e+04, H_emp=0.1083, H_exact=0.1083
tau=0.07499, rho=0.004293, beta_g_exact=2849, H_emp=0.1083, H_exact=0.1083
tau=0.3162, rho=0.01798, beta_g_exact=680.3, H_emp=0.1082, H_exact=0.1081
tau=1.334, rho=0.07365, beta_g_exact=166.1, H_emp=0.1061, H_exact=0.1057
tau=5.623, rho=0.2757, beta_g_exact=44.36, H_emp=0.08537, H_exact=0.08218
tau=23.71, rho=0.7435, beta_g_exact=16.45, H_emp=0.04631, H_exact=0
tau=100, rho=0.9968, beta_g_exact=12.27, H_emp=0.03749, H_exact=0
```

## 2. `outputs/rf_mcl_results/gmm_v3/SUMMARY.txt`

```text
RF-MCL GMM expert training v3
{
  "d": 64,
  "p": 256,
  "K": 4,
  "mu": 1.0,
  "sigma0": 0.5,
  "t": 2.05,
  "n_train": 6000,
  "n_test": 3000,
  "batch_size": 256,
  "steps": 4000,
  "lr": 0.003,
  "weight_decay": 0.0,
  "init_std": 0.001,
  "seed": 0,
  "calib_steps": 1200,
  "warmup_steps": 800,
  "ramp_steps": 1000,
  "eval_every": 200,
  "power_iters": 30,
  "ridge": 1e-05,
  "beta_target_mult": 3.0,
  "beta_glass_safety": 0.45,
  "cold_mult_glass": 1.3,
  "fixed_good_mult": 1.2,
  "device": "auto",
  "outdir": "./gmm_v3",
  "beta_grid": false
}

Calibration channels:
  lambda_free: 2.1210267543792725
  lambda_class: 1.3606247901916504
  lambda_trans: 0.15873096883296967
  beta_free: 0.23573488593085157
  beta_class: 0.36747823764800647
  beta_trans: 3.149983923581843
  free_align_m: 0.6276569366455078

Glass calibration:
  alpha: 0.1250994932445351
  r_t: 0.016851957422000366
  y_alpha: 0.44862007180556196
  beta_glass_exact: 72.93263402768928
  beta_glass_gauss: 41.976615350346535

Chosen beta_target: 1.1024347129440195

Final metrics by variant:

[uniform]
  beta: 0.0
  test_best_mse: 0.037869133055210114
  test_class_mi_norm: 0.00033348752185702324
  test_class_purity: 0.019531270692823455
  test_usage_entropy: 0.9988049864768982
  test_eff_frac_min: 1.0
  test_expert_align_m: 0.12226272374391556
  test_expert_diversity: 0.0002781408547889441

[hard_cold]
  beta: 94.81242423599606
  test_best_mse: 0.03764839470386505
  test_class_mi_norm: 0.0
  test_class_purity: 0.017578125
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0
  test_expert_align_m: 0.06348235160112381
  test_expert_diversity: 0.8345741033554077

[fixed_class]
  beta: 0.44097388517760777
  test_best_mse: 0.03758814185857773
  test_class_mi_norm: 0.0017853026511147618
  test_class_purity: 0.04492187750292942
  test_usage_entropy: 0.9823706746101379
  test_eff_frac_min: 0.9999747276306152
  test_expert_align_m: 0.22223341464996338
  test_expert_diversity: 0.014126280322670937

[fixed_good]
  beta: 1.1024347129440195
  test_best_mse: 0.03690845146775246
  test_class_mi_norm: 0.0
  test_class_purity: 0.00390625
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0032848077826201916
  test_expert_align_m: 0.05749950930476189
  test_expert_diversity: 0.6396310925483704

[theory_anneal]
  beta: 1.1024347129440195
  test_best_mse: 0.03382330387830734
  test_class_mi_norm: 0.00028080877382308245
  test_class_purity: 0.019531243189703673
  test_usage_entropy: 0.9993184208869934
  test_eff_frac_min: 0.9355311393737793
  test_expert_align_m: 0.9604715704917908
  test_expert_diversity: 0.02360406517982483
```

## 3. `outputs/rf_mcl_results/gmm_v3_grid/SUMMARY.txt`

```text
RF-MCL GMM expert training v3
{
  "d": 64,
  "p": 256,
  "K": 4,
  "mu": 1.0,
  "sigma0": 0.5,
  "t": 2.05,
  "n_train": 6000,
  "n_test": 3000,
  "batch_size": 256,
  "steps": 4000,
  "lr": 0.003,
  "weight_decay": 0.0,
  "init_std": 0.001,
  "seed": 0,
  "calib_steps": 1200,
  "warmup_steps": 800,
  "ramp_steps": 1000,
  "eval_every": 200,
  "power_iters": 30,
  "ridge": 1e-05,
  "beta_target_mult": 3.0,
  "beta_glass_safety": 0.45,
  "cold_mult_glass": 1.3,
  "fixed_good_mult": 1.2,
  "device": "auto",
  "outdir": "./gmm_v3_grid",
  "beta_grid": true
}

Calibration channels:
  lambda_free: 2.1210267543792725
  lambda_class: 1.3606247901916504
  lambda_trans: 0.15873096883296967
  beta_free: 0.23573488593085157
  beta_class: 0.36747823764800647
  beta_trans: 3.149983923581843
  free_align_m: 0.6276569366455078

Glass calibration:
  alpha: 0.1250994932445351
  r_t: 0.016851957422000366
  y_alpha: 0.44862007180556196
  beta_glass_exact: 72.93263402768928
  beta_glass_gauss: 41.976615350346535

Chosen beta_target: 1.1024347129440195

Final metrics by variant:

[uniform]
  beta: 0.0
  test_best_mse: 0.037869133055210114
  test_class_mi_norm: 0.00033348752185702324
  test_class_purity: 0.019531270692823455
  test_usage_entropy: 0.9988049864768982
  test_eff_frac_min: 1.0
  test_expert_align_m: 0.12226272374391556
  test_expert_diversity: 0.0002781408547889441

[hard_cold]
  beta: 94.81242423599606
  test_best_mse: 0.03764839470386505
  test_class_mi_norm: 0.0
  test_class_purity: 0.017578125
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0
  test_expert_align_m: 0.06348235160112381
  test_expert_diversity: 0.8345741033554077

[fixed_class]
  beta: 0.44097388517760777
  test_best_mse: 0.03758814185857773
  test_class_mi_norm: 0.0017853026511147618
  test_class_purity: 0.04492187750292942
  test_usage_entropy: 0.9823706746101379
  test_eff_frac_min: 0.9999747276306152
  test_expert_align_m: 0.22223341464996338
  test_expert_diversity: 0.014126280322670937

[fixed_good]
  beta: 1.1024347129440195
  test_best_mse: 0.03690845146775246
  test_class_mi_norm: 0.0
  test_class_purity: 0.00390625
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0032848077826201916
  test_expert_align_m: 0.05749950930476189
  test_expert_diversity: 0.6396310925483704

[theory_anneal]
  beta: 1.1024347129440195
  test_best_mse: 0.03382330387830734
  test_class_mi_norm: 0.00028080877382308245
  test_class_purity: 0.019531243189703673
  test_usage_entropy: 0.9993184208869934
  test_eff_frac_min: 0.9355311393737793
  test_expert_align_m: 0.9604715704917908
  test_expert_diversity: 0.02360406517982483

[grid_0.0918696]
  beta: 0.0918696
  test_best_mse: 0.03793059289455414
  test_class_mi_norm: 0.0009510573581792414
  test_class_purity: 0.04003907449077815
  test_usage_entropy: 0.9979106783866882
  test_eff_frac_min: 1.0
  test_expert_align_m: 0.14045946300029755
  test_expert_diversity: 0.00027728264103643596

[grid_0.275609]
  beta: 0.275609
  test_best_mse: 0.037879157811403275
  test_class_mi_norm: 0.0020403305534273386
  test_class_purity: 0.0527343615249265
  test_usage_entropy: 0.9782653450965881
  test_eff_frac_min: 0.9999995231628418
  test_expert_align_m: 0.16256053745746613
  test_expert_diversity: 0.00035314884735271335

[grid_0.459348]
  beta: 0.459348
  test_best_mse: 0.037595875561237335
  test_class_mi_norm: 0.0004873896250501275
  test_class_purity: 0.020507825363893062
  test_usage_entropy: 0.9364111423492432
  test_eff_frac_min: 0.9999685287475586
  test_expert_align_m: 0.2541366517543793
  test_expert_diversity: 0.014940083958208561

[grid_1.10243]
  beta: 1.10243
  test_best_mse: 0.037217773497104645
  test_class_mi_norm: 0.0
  test_class_purity: 0.0224609375
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0014915213687345386
  test_expert_align_m: 0.05531072989106178
  test_expert_diversity: 0.6183330416679382

[grid_58.3461]
  beta: 58.3461
  test_best_mse: 0.03762182220816612
  test_class_mi_norm: 0.0
  test_class_purity: 0.0087890625
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0
  test_expert_align_m: 0.06216323375701904
  test_expert_diversity: 0.8173807263374329

[grid_87.5192]
  beta: 87.5192
  test_best_mse: 0.037528738379478455
  test_class_mi_norm: 0.0
  test_class_purity: 0.0224609375
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0
  test_expert_align_m: 0.062072351574897766
  test_expert_diversity: 0.8213564157485962
```

## 4. `outputs/rf_mcl_results/rf_mcl_cifar_quick/SUMMARY.txt`

```text
RF-MCL CIFAR expert training v2
{
  "data_root": "./data",
  "classes": [
    "automobile",
    "horse"
  ],
  "d_mode": "rp",
  "pca_dim": 512,
  "rp_dim": 128,
  "p": 128,
  "K": 4,
  "t": 1.5,
  "n_train": 1200,
  "n_test": 600,
  "batch_size": 96,
  "steps": 500,
  "lr": 0.002,
  "weight_decay": 0.0,
  "init_std": 0.001,
  "seed": 0,
  "calib_steps": 150,
  "warmup_steps": 100,
  "ramp_steps": 150,
  "eval_every": 100,
  "power_iters": 8,
  "ridge": 1e-05,
  "beta_target_mult": 3.0,
  "beta_glass_safety": 0.45,
  "cold_mult_glass": 1.2,
  "device": "auto",
  "outdir": "./rf_mcl_cifar_quick",
  "no_download": false,
  "beta_grid": false
}

Data info:
{
  "classes": [
    "automobile",
    "horse"
  ],
  "raw_dim": 3072,
  "global_std": 0.5141243934631348,
  "d_mode": "rp",
  "d": 128
}

Channels:
  lambda_free: 82.81370544433594
  lambda_class: 2.8599095344543457
  lambda_trans: 2.783156633377075
  beta_free: 0.00603764796318741
  beta_class: 0.17483070494931222
  beta_trans: 0.17965212378047563
  free_align_m: 0.08050838857889175

Glass:
  alpha: 0.04997601293137614
  r_t: 0.05239569649125595
  v_emp: 0.2155480980873108
  beta_glass_emp: 0.6809632942542221
  E_mean_per_d: 0.1018686443567276

Chosen beta_target: 0.30643348241439994

Final metrics by variant:

[uniform]
  beta: 0.0
  test_best_mse: 0.6218883395195007
  test_class_mi_norm: 0.003688820404931903
  test_class_purity: 0.0733333399337548
  test_usage_entropy: 0.9966471791267395
  test_eff_frac_min: 1.0000001192092896
  test_expert_align_m: 0.08321640640497208
  test_expert_diversity: 0.0015125134959816933

[hard_cold]
  beta: 0.8171559531050665
  test_best_mse: 0.5439487099647522
  test_class_mi_norm: 0.0
  test_class_purity: 0.009999990463256836
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.022574778646230698
  test_expert_align_m: 0.07079727947711945
  test_expert_diversity: 0.4614320397377014

[fixed_class]
  beta: 0.20979684593917466
  test_best_mse: 0.628340482711792
  test_class_mi_norm: 0.0009890111396089196
  test_class_purity: 0.06999997711849365
  test_usage_entropy: 0.5260510444641113
  test_eff_frac_min: 0.9999986886978149
  test_expert_align_m: 0.09148932248353958
  test_expert_diversity: 0.0016588590806350112

[fixed_good]
  beta: 0.30643348241439994
  test_best_mse: 0.6218097805976868
  test_class_mi_norm: 0.0
  test_class_purity: 0.0366666316986084
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.9997521042823792
  test_expert_align_m: 0.07535939663648605
  test_expert_diversity: 0.00843422207981348

[theory_anneal]
  beta: 0.30643348241439994
  test_best_mse: 0.6159740090370178
  test_class_mi_norm: 0.0
  test_class_purity: 0.043333351612091064
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.9999799728393555
  test_expert_align_m: 0.08394868671894073
  test_expert_diversity: 0.003021989716216922
```

## 5. `scripts/rf_mcl_cifar/SUMMARY.txt`

```text
RF-MCL CIFAR expert training v2
{
  "data_root": "./data",
  "classes": [
    "automobile",
    "horse"
  ],
  "d_mode": "full",
  "pca_dim": 512,
  "rp_dim": 512,
  "p": 512,
  "K": 4,
  "t": 1.5,
  "n_train": 8000,
  "n_test": 2500,
  "batch_size": 192,
  "steps": 3000,
  "lr": 0.002,
  "weight_decay": 0.0,
  "init_std": 0.001,
  "seed": 0,
  "calib_steps": 900,
  "warmup_steps": 600,
  "ramp_steps": 900,
  "eval_every": 200,
  "power_iters": 30,
  "ridge": 1e-05,
  "beta_target_mult": 3.0,
  "beta_glass_safety": 0.45,
  "cold_mult_glass": 1.2,
  "device": "auto",
  "outdir": "./rf_mcl_cifar",
  "no_download": false,
  "beta_grid": false
}

Data info:
{
  "classes": [
    "automobile",
    "horse"
  ],
  "raw_dim": 3072,
  "global_std": 0.5104711055755615,
  "d_mode": "full",
  "d": 3072
}

Channels:
  lambda_free: 805.7389526367188
  lambda_class: 4.28288459777832
  lambda_trans: 4.108233451843262
  beta_free: 0.000620548377813668
  beta_class: 0.1167437479541963
  beta_trans: 0.1217068128822138
  free_align_m: 0.01765316352248192

Glass:
  alpha: 0.0026998859505540456
  r_t: 0.05239569649125595
  v_emp: 6.799816767374675
  beta_glass_emp: 0.028179877591121005
  E_mean_per_d: 0.10315446058909099

Chosen beta_target: 0.3502312438625889
WARNING: class/glass window narrow or absent; target uses class multiplier.

Final metrics by variant:

[uniform]
  beta: 0.0
  test_best_mse: 0.8576539754867554
  test_class_mi_norm: 0.0025528487749397755
  test_class_purity: 0.0527343814028427
  test_usage_entropy: 0.9990808963775635
  test_eff_frac_min: 1.0
  test_expert_align_m: 0.018154282122850418
  test_expert_diversity: 0.0002826043637469411

[hard_cold]
  beta: 0.5253468657938833
  test_best_mse: 0.8574603796005249
  test_class_mi_norm: 0.0
  test_class_purity: 0.037109375
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0009765625
  test_expert_align_m: 0.009310772642493248
  test_expert_diversity: 0.7948613166809082

[fixed_class]
  beta: 0.14009249754503555
  test_best_mse: 0.8559088706970215
  test_class_mi_norm: 0.0
  test_class_purity: 0.0234375
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0010023070499300957
  test_expert_align_m: 0.008717229589819908
  test_expert_diversity: 0.6824442744255066

[fixed_good]
  beta: 0.3502312438625889
  test_best_mse: 0.8564081192016602
  test_class_mi_norm: 0.0
  test_class_purity: 0.078125
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0009765625
  test_expert_align_m: 0.009158642031252384
  test_expert_diversity: 0.7778480648994446

[theory_anneal]
  beta: 0.3502312438625889
  test_best_mse: 0.8555857539176941
  test_class_mi_norm: 0.0
  test_class_purity: 0.0234375
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.03534277155995369
  test_expert_align_m: 0.00824721995741129
  test_expert_diversity: 0.22854985296726227
```

## 6. `scripts/rf_mcl_cifar_pca/SUMMARY.txt`

```text
RF-MCL CIFAR expert training v2
{
  "data_root": "./data",
  "classes": [
    "automobile",
    "horse"
  ],
  "d_mode": "pca",
  "pca_dim": 512,
  "rp_dim": 512,
  "p": 512,
  "K": 4,
  "t": 1.5,
  "n_train": 6000,
  "n_test": 2000,
  "batch_size": 192,
  "steps": 3000,
  "lr": 0.002,
  "weight_decay": 0.0,
  "init_std": 0.001,
  "seed": 0,
  "calib_steps": 900,
  "warmup_steps": 600,
  "ramp_steps": 900,
  "eval_every": 200,
  "power_iters": 20,
  "ridge": 1e-05,
  "beta_target_mult": 3.0,
  "beta_glass_safety": 0.45,
  "cold_mult_glass": 1.2,
  "device": "auto",
  "outdir": "./rf_mcl_cifar_pca",
  "no_download": false,
  "beta_grid": false
}

Data info:
{
  "classes": [
    "automobile",
    "horse"
  ],
  "raw_dim": 3072,
  "global_std": 0.5098379254341125,
  "d_mode": "pca",
  "d": 512
}

Channels:
  lambda_free: 172.7288055419922
  lambda_class: 2.1545116901397705
  lambda_trans: 2.3518309593200684
  beta_free: 0.0028947111538870874
  beta_class: 0.23207114739184695
  beta_trans: 0.21260031381866296
  free_align_m: 0.045483365654945374

Glass:
  alpha: 0.015637436655566887
  r_t: 0.05239569649125595
  v_emp: 0.47297659516334534
  beta_glass_emp: 0.25714492307089903
  E_mean_per_d: 0.10487639904022217

Chosen beta_target: 0.6962134421755408
WARNING: class/glass window narrow or absent; target uses class multiplier.

Final metrics by variant:

[uniform]
  beta: 0.0
  test_best_mse: 0.3421788811683655
  test_class_mi_norm: 0.0009837179677560925
  test_class_purity: 0.033203136816155165
  test_usage_entropy: 0.9992148876190186
  test_eff_frac_min: 1.0
  test_expert_align_m: 0.04523086175322533
  test_expert_diversity: 0.000317790632834658

[hard_cold]
  beta: 1.0443201632633112
  test_best_mse: 0.33721041679382324
  test_class_mi_norm: 0.0
  test_class_purity: 0.046875
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0009765625
  test_expert_align_m: 0.04390304163098335
  test_expert_diversity: 0.8234187364578247

[fixed_class]
  beta: 0.27848537687021635
  test_best_mse: 0.3307521641254425
  test_class_mi_norm: 0.0
  test_class_purity: 0.044921875
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0009765625
  test_expert_align_m: 0.04368682950735092
  test_expert_diversity: 0.7423891425132751

[fixed_good]
  beta: 0.6962134421755408
  test_best_mse: 0.33436885476112366
  test_class_mi_norm: 0.0
  test_class_purity: 0.017578125
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.0009765625
  test_expert_align_m: 0.04376531019806862
  test_expert_diversity: 0.8049576282501221

[theory_anneal]
  beta: 0.6962134421755408
  test_best_mse: 0.3204823434352875
  test_class_mi_norm: 0.0
  test_class_purity: 0.021484375
  test_usage_entropy: -0.0
  test_eff_frac_min: 0.000977285555563867
  test_expert_align_m: 0.0428081713616848
  test_expert_diversity: 0.3459501564502716
```

