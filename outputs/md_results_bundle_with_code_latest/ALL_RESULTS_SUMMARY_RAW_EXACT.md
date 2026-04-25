# ALL RESULTS SUMMARY (RAW EXACT)

Total files: 4

## 1. `outputs/rf_mcl_results/gmm_v5_quick/SUMMARY.txt`

```text
RF-MCL GMM v5 router (target=eps, residualized)
{
  "d": 64,
  "p": 256,
  "C": 4,
  "K": 4,
  "mu": 1.0,
  "sigma0": 0.5,
  "t": 2.05,
  "n_train": 2500,
  "n_test": 1200,
  "n_calib": 1200,
  "batch_size": 256,
  "steps": 1500,
  "lr": 0.003,
  "init_std": 0.001,
  "activation": "erf",
  "seed": 0,
  "warmup_steps": 800,
  "ramp_steps": 1000,
  "eval_every": 150,
  "power_iters": 30,
  "ridge": 1e-05,
  "router_steps": 600,
  "router_lr": 0.002,
  "router_batch_size": 256,
  "beta_grid": true,
  "quick": true,
  "device": "cuda",
  "outdir": "./outputs/rf_mcl_results/gmm_v5_quick"
}

Calibration:
  lambda_free: 0.032435815781354904
  lambda_class: 0.7549533247947693
  lambda_trans: 0.19845962524414062
  beta_free: 15.4150585689112
  beta_class: 0.6622925995271599
  beta_trans: 2.519404132628544
  beta_glass_emp: 43.1190292929964
  alpha_log_n_over_d: 0.11078245055900143
  residual_mse: 0.03698962926864624
  v_emp: 0.00011916892253793776

Final router metrics:

[uniform] beta_eval=0.6622925995271599
  oracle_best_mse: 0.05286922678351402
  soft_oracle_mse: 0.052872709929943085
  mean_expert_mse: 0.052872709929943085
  router_mix_mse: 0.05287270247936249
  router_soft_mse: 0.052872709929943085
  oracle_class_mi_norm: 0.004262005346577206
  router_class_mi_norm: 0.006515003645017515
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 2.3153334804959513e-09
  teacher_status: near_uniform (router likely uninformative)

[fixed_class] beta_eval=0.7947511194325919
  oracle_best_mse: 0.052856624126434326
  soft_oracle_mse: 0.052860062569379807
  mean_expert_mse: 0.052860062569379807
  router_mix_mse: 0.052860062569379807
  router_soft_mse: 0.052860062569379807
  oracle_class_mi_norm: 0.0009116130026538501
  router_class_mi_norm: 0.004569649160162039
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 4.0552396995963136e-09
  teacher_status: near_uniform (router likely uninformative)

[fixed_good] beta_eval=1.9868777985814798
  oracle_best_mse: 0.05283534526824951
  soft_oracle_mse: 0.05283890292048454
  mean_expert_mse: 0.05283890292048454
  router_mix_mse: 0.05283890292048454
  router_soft_mse: 0.05283890292048454
  oracle_class_mi_norm: 0.0024129711240249412
  router_class_mi_norm: 0.002819953157121378
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: -1.4516279289722434e-09
  teacher_status: near_uniform (router likely uninformative)

[hard_cold] beta_eval=51.74283515159568
  oracle_best_mse: 0.05289638042449951
  soft_oracle_mse: 0.052904028445482254
  mean_expert_mse: 0.052904028445482254
  router_mix_mse: 0.05290402099490166
  router_soft_mse: 0.052904028445482254
  oracle_class_mi_norm: 0.011061521312620034
  router_class_mi_norm: 0.01157616640738589
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 1.216797471670361e-07
  teacher_status: near_uniform (router likely uninformative)

[theory_anneal] beta_eval=1.9868777985814798
  oracle_best_mse: 0.052894849330186844
  soft_oracle_mse: 0.05289826914668083
  mean_expert_mse: 0.05289826914668083
  router_mix_mse: 0.05289826542139053
  router_soft_mse: 0.05289826914668083
  oracle_class_mi_norm: 0.0030442201323478127
  router_class_mi_norm: 0.00637081264570615
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 3.711742690981623e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_0.200] beta_eval=0.2
  oracle_best_mse: 0.052852265536785126
  soft_oracle_mse: 0.05285574123263359
  mean_expert_mse: 0.05285574123263359
  router_mix_mse: 0.05285574123263359
  router_soft_mse: 0.05285574123263359
  oracle_class_mi_norm: 0.0013350244930957088
  router_class_mi_norm: 0.009354601524843647
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 3.4277678473415563e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_0.400] beta_eval=0.4
  oracle_best_mse: 0.05284544825553894
  soft_oracle_mse: 0.052848901599645615
  mean_expert_mse: 0.052848901599645615
  router_mix_mse: 0.05284889414906502
  router_soft_mse: 0.052848901599645615
  oracle_class_mi_norm: 0.0031996008922854035
  router_class_mi_norm: 0.0015785027321786411
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 3.054065889074309e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_0.700] beta_eval=0.7
  oracle_best_mse: 0.0528658889234066
  soft_oracle_mse: 0.05286918953061104
  mean_expert_mse: 0.05286918953061104
  router_mix_mse: 0.05286918953061104
  router_soft_mse: 0.05286918953061104
  oracle_class_mi_norm: 0.0016092310682167211
  router_class_mi_norm: 0.002247374013852587
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 1.639749669379853e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_1.100] beta_eval=1.1
  oracle_best_mse: 0.05289189890027046
  soft_oracle_mse: 0.052895303815603256
  mean_expert_mse: 0.052895303815603256
  router_mix_mse: 0.05289530009031296
  router_soft_mse: 0.052895303815603256
  oracle_class_mi_norm: 0.002333284639590988
  router_class_mi_norm: 0.003531764776006653
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 1.4543898307906034e-09
  teacher_status: near_uniform (router likely uninformative)
```

## 2. `outputs/rf_mcl_results/cifar_v5_quick/SUMMARY.txt`

```text
RF-MCL CIFAR v5 router (target=eps, residualized)
{
  "data_root": "./data",
  "classes": "automobile,horse",
  "d_mode": "pca",
  "pca_dim": 256,
  "rp_dim": 512,
  "p": 256,
  "K": 4,
  "t": 1.5,
  "n_train": 3000,
  "n_test": 1000,
  "n_calib": 1000,
  "batch_size": 192,
  "steps": 1200,
  "lr": 0.002,
  "init_std": 0.001,
  "activation": "erf",
  "seed": 0,
  "warmup_steps": 600,
  "ramp_steps": 900,
  "eval_every": 150,
  "power_iters": 30,
  "ridge": 1e-05,
  "router_steps": 500,
  "router_lr": 0.002,
  "router_batch_size": 192,
  "beta_grid": true,
  "quick": true,
  "no_download": false,
  "device": "cuda",
  "outdir": "./outputs/rf_mcl_results/cifar_v5_quick"
}

Data info:
{
  "classes": [
    "automobile",
    "horse"
  ],
  "raw_dim": 3072,
  "global_std": 0.2588672935962677,
  "train_n": 3000,
  "test_n": 1000,
  "d_mode": "pca",
  "d_latent": 256,
  "p": 256,
  "K": 4,
  "t": 1.5,
  "target": "eps"
}

Calibration:
  lambda_free: 0.0841887965798378
  lambda_class: 0.9163100719451904
  lambda_trans: 1.1533339023590088
  beta_free: 5.9390325115983895
  beta_class: 0.5456668166252654
  beta_trans: 0.4335257976699626
  beta_glass_emp: 7.644205959364432
  alpha_log_n_over_d: 0.026983419058523972
  residual_mse: 0.21530278027057648
  v_emp: 0.0009235538309440017

Final router metrics:

[uniform] beta_eval=0.5456668166252654
  oracle_best_mse: 0.314144104719162
  soft_oracle_mse: 0.3141525983810425
  mean_expert_mse: 0.3141525983810425
  router_mix_mse: 0.3141525983810425
  router_soft_mse: 0.3141525983810425
  oracle_class_mi_norm: 0.0019137840510227984
  router_class_mi_norm: 0.0020103377235680714
  teacher_entropy_norm: 1.0000001192092896
  router_vs_teacher_ce: 1.3862944841384888
  router_vs_teacher_kl: -1.3343240401475498e-10
  teacher_status: near_uniform (router likely uninformative)

[fixed_class] beta_eval=0.6548001799503185
  oracle_best_mse: 0.31402069330215454
  soft_oracle_mse: 0.3140294849872589
  mean_expert_mse: 0.3140294849872589
  router_mix_mse: 0.3140294849872589
  router_soft_mse: 0.3140294849872589
  oracle_class_mi_norm: 0.008102604980233666
  router_class_mi_norm: 0.0006657645691993872
  teacher_entropy_norm: 1.0000001192092896
  router_vs_teacher_ce: 1.3862944841384888
  router_vs_teacher_kl: -1.0790113247338695e-09
  teacher_status: near_uniform (router likely uninformative)

[fixed_good] beta_eval=1.6370004498757962
  oracle_best_mse: 0.3141254484653473
  soft_oracle_mse: 0.31413429975509644
  mean_expert_mse: 0.31413429975509644
  router_mix_mse: 0.31413429975509644
  router_soft_mse: 0.31413429975509644
  oracle_class_mi_norm: 0.002657114787682777
  router_class_mi_norm: 0.00011800572289679685
  teacher_entropy_norm: 1.0000001192092896
  router_vs_teacher_ce: 1.3862944841384888
  router_vs_teacher_kl: -2.6858468782364753e-09
  teacher_status: near_uniform (router likely uninformative)

[hard_cold] beta_eval=9.173047151237318
  oracle_best_mse: 0.31416523456573486
  soft_oracle_mse: 0.31417423486709595
  mean_expert_mse: 0.31417423486709595
  router_mix_mse: 0.31417420506477356
  router_soft_mse: 0.31417423486709595
  oracle_class_mi_norm: 0.0004628783163010913
  router_class_mi_norm: 0.005623688337065589
  teacher_entropy_norm: 1.0000001192092896
  router_vs_teacher_ce: 1.3862944841384888
  router_vs_teacher_kl: 1.8962259318300312e-08
  teacher_status: near_uniform (router likely uninformative)

[theory_anneal] beta_eval=1.6370004498757962
  oracle_best_mse: 0.314110666513443
  soft_oracle_mse: 0.31411927938461304
  mean_expert_mse: 0.31411927938461304
  router_mix_mse: 0.31411927938461304
  router_soft_mse: 0.31411927938461304
  oracle_class_mi_norm: 0.0009855010322856969
  router_class_mi_norm: 0.00726398476648071
  teacher_entropy_norm: 1.0000001192092896
  router_vs_teacher_ce: 1.3862944841384888
  router_vs_teacher_kl: -1.0549773277190866e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_0.200] beta_eval=0.2
  oracle_best_mse: 0.31414735317230225
  soft_oracle_mse: 0.31415584683418274
  mean_expert_mse: 0.31415584683418274
  router_mix_mse: 0.31415584683418274
  router_soft_mse: 0.3141558766365051
  oracle_class_mi_norm: 0.0020188683328832406
  router_class_mi_norm: 0.0005527666076801537
  teacher_entropy_norm: 1.0000001192092896
  router_vs_teacher_ce: 1.3862944841384888
  router_vs_teacher_kl: 6.581035161268289e-10
  teacher_status: near_uniform (router likely uninformative)

[grid_0.400] beta_eval=0.4
  oracle_best_mse: 0.31415170431137085
  soft_oracle_mse: 0.3141605854034424
  mean_expert_mse: 0.3141605854034424
  router_mix_mse: 0.3141605854034424
  router_soft_mse: 0.3141605854034424
  oracle_class_mi_norm: 0.0012125406216939865
  router_class_mi_norm: 0.006883877432941737
  teacher_entropy_norm: 1.0000001192092896
  router_vs_teacher_ce: 1.3862944841384888
  router_vs_teacher_kl: 1.1411206424227771e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_0.700] beta_eval=0.7
  oracle_best_mse: 0.31400954723358154
  soft_oracle_mse: 0.3140180706977844
  mean_expert_mse: 0.3140180706977844
  router_mix_mse: 0.3140180706977844
  router_soft_mse: 0.3140180706977844
  oracle_class_mi_norm: 0.0010854191924823025
  router_class_mi_norm: 0.0012618185818984635
  teacher_entropy_norm: 1.0000001192092896
  router_vs_teacher_ce: 1.3862944841384888
  router_vs_teacher_kl: -1.2253029701980722e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_1.100] beta_eval=1.1
  oracle_best_mse: 0.3140150308609009
  soft_oracle_mse: 0.31402382254600525
  mean_expert_mse: 0.31402385234832764
  router_mix_mse: 0.31402382254600525
  router_soft_mse: 0.31402382254600525
  oracle_class_mi_norm: 0.0039465733065444045
  router_class_mi_norm: 0.0033471669866953213
  teacher_entropy_norm: 1.0000001192092896
  router_vs_teacher_ce: 1.3862944841384888
  router_vs_teacher_kl: 6.536959168412793e-11
  teacher_status: near_uniform (router likely uninformative)
```

## 3. `outputs/rf_mcl_results/gmm_v5_serious/SUMMARY.txt`

```text
RF-MCL GMM v5 router (target=eps, residualized)
{
  "d": 64,
  "p": 256,
  "C": 4,
  "K": 4,
  "mu": 1.0,
  "sigma0": 0.5,
  "t": 2.05,
  "n_train": 6000,
  "n_test": 3000,
  "n_calib": 2500,
  "batch_size": 256,
  "steps": 4000,
  "lr": 0.003,
  "init_std": 0.001,
  "activation": "erf",
  "seed": 0,
  "warmup_steps": 800,
  "ramp_steps": 1000,
  "eval_every": 200,
  "power_iters": 30,
  "ridge": 1e-05,
  "router_steps": 1200,
  "router_lr": 0.002,
  "router_batch_size": 256,
  "beta_grid": true,
  "quick": false,
  "device": "cuda",
  "outdir": "./outputs/rf_mcl_results/gmm_v5_serious"
}

Calibration:
  lambda_free: 0.02505200356245041
  lambda_class: 0.5840945243835449
  lambda_trans: 0.17778773605823517
  beta_free: 19.95848351025436
  beta_class: 0.8560258299419969
  beta_trans: 2.812342465715536
  beta_glass_emp: 36.63802155673225
  alpha_log_n_over_d: 0.12225071891962956
  residual_mse: 0.042292285710573196
  v_emp: 0.00018214505689684302

Final router metrics:

[uniform] beta_eval=0.8560258299419969
  oracle_best_mse: 0.04944416135549545
  soft_oracle_mse: 0.04944484308362007
  mean_expert_mse: 0.04944484308362007
  router_mix_mse: 0.04944484308362007
  router_soft_mse: 0.04944484308362007
  oracle_class_mi_norm: 0.0008075687568428219
  router_class_mi_norm: 0.008158110054111607
  teacher_entropy_norm: 0.9999998807907104
  router_vs_teacher_ce: 1.3862942457199097
  router_vs_teacher_kl: 2.883699390388017e-10
  teacher_status: near_uniform (router likely uninformative)

[fixed_class] beta_eval=1.0272309959303962
  oracle_best_mse: 0.04941752925515175
  soft_oracle_mse: 0.049418188631534576
  mean_expert_mse: 0.049418188631534576
  router_mix_mse: 0.049418188631534576
  router_soft_mse: 0.049418188631534576
  oracle_class_mi_norm: 0.0006099037146625612
  router_class_mi_norm: 0.002723298000543769
  teacher_entropy_norm: 0.9999998807907104
  router_vs_teacher_ce: 1.3862942457199097
  router_vs_teacher_kl: 9.540539469554687e-10
  teacher_status: near_uniform (router likely uninformative)

[fixed_good] beta_eval=2.568077489825991
  oracle_best_mse: 0.0494086816906929
  soft_oracle_mse: 0.049409378319978714
  mean_expert_mse: 0.049409378319978714
  router_mix_mse: 0.049409378319978714
  router_soft_mse: 0.049409378319978714
  oracle_class_mi_norm: 0.0005177632548445532
  router_class_mi_norm: 0.011563066682020659
  teacher_entropy_norm: 0.9999998807907104
  router_vs_teacher_ce: 1.3862942457199097
  router_vs_teacher_kl: -5.04095487574574e-10
  teacher_status: near_uniform (router likely uninformative)

[hard_cold] beta_eval=43.965625868078696
  oracle_best_mse: 0.049453724175691605
  soft_oracle_mse: 0.04945540428161621
  mean_expert_mse: 0.04945540428161621
  router_mix_mse: 0.04945540428161621
  router_soft_mse: 0.04945540428161621
  oracle_class_mi_norm: 0.0014963102760007262
  router_class_mi_norm: 0.011193221607360158
  teacher_entropy_norm: 0.9999998807907104
  router_vs_teacher_ce: 1.3862942457199097
  router_vs_teacher_kl: 1.1210630646019126e-08
  teacher_status: near_uniform (router likely uninformative)

[theory_anneal] beta_eval=2.568077489825991
  oracle_best_mse: 0.04940691217780113
  soft_oracle_mse: 0.049407582730054855
  mean_expert_mse: 0.049407582730054855
  router_mix_mse: 0.049407582730054855
  router_soft_mse: 0.049407582730054855
  oracle_class_mi_norm: 0.0013911909078354063
  router_class_mi_norm: 0.009594670834869338
  teacher_entropy_norm: 0.9999998807907104
  router_vs_teacher_ce: 1.3862942457199097
  router_vs_teacher_kl: -2.6401807406983835e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_0.200] beta_eval=0.2
  oracle_best_mse: 0.04941989481449127
  soft_oracle_mse: 0.04942058399319649
  mean_expert_mse: 0.04942058399319649
  router_mix_mse: 0.04942058399319649
  router_soft_mse: 0.04942058399319649
  oracle_class_mi_norm: 0.0008854982896779873
  router_class_mi_norm: 0.012021324933539555
  teacher_entropy_norm: 0.9999998807907104
  router_vs_teacher_ce: 1.3862942457199097
  router_vs_teacher_kl: -2.1358133039939275e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_0.400] beta_eval=0.4
  oracle_best_mse: 0.04943924397230148
  soft_oracle_mse: 0.04943990707397461
  mean_expert_mse: 0.04943990707397461
  router_mix_mse: 0.04943990707397461
  router_soft_mse: 0.04943990707397461
  oracle_class_mi_norm: 0.0015570140528183582
  router_class_mi_norm: 0.011320026665204916
  teacher_entropy_norm: 0.9999998807907104
  router_vs_teacher_ce: 1.3862942457199097
  router_vs_teacher_kl: -2.493397266434272e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_0.700] beta_eval=0.7
  oracle_best_mse: 0.049394164234399796
  soft_oracle_mse: 0.04939482733607292
  mean_expert_mse: 0.04939482733607292
  router_mix_mse: 0.04939482733607292
  router_soft_mse: 0.04939482733607292
  oracle_class_mi_norm: 0.001043436089632928
  router_class_mi_norm: 0.005234640934970383
  teacher_entropy_norm: 0.9999998807907104
  router_vs_teacher_ce: 1.3862942457199097
  router_vs_teacher_kl: -4.369170025775304e-10
  teacher_status: near_uniform (router likely uninformative)

[grid_1.100] beta_eval=1.1
  oracle_best_mse: 0.049420975148677826
  soft_oracle_mse: 0.049421656876802444
  mean_expert_mse: 0.049421656876802444
  router_mix_mse: 0.04942164942622185
  router_soft_mse: 0.049421656876802444
  oracle_class_mi_norm: 0.0005521806363132509
  router_class_mi_norm: 0.014672985474924129
  teacher_entropy_norm: 0.9999998807907104
  router_vs_teacher_ce: 1.3862942457199097
  router_vs_teacher_kl: -4.664377772911621e-10
  teacher_status: near_uniform (router likely uninformative)
```

## 4. `outputs/rf_mcl_results/cifar_v5_serious/SUMMARY.txt`

```text
RF-MCL CIFAR v5 router (target=eps, residualized)
{
  "data_root": "./data",
  "classes": "automobile,horse",
  "d_mode": "pca",
  "pca_dim": 512,
  "rp_dim": 512,
  "p": 512,
  "K": 4,
  "t": 1.5,
  "n_train": 8000,
  "n_test": 2500,
  "n_calib": 2048,
  "batch_size": 192,
  "steps": 3000,
  "lr": 0.002,
  "init_std": 0.001,
  "activation": "erf",
  "seed": 0,
  "warmup_steps": 600,
  "ramp_steps": 900,
  "eval_every": 200,
  "power_iters": 30,
  "ridge": 1e-05,
  "router_steps": 1000,
  "router_lr": 0.002,
  "router_batch_size": 192,
  "beta_grid": true,
  "quick": false,
  "no_download": false,
  "device": "cuda",
  "outdir": "./outputs/rf_mcl_results/cifar_v5_serious"
}

Data info:
{
  "classes": [
    "automobile",
    "horse"
  ],
  "raw_dim": 3072,
  "global_std": 0.2596452534198761,
  "train_n": 8000,
  "test_n": 2000,
  "d_mode": "pca",
  "d_latent": 512,
  "p": 512,
  "K": 4,
  "t": 1.5,
  "target": "eps"
}

Calibration:
  lambda_free: 0.08049505203962326
  lambda_class: 2.4786269664764404
  lambda_trans: 0.9780968427658081
  beta_free: 6.211561920028049
  beta_class: 0.20172458654026046
  beta_trans: 0.5111968244229556
  beta_glass_emp: 7.603144665588635
  alpha_log_n_over_d: 0.014891833957342575
  residual_mse: 0.21696174144744873
  v_emp: 0.0005152187659405172

Final router metrics:

[uniform] beta_eval=0.20172458654026046
  oracle_best_mse: 0.30773627758026123
  soft_oracle_mse: 0.3077402114868164
  mean_expert_mse: 0.3077402114868164
  router_mix_mse: 0.3077402114868164
  router_soft_mse: 0.3077402114868164
  oracle_class_mi_norm: 0.0001205324232909023
  router_class_mi_norm: 0.0004921255508852326
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 8.647338400891158e-10
  teacher_status: near_uniform (router likely uninformative)

[fixed_class] beta_eval=0.24206950384831255
  oracle_best_mse: 0.3078598082065582
  soft_oracle_mse: 0.30786362290382385
  mean_expert_mse: 0.30786362290382385
  router_mix_mse: 0.30786362290382385
  router_soft_mse: 0.30786362290382385
  oracle_class_mi_norm: 0.001174200525426708
  router_class_mi_norm: 0.0010700017894338656
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: -2.526959475002144e-10
  teacher_status: near_uniform (router likely uninformative)

[fixed_good] beta_eval=0.6051737596207813
  oracle_best_mse: 0.30777937173843384
  soft_oracle_mse: 0.3077832758426666
  mean_expert_mse: 0.3077832758426666
  router_mix_mse: 0.3077832758426666
  router_soft_mse: 0.3077832758426666
  oracle_class_mi_norm: 8.111188817616112e-05
  router_class_mi_norm: 0.0006231719099415347
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: -3.089829503366559e-10
  teacher_status: near_uniform (router likely uninformative)

[hard_cold] beta_eval=9.123773598706363
  oracle_best_mse: 0.30773305892944336
  soft_oracle_mse: 0.30773693323135376
  mean_expert_mse: 0.30773693323135376
  router_mix_mse: 0.30773690342903137
  router_soft_mse: 0.30773693323135376
  oracle_class_mi_norm: 0.0007093398418786432
  router_class_mi_norm: 0.0030450295513917594
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 1.3495499828763968e-08
  teacher_status: near_uniform (router likely uninformative)

[theory_anneal] beta_eval=0.6051737596207813
  oracle_best_mse: 0.3077799081802368
  soft_oracle_mse: 0.307783842086792
  mean_expert_mse: 0.307783842086792
  router_mix_mse: 0.307783842086792
  router_soft_mse: 0.307783842086792
  oracle_class_mi_norm: 0.0008481055200273808
  router_class_mi_norm: 0.0008735405035447927
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: 2.8707683452644517e-10
  teacher_status: near_uniform (router likely uninformative)

[grid_0.200] beta_eval=0.2
  oracle_best_mse: 0.3077773451805115
  soft_oracle_mse: 0.3077811300754547
  mean_expert_mse: 0.3077811300754547
  router_mix_mse: 0.3077811300754547
  router_soft_mse: 0.3077811300754547
  oracle_class_mi_norm: 0.0012338530594058292
  router_class_mi_norm: 0.0009179827862355511
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: -1.0724342525136876e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_0.400] beta_eval=0.4
  oracle_best_mse: 0.3078039586544037
  soft_oracle_mse: 0.3078078031539917
  mean_expert_mse: 0.3078078031539917
  router_mix_mse: 0.3078078031539917
  router_soft_mse: 0.3078078031539917
  oracle_class_mi_norm: 0.000581349676051375
  router_class_mi_norm: 0.0016891136407756952
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: -6.986971556877108e-10
  teacher_status: near_uniform (router likely uninformative)

[grid_0.700] beta_eval=0.7
  oracle_best_mse: 0.3077862858772278
  soft_oracle_mse: 0.30779021978378296
  mean_expert_mse: 0.30779021978378296
  router_mix_mse: 0.30779021978378296
  router_soft_mse: 0.30779021978378296
  oracle_class_mi_norm: 0.0011222895803127363
  router_class_mi_norm: 0.0011494290873699173
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: -1.5296235389428148e-09
  teacher_status: near_uniform (router likely uninformative)

[grid_1.100] beta_eval=1.1
  oracle_best_mse: 0.30775684118270874
  soft_oracle_mse: 0.30776068568229675
  mean_expert_mse: 0.30776068568229675
  router_mix_mse: 0.30776065587997437
  router_soft_mse: 0.30776068568229675
  oracle_class_mi_norm: 0.00138050315910529
  router_class_mi_norm: 0.0034862046197707405
  teacher_entropy_norm: 1.0
  router_vs_teacher_ce: 1.3862943649291992
  router_vs_teacher_kl: -8.515252947205454e-10
  teacher_status: near_uniform (router likely uninformative)
```
