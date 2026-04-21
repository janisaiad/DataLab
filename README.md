# Mixture Score-Based Diffusion Models with Multiple Choice Learning

This repository implements a **score-based diffusion model on MNIST** and extends it with
**Multiple Choice Learning (MCL)** over $K$ expert score networks trained under a
Winner-Takes-All rule. We study and mitigate the **expert collapse** of hard WTA
(three variants: Annealed WTA, Relaxed WTA, Resilient MCL), compare **five inference routing
strategies** for the multi-expert sampler, and analyse the resulting **quality vs. diversity**
trade-off (FID, Precision, Recall + inter- vs. intra-class diversity).

The five deliverables required by the assignment are mapped directly to the code:

| Deliverable                                       | Where                                                                 |
|---|---|
| 1. Baseline single-model diffusion                 | `src/model.py` (`ScoreNet`), `src/train.py --mode baseline`, `src/diffusion.py` (Euler & Heun) |
| 2. MCL with $K\ge 2$ experts (4 training variants) | `src/train.py --mode mcl --mcl_variant {hard,annealed,relaxed,resilient}_wta` |
| 3. Empirical study of expert specialization        | `src/analyze.py` → `outputs/<tag>/analysis/*.png`                     |
| 4. Five multi-expert inference strategies          | `src/sample.py`, `src/gating.py`                                      |
| 5. Quality vs. diversity (FID, P, R + extras)      | `src/evaluate.py`, `tools/make_outputs_extra.py`                      |

Headline result on MNIST (annealed WTA, $K=4$, 2,048 samples, 200 ODE steps, A100):
**FID 88.3 with the learned gating**, vs. 111.4 for the monolithic baseline (Heun) — a
23-point FID improvement at higher precision (0.64 vs. 0.43). Full tables under
[Results](#results).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+. MNIST is downloaded automatically on first run.
GPU (CUDA / MPS) is auto-detected with CPU fallback.

## Quick start — reproduce the main experiment

The "main experiment" of the assignment — baseline + K=4 annealed WTA + gating + 5 inference
strategies + FID/P/R + analysis plots — runs in a single command:

```bash
python run_pipeline.py        # subprocess-based, mirrors the manual step-by-step flow
# or, faster on GPU (single process, AMP, models stay loaded across stages):
python run_all.py
```

This produces, under ~30 min on an A100:

- `checkpoints/shared/baseline_final.pt`, `checkpoints/shared/mnist_classifier.pt`
- `checkpoints/annealed_wta_K4/{mcl_K4_final.pt, gating_K4.pt, mcl_K4_log.json}`
- `outputs/annealed_wta_K4/metrics.json` (FID, Precision, Recall for the 7 strategies)
- `outputs/annealed_wta_K4/{baseline_euler,mcl_<strategy>}.png` (sample grids)
- `outputs/annealed_wta_K4/analysis/{expert_vs_digit,expert_vs_sigma,multi_expert_grid,trajectory,strategy_comparison}.png`

Cross-variant ablations (hard / relaxed / resilient WTA, K-sweep, batch-size sweep) are
launched with `python run_variant.py --variant <variant> [--K <K>] [--batch_size <B>]`,
or via the SLURM submission scripts in `slurm/`. The aggregated FID table is in the
[Results](#results) section.

## Project Structure

```
├── run_pipeline.py    # Subprocess-based full pipeline
├── run_all.py         # Single-process pipeline (same stages, in-memory)
├── report.pdf         # Compiled report
├── requirements.txt   # Dependencies
├── src/
│   ├── model.py       # ScoreNet (U-Net), GatingNet, and ScoringHead architectures
│   ├── diffusion.py   # Noise schedule, forward process, Euler & Heun ODE samplers
│   ├── train.py       # Training: baseline and MCL (4 variants: hard/annealed/relaxed/resilient)
│   ├── sample.py      # Sampling with 5 multi-expert strategies
│   ├── gating.py      # Train a learned gating network for expert routing
│   ├── evaluate.py    # FID, Precision, and Recall metrics
│   ├── analyze.py     # Specialization analysis and visualization
│   └── utils.py       # Data loading, image grids, seeding, EMA
├── checkpoints/       # Saved model weights (gitignored)
└── outputs/           # Generated images, metrics, analysis plots (gitignored)
    ├── metrics.json
    ├── training_curves.png
    ├── expert_usage_training.png
    ├── metrics_comparison.png
    └── analysis/
        ├── expert_vs_digit.png
        ├── expert_vs_sigma.png
        ├── multi_expert_grid.png
        ├── trajectory.png
        └── strategy_comparison.png
```

## Step-by-Step Reproduction

Each script auto-detects CUDA/MPS. Use `--device cpu` to force CPU.

### 1. Train the baseline diffusion model

```bash
python -m src.train --mode baseline --epochs 200 --base_ch 32 --ch_mult 1 2 \
    --time_dim 128 --lr 3e-4 --ema_decay 0.999 --out_dir checkpoints
```

### 2. Train MCL with K=4 experts

Choose a variant with `--mcl_variant`:

```bash
# Annealed Winner-Takes-All
python -m src.train --mode mcl --K 4 --mcl_variant annealed_wta --epochs 200 \
    --base_ch 32 --ch_mult 1 2 --time_dim 128 --lr 3e-4 --ema_decay 0.999

# Hard Winner-Takes-All
python -m src.train --mode mcl --K 4 --mcl_variant hard_wta --epochs 200 ...

# Relaxed Winner-Takes-All
python -m src.train --mode mcl --K 4 --mcl_variant relaxed_wta --relaxed_alpha 0.1 --epochs 200 ...

# Resilient MCL
python -m src.train --mode mcl --K 4 --mcl_variant resilient_mcl --epochs 200 ...
```

### 3. Train the gating network

```bash
python -m src.gating --mcl_ckpt checkpoints/mcl_K4_final.pt \
    --epochs 25 --collect_batches 80
```

### 4. Generate samples

```bash
# Baseline (Euler and Heun solvers)
python -m src.sample --checkpoint checkpoints/baseline_final.pt \
    --mode baseline --solver euler --num_samples 2048
python -m src.sample --checkpoint checkpoints/baseline_final.pt \
    --mode baseline --solver heun --num_samples 2048

# MCL, single expert
python -m src.sample --checkpoint checkpoints/mcl_K4_final.pt \
    --mode mcl --strategy single_expert --expert_id 0 --num_samples 2048

# MCL, learned gating
python -m src.sample --checkpoint checkpoints/mcl_K4_final.pt \
    --mode mcl --strategy gated --gating_ckpt checkpoints/gating_K4.pt \
    --num_samples 2048

# Other strategies: random_expert, best_expert, mixture_score
```

### 5. Evaluate (FID, Precision, Recall)

```bash
python -m src.evaluate --samples_pt outputs/mcl_K4_gated_euler_n2048.pt
```

### 6. Analyze expert specialization

```bash
python -m src.analyze --mcl_ckpt checkpoints/mcl_K4_final.pt \
    --out_dir outputs/analysis
```

## Method

### Score-Based Diffusion (Baseline)

A U-Net (`ScoreNet`) is trained to predict the noise $\epsilon$ added to data via a variance-exploding forward process $x_t = x_0 + \sigma\epsilon$, with $\sigma$ sampled log-uniformly from $[\sigma_{\min}, \sigma_{\max}]$. The loss is denoising score matching (MSE on noise prediction). Sampling solves the probability-flow ODE $dx/d\sigma = \epsilon_\theta(x, \sigma)$ from $\sigma_{\max}$ to 0, discretized with Euler or Heun's method.

### Multiple Choice Learning

$K=4$ independent expert networks are trained with a **Winner-Takes-All** rule. Four training variants are supported:

| Variant | Flag | Description |
|---|---|---|
| **Hard WTA** | `hard_wta` | Only the winner gets gradients. Prone to expert collapse. |
| **Annealed WTA** | `annealed_wta` | Soft-to-hard annealing (τ: 10→0.01). All experts train early, competition sharpens gradually. |
| **Relaxed WTA** | `relaxed_wta` | Winner gets weight 1, losers get weight α (default 0.1). |
| **Resilient MCL** | `resilient_mcl` | Learned scoring heads predict expert competence, preventing dead experts. |

With hard WTA, only 1 of 4 experts survives (hypothesis collapse). With annealed WTA, **3 of 4 experts** learn meaningful score functions.

### Inference Routing Strategies

| Strategy | Description |
|---|---|
| `single_expert` | One fixed expert for the entire ODE trajectory |
| `random_expert` | Uniformly random expert at each ODE step |
| `best_expert` | All $K$ experts evaluated; smallest prediction norm wins |
| `mixture_score` | Average predictions of all $K$ experts |
| `gated` | Learned gating network selects expert per step and sample |

## Results

Quantitative evaluation on 2,048 generated samples (Annealed WTA, Euler solver, 200 ODE steps, seed 42, NVIDIA A100-80GB on MesoNet Juliet, SLURM run of 2026-04-13; see `outputs/metrics.json`):

| Strategy | FID ↓ | Precision ↑ | Recall ↑ |
|---|:---:|:---:|:---:|
| Baseline (Euler) | 113.62 | 0.432 | 0.833 |
| Baseline (Heun) | 111.44 | 0.434 | 0.833 |
| Single Expert | 339.12 | 0.627 | 0.901 |
| Random Expert | 81.67 | 0.374 | 0.866 |
| Best Expert | 287.86 | 0.380 | 0.804 |
| Mixture Score | 113.40 | 0.341 | 0.814 |
| **Learned Gating** | **88.25** | **0.644** | **0.836** |

### Cross-variant ablation (FID per routing strategy, K=4)

| Strategy | Hard WTA | Annealed WTA | Relaxed WTA | Resilient MCL |
|---|:---:|:---:|:---:|:---:|
| Baseline (Euler) | 113.62 | 113.62 | 113.62 | 113.62 |
| Baseline (Heun)  | 111.44 | 111.44 | 111.44 | 111.44 |
| Single Expert    |   53.80 | 339.12 | 116.26 |  13.42 |
| Random per-step  | 2169.02 |  81.67 |  94.04 | 2442.67 |
| Best per-step    | 2547.13 | 287.86 | 423.31 | 2549.25 |
| Mixture Score    | 1860.00 | 113.40 | 110.68 | 2445.87 |
| **Learned Gating** | **49.50** | **88.25** | **116.48** | **13.49** |

See `outputs_{hard,relaxed,resilient}_wta/metrics.json` for the raw numbers.

**Key findings:**
- **Annealed WTA prevents expert collapse**: 3 of 4 experts learn meaningful score functions (usage: 40%/31%/28%/0%), vs. only 1 under hard WTA.
- **Learned gating strictly beats the monolithic baseline under annealed WTA** (FID 88.25 vs. 111.44 with Heun, a 23-FID improvement), with Precision 0.644 and Recall 0.836.
- **Global best FID is Resilient MCL + Gating** (FID 13.49), but the near-identical single-expert FID (13.42) reveals that this run has effectively collapsed to one survivor: an upper-bound benchmark, not a true mixture.
- **Per-step heuristics are regime-dependent**: catastrophic under hard/resilient WTA (FID > 1800), on-manifold under annealed/relaxed WTA.
- **FID metric caveat**: the FID values above are computed in the embedding space of a small MNIST classifier (5-epoch CNN), not the ImageNet-pretrained Inception features used elsewhere in the literature. They are therefore internally comparable across strategies but not directly comparable to MNIST FIDs reported in GAN/diffusion papers.

## Hyperparameters (as used in `run_pipeline.py`)

| Parameter | Value | Description |
|---|---|---|
| `base_ch` | 32 | Base channel count of the U-Net |
| `ch_mult` | (1, 2) | Channel multipliers per resolution level |
| `time_dim` | 128 | Time embedding dimension |
| `dropout` | 0.05 | Dropout rate in residual blocks |
| `sigma_min` | 0.01 | Minimum noise level |
| `sigma_max` | 80.0 | Maximum noise level |
| `lr` | 3e-4 | Learning rate (Adam) |
| `ema_decay` | 0.999 | EMA decay for inference weights |
| `K` | 4 | Number of MCL experts |
| `mcl_variant` | `annealed_wta` | MCL training variant |
| `anneal_tau_max` | 10.0 | Initial temperature (soft assignment) |
| `anneal_tau_min` | 0.01 | Final temperature (hard WTA) |
| `epochs` | 200 | Training epochs (baseline & MCL) |
| `batch_size` | 256 | Training batch size |
| `num_steps` | 200 | ODE integration steps at inference |

## Report

See [`report.pdf`](report.pdf) for the full mathematical formulation, implementation details, expert-collapse analysis, quantitative and qualitative results, and the inter-class vs. intra-class diversity discussion.
