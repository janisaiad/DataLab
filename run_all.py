"""Integrated pipeline tuned for GPU (SLURM): AMP + multi-worker loading."""

import os, sys, json, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import ScoreNet, GatingNet
from src.diffusion import get_sigmas, sample_sigma_train, add_noise, euler_sample, heun_sample
from src.utils import get_mnist_loaders, save_image_grid, set_seed, EMA
from src.sample import generate_baseline, generate_mcl
from src.evaluate import evaluate, train_classifier, extract_features, compute_fid, compute_precision_recall
from src.analyze import (
    expert_vs_digit, plot_expert_vs_digit,
    expert_vs_sigma, plot_expert_vs_sigma,
    same_noise_multi_expert, plot_multi_expert_grid,
    compare_strategies, plot_strategy_comparison,
    plot_trajectory,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
K = 4
MCL_VARIANT = "annealed_wta"
TAG = f"{MCL_VARIANT}_K{K}"
SHARED_CKPT = "checkpoints/shared"
CKPT = f"checkpoints/{TAG}"
OUT = f"outputs/{TAG}"
ANALYSIS = f"{OUT}/analysis"

ARCH = dict(base_ch=32, ch_mult=(1, 2), num_res_blocks=2, time_dim=128, dropout=0.05)
SIGMA_MIN, SIGMA_MAX = 0.01, 80.0
ANNEAL_TAU_MAX = 10.0
ANNEAL_TAU_MIN = 0.01
RELAXED_ALPHA = 0.1

BASELINE_EPOCHS = 200
MCL_EPOCHS = 200
GATING_EPOCHS = 25
BATCH_SIZE = 512
LR = 3e-4
EMA_DECAY = 0.999
NUM_STEPS = 200
N_EVAL = 2048
NUM_WORKERS = 4
USE_AMP = True

for d in [SHARED_CKPT, CKPT, OUT, ANALYSIS]:
    os.makedirs(d, exist_ok=True)

set_seed(SEED)
device = torch.device(DEVICE)
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

train_loader, test_loader = get_mnist_loaders(BATCH_SIZE, num_workers=NUM_WORKERS)


print(f"\n{'='*60}\n  1/7  Baseline training ({BASELINE_EPOCHS} epochs)\n{'='*60}")
t0 = time.time()

model = ScoreNet(**ARCH).to(device)
ema = EMA(model, decay=EMA_DECAY)
opt = torch.optim.Adam(model.parameters(), lr=LR)
scaler_bl = GradScaler(enabled=USE_AMP)
print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

bl_log = {"loss": []}
for epoch in range(1, BASELINE_EPOCHS + 1):
    model.train()
    eloss = 0.0
    for images, _ in train_loader:
        x0 = images.to(device, non_blocking=True)
        sigma = sample_sigma_train(x0.shape[0], SIGMA_MIN, SIGMA_MAX, device)
        xt, eps = add_noise(x0, sigma)
        with autocast("cuda", enabled=USE_AMP):
            loss = F.mse_loss(model(xt, sigma), eps)
        opt.zero_grad()
        scaler_bl.scale(loss).backward()
        scaler_bl.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_bl.step(opt); scaler_bl.update()
        ema.update(model)
        eloss += loss.item()
    avg = eloss / len(train_loader)
    bl_log["loss"].append(avg)
    if epoch % 5 == 0 or epoch == BASELINE_EPOCHS:
        samples = generate_baseline(
            ema.shadow, 64, NUM_STEPS, device, SIGMA_MIN, SIGMA_MAX, seed=SEED)
        save_image_grid(samples, f"{SHARED_CKPT}/baseline_samples_ep{epoch}.png")
        print(f"  epoch {epoch:3d}/{BASELINE_EPOCHS}  loss={avg:.4f}")

ckpt_bl = {"model": model.state_dict(), "ema": ema.state_dict(),
           "args": {**ARCH, "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX,
                    "ch_mult": list(ARCH["ch_mult"])}}
torch.save(ckpt_bl, f"{SHARED_CKPT}/baseline_final.pt")
with open(f"{SHARED_CKPT}/baseline_log.json", "w") as f:
    json.dump(bl_log, f)
print(f"  Baseline done in {(time.time()-t0)/60:.1f} min")

baseline_model = copy.deepcopy(ema.shadow).eval()


print(f"\n{'='*60}\n  2/7  MCL training K={K} ({MCL_EPOCHS} epochs, {MCL_VARIANT})\n{'='*60}")
t0 = time.time()

experts = nn.ModuleList([ScoreNet(**ARCH) for _ in range(K)]).to(device)
emas_mcl = [EMA(experts[k], decay=EMA_DECAY) for k in range(K)]
opts = [torch.optim.Adam(experts[k].parameters(), lr=LR) for k in range(K)]
scalers_mcl = [GradScaler(enabled=USE_AMP) for _ in range(K)]
total_p = sum(p.numel() for p in experts.parameters())
print(f"  total params: {total_p:,}  ({total_p//K:,} per expert)")

if MCL_VARIANT == "resilient_mcl":
    from src.model import ScoringHead
    scoring_heads = nn.ModuleList([ScoringHead() for _ in range(K)]).to(device)
    score_opt = torch.optim.Adam(scoring_heads.parameters(), lr=LR)
    total_p += sum(p.numel() for p in scoring_heads.parameters())

mcl_log = {"loss": [], "expert_usage": [], "variant": MCL_VARIANT}
for epoch in range(1, MCL_EPOCHS + 1):
    experts.train()
    eloss = 0.0
    usage = torch.zeros(K)

    if MCL_VARIANT == "annealed_wta":
        progress = (epoch - 1) / max(MCL_EPOCHS - 1, 1)
        tau = ANNEAL_TAU_MAX * (ANNEAL_TAU_MIN / ANNEAL_TAU_MAX) ** progress
    if MCL_VARIANT == "resilient_mcl":
        scoring_heads.train()

    for images, _ in train_loader:
        x0 = images.to(device, non_blocking=True)
        B = x0.shape[0]
        sigma = sample_sigma_train(B, SIGMA_MIN, SIGMA_MAX, device)
        xt, eps = add_noise(x0, sigma)

        with torch.no_grad(), autocast("cuda", enabled=USE_AMP):
            ls = torch.stack([
                (experts[k](xt, sigma) - eps).pow(2).sum(dim=(1,2,3))
                for k in range(K)
            ], dim=1)

        if MCL_VARIANT == "hard_wta":
            winners = ls.argmin(dim=1)
            bloss = 0.0
            for k in range(K):
                mask = winners == k
                n = mask.sum().item()
                if n == 0:
                    continue
                opts[k].zero_grad()
                with autocast("cuda", enabled=USE_AMP):
                    lk = F.mse_loss(experts[k](xt[mask], sigma[mask]), eps[mask])
                scalers_mcl[k].scale(lk).backward()
                scalers_mcl[k].unscale_(opts[k])
                nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                scalers_mcl[k].step(opts[k]); scalers_mcl[k].update()
                emas_mcl[k].update(experts[k])
                bloss += lk.item() * n
                usage[k] += n

        elif MCL_VARIANT == "annealed_wta":
            weights = F.softmax(-ls / (tau + 1e-8), dim=1)
            winners = ls.argmin(dim=1)
            bloss = 0.0
            for k in range(K):
                w_k = weights[:, k]
                if w_k.sum().item() < 1e-8:
                    continue
                opts[k].zero_grad()
                with autocast("cuda", enabled=USE_AMP):
                    eps_pred = experts[k](xt, sigma)
                    per_sample = (eps_pred - eps).pow(2).mean(dim=(1, 2, 3))
                    lk = (w_k * per_sample).sum() / (w_k.sum() + 1e-8)
                scalers_mcl[k].scale(lk).backward()
                scalers_mcl[k].unscale_(opts[k])
                nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                scalers_mcl[k].step(opts[k]); scalers_mcl[k].update()
                emas_mcl[k].update(experts[k])
                bloss += lk.item() * B / K
                usage[k] += (winners == k).sum().item()

        elif MCL_VARIANT == "relaxed_wta":
            winners = ls.argmin(dim=1)
            bloss = 0.0
            for k in range(K):
                mask_win = (winners == k)
                mask_lose = ~mask_win
                n_win, n_lose = mask_win.sum().item(), mask_lose.sum().item()
                if n_win == 0 and n_lose == 0:
                    continue
                opts[k].zero_grad()
                with autocast("cuda", enabled=USE_AMP):
                    eps_pred = experts[k](xt, sigma)
                    per_sample = (eps_pred - eps).pow(2).mean(dim=(1, 2, 3))
                    loss_w = per_sample[mask_win].sum() if n_win > 0 else 0.0
                    loss_l = RELAXED_ALPHA * per_sample[mask_lose].sum() if n_lose > 0 else 0.0
                    total_w = n_win + RELAXED_ALPHA * n_lose
                    lk = (loss_w + loss_l) / (total_w + 1e-8)
                scalers_mcl[k].scale(lk).backward()
                scalers_mcl[k].unscale_(opts[k])
                nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                scalers_mcl[k].step(opts[k]); scalers_mcl[k].update()
                emas_mcl[k].update(experts[k])
                bloss += lk.item() * n_win if n_win > 0 else 0.0
                usage[k] += n_win

        elif MCL_VARIANT == "resilient_mcl":
            with torch.no_grad():
                scores = torch.stack([scoring_heads[k](xt, sigma) for k in range(K)], dim=1)
            winners = scores.argmax(dim=1)
            bloss = 0.0
            for k in range(K):
                mask = winners == k
                n = mask.sum().item()
                if n == 0:
                    continue
                opts[k].zero_grad()
                with autocast("cuda", enabled=USE_AMP):
                    lk = F.mse_loss(experts[k](xt[mask], sigma[mask]), eps[mask])
                scalers_mcl[k].scale(lk).backward()
                scalers_mcl[k].unscale_(opts[k])
                nn.utils.clip_grad_norm_(experts[k].parameters(), 1.0)
                scalers_mcl[k].step(opts[k]); scalers_mcl[k].update()
                emas_mcl[k].update(experts[k])
                bloss += lk.item() * n
                usage[k] += n
            best_expert = ls.argmin(dim=1)
            score_opt.zero_grad()
            with autocast("cuda", enabled=USE_AMP):
                score_logits = torch.stack([scoring_heads[k](xt, sigma) for k in range(K)], dim=1)
                s_loss = F.cross_entropy(score_logits, best_expert)
            s_loss.backward()
            nn.utils.clip_grad_norm_(scoring_heads.parameters(), 1.0)
            score_opt.step()

        eloss += bloss / B

    avg = eloss / len(train_loader)
    uf = (usage / max(usage.sum(), 1)).tolist()
    mcl_log["loss"].append(avg)
    mcl_log["expert_usage"].append(uf)
    if epoch % 5 == 0 or epoch == MCL_EPOCHS:
        extra = f"  τ={tau:.3f}" if MCL_VARIANT == "annealed_wta" else ""
        print(f"  epoch {epoch:3d}/{MCL_EPOCHS}  loss={avg:.4f}  usage={[f'{u:.2f}' for u in uf]}{extra}")

ckpt_mcl = {"experts": experts.state_dict(),
            "emas": [e.state_dict() for e in emas_mcl],
            "args": {**ARCH, "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX,
                     "K": K, "ch_mult": list(ARCH["ch_mult"])},
            "mcl_variant": MCL_VARIANT}
if MCL_VARIANT == "resilient_mcl":
    ckpt_mcl["scoring_heads"] = scoring_heads.state_dict()
torch.save(ckpt_mcl, f"{CKPT}/mcl_K{K}_final.pt")
with open(f"{CKPT}/mcl_K{K}_log.json", "w") as f:
    json.dump(mcl_log, f)
print(f"  MCL ({MCL_VARIANT}) done in {(time.time()-t0)/60:.1f} min")

ema_experts = nn.ModuleList([emas_mcl[k].shadow for k in range(K)]).eval()


print(f"\n{'='*60}\n  3/7  Gating network ({GATING_EPOCHS} epochs)\n{'='*60}")
t0 = time.time()

print("  Collecting winner labels ...")
xts, sigs, wins = [], [], []
for i, (images, _) in enumerate(train_loader):
    if i >= 50:
        break
    x0 = images.to(device)
    sigma = sample_sigma_train(x0.shape[0], SIGMA_MIN, SIGMA_MAX, device)
    xt, eps = add_noise(x0, sigma)
    with torch.no_grad():
        ls = torch.stack([
            (ema_experts[k](xt, sigma) - eps).pow(2).sum(dim=(1,2,3))
            for k in range(K)
        ], dim=1)
    xts.append(xt.cpu()); sigs.append(sigma.cpu()); wins.append(ls.argmin(1).cpu())

ds = torch.utils.data.TensorDataset(torch.cat(xts), torch.cat(sigs), torch.cat(wins))
gl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, drop_last=True)

gating = GatingNet(K=K).to(device)
gopt = torch.optim.Adam(gating.parameters(), lr=1e-3)

for epoch in range(1, GATING_EPOCHS + 1):
    gating.train()
    correct = total = 0
    for xb, sb, wb in gl:
        logits = gating(xb.to(device), sb.to(device))
        loss = F.cross_entropy(logits, wb.to(device))
        gopt.zero_grad(); loss.backward(); gopt.step()
        correct += (logits.argmax(1) == wb.to(device)).sum().item()
        total += xb.shape[0]
    if epoch % 5 == 0 or epoch == GATING_EPOCHS:
        print(f"  epoch {epoch:3d}/{GATING_EPOCHS}  acc={correct/total:.3f}")

torch.save({"gating_net": gating.state_dict(), "K": K}, f"{CKPT}/gating_K{K}.pt")
gating.eval()
print(f"  Gating done in {(time.time()-t0)/60:.1f} min")
del xts, sigs, wins, ds, gl


print(f"\n{'='*60}\n  4/7  Generating samples (N={N_EVAL})\n{'='*60}")
t0 = time.time()

gen = {}

print("  baseline (euler) ...", flush=True)
gen["baseline_euler"] = generate_baseline(
    baseline_model, N_EVAL, NUM_STEPS, device, SIGMA_MIN, SIGMA_MAX, "euler", SEED)
save_image_grid(gen["baseline_euler"][:64], f"{OUT}/baseline_euler.png")

print("  baseline (heun) ...", flush=True)
gen["baseline_heun"] = generate_baseline(
    baseline_model, N_EVAL, NUM_STEPS, device, SIGMA_MIN, SIGMA_MAX, "heun", SEED)

for strat in ["single_expert", "random_expert", "best_expert", "mixture_score", "gated"]:
    print(f"  mcl {strat} ...", flush=True)
    kw = {"strategy": strat, "expert_id": 0, "seed": 0}
    if strat == "gated":
        kw["gating_net"] = gating
    gen[strat] = generate_mcl(
        ema_experts, K, num_samples=N_EVAL, num_steps=NUM_STEPS,
        device=device, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, **kw)
    save_image_grid(gen[strat][:64], f"{OUT}/mcl_{strat}.png")

for eid in range(K):
    print(f"  expert {eid} grid ...", flush=True)
    s = generate_mcl(
        ema_experts, K, strategy="single_expert", expert_id=eid,
        num_samples=64, num_steps=NUM_STEPS, device=device,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, seed=0)
    save_image_grid(s, f"{OUT}/expert_{eid}_grid.png")

for name, imgs in gen.items():
    torch.save(imgs.cpu(), f"{OUT}/{name}.pt")

print(f"  Sampling done in {(time.time()-t0)/60:.1f} min")


print(f"\n{'='*60}\n  5/7  Evaluation (FID, Precision, Recall)\n{'='*60}")
t0 = time.time()

real_imgs = torch.cat([x for x, _ in test_loader])[:N_EVAL]
clf = train_classifier(device)
feats_real = extract_features(clf, real_imgs, device=str(device))

all_metrics = {}
for name, imgs in gen.items():
    feats_gen = extract_features(clf, imgs.cpu()[:N_EVAL], device=str(device))
    fid = compute_fid(feats_real, feats_gen)
    prec, rec = compute_precision_recall(feats_real, feats_gen, k=5)
    all_metrics[name] = {"fid": fid, "precision": prec, "recall": rec}
    print(f"  {name:20s}  FID={fid:8.2f}  P={prec:.4f}  R={rec:.4f}")

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"  Evaluation done in {(time.time()-t0)/60:.1f} min")


print(f"\n{'='*60}\n  6/7  Analysis & visualisation\n{'='*60}")
t0 = time.time()

print("  expert vs digit ...", flush=True)
counts = expert_vs_digit(ema_experts, K, train_loader, SIGMA_MIN, SIGMA_MAX, device, 30)
plot_expert_vs_digit(counts, f"{ANALYSIS}/expert_vs_digit.png")

print("  expert vs sigma ...", flush=True)
centres, usage = expert_vs_sigma(ema_experts, K, train_loader, SIGMA_MIN, SIGMA_MAX, device, num_batches=30)
plot_expert_vs_sigma(centres, usage, f"{ANALYSIS}/expert_vs_sigma.png")

print("  multi-expert grid ...", flush=True)
all_s = same_noise_multi_expert(ema_experts, K, device, SIGMA_MIN, SIGMA_MAX,
                                 num_samples=8, num_steps=NUM_STEPS, seed=SEED)
plot_multi_expert_grid(all_s, f"{ANALYSIS}/multi_expert_grid.png")

print("  trajectory ...", flush=True)
_, traj = generate_mcl(ema_experts, K, strategy="single_expert", expert_id=0,
                       num_samples=1, num_steps=NUM_STEPS, device=device,
                       sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, seed=SEED,
                       return_trajectory=True)
plot_trajectory(traj, f"{ANALYSIS}/trajectory.png")

print("  strategy comparison ...", flush=True)
results = compare_strategies(
    ema_experts, K, device, SIGMA_MIN, SIGMA_MAX, seed=SEED,
    baseline_model=baseline_model,
)
plot_strategy_comparison(results, f"{ANALYSIS}/strategy_comparison.png")

print(f"  Analysis done in {(time.time()-t0)/60:.1f} min")


print(f"\n{'='*60}\n  7/7  Summary figures\n{'='*60}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, len(bl_log["loss"])+1), bl_log["loss"], "b-", linewidth=2)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Baseline Training Loss"); axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(mcl_log["loss"])+1), mcl_log["loss"], "r-", linewidth=2)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE Loss (winner)")
axes[1].set_title(f"MCL K={K} Training Loss"); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/training_curves.png", dpi=150, bbox_inches="tight"); plt.close()

usage_arr = np.array(mcl_log["expert_usage"])
fig, ax = plt.subplots(figsize=(8, 4))
for k in range(K):
    ax.plot(range(1, len(usage_arr)+1), usage_arr[:, k], label=f"Expert {k}", linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Usage fraction")
ax.set_title("Expert Usage During Training"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/expert_usage_training.png", dpi=150, bbox_inches="tight"); plt.close()

names = list(all_metrics.keys())
fids = [all_metrics[n]["fid"] for n in names]
precs = [all_metrics[n]["precision"] for n in names]
recs = [all_metrics[n]["recall"] for n in names]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = range(len(names))
colors = ["#4c72b0", "#4c72b0", "#dd8452", "#dd8452", "#dd8452", "#dd8452", "#55a868"]

axes[0].bar(x, fids, color=colors)
axes[0].set_xticks(list(x)); axes[0].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
axes[0].set_ylabel("FID ↓"); axes[0].set_title("FID (lower is better)")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(x, precs, color=colors)
axes[1].set_xticks(list(x)); axes[1].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
axes[1].set_ylabel("Precision ↑"); axes[1].set_title("Precision")
axes[1].grid(True, alpha=0.3, axis="y")

axes[2].bar(x, recs, color=colors)
axes[2].set_xticks(list(x)); axes[2].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
axes[2].set_ylabel("Recall ↑"); axes[2].set_title("Recall")
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT}/metrics_comparison.png", dpi=150, bbox_inches="tight"); plt.close()

print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETE")
print(f"  Checkpoints:  {CKPT}/")
print(f"  Outputs:      {OUT}/")
print(f"  Analysis:     {ANALYSIS}/")
print(f"{'='*60}")
