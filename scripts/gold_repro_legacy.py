#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
import torchvision
import torchvision.transforms as T

from src.evaluate import evaluate
from src.model import GatingNet
from src.sample import generate_baseline, generate_mcl, load_baseline, load_mcl
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gold legacy repro with fixed checkpoints and legacy sampler")
    p.add_argument("--baseline_ckpt", type=str, default="checkpoints/shared/baseline_final.pt")
    p.add_argument("--mcl_ckpt", type=str, default="checkpoints/annealed_wta_K4/mcl_K4_final.pt")
    p.add_argument("--gating_ckpt", type=str, default="checkpoints/annealed_wta_K4/gating_K4.pt")
    p.add_argument("--out_dir", type=str, default="outputs/gold_repro_legacy")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--num_samples", type=int, default=2048)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--solver_baseline", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument("--solver_mcl", type=str, default="euler", choices=["euler", "heun"])
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    return p.parse_args()


def parse_seeds(spec: str) -> list[int]:
    vals = []
    for t in spec.split(","):
        t = t.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("No seeds provided")
    return vals


def to_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_real_images(num_samples: int) -> torch.Tensor:
    tfm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfm)
    return torch.stack([ds[i][0] for i in range(num_samples)])


def aggregate(per_seed: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    metrics = {}
    names = sorted(next(iter(per_seed.values())).keys())
    keys = ["fid", "precision", "recall"]
    for name in names:
        metrics[name] = {}
        for k in keys:
            vals = [per_seed[s][name][k] for s in per_seed]
            metrics[name][f"{k}_mean"] = float(mean(vals))
            metrics[name][f"{k}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return metrics


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = to_device(args.device)
    seeds = parse_seeds(args.seeds)

    baseline_model, a_bl = load_baseline(args.baseline_ckpt, device=device)
    experts, K, a_mcl = load_mcl(args.mcl_ckpt, device=device)
    g_ckpt = torch.load(args.gating_ckpt, map_location=device, weights_only=False)
    gating_net = GatingNet(K).to(device)
    gating_net.load_state_dict(g_ckpt["gating_net"])
    gating_net.eval()

    real = load_real_images(args.num_samples)
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    generated_dir = out / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        set_seed(seed)
        key = f"seed_{seed}"
        per_seed[key] = {}

        baseline_heun = generate_baseline(
            baseline_model,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=device,
            sigma_min=a_bl["sigma_min"],
            sigma_max=a_bl["sigma_max"],
            solver=args.solver_baseline,
            seed=seed,
        )
        single_expert = generate_mcl(
            experts,
            K,
            strategy="single_expert",
            expert_id=0,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=device,
            sigma_min=a_mcl["sigma_min"],
            sigma_max=a_mcl["sigma_max"],
            solver=args.solver_mcl,
            seed=seed,
        )
        random_expert = generate_mcl(
            experts,
            K,
            strategy="random_expert",
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=device,
            sigma_min=a_mcl["sigma_min"],
            sigma_max=a_mcl["sigma_max"],
            solver=args.solver_mcl,
            seed=seed,
        )
        best_expert = generate_mcl(
            experts,
            K,
            strategy="best_expert",
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=device,
            sigma_min=a_mcl["sigma_min"],
            sigma_max=a_mcl["sigma_max"],
            solver=args.solver_mcl,
            seed=seed,
        )
        mixture_score = generate_mcl(
            experts,
            K,
            strategy="mixture_score",
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=device,
            sigma_min=a_mcl["sigma_min"],
            sigma_max=a_mcl["sigma_max"],
            solver=args.solver_mcl,
            seed=seed,
        )
        gated = generate_mcl(
            experts,
            K,
            strategy="gated",
            gating_net=gating_net,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=device,
            sigma_min=a_mcl["sigma_min"],
            sigma_max=a_mcl["sigma_max"],
            solver=args.solver_mcl,
            seed=seed,
        )

        generated = {
            "baseline_heun": baseline_heun.cpu(),
            "single_expert": single_expert.cpu(),
            "random_expert": random_expert.cpu(),
            "best_expert": best_expert.cpu(),
            "mixture_score": mixture_score.cpu(),
            "gated": gated.cpu(),
        }
        for name, ten in generated.items():
            torch.save(ten, generated_dir / f"{name}_n{args.num_samples}_{key}.pt")
            per_seed[key][name] = evaluate(real, ten[: args.num_samples], device=str(device), k=5)

    agg = aggregate(per_seed)
    payload = {
        "config": {
            "baseline_ckpt": args.baseline_ckpt,
            "mcl_ckpt": args.mcl_ckpt,
            "gating_ckpt": args.gating_ckpt,
            "device": str(device),
            "num_samples": args.num_samples,
            "num_steps": args.num_steps,
            "solver_baseline": args.solver_baseline,
            "solver_mcl": args.solver_mcl,
            "seeds": seeds,
        },
        "per_seed": per_seed,
        "aggregate": agg,
    }
    (out / "metrics_per_seed.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Gold Legacy Repro", ""]
    lines.append(f"- Device: `{device}`")
    lines.append(f"- Seeds: `{seeds}`")
    lines.append(f"- Samples per seed: `{args.num_samples}`")
    lines.append(f"- Steps: `{args.num_steps}`")
    lines.append("")
    lines.append("## Aggregate (mean +/- std)")
    lines.append("")
    for name in ["baseline_heun", "single_expert", "random_expert", "best_expert", "mixture_score", "gated"]:
        m = agg[name]
        lines.append(
            f"- `{name}`: "
            f"FID={m['fid_mean']:.3f} +/- {m['fid_std']:.3f}, "
            f"P={m['precision_mean']:.4f} +/- {m['precision_std']:.4f}, "
            f"R={m['recall_mean']:.4f} +/- {m['recall_std']:.4f}"
        )
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out / "metrics_per_seed.json").as_posix())
    print((out / "SUMMARY.md").as_posix())


if __name__ == "__main__":
    main()
