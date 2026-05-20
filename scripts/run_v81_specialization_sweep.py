#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class Job:
    name: str
    beta_rho: float
    hard_wta: bool
    diversity_weight: float
    balance_weight: float
    seed: int


def sh(cmd: List[str], cwd: Path) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run V8.1 specialization sweep")
    ap.add_argument("--hours-budget", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base-outdir", default="outputs/v81_specialization")
    ap.add_argument("--baseline-steps", type=int, default=10000)
    ap.add_argument("--mcl-steps", type=int, default=20000)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--sample-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_root = Path(args.base_outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs: List[Job] = []
    for rho in [1.0, 3.0, 10.0, 30.0]:
        jobs.append(
            Job(
                name=f"rho{rho:g}_soft",
                beta_rho=rho,
                hard_wta=False,
                diversity_weight=0.01,
                balance_weight=0.05,
                seed=0,
            )
        )
        jobs.append(
            Job(
                name=f"rho{rho:g}_hard",
                beta_rho=rho,
                hard_wta=True,
                diversity_weight=0.02,
                balance_weight=0.10,
                seed=0,
            )
        )

    deadline = time.time() + args.hours_budget * 3600.0
    master_rows = []
    for i, job in enumerate(jobs, start=1):
        now = time.time()
        if now >= deadline:
            master_rows.append({"job": job.name, "status": "skipped_budget_exhausted"})
            break

        outdir = out_root / job.name
        cmd = [
            "python",
            "scripts/run_variantv8.py",
            "--dataset", "mnist",
            "--classes", "all",
            "--image-size", "28",
            "--K", "4",
            "--outdir", str(outdir),
            "--device", args.device,
            "--all",
            "--baseline-steps", str(args.baseline_steps),
            "--mcl-steps", str(args.mcl_steps),
            "--batch-size", str(args.batch_size),
            "--num-samples", str(args.num_samples),
            "--sample-steps", str(args.sample_steps),
            "--sample-batch-size", "64",
            "--pca-dim-router", "64",
            "--beta-rho", str(job.beta_rho),
            "--beta-max", "80",
            "--balance-weight", str(job.balance_weight),
            "--diversity-weight", str(job.diversity_weight),
            "--paired-batches", "5",
            "--seed", str(job.seed),
        ]
        if job.hard_wta:
            cmd.append("--hard-wta")
        t0 = time.time()
        rc = sh(cmd, repo)
        retry_mode = "none"
        if rc != 0 and args.device == "cuda":
            retry_mode = "cpu_fallback"
            cpu_cmd = cmd.copy()
            dev_idx = cpu_cmd.index("--device")
            cpu_cmd[dev_idx + 1] = "cpu"
            b_idx = cpu_cmd.index("--batch-size")
            cpu_cmd[b_idx + 1] = str(max(32, int(args.batch_size // 2)))
            bs_idx = cpu_cmd.index("--baseline-steps")
            ms_idx = cpu_cmd.index("--mcl-steps")
            cpu_cmd[bs_idx + 1] = str(max(2000, int(args.baseline_steps * 0.4)))
            cpu_cmd[ms_idx + 1] = str(max(4000, int(args.mcl_steps * 0.4)))
            rc = sh(cpu_cmd, repo)
        elapsed = time.time() - t0
        master_rows.append(
            {
                "job": job.name,
                "index": i,
                "return_code": rc,
                "elapsed_s": round(elapsed, 2),
                "outdir": str(outdir),
                "retry_mode": retry_mode,
                **asdict(job),
            }
        )

    (out_root / "SWEEP_RESULTS.json").write_text(json.dumps(master_rows, indent=2), encoding="utf-8")
    print(json.dumps(master_rows, indent=2))


if __name__ == "__main__":
    main()
