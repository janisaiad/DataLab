#!/usr/bin/env python3
from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class Job:
    name: str
    args: List[str]
    priority: int


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def pick_device() -> str:
    # Prefer CUDA only if there is enough free VRAM.
    try:
        q = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        if not q:
            return "cpu"
        total_s, used_s = q.splitlines()[0].split(",")
        total = float(total_s.strip())
        used = float(used_s.strip())
        free = total - used
        return "cuda" if free >= 3000.0 else "cpu"
    except Exception:
        return "cpu"


def run() -> None:
    root = Path("/Data/janis.aiad/DataLab")
    script = root / "scripts" / "theory_beta_router_tests.py"
    out_root = root / "outputs" / "theory_beta_8h_auto"
    out_root.mkdir(parents=True, exist_ok=True)
    master_log = out_root / "master_log.jsonl"

    max_wall_s = 8 * 3600
    t0 = time.time()

    device = pick_device()

    # Denser, information-first plan for ~10h walltime.
    jobs = [
        Job(
            name="mnist_all_pca64_seed0",
            priority=1,
            args=[
                "--dataset", "mnist",
                "--classes", "all",
                "--feature-mode", "pca",
                "--pca-dim", "64",
                "--n-train", "30000",
                "--n-test", "5000",
                "--train-for-A", "8000",
                "--times", "0.15,0.25,0.35,0.5,0.7,0.85,1.1,1.3,1.5,1.8,2.05,2.3,2.5,2.8,3.0",
                "--seed", "0",
                "--device", device,
                "--outdir", str(out_root / "mnist_all_pca64_seed0"),
            ],
        ),
        Job(
            name="mnist_all_pca64_seed1",
            priority=2,
            args=[
                "--dataset", "mnist", "--classes", "all", "--feature-mode", "pca", "--pca-dim", "64",
                "--n-train", "30000", "--n-test", "5000", "--train-for-A", "8000",
                "--times", "0.15,0.25,0.35,0.5,0.7,0.85,1.1,1.3,1.5,1.8,2.05,2.3,2.5,2.8,3.0",
                "--seed", "1", "--device", device,
                "--outdir", str(out_root / "mnist_all_pca64_seed1"),
            ],
        ),
        Job(
            name="mnist_all_pca64_seed2",
            priority=3,
            args=[
                "--dataset", "mnist", "--classes", "all", "--feature-mode", "pca", "--pca-dim", "64",
                "--n-train", "30000", "--n-test", "5000", "--train-for-A", "8000",
                "--times", "0.15,0.25,0.35,0.5,0.7,0.85,1.1,1.3,1.5,1.8,2.05,2.3,2.5,2.8,3.0",
                "--seed", "2", "--device", device,
                "--outdir", str(out_root / "mnist_all_pca64_seed2"),
            ],
        ),
        Job(
            name="mnist_all_pca64_seed3",
            priority=4,
            args=[
                "--dataset", "mnist", "--classes", "all", "--feature-mode", "pca", "--pca-dim", "64",
                "--n-train", "30000", "--n-test", "5000", "--train-for-A", "8000",
                "--times", "0.15,0.25,0.35,0.5,0.7,0.85,1.1,1.3,1.5,1.8,2.05,2.3,2.5,2.8,3.0",
                "--seed", "3", "--device", device,
                "--outdir", str(out_root / "mnist_all_pca64_seed3"),
            ],
        ),
        Job(
            name="mnist_all_pca128_seed0",
            priority=5,
            args=[
                "--dataset", "mnist", "--classes", "all", "--feature-mode", "pca", "--pca-dim", "128",
                "--n-train", "30000", "--n-test", "5000", "--train-for-A", "8000",
                "--times", "0.15,0.25,0.35,0.5,0.7,0.85,1.1,1.3,1.5,1.8,2.05,2.3,2.5,2.8,3.0",
                "--seed", "0", "--device", device,
                "--outdir", str(out_root / "mnist_all_pca128_seed0"),
            ],
        ),
        Job(
            name="cifar_auto_horse_pca128_seed0",
            priority=6,
            args=[
                "--dataset", "cifar10", "--classes", "automobile,horse", "--feature-mode", "pca", "--pca-dim", "128",
                "--n-train", "10000", "--n-test", "2000", "--train-for-A", "4000",
                "--times", "0.15,0.25,0.35,0.5,0.7,0.85,1.1,1.3,1.5,1.8,2.05,2.3,2.5,2.8,3.0",
                "--seed", "0", "--device", device,
                "--outdir", str(out_root / "cifar_auto_horse_pca128_seed0"),
            ],
        ),
        Job(
            name="cifar_auto_horse_pca128_seed1",
            priority=7,
            args=[
                "--dataset", "cifar10", "--classes", "automobile,horse", "--feature-mode", "pca", "--pca-dim", "128",
                "--n-train", "10000", "--n-test", "2000", "--train-for-A", "4000",
                "--times", "0.15,0.25,0.35,0.5,0.7,0.85,1.1,1.3,1.5,1.8,2.05,2.3,2.5,2.8,3.0",
                "--seed", "1", "--device", device,
                "--outdir", str(out_root / "cifar_auto_horse_pca128_seed1"),
            ],
        ),
        Job(
            name="cifar_auto_horse_pca256_seed0",
            priority=8,
            args=[
                "--dataset", "cifar10", "--classes", "automobile,horse", "--feature-mode", "pca", "--pca-dim", "256",
                "--n-train", "12000", "--n-test", "2500", "--train-for-A", "5000",
                "--times", "0.15,0.25,0.35,0.5,0.7,0.85,1.1,1.3,1.5,1.8,2.05,2.3,2.5,2.8,3.0",
                "--seed", "0", "--device", device,
                "--outdir", str(out_root / "cifar_auto_horse_pca256_seed0"),
            ],
        ),
        Job(
            name="cifar_all_pca192_heavy",
            priority=9,
            args=[
                "--dataset", "cifar10", "--classes", "all", "--feature-mode", "pca", "--pca-dim", "192",
                "--n-train", "50000", "--n-test", "10000", "--train-for-A", "10000",
                "--times", "0.2,0.35,0.5,0.7,0.85,1.1,1.3,1.5,1.8,2.05,2.3,2.5,2.8,3.0",
                "--seed", "0", "--device", device,
                "--outdir", str(out_root / "cifar_all_pca192_heavy"),
            ],
        ),
    ]

    jobs = sorted(jobs, key=lambda x: x.priority)

    with master_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_iso(), "event": "start", "max_wall_s": max_wall_s, "device": device, "jobs": [asdict(j) for j in jobs]}) + "\n")

    for job in jobs:
        elapsed = time.time() - t0
        remain = max_wall_s - elapsed
        if remain <= 120:
            with master_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": now_iso(), "event": "stop_no_time_left", "elapsed_s": elapsed}) + "\n")
            break

        cmd = ["python", str(script)] + job.args
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        with master_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now_iso(), "event": "job_start", "job": job.name, "remaining_s": remain, "cmd": cmd_str}) + "\n")

        start = time.time()
        timeout = max(60, int(remain - 30))
        status = "ok"
        err = ""
        try:
            subprocess.run(cmd, cwd=str(root), check=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
        except subprocess.CalledProcessError as e:
            status = "failed"
            err = str(e)
        dur = time.time() - start

        with master_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now_iso(), "event": "job_end", "job": job.name, "status": status, "duration_s": dur, "error": err}) + "\n")

    with master_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_iso(), "event": "done", "elapsed_s": time.time() - t0}) + "\n")


if __name__ == "__main__":
    run()

