#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_checksums(base: Path, files: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rel in files:
        p = base / rel
        if p.exists():
            out[rel] = sha256_of_file(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze V8 positive control artifact set.")
    ap.add_argument("--src", default="outputs/v8_mnist_gold")
    ap.add_argument("--dst", default="outputs/v8_mnist_gold_frozen")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    key_files = [
        "SUMMARY_v8.txt",
        "config.json",
        "data_info.json",
        "beta_calibrator.json",
        "metrics.json",
        "router_calibration_by_t.csv",
        "baseline_final.pt",
        "mcl_final.pt",
        "router_risk_table.pt",
    ]

    copied: List[str] = []
    for rel in key_files:
        src_file = src / rel
        if src_file.exists():
            shutil.copy2(src_file, dst / rel)
            copied.append(rel)

    manifest = {
        "source_dir": str(src.resolve()),
        "frozen_dir": str(dst.resolve()),
        "copied_files": copied,
        "checksums_sha256": collect_checksums(dst, copied),
    }
    (dst / "FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
