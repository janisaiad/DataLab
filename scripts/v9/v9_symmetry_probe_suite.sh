#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
SCRIPT=scripts/v9/v9_symmetry_router_probe.py
LOG=outputs/v9_symmetry_probe_suite.log
mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

echo "[sym-probe] $(date -Is) start"

run_one() {
  local name="$1"
  shift
  echo "[sym-probe] $(date -Is) BEGIN $name"
  if python "$SCRIPT" "$@"; then
    echo "[sym-probe] $(date -Is) OK $name"
  else
    echo "[sym-probe] $(date -Is) FAIL $name"
  fi
}

python -m py_compile "$SCRIPT"

run_one gmm \
  --dataset gmm --C 4 --K 4 --d-latent 64 \
  --n-train 6000 --n-test 3000 \
  --t-grid 0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,3.0 \
  --rho-list 0.5,1.0,2.0 --mcl-steps 3000 \
  --device cuda --outdir outputs/v9_symmetry_gmm

run_one mnist \
  --dataset mnist --classes 0,1,2,3,4,5,6,7,8,9 \
  --d-latent 128 --K 4 --n-train 12000 --n-test 2048 \
  --t-grid 0.6,1.0,1.4,1.8,2.2,2.6,3.0 \
  --rho-list 0.5,1.0,2.0 --mcl-steps 3000 \
  --device cuda --outdir outputs/v9_symmetry_mnist

run_one cifar \
  --dataset cifar10 --classes automobile,horse \
  --d-latent 128 --K 2 --n-train 8000 --n-test 2000 \
  --t-grid 0.6,1.0,1.4,1.8,2.2,2.6,3.0 \
  --rho-list 0.5,1.0,2.0 --mcl-steps 3000 \
  --device cuda --outdir outputs/v9_symmetry_cifar_auto_horse

# Aggregate report
python3 <<'PY'
from pathlib import Path
import pandas as pd

root = Path("outputs")
runs = [
    ("gmm", root / "v9_symmetry_gmm"),
    ("mnist", root / "v9_symmetry_mnist"),
    ("cifar_auto_horse", root / "v9_symmetry_cifar_auto_horse"),
]
lines = ["# Rapport symmetry-breaking probe V9", ""]
for name, d in runs:
    lines.append(f"## {name}")
    if not d.exists():
        lines.append("*(non exécuté)*\n")
        continue
    readme = d / "README_SUMMARY.md"
    if readme.exists():
        lines.append(readme.read_text(encoding="utf-8"))
    else:
        lines.append("*(README_SUMMARY.md manquant)*")
    lines.append("")
out = root / "V9_SYMMETRY_PROBE_REPORT.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out}")
PY

echo "[sym-probe] $(date -Is) finished"
