#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "md_results_bundle_with_code_latest"


CSV_FILES = [
    "outputs/rf_mcl_results/gmm_v5_quick/metrics_all.csv",
    "outputs/rf_mcl_results/gmm_v5_quick/router_final_metrics.csv",
    "outputs/rf_mcl_results/cifar_v5_quick/metrics_all.csv",
    "outputs/rf_mcl_results/cifar_v5_quick/router_final_metrics.csv",
    "outputs/rf_mcl_results/gmm_v5_serious/metrics_all.csv",
    "outputs/rf_mcl_results/gmm_v5_serious/router_final_metrics.csv",
    "outputs/rf_mcl_results/cifar_v5_serious/metrics_all.csv",
    "outputs/rf_mcl_results/cifar_v5_serious/router_final_metrics.csv",
]

SUMMARY_FILES = [
    "outputs/rf_mcl_results/gmm_v5_quick/SUMMARY.txt",
    "outputs/rf_mcl_results/cifar_v5_quick/SUMMARY.txt",
    "outputs/rf_mcl_results/gmm_v5_serious/SUMMARY.txt",
    "outputs/rf_mcl_results/cifar_v5_serious/SUMMARY.txt",
]

PY_FILES = [
    "scripts/rf_mcl_gmm_v4_router.py",
    "scripts/rf_mcl_cifar_v3_router.py",
    "scripts/rf_mcl_gmm_v5_router.py",
    "scripts/rf_mcl_cifar_v5_router.py",
]


def read_text(path: Path) -> str:
    if not path.exists():
        return f"[MISSING FILE] {path.as_posix()}\n"
    return path.read_text(encoding="utf-8", errors="replace")


def write_raw_bundle(out_file: Path, title: str, files: list[str], fence: str):
    lines = [title, "", f"Total files: {len(files)}", ""]
    for idx, rel in enumerate(files, start=1):
        src = ROOT / rel
        lines.append(f"## {idx}. `{rel}`")
        lines.append("")
        lines.append(f"```{fence}")
        lines.append(read_text(src))
        lines.append("```")
        lines.append("")
    out_file.write_text("\n".join(lines), encoding="utf-8")


def build_index():
    lines = [
        "# RESULTS + CODE BUNDLE INDEX",
        "",
        f"- CSV files: {len(CSV_FILES)}",
        f"- SUMMARY files: {len(SUMMARY_FILES)}",
        f"- Python files: {len(PY_FILES)}",
        "",
        "## Main outputs",
        "",
        "- `ALL_RESULTS_CSV_RAW_EXACT.md`",
        "- `ALL_RESULTS_SUMMARY_RAW_EXACT.md`",
        "- `ALL_PYTHON_CODE_RAW_EXACT.md`",
        "",
        "## Included CSV files",
        "",
    ]
    for rel in CSV_FILES:
        lines.append(f"- `{rel}`")
    lines.extend(["", "## Included SUMMARY files", ""])
    for rel in SUMMARY_FILES:
        lines.append(f"- `{rel}`")
    lines.extend(["", "## Included Python files", ""])
    for rel in PY_FILES:
        lines.append(f"- `{rel}`")
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_raw_bundle(
        OUT / "ALL_RESULTS_CSV_RAW_EXACT.md",
        "# ALL RESULTS CSV (RAW EXACT)",
        CSV_FILES,
        "csv",
    )
    write_raw_bundle(
        OUT / "ALL_RESULTS_SUMMARY_RAW_EXACT.md",
        "# ALL RESULTS SUMMARY (RAW EXACT)",
        SUMMARY_FILES,
        "text",
    )
    write_raw_bundle(
        OUT / "ALL_PYTHON_CODE_RAW_EXACT.md",
        "# ALL PYTHON CODE (RAW EXACT)",
        PY_FILES,
        "python",
    )
    build_index()
    print(OUT.as_posix())


if __name__ == "__main__":
    main()

