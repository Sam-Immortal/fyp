"""Compare two metrics_report.json files side by side.

Usage:
    python -m app.audio.metrics_compare A.json B.json [--out-md OUT.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(val, precision: int = 4) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def compare_reports(a: dict, b: dict, label_a: str, label_b: str) -> str:
    lines = []
    lines.append(f"# Metrics Comparison: {label_a} vs {label_b}\n")

    # Headline metrics
    lines.append("## Headline Metrics\n")
    lines.append("| Metric | " + label_a + " | " + label_b + " | Difference |\n")
    lines.append("|--------|" + "----------|" * 3 + "\n")

    # EER
    eer_a = a.get("threshold_free", {}).get("eer_percent")
    eer_b = b.get("threshold_free", {}).get("eer_percent")
    diff_eer = (eer_b - eer_a) if (eer_a is not None and eer_b is not None) else None
    lines.append(f"| EER (%) | {fmt(eer_a)} | {fmt(eer_b)} | {fmt(diff_eer)} |\n")

    # EER threshold
    eer_thr_a = a.get("threshold_free", {}).get("eer_threshold")
    eer_thr_b = b.get("threshold_free", {}).get("eer_threshold")
    diff_eer_thr = (eer_thr_b - eer_thr_a) if (eer_thr_a is not None and eer_thr_b is not None) else None
    lines.append(f"| EER threshold | {fmt(eer_thr_a, 6)} | {fmt(eer_thr_b, 6)} | {fmt(diff_eer_thr, 6)} |\n")

    # AUC
    auc_a = a.get("threshold_free", {}).get("auc")
    auc_b = b.get("threshold_free", {}).get("auc")
    diff_auc = (auc_b - auc_a) if (auc_a is not None and auc_b is not None) else None
    lines.append(f"| AUC | {fmt(auc_a, 6)} | {fmt(auc_b, 6)} | {fmt(diff_auc, 6)} |\n")

    # Counts
    lines.append("\n## Counts\n")
    lines.append("| Count | " + label_a + " | " + label_b + " |\n")
    lines.append("|-------|" + "----------|" * 2 + "\n")
    for key in ["total", "bonafide", "spoof", "spoof_share"]:
        ca = a.get("counts", {}).get(key)
        cb = b.get("counts", {}).get(key)
        lines.append(f"| {key} | {fmt(ca)} | {fmt(cb)} |\n")

    # Recall/FPR at 0.75 (and other common thresholds)
    lines.append("\n## Metrics at Threshold 0.75\n")
    lines.append("| Metric | " + label_a + " | " + label_b + " | Difference |\n")
    lines.append("|--------|" + "----------|" * 3 + "\n")

    thr_key = "0.750000"
    for metric in ["recall_tpr", "fpr", "precision", "f1_spoof", "accuracy", "balanced_accuracy", "mcc"]:
        ma = a.get("thresholds", {}).get(thr_key, {}).get(metric)
        mb = b.get("thresholds", {}).get(thr_key, {}).get(metric)
        diff = (mb - ma) if (ma is not None and mb is not None) else None
        lines.append(f"| {metric} | {fmt(ma)} | {fmt(mb)} | {fmt(diff)} |\n")

    # Per-attack comparison (if both have per_attack)
    per_attack_a = a.get("per_attack", {})
    per_attack_b = b.get("per_attack", {})
    common_attacks = sorted(set(per_attack_a.keys()) & set(per_attack_b.keys()))
    if common_attacks:
        lines.append("\n## Per-Attack EER vs Bonafide\n")
        lines.append("| Attack | " + label_a + " EER (%) | " + label_b + " EER (%) | Difference |\n")
        lines.append("|--------|" + "------------------|" * 3 + "\n")
        for att in common_attacks:
            eer_a = per_attack_a[att].get("eer_vs_bonafide")
            eer_b = per_attack_b[att].get("eer_vs_bonafide")
            diff = (eer_b - eer_a) if (eer_a is not None and eer_b is not None) else None
            lines.append(f"| {att} | {fmt(eer_a)} | {fmt(eer_b)} | {fmt(diff)} |\n")

    # Per-type comparison
    per_type_a = a.get("per_type", {})
    per_type_b = b.get("per_type", {})
    common_types = sorted(set(per_type_a.keys()) & set(per_type_b.keys()))
    if common_types:
        lines.append("\n## Per-Type EER vs Bonafide\n")
        lines.append("| Type | " + label_a + " EER (%) | " + label_b + " EER (%) | Difference |\n")
        lines.append("|------|" + "------------------|" * 3 + "\n")
        for typ in common_types:
            eer_a = per_type_a[typ].get("eer_vs_bonafide")
            eer_b = per_type_b[typ].get("eer_vs_bonafide")
            diff = (eer_b - eer_a) if (eer_a is not None and eer_b is not None) else None
            lines.append(f"| {typ} | {fmt(eer_a)} | {fmt(eer_b)} | {fmt(diff)} |\n")

    # Group-by comparison (if both have group_by)
    group_a = a.get("group_by", {})
    group_b = b.get("group_by", {})
    common_cols = sorted(set(group_a.keys()) & set(group_b.keys()))
    for col in common_cols:
        groups_a = group_a[col]
        groups_b = group_b[col]
        common_groups = sorted(set(groups_a.keys()) & set(groups_b.keys()))
        if common_groups:
            lines.append(f"\n## Group-by: {col}\n")
            lines.append("| Group | " + label_a + " EER (%) | " + label_b + " EER (%) | Difference |\n")
            lines.append("|-------|" + "------------------|" * 3 + "\n")
            for grp in common_groups:
                eer_a = groups_a[grp].get("eer")
                eer_b = groups_b[grp].get("eer")
                diff = (eer_b - eer_a) if (eer_a is not None and eer_b is not None) else None
                lines.append(f"| {grp} | {fmt(eer_a)} | {fmt(eer_b)} | {fmt(diff)} |\n")

    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two metrics_report.json files")
    parser.add_argument("file_a", type=Path, help="First metrics_report.json (e.g., 2019)")
    parser.add_argument("file_b", type=Path, help="Second metrics_report.json (e.g., 2021)")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional output markdown file")
    parser.add_argument("--label-a", type=str, default="A", help="Label for file A")
    parser.add_argument("--label-b", type=str, default="B", help="Label for file B")
    args = parser.parse_args()

    if not args.file_a.exists():
        print(f"File not found: {args.file_a}", file=sys.stderr)
        sys.exit(1)
    if not args.file_b.exists():
        print(f"File not found: {args.file_b}", file=sys.stderr)
        sys.exit(1)

    a = load_json(args.file_a)
    b = load_json(args.file_b)

    md = compare_reports(a, b, args.label_a, args.label_b)

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Comparison written to {args.out_md}")
    else:
        print(md)


if __name__ == "__main__":
    main()