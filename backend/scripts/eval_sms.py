#!/usr/bin/env python3
"""Evaluate SMS scam classifier on labelled dataset.

Usage:
    python scripts/eval_sms.py --split test --delay 15 --model mistralai/Mistral-7B-Instruct-v0.3
    python scripts/eval_sms.py --split dev --limit 10 --out reports/sms_eval
    python scripts/eval_sms.py --split all --resume
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.nlp.worker import (
    build_scam_prompt,
    call_llm_raw,
    parse_llm_response,
    NlpParseError,
)


@dataclass
class EvalRow:
    id: int
    label: str
    category: str
    split: str
    sender: str
    score: float = 0.0
    risk_level: str | None = None
    patterns: dict | None = None
    explanation: str = ""
    status: str = "pending"
    error: str | None = None
    total_tokens: int = 0
    seconds: float = 0.0


def load_csv(csv_path: Path, split: str, limit: int | None) -> list[tuple[EvalRow, str]]:
    """Load rows from CSV, filtering by split. Returns (EvalRow, text) pairs."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split != "all" and row["split"] != split:
                continue
            eval_row = EvalRow(
                id=int(row["id"]),
                label=row["label"],
                category=row["category"],
                split=row["split"],
                sender=row["sender"],
            )
            rows.append((eval_row, row["text"]))
            if limit and len(rows) >= limit:
                break
    return rows


def load_existing_results(out_path: Path) -> dict[int, EvalRow]:
    """Load already-completed rows from existing results CSV.
    Keeps only the LAST row per id (handles failed->completed transitions)."""
    completed = {}
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] == "completed":
                    completed[int(row["id"])] = EvalRow(
                        id=int(row["id"]),
                        label=row["label"],
                        category=row["category"],
                        split=row["split"],
                        sender=row["sender"],
                        score=float(row["score"]),
                        risk_level=row["risk_level"] if row["risk_level"] else None,
                        patterns=json.loads(row["patterns"]) if row["patterns"] else None,
                        explanation=row["explanation"],
                        status=row["status"],
                        error=row["error"] if row["error"] else None,
                        total_tokens=int(row["total_tokens"]),
                        seconds=float(row["seconds"]),
                    )
    return completed


def save_row(out_path: Path, row: EvalRow, fieldnames: list[str]) -> None:
    """Append a single row to results CSV."""
    file_exists = out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "id": row.id,
            "label": row.label,
            "category": row.category,
            "split": row.split,
            "sender": row.sender,
            "score": row.score,
            "risk_level": row.risk_level or "",
            "patterns": json.dumps(row.patterns) if row.patterns else "",
            "explanation": row.explanation,
            "status": row.status,
            "error": row.error or "",
            "total_tokens": row.total_tokens,
            "seconds": row.seconds,
        })


async def process_row(
    row: EvalRow,
    text: str,
    model: str,
    threshold: float,
) -> EvalRow:
    """Process a single row: build prompt, call LLM, parse response."""
    start = time.perf_counter()

    messages = build_scam_prompt(text, row.sender)

    try:
        content, usage = await call_llm_raw(messages, model=model)
        result = parse_llm_response(content)

        row.score = result.scam_probability
        row.risk_level = result.risk_level
        row.patterns = result.patterns
        row.explanation = result.explanation
        row.status = "completed"
        row.error = None
        row.total_tokens = usage.get("total_tokens", 0)

    except NlpParseError as e:
        row.status = "failed"
        row.error = f"Parse error: {e}"
        row.score = 0.0
        row.risk_level = None
        row.patterns = None
        row.total_tokens = 0

    except RuntimeError as e:
        row.status = "failed"
        row.error = f"API error: {e}"
        row.score = 0.0
        row.risk_level = None
        row.patterns = None
        row.total_tokens = 0

    except Exception as e:
        row.status = "failed"
        row.error = f"Unexpected error: {type(e).__name__}: {e}"
        row.score = 0.0
        row.risk_level = None
        row.patterns = None
        row.total_tokens = 0

    row.seconds = time.perf_counter() - start
    return row


def compute_metrics(rows: list[EvalRow], threshold: float, failed_as_miss: bool = False) -> dict:
    """Compute evaluation metrics."""
    total_rows = len(rows)
    completed_rows = [r for r in rows if r.status == "completed"]
    failed_rows = [r for r in rows if r.status == "failed"]

    # --- Score-only rule (threshold-based) ---
    if failed_as_miss:
        eval_rows = [r for r in rows if r.status in ("completed", "failed")]
        y_true = [1 if r.label == "scam" else 0 for r in eval_rows]
        y_pred = []
        for r in eval_rows:
            if r.status == "failed":
                # Failed scam counts as FN (predict legit=0), failed legit counts as FP (predict scam=1)
                y_pred.append(0 if r.label == "scam" else 1)
            else:
                y_pred.append(1 if r.score >= threshold else 0)
    else:
        eval_rows = completed_rows
        y_true = [1 if r.label == "scam" else 0 for r in eval_rows]
        y_pred = [1 if r.score >= threshold else 0 for r in eval_rows]

    if not eval_rows:
        return {
            "total_rows": total_rows,
            "completed": len(completed_rows),
            "failed": len(failed_rows),
            "error": "No rows to evaluate",
        }

    # Confusion matrix (score-only rule)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(eval_rows)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Mean scores
    scam_scores = [r.score for r in eval_rows if r.label == "scam"]
    legit_scores = [r.score for r in eval_rows if r.label == "legit"]
    mean_scam = sum(scam_scores) / len(scam_scores) if scam_scores else 0.0
    mean_legit = sum(legit_scores) / len(legit_scores) if legit_scores else 0.0

    # ROC-AUC if sklearn available
    roc_auc = None
    try:
        from sklearn.metrics import roc_auc_score
        scores = [r.score for r in eval_rows]
        if len(set(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, scores)
    except ImportError:
        pass

    # --- App rule: predicted scam = (risk_level in high/critical) OR score >= threshold ---
    if failed_as_miss:
        eval_rows_app = [r for r in rows if r.status in ("completed", "failed")]
        y_true_app = [1 if r.label == "scam" else 0 for r in eval_rows_app]
        y_pred_app = []
        for r in eval_rows_app:
            if r.status == "failed":
                # Failed scam counts as FN, failed legit counts as FP
                y_pred_app.append(0 if r.label == "scam" else 1)
            else:
                y_pred_app.append(1 if (r.risk_level in ("high", "critical") or r.score >= threshold) else 0)
    else:
        eval_rows_app = completed_rows
        y_true_app = [1 if r.label == "scam" else 0 for r in eval_rows_app]
        y_pred_app = [
            1 if (r.risk_level in ("high", "critical") or r.score >= threshold) else 0
            for r in eval_rows_app
        ]

    tp_app = sum(1 for yt, yp in zip(y_true_app, y_pred_app) if yt == 1 and yp == 1)
    fp_app = sum(1 for yt, yp in zip(y_true_app, y_pred_app) if yt == 0 and yp == 1)
    tn_app = sum(1 for yt, yp in zip(y_true_app, y_pred_app) if yt == 0 and yp == 0)
    fn_app = sum(1 for yt, yp in zip(y_true_app, y_pred_app) if yt == 1 and yp == 0)

    total_app = len(eval_rows_app)
    recall_app = tp_app / (tp_app + fn_app) if (tp_app + fn_app) > 0 else 0.0
    fpr_app = fp_app / (fp_app + tn_app) if (fp_app + tn_app) > 0 else 0.0
    precision_app = tp_app / (tp_app + fp_app) if (tp_app + fp_app) > 0 else 0.0
    accuracy_app = (tp_app + tn_app) / total_app if total_app > 0 else 0.0

    # Error rate per category - use ALL rows (completed + failed) for denominator
    category_errors = {}
    for r in rows:
        cat = r.category
        if cat not in category_errors:
            category_errors[cat] = {"total": 0, "errors": 0}
        category_errors[cat]["total"] += 1
        if r.status == "failed":
            category_errors[cat]["errors"] += 1

    error_rate_by_cat = {
        cat: (data["errors"] / data["total"] if data["total"] > 0 else 0.0)
        for cat, data in category_errors.items()
    }

    # Wrong predictions (score-only rule)
    wrong = []
    for r in eval_rows:
        pred = "scam" if r.score >= threshold else "legit"
        if pred != r.label:
            wrong.append({
                "id": r.id,
                "category": r.category,
                "score": round(r.score, 4),
                "risk_level": r.risk_level,
                "explanation": r.explanation[:100] + "..." if len(r.explanation) > 100 else r.explanation,
            })

    # Wrong predictions (app rule)
    wrong_app = []
    for r in eval_rows_app:
        pred_app = "scam" if (r.risk_level in ("high", "critical") or r.score >= threshold) else "legit"
        if pred_app != r.label:
            wrong_app.append({
                "id": r.id,
                "category": r.category,
                "score": round(r.score, 4),
                "risk_level": r.risk_level,
                "explanation": r.explanation[:100] + "..." if len(r.explanation) > 100 else r.explanation,
            })

    return {
        "total_rows": total_rows,
        "completed": len(completed_rows),
        "failed": len(failed_rows),
        "threshold": threshold,
        # Score-only rule
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "accuracy": round(accuracy, 4),
        # App rule
        "confusion_matrix_app": {"tp": tp_app, "fp": fp_app, "tn": tn_app, "fn": fn_app},
        "recall_app": round(recall_app, 4),
        "fpr_app": round(fpr_app, 4),
        "precision_app": round(precision_app, 4),
        "accuracy_app": round(accuracy_app, 4),
        "mean_scam_score": round(mean_scam, 4),
        "mean_legit_score": round(mean_legit, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "error_rate_by_category": error_rate_by_cat,
        "wrong_predictions": wrong,
        "wrong_predictions_app": wrong_app,
    }


def print_metrics(metrics: dict, model: str, split: str) -> None:
    """Print metrics to console."""
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Split: {split}")
    print(f"Threshold: {metrics['threshold']}")
    print(f"Total rows: {metrics['total_rows']}")
    print(f"Completed: {metrics['completed']}, Failed: {metrics['failed']}")

    # Score-only rule
    print(f"\n--- Score-only rule (score >= threshold) ---")
    cm = metrics['confusion_matrix']
    print(f"Confusion Matrix:")
    print(f"  TP={cm['tp']}  FP={cm['fp']}")
    print(f"  FN={cm['fn']}  TN={cm['tn']}")
    print(f"Recall (TPR): {metrics['recall']:.4f}")
    print(f"False Positive Rate: {metrics['fpr']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    # App rule
    print(f"\n--- App rule (risk_level in high/critical OR score >= threshold) ---")
    cm_app = metrics['confusion_matrix_app']
    print(f"Confusion Matrix:")
    print(f"  TP={cm_app['tp']}  FP={cm_app['fp']}")
    print(f"  FN={cm_app['fn']}  TN={cm_app['tn']}")
    print(f"Recall (TPR): {metrics['recall_app']:.4f}")
    print(f"False Positive Rate: {metrics['fpr_app']:.4f}")
    print(f"Precision: {metrics['precision_app']:.4f}")
    print(f"Accuracy: {metrics['accuracy_app']:.4f}")

    print(f"\nMean score (scam): {metrics['mean_scam_score']:.4f}")
    print(f"Mean score (legit): {metrics['mean_legit_score']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    else:
        print("ROC-AUC: N/A (sklearn not installed)")

    print(f"\nError rate by category:")
    for cat, rate in sorted(metrics['error_rate_by_category'].items()):
        print(f"  {cat}: {rate:.2%}")

    print(f"\nWrong predictions (score-only rule) ({len(metrics['wrong_predictions'])}):")
    for w in metrics['wrong_predictions'][:20]:
        risk = w.get('risk_level', 'N/A')
        print(f"  ID {w['id']:4d} | {w['category']:<25s} | score={w['score']:.4f} | risk={risk} | {w['explanation']}")
    if len(metrics['wrong_predictions']) > 20:
        print(f"  ... and {len(metrics['wrong_predictions']) - 20} more")

    print(f"\nWrong predictions (app rule) ({len(metrics['wrong_predictions_app'])}):")
    for w in metrics['wrong_predictions_app'][:20]:
        risk = w.get('risk_level', 'N/A')
        print(f"  ID {w['id']:4d} | {w['category']:<25s} | score={w['score']:.4f} | risk={risk} | {w['explanation']}")
    if len(metrics['wrong_predictions_app']) > 20:
        print(f"  ... and {len(metrics['wrong_predictions_app']) - 20} more")


def write_report(
    out_dir: Path,
    model_slug: str,
    split: str,
    rows: list[EvalRow],
    threshold: float,
    metrics: dict,
) -> Path:
    """Write markdown report."""
    report_path = out_dir / f"{model_slug}_{split}_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# SMS Evaluation Report\n\n")
        f.write(f"**Model:** {model_slug}\n")
        f.write(f"**Split:** {split}\n")
        f.write(f"**Threshold:** {threshold}\n")
        f.write(f"**Total rows:** {metrics['total_rows']}\n")
        f.write(f"**Completed:** {metrics['completed']}\n")
        f.write(f"**Failed:** {metrics['failed']}\n\n")

        # Score-only rule
        f.write(f"## Score-only Rule (score >= threshold)\n\n")
        cm = metrics['confusion_matrix']
        f.write(f"| | Pred Scam | Pred Legit |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| **Actual Scam** | {cm['tp']} | {cm['fn']} |\n")
        f.write(f"| **Actual Legit** | {cm['fp']} | {cm['tn']} |\n\n")
        f.write(f"- **Recall (TPR):** {metrics['recall']:.4f}\n")
        f.write(f"- **False Positive Rate:** {metrics['fpr']:.4f}\n")
        f.write(f"- **Precision:** {metrics['precision']:.4f}\n")
        f.write(f"- **Accuracy:** {metrics['accuracy']:.4f}\n\n")

        # App rule
        f.write(f"## App Rule (risk_level in high/critical OR score >= threshold)\n\n")
        cm_app = metrics['confusion_matrix_app']
        f.write(f"| | Pred Scam | Pred Legit |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| **Actual Scam** | {cm_app['tp']} | {cm_app['fn']} |\n")
        f.write(f"| **Actual Legit** | {cm_app['fp']} | {cm_app['tn']} |\n\n")
        f.write(f"- **Recall (TPR):** {metrics['recall_app']:.4f}\n")
        f.write(f"- **False Positive Rate:** {metrics['fpr_app']:.4f}\n")
        f.write(f"- **Precision:** {metrics['precision_app']:.4f}\n")
        f.write(f"- **Accuracy:** {metrics['accuracy_app']:.4f}\n\n")

        f.write(f"## Overall Metrics\n\n")
        f.write(f"- **Mean score (scam):** {metrics['mean_scam_score']:.4f}\n")
        f.write(f"- **Mean score (legit):** {metrics['mean_legit_score']:.4f}\n")
        if metrics['roc_auc'] is not None:
            f.write(f"- **ROC-AUC:** {metrics['roc_auc']:.4f}\n")
        else:
            f.write(f"- **ROC-AUC:** N/A (sklearn not installed)\n")

        f.write(f"\n## Error Rate by Category\n\n")
        f.write(f"| Category | Error Rate |\n")
        f.write(f"|---|---|\n")
        for cat, rate in sorted(metrics['error_rate_by_category'].items()):
            f.write(f"| {cat} | {rate:.2%} |\n")

        f.write(f"\n## Wrong Predictions (score-only rule)\n\n")
        f.write(f"| ID | Category | Score | Risk Level | Explanation |\n")
        f.write(f"|---|---|---|---|---|\n")
        for w in metrics['wrong_predictions']:
            risk = w.get('risk_level', 'N/A')
            f.write(f"| {w['id']} | {w['category']} | {w['score']:.4f} | {risk} | {w['explanation']} |\n")

        f.write(f"\n## Wrong Predictions (app rule)\n\n")
        f.write(f"| ID | Category | Score | Risk Level | Explanation |\n")
        f.write(f"|---|---|---|---|---|\n")
        for w in metrics['wrong_predictions_app']:
            risk = w.get('risk_level', 'N/A')
            f.write(f"| {w['id']} | {w['category']} | {w['score']:.4f} | {risk} | {w['explanation']} |\n")

        if metrics['failed'] > 0:
            f.write(f"\n## Failed Rows as Misses (failed_as_miss=True)\n\n")
            metrics_fail = compute_metrics(rows, threshold, failed_as_miss=True)
            f.write(f"- **Recall:** {metrics_fail['recall']:.4f}\n")
            f.write(f"- **FPR:** {metrics_fail['fpr']:.4f}\n")
            f.write(f"- **Precision:** {metrics_fail['precision']:.4f}\n")

    return report_path


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    settings = get_settings()

    # Resolve paths
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / "data" / "sms_eval" / "sms_labelled.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else (script_dir.parent / "reports" / "sms_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_slug = args.model.replace("/", "-").replace(":", "-")
    results_path = out_dir / f"{model_slug}_{args.split}.csv"

    # Load data once
    rows_with_text = load_csv(csv_path, args.split, args.limit)
    if not rows_with_text:
        print("No rows to process", file=sys.stderr)
        return 1

    # Load existing results for resume
    completed = load_existing_results(results_path)
    print(f"Loaded {len(rows_with_text)} rows from CSV, {len(completed)} already completed")

    # Filter out already completed
    pending_with_text = [(r, t) for r, t in rows_with_text if r.id not in completed]
    print(f"Processing {len(pending_with_text)} pending rows...")

    # Process rows
    fieldnames = [
        "id", "label", "category", "split", "sender",
        "score", "risk_level", "patterns", "explanation",
        "status", "error", "total_tokens", "seconds"
    ]

    for i, (row, text) in enumerate(pending_with_text):
        print(f"  [{i+1}/{len(pending_with_text)}] ID {row.id} ({row.category}, {row.split})...", end=" ", flush=True)

        processed = await process_row(row, text, args.model, settings.safety_threshold)

        save_row(results_path, processed, fieldnames)

        status_str = "OK" if processed.status == "completed" else "FAIL"
        print(f"{status_str} score={processed.score:.3f} status={processed.status} tokens={processed.total_tokens} time={processed.seconds:.1f}s")

        if args.delay > 0 and i < len(pending_with_text) - 1:
            await asyncio.sleep(args.delay)

    # Reload all results from CSV for metrics (keeps last row per id)
    final_rows = []
    with open(results_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            final_rows.append(EvalRow(
                id=int(r["id"]),
                label=r["label"],
                category=r["category"],
                split=r["split"],
                sender=r["sender"],
                score=float(r["score"]),
                risk_level=r["risk_level"] if r["risk_level"] else None,
                patterns=json.loads(r["patterns"]) if r["patterns"] else None,
                explanation=r["explanation"],
                status=r["status"],
                error=r["error"] if r["error"] else None,
                total_tokens=int(r["total_tokens"]),
                seconds=float(r["seconds"]),
            ))

    # Deduplicate: keep last row per id
    seen = {}
    for r in final_rows:
        seen[r.id] = r
    final_rows = list(seen.values())

    # Compute and print metrics
    metrics = compute_metrics(final_rows, settings.safety_threshold)
    print_metrics(metrics, args.model, args.split)

    # Write report
    report_path = write_report(out_dir, model_slug, args.split, final_rows, settings.safety_threshold, metrics)
    print(f"\nReport written to: {report_path}")
    print(f"Results CSV: {results_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SMS scam classifier")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test", help="Data split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--delay", type=float, default=15.0, help="Delay between requests (seconds)")
    parser.add_argument("--model", type=str, default=None, help="Model name (default: settings hf_llm_model)")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results (default: always resumes)")

    args = parser.parse_args()

    if args.model is None:
        settings = get_settings()
        args.model = settings.hf_llm_model

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())