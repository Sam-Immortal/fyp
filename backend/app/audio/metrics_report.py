"""Metrics report for ASVspoof LA eval scores CSV (2019 and 2021).

Computes all metrics from the per-file scores CSV without re-scoring.
Outputs JSON, Markdown, and optional PNG plots.

Run as:
    python -m app.audio.metrics_report --scores X.csv --out-dir Y
        [--thresholds 0.75] [--dev-scores dev.csv]
        [--group-by COL [COL...]] [--sampled]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import auc as sklearn_auc, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


ATTACK_TYPE_MAP = {
    # From ASVspoof 2019 database paper (Table 1) and arXiv 1911.01601
    # TTS attacks
    "A01": "TTS", "A02": "TTS", "A03": "TTS", "A04": "TTS",
    "A07": "TTS", "A08": "TTS", "A09": "TTS", "A10": "TTS",
    "A11": "TTS", "A12": "TTS", "A16": "TTS",
    # VC attacks
    "A05": "VC", "A06": "VC",
    "A17": "VC", "A18": "VC", "A19": "VC",
    # Hybrid
    "A13": "hybrid", "A14": "hybrid", "A15": "hybrid",
}

# A16 same algorithm as training A04; A19 same as training A06
ATTACK_NOTES = {
    "A16": "same algorithm as training attack A04",
    "A19": "same algorithm as training attack A06",
}

FULL_EVAL_COUNTS = {
    "total": 71237,
    "bonafide": 7355,
    "spoof": 63882,
}

EXPECTED_ATTACKS = [f"A{i:02d}" for i in range(1, 20)]  # A01-A19


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_scores(
    csv_path: Path
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    list[str], list[str], list[str],
    list[str], list[str], list[str], list[str]
]:
    """Load scores CSV.

    Returns (scores, labels, durations, attack_ids, utt_ids, file_names,
             codecs, transmissions, trims, subsets).
    Missing columns (for 2019 data) return empty lists.
    """
    scores = []
    labels = []
    durations = []
    attack_ids = []
    utt_ids = []
    file_names = []
    codecs = []
    transmissions = []
    trims = []
    subsets = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_extra = "codec" in reader.fieldnames
        for row in reader:
            scores.append(float(row["score"]))
            labels.append(int(row["label"]))
            durations.append(float(row["duration_s"]))
            attack_ids.append(row["attack_id"])
            utt_ids.append(row["utt_id"])
            file_names.append(row["file_name"])
            if has_extra:
                codecs.append(row.get("codec", ""))
                transmissions.append(row.get("transmission", ""))
                trims.append(row.get("trim", ""))
                subsets.append(row.get("subset", ""))
            else:
                codecs.append("")
                transmissions.append("")
                trims.append("")
                subsets.append("")
    return (
        np.array(scores, dtype=np.float64),
        np.array(labels, dtype=np.int32),
        np.array(durations, dtype=np.float64),
        attack_ids,
        utt_ids,
        file_names,
        codecs,
        transmissions,
        trims,
        subsets,
    )


def compute_eer_numpy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute EER (%) and threshold from scratch in numpy. Positive=spoof (1)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need both spoof and bonafide samples for EER")

    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    order = np.argsort(-all_scores, kind="mergesort")
    s = all_scores[order]
    y = all_labels[order]

    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(1 - y)
    p = float(len(pos))
    n = float(len(neg))

    tpr = tp_cum / p
    fpr = fp_cum / n
    fnr = 1 - tpr

    diff = np.abs(fpr - fnr)
    idx = int(np.argmin(diff))
    eer = (fpr[idx] + fnr[idx]) / 2.0 * 100.0
    eer_thr = float(s[idx])
    return float(eer), eer_thr


def compute_auc_numpy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC from scratch using Mann-Whitney U (handles ties correctly)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need both classes for AUC")

    neg_sorted = np.sort(neg)
    less = np.searchsorted(neg_sorted, pos, side="left")
    equal = np.searchsorted(neg_sorted, pos, side="right") - less
    auc_val = float((less + 0.5 * equal).sum()) / (len(pos) * len(neg))
    return auc_val


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for DET/ROC plots."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    order = np.argsort(-all_scores, kind="mergesort")
    s = all_scores[order]
    y = all_labels[order]

    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(1 - y)
    p = float(len(pos))
    n = float(len(neg))

    tpr = tp_cum / p
    fpr = fp_cum / n

    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    thresholds = np.concatenate([[s[0] + 1e-9], s]) if len(s) > 0 else np.array([1.0, 0.0])
    return fpr, tpr, thresholds


def confusion_at_threshold(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred = (scores >= thr).astype(np.int32)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    p = tp + fn
    n = fp + tn
    total = p + n

    acc = (tp + tn) / total if total else 0.0
    tpr = tp / p if p else 0.0
    tnr = tn / n if n else 0.0
    fpr = fp / n if n else 0.0
    fnr = fn / p if p else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    bal_acc = (tpr + tnr) / 2.0
    f1_spoof = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_bonafide = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    mcc_num = tp * tn - fp * fn
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 1.0
    mcc = mcc_num / mcc_den if mcc_den else 0.0
    always_spoof_acc = p / total if total else 0.0

    tpr_lo, tpr_hi = wilson_interval(tpr, p)
    fpr_lo, fpr_hi = wilson_interval(fpr, n)

    return {
        "threshold": thr,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall_tpr": tpr,
        "specificity_tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "f1_spoof": f1_spoof,
        "f1_bonafide": f1_bonafide,
        "mcc": mcc,
        "always_spoof_baseline_accuracy": always_spoof_acc,
        "recall_wilson_95": [tpr_lo, tpr_hi],
        "fpr_wilson_95": [fpr_lo, fpr_hi],
    }


def group_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    group_values: list[str],
    thresholds: list[float],
    bonafide_scores: np.ndarray,
    is_attack_id: bool = False,
) -> dict:
    """Metrics per group value.

    For attack_id: compare that attack's spoofs against ALL bonafide.
    For other columns: use only rows of that group.
    """
    out = {}
    unique_vals = sorted(set(v for v in group_values if v))
    for val in unique_vals:
        mask = np.array([v == val for v in group_values], dtype=bool)
        if not mask.any():
            continue
        g_scores = scores[mask]
        g_labels = labels[mask]
        n = int(mask.sum())
        if n == 0:
            continue
        n_spoof = int(np.sum(g_labels == 1))
        n_bonafide = int(np.sum(g_labels == 0))
        spoof_share = n_spoof / n if n else 0.0
        mean_score = float(g_scores.mean())

        # EER and AUC
        try:
            if is_attack_id:
                eer, _ = compute_eer_numpy(
                    np.concatenate([g_scores[g_labels == 1], bonafide_scores]),
                    np.concatenate([np.ones(n_spoof), np.zeros(len(bonafide_scores))]),
                )
            else:
                eer, _ = compute_eer_numpy(g_scores, g_labels)
        except ValueError:
            eer = None
        try:
            if not is_attack_id:
                auc_val = compute_auc_numpy(g_scores, g_labels)
            else:
                auc_val = None
        except ValueError:
            auc_val = None

        # Recall/specificity/FPR/FNR/F1 at each threshold
        thresholds_metrics = {}
        for thr in thresholds:
            tp = int(np.sum((g_scores >= thr) & (g_labels == 1)))
            fn = int(np.sum((g_scores < thr) & (g_labels == 1)))
            fp = int(np.sum((g_scores >= thr) & (g_labels == 0)))
            tn = int(np.sum((g_scores < thr) & (g_labels == 0)))
            p = tp + fn
            n_neg = fp + tn
            tpr = tp / p if p else 0.0
            tnr = tn / n_neg if n_neg else 0.0
            fpr = fp / n_neg if n_neg else 0.0
            fnr = fn / p if p else 0.0
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            f1_spoof = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            f1_bonafide = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
            thresholds_metrics[f"{thr:.6f}"] = {
                "recall_tpr": tpr,
                "specificity_tnr": tnr,
                "fpr": fpr,
                "fnr": fnr,
                "precision": prec,
                "f1_spoof": f1_spoof,
                "f1_bonafide": f1_bonafide,
            }

        out[val] = {
            "n": n,
            "n_spoof": n_spoof,
            "n_bonafide": n_bonafide,
            "spoof_share": spoof_share,
            "mean_score": mean_score,
            "eer": eer,
            "auc": auc_val,
            "thresholds": thresholds_metrics,
        }
    return out


def attack_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    attack_ids: list[str],
    thresholds: list[float],
    bonafide_scores: np.ndarray,
) -> dict:
    """Per-attack metrics (uses group_metrics with is_attack_id=True)."""
    return group_metrics(scores, labels, attack_ids, thresholds, bonafide_scores, is_attack_id=True)


def type_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    attack_ids: list[str],
    thresholds: list[float],
    bonafide_scores: np.ndarray,
) -> dict:
    """Per-type metrics (TTS/VC/hybrid)."""
    out = {}
    for typ in ("TTS", "VC", "hybrid"):
        mask = np.array([ATTACK_TYPE_MAP.get(a, "unknown") == typ for a in attack_ids], dtype=bool)
        if not mask.any():
            out[typ] = {"n": 0}
            continue
        typ_scores = scores[mask]
        typ_labels = labels[mask]
        n = int(mask.sum())
        mean_score = float(typ_scores.mean())
        recalls = {}
        for thr in thresholds:
            tp = int(np.sum((typ_scores >= thr) & (typ_labels == 1)))
            p = int(np.sum(typ_labels == 1))
            recalls[f"recall_at_{thr:.3f}"] = tp / p if p else 0.0
        try:
            eer, _ = compute_eer_numpy(
                np.concatenate([typ_scores[typ_labels == 1], bonafide_scores]),
                np.concatenate([np.ones(np.sum(typ_labels == 1)), np.zeros(len(bonafide_scores))]),
            )
        except ValueError:
            eer = None
        out[typ] = {
            "n": n,
            "mean_score": mean_score,
            "recalls": recalls,
            "eer_vs_bonafide": eer,
        }
    return out


def duration_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    durations: np.ndarray,
    window_modes: list[str],
    thresholds: list[float],
) -> dict:
    """Duration statistics by class and terciles."""
    bonafide = labels == 0
    spoof = labels == 1

    def stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
        return {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "max": float(arr.max()),
        }

    modes = np.array(window_modes)
    bonafide_modes = modes[bonafide]
    spoof_modes = modes[spoof]

    def mode_share(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {"padded": 0.0, "truncated": 0.0, "exact": 0.0}
        total = len(arr)
        return {
            "padded": float(np.sum(arr == "padded")) / total,
            "truncated": float(np.sum(arr == "truncated")) / total,
            "exact": float(np.sum(arr == "exact")) / total,
        }

    all_dur = durations
    if len(all_dur) >= 3:
        q1, q2 = np.percentile(all_dur, [33.33, 66.66])
    else:
        q1 = q2 = float(np.median(all_dur)) if len(all_dur) > 0 else 0.0

    tercile_results = {}
    for lo, hi, name in [(0.0, q1, "low"), (q1, q2, "mid"), (q2, float("inf"), "high")]:
        mask = (durations >= lo) & (durations < hi)
        if not mask.any():
            tercile_results[name] = {"n": 0, "eer": None, "recall_at_0.75": 0.0, "fpr_at_0.75": 0.0}
            continue
        bin_scores = scores[mask]
        bin_labels = labels[mask]
        try:
            eer, _ = compute_eer_numpy(bin_scores, bin_labels)
        except ValueError:
            eer = None
        tp = int(np.sum((bin_scores >= 0.75) & (bin_labels == 1)))
        fp = int(np.sum((bin_scores >= 0.75) & (bin_labels == 0)))
        p = int(np.sum(bin_labels == 1))
        n = int(np.sum(bin_labels == 0))
        recall = tp / p if p else 0.0
        fpr = fp / n if n else 0.0
        tercile_results[name] = {
            "n": int(mask.sum()),
            "duration_range": [float(lo), float(hi) if hi != float("inf") else None],
            "eer": eer,
            "recall_at_0.75": recall,
            "fpr_at_0.75": fpr,
        }

    return {
        "by_class": {
            "bonafide": {
                "duration": stats(durations[bonafide]),
                "window_mode_share": mode_share(bonafide_modes),
            },
            "spoof": {
                "duration": stats(durations[spoof]),
                "window_mode_share": mode_share(spoof_modes),
            },
        },
        "by_tercile": tercile_results,
    }


def score_percentiles(scores: np.ndarray, labels: np.ndarray) -> dict:
    pcts = [1, 5, 25, 50, 75, 95, 99]
    bonafide = scores[labels == 0]
    spoof = scores[labels == 1]
    return {
        "bonafide": {f"p{p}": float(np.percentile(bonafide, p)) if len(bonafide) else 0.0 for p in pcts},
        "spoof": {f"p{p}": float(np.percentile(spoof, p)) if len(spoof) else 0.0 for p in pcts},
    }


def illustrative_precision(tpr: float, fpr: float, prevalences: list[float]) -> dict:
    """Compute precision at given spoof prevalences from TPR and FPR.

    Precision = (TPR * prev) / (TPR * prev + FPR * (1 - prev))
    """
    out = {}
    for prev in prevalences:
        if prev == 0:
            out[f"prev_{int(prev*100)}%"] = 0.0
        else:
            num = tpr * prev
            den = tpr * prev + fpr * (1 - prev)
            out[f"prev_{int(prev*100)}%"] = num / den if den else 0.0
    return out


def write_plots(
    out_dir: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
) -> None:
    """Write histogram and DET curve PNGs if matplotlib available."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    bins = np.linspace(0, 1, 101)
    ax.hist(pos, bins=bins, alpha=0.7, label="Spoof", density=True, log=True)
    ax.hist(neg, bins=bins, alpha=0.7, label="Bonafide", density=True, log=True)
    ax.set_xlabel("Anomaly score (P(spoof))")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Score Distribution by Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_histogram.png", dpi=150)
    plt.close(fig)

    fnr = 1 - tpr
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, fnr)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FPR (log)")
    ax.set_ylabel("FNR (log)")
    ax.set_title("DET Curve")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(out_dir / "det_curve.png", dpi=150)
    plt.close(fig)


def min_t_dcf_note() -> str:
    """Return note about min t-DCF."""
    asv_scores_path = Path("data/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
    if asv_scores_path.exists():
        return (
            "min t-DCF: not computed. ASV scores file exists at "
            f"{asv_scores_path} but the official evaluate_tDCF_asvspoof19.py "
            "script and cost parameters (C_miss, C_fa, P_target, P_spoof) from the "
            "ASVspoof 2019 evaluation plan are not available in this repository. "
            "NEVER guess parameters."
        )
    return (
        "min t-DCF: not computed. ASV scores file not found at "
        f"{asv_scores_path}. Official implementation and cost parameters "
        "from ASVspoof 2019 evaluation plan required."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Metrics report from scores CSV")
    parser.add_argument("--scores", type=Path, required=True, help="Scores CSV from eval_scores.py")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for reports")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.75], help="Thresholds to evaluate")
    parser.add_argument("--dev-scores", type=Path, default=None, help="Optional dev scores CSV for dev EER threshold")
    parser.add_argument("--group-by", type=str, nargs="*", default=[],
                        help="Columns to group by (codec, transmission, attack_id, trim, subset)")
    parser.add_argument("--sampled", action="store_true",
                        help="Print caution that metrics reflect sampled class mix")
    args = parser.parse_args()

    if not args.scores.exists():
        print(f"Scores file not found: {args.scores}", file=sys.stderr)
        sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    (scores, labels, durations, attack_ids, utt_ids, file_names,
     codecs, transmissions, trims, subsets) = load_scores(args.scores)
    bonafide_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]

    total = len(labels)
    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)
    spoof_share = n_spoof / total if total else 0.0

    attack_counts: dict[str, int] = {}
    for att in attack_ids:
        if att != "-":
            attack_counts[att] = attack_counts.get(att, 0) + 1

    counts_match = (
        total == FULL_EVAL_COUNTS["total"] and
        n_bonafide == FULL_EVAL_COUNTS["bonafide"] and
        n_spoof == FULL_EVAL_COUNTS["spoof"]
    )
    counts_info = {
        "total": total,
        "bonafide": n_bonafide,
        "spoof": n_spoof,
        "spoof_share": spoof_share,
        "per_attack": attack_counts,
        "matches_full_eval": counts_match,
        "note": "Matches full eval counts" if counts_match else "Partial CSV - counts differ from full eval (expected)."
    }

    try:
        eer, eer_thr = compute_eer_numpy(scores, labels)
        auc_val = compute_auc_numpy(scores, labels)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    sklearn_cross_check = None
    if SKLEARN_AVAILABLE:
        fpr_sk, tpr_sk, thr_sk = roc_curve(labels, scores, pos_label=1)
        eer_sk_idx = np.argmin(np.abs(fpr_sk - (1 - tpr_sk)))
        eer_sk = (fpr_sk[eer_sk_idx] + (1 - tpr_sk[eer_sk_idx])) / 2 * 100
        auc_sk = sklearn_auc(fpr_sk, tpr_sk)
        eer_diff = abs(eer - eer_sk)
        auc_diff = abs(auc_val - auc_sk)
        passed = (eer_diff <= 0.02) and (auc_diff <= 1e-4)
        sklearn_cross_check = {
            "sklearn_eer": float(eer_sk),
            "sklearn_auc": float(auc_sk),
            "eer_diff_pp": float(eer_diff),
            "auc_diff": float(auc_diff),
            "passed": passed,
            "note": "PASS" if passed else (
                f"FAIL: EER diff {eer_diff:.4f}pp > 0.02pp or AUC diff {auc_diff:.6f} > 1e-4"
            )
        }
        if not passed:
            print(f"WARNING: sklearn cross-check FAILED: {sklearn_cross_check['note']}", file=sys.stderr)
    else:
        sklearn_cross_check = {"available": False, "note": "scikit-learn not installed (eval extra not used)"}

    eval_thresholds = list(args.thresholds)
    if eer_thr not in eval_thresholds:
        eval_thresholds.append(eer_thr)

    dev_eer_thr = None
    if args.dev_scores and args.dev_scores.exists():
        dev_scores, dev_labels, _, _, _, _, _, _, _, _ = load_scores(args.dev_scores)
        try:
            _, dev_eer_thr = compute_eer_numpy(dev_scores, dev_labels)
            if dev_eer_thr not in eval_thresholds:
                eval_thresholds.append(dev_eer_thr)
        except ValueError:
            pass

    threshold_metrics = {}
    for thr in sorted(set(eval_thresholds)):
        threshold_metrics[f"{thr:.6f}"] = confusion_at_threshold(scores, labels, thr)

    per_attack = attack_metrics(scores, labels, attack_ids, eval_thresholds, bonafide_scores)
    per_type = type_metrics(scores, labels, attack_ids, eval_thresholds, bonafide_scores)

    window_modes = []
    with open(args.scores, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            window_modes.append(row["window_mode"])
    duration_info = duration_metrics(scores, labels, durations, window_modes, eval_thresholds)

    percentiles = score_percentiles(scores, labels)

    t_dcf_note = min_t_dcf_note()

    fpr_arr, tpr_arr, _ = roc_points(scores, labels)

    # Group-by metrics
    group_by_results = {}
    for col in args.group_by:
        if col == "codec":
            group_by_results[col] = group_metrics(scores, labels, codecs, eval_thresholds, bonafide_scores, is_attack_id=False)
        elif col == "transmission":
            group_by_results[col] = group_metrics(scores, labels, transmissions, eval_thresholds, bonafide_scores, is_attack_id=False)
        elif col == "attack_id":
            group_by_results[col] = group_metrics(scores, labels, attack_ids, eval_thresholds, bonafide_scores, is_attack_id=True)
        elif col == "trim":
            group_by_results[col] = group_metrics(scores, labels, trims, eval_thresholds, bonafide_scores, is_attack_id=False)
        elif col == "subset":
            group_by_results[col] = group_metrics(scores, labels, subsets, eval_thresholds, bonafide_scores, is_attack_id=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scores_file": str(args.scores),
        "sampled": args.sampled,
        "counts": counts_info,
        "threshold_free": {
            "eer_percent": eer,
            "eer_threshold": eer_thr,
            "auc": auc_val,
            "sklearn_cross_check": sklearn_cross_check,
        },
        "thresholds": threshold_metrics,
        "per_attack": per_attack,
        "per_type": per_type,
        "duration": duration_info,
        "score_percentiles": percentiles,
        "min_t_dcf": t_dcf_note,
        "group_by": group_by_results,
    }

    json_path = args.out_dir / "metrics_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = args.out_dir / "metrics_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ASVspoof LA Eval Metrics Report\n\n")
        f.write(f"Generated: {report['generated_at']}\n")
        f.write(f"Scores file: {args.scores}\n")
        if args.sampled:
            f.write("**CAUTION: This report uses a stratified sample. Accuracy, precision, F1, and MCC reflect the sample's class mix, not the population.**\n")
        f.write("\n")

        f.write("## 1. Counts\n\n")
        c = counts_info
        f.write(f"- Total: {c['total']}\n")
        f.write(f"- Bonafide: {c['bonafide']} ({c['bonafide']/c['total']*100:.2f}%)\n")
        f.write(f"- Spoof: {c['spoof']} ({c['spoof']*100:.2f}%)\n")
        f.write(f"- Spoof share: {c['spoof_share']:.4f}\n")
        f.write(f"- Matches full eval: {c['matches_full_eval']}\n")
        f.write(f"- Note: {c['note']}\n\n")
        f.write("### Per-attack counts\n\n")
        f.write("| Attack | Count |\n|--------|-------|\n")
        for att in sorted(attack_counts):
            f.write(f"| {att} | {attack_counts[att]} |\n")
        f.write("\n")

        f.write("## 2. Threshold-free Metrics\n\n")
        tf = report["threshold_free"]
        f.write(f"- EER: {tf['eer_percent']:.4f}% (at threshold {tf['eer_threshold']:.6f})\n")
        f.write(f"- AUC: {tf['auc']:.6f}\n")
        if SKLEARN_AVAILABLE:
            sc = tf["sklearn_cross_check"]
            f.write(f"- sklearn EER: {sc['sklearn_eer']:.4f}%\n")
            f.write(f"- sklearn AUC: {sc['sklearn_auc']:.6f}\n")
            f.write(f"- EER diff: {sc['eer_diff_pp']:.4f} pp\n")
            f.write(f"- AUC diff: {sc['auc_diff']:.6f}\n")
            f.write(f"- Cross-check: {sc['note']}\n")
        else:
            f.write("- sklearn cross-check: skipped (scikit-learn not installed)\n")
        f.write("\n")

        f.write("## 3. Metrics at Thresholds\n\n")
        for thr_key, m in threshold_metrics.items():
            f.write(f"### Threshold = {thr_key}\n\n")
            f.write(f"- Confusion: TP={m['TP']} FN={m['FN']} FP={m['FP']} TN={m['TN']}\n")
            f.write(f"- Accuracy: {m['accuracy']:.6f}\n")
            f.write(f"- Balanced accuracy: {m['balanced_accuracy']:.6f}\n")
            f.write(f"- Precision: {m['precision']:.6f}\n")
            f.write(f"- Recall (TPR): {m['recall_tpr']:.6f} (Wilson 95%: [{m['recall_wilson_95'][0]:.6f}, {m['recall_wilson_95'][1]:.6f}])\n")
            f.write(f"- Specificity (TNR): {m['specificity_tnr']:.6f}\n")
            f.write(f"- FPR: {m['fpr']:.6f} (Wilson 95%: [{m['fpr_wilson_95'][0]:.6f}, {m['fpr_wilson_95'][1]:.6f}])\n")
            f.write(f"- FNR: {m['fnr']:.6f}\n")
            f.write(f"- F1 (spoof): {m['f1_spoof']:.6f}\n")
            f.write(f"- F1 (bonafide): {m['f1_bonafide']:.6f}\n")
            f.write(f"- MCC: {m['mcc']:.6f}\n")
            f.write(f"- Always-spoof baseline accuracy: {m['always_spoof_baseline_accuracy']:.6f}\n\n")

        # Illustrative precision at assumed spoof prevalence
        f.write("## 4. Illustrative Precision at Assumed Spoof Prevalence\n\n")
        f.write("Computed from TPR and FPR at each threshold: Precision = (TPR * prev) / (TPR * prev + FPR * (1 - prev))\n\n")
        f.write("| Threshold | Prevalence | Precision |\n")
        f.write("|-----------|------------|-----------|\n")
        for thr_key, m in threshold_metrics.items():
            tpr = m["recall_tpr"]
            fpr = m["fpr"]
            for prev in [0.5, 0.1, 0.01]:
                prec = illustrative_precision(tpr, fpr, [prev])[f"prev_{int(prev*100)}%"]
                f.write(f"| {thr_key} | {prev:.0%} | {prec:.6f} |\n")
        f.write("\n")

        f.write("## 5. Per-Attack Metrics\n\n")
        f.write("| Attack | Type | n | Mean Score | EER vs Bonafide | Note |\n")
        f.write("|--------|------|---|------------|------------------|------|\n")
        for att in sorted(per_attack):
            a = per_attack[att]
            atype = a.get("type", ATTACK_TYPE_MAP.get(att, "unknown"))
            note = a.get("note", ATTACK_NOTES.get(att, ""))
            eer_val = a.get("eer")
            eer_str = f"{eer_val:.4f}%" if eer_val is not None else "n/a"
            f.write(
                f"| {att} | {atype} | {a['n']} | {a['mean_score']:.6f} | "
                f"{eer_str} | {note} |\n"
            )
            # group_metrics stores per-threshold metrics in "thresholds"
            if "thresholds" in a:
                for thr_key, tm in sorted(a["thresholds"].items()):
                    f.write(f"| | | | Recall @ {thr_key}: {tm['recall_tpr']:.6f} | |\n")
            elif "recalls" in a:
                for thr_key in sorted(set([f"{t:.3f}" for t in eval_thresholds])):
                    rkey = f"recall_at_{thr_key}"
                    if rkey in a["recalls"]:
                        f.write(f"| | | | Recall @ {thr_key}: {a['recalls'][rkey]:.6f} | |\n")
        f.write("\n")

        f.write("### Per-Type Metrics\n\n")
        f.write("| Type | n | Mean Score | EER vs Bonafide |\n")
        f.write("|------|---|------------|------------------|\n")
        for typ in ["TTS", "VC", "hybrid"]:
            t = per_type.get(typ, {"n": 0})
            f.write(f"| {typ} | {t.get('n', 0)} | {t.get('mean_score', 0):.6f} | {t.get('eer_vs_bonafide', 'N/A')} |\n")
        f.write("\n")

        # Group-by tables
        if group_by_results:
            f.write("## 6. Group-By Metrics\n\n")
            for col, groups in group_by_results.items():
                f.write(f"### Grouped by {col}\n\n")
                f.write("| Group | n | Spoof | Bonafide | Spoof Share | Mean Score | EER | AUC |\n")
                f.write("|-------|---|-------|----------|-------------|------------|-----|-----|\n")
                for grp, vals in sorted(groups.items()):
                    eer_str = f"{vals['eer']:.4f}%" if vals['eer'] is not None else "n/a"
                    auc_str = f"{vals['auc']:.6f}" if vals['auc'] is not None else "n/a"
                    f.write(
                        f"| {grp} | {vals['n']} | {vals['n_spoof']} | {vals['n_bonafide']} | "
                        f"{vals['spoof_share']:.4f} | {vals['mean_score']:.6f} | "
                        f"{eer_str} | {auc_str} |\n"
                    )
                f.write("\n")
                # Threshold-specific metrics per group
                f.write(f"#### Per-threshold metrics by {col}\n\n")
                for grp, vals in sorted(groups.items()):
                    f.write(f"##### {grp}\n\n")
                    f.write("| Threshold | Recall (TPR) | Specificity (TNR) | FPR | FNR | Precision | F1 (spoof) | F1 (bonafide) |\n")
                    f.write("|-----------|--------------|-------------------|-----|-----|-----------|------------|---------------|\n")
                    for thr_key, tm in sorted(vals["thresholds"].items()):
                        f.write(
                            f"| {thr_key} | {tm['recall_tpr']:.6f} | {tm['specificity_tnr']:.6f} | "
                            f"{tm['fpr']:.6f} | {tm['fnr']:.6f} | {tm['precision']:.6f} | "
                            f"{tm['f1_spoof']:.6f} | {tm['f1_bonafide']:.6f} |\n"
                        )
                    f.write("\n")

        f.write("## 7. Duration Check\n\n")
        d = duration_info
        for cls in ["bonafide", "spoof"]:
            cls_d = d["by_class"][cls]
            f.write(f"### {cls.capitalize()}\n\n")
            dur = cls_d["duration"]
            f.write(f"- Duration (s): min={dur['min']:.3f} median={dur['median']:.3f} mean={dur['mean']:.3f} max={dur['max']:.3f}\n")
            modes = cls_d["window_mode_share"]
            f.write(f"- Window mode share: padded={modes['padded']:.3f} truncated={modes['truncated']:.3f} exact={modes['exact']:.3f}\n\n")

        f.write("### Duration Terciles\n\n")
        f.write("| Tercile | n | Duration Range | EER | Recall@0.75 | FPR@0.75 |\n")
        f.write("|---------|---|----------------|-----|-------------|----------|\n")
        for name in ["low", "mid", "high"]:
            t = d["by_tercile"][name]
            rng = t["duration_range"]
            rng_str = f"[{rng[0]:.3f}, {rng[1]:.3f})" if rng[1] else f"[{rng[0]:.3f}, inf)"
            eer_str = f"{t['eer']:.4f}%" if t['eer'] is not None else "N/A"
            f.write(f"| {name} | {t['n']} | {rng_str} | {eer_str} | {t['recall_at_0.75']:.6f} | {t['fpr_at_0.75']:.6f} |\n")
        f.write("\n")

        f.write("## 8. Score Percentiles\n\n")
        f.write("| Class | P1 | P5 | P25 | P50 | P75 | P95 | P99 |\n")
        f.write("|-------|----|----|-----|-----|-----|-----|-----|\n")
        for cls in ["bonafide", "spoof"]:
            p = percentiles[cls]
            f.write(f"| {cls} | {p['p1']:.6f} | {p['p5']:.6f} | {p['p25']:.6f} | {p['p50']:.6f} | {p['p75']:.6f} | {p['p95']:.6f} | {p['p99']:.6f} |\n")
        f.write("\n")

        f.write("## 9. Min t-DCF\n\n")
        f.write(f"{t_dcf_note}\n")

    if MATPLOTLIB_AVAILABLE:
        write_plots(args.out_dir, scores, labels, fpr_arr, tpr_arr)

    print(json.dumps({
        "eer_percent": eer,
        "eer_threshold": eer_thr,
        "auc": auc_val,
        "counts": {"total": total, "bonafide": n_bonafide, "spoof": n_spoof},
        "sklearn_cross_check": "PASS" if (sklearn_cross_check and sklearn_cross_check.get("passed")) else "SKIP/FAIL",
    }, indent=2))

    print(f"\nReports written to: {args.out_dir}")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    if MATPLOTLIB_AVAILABLE:
        print(f"  - {args.out_dir}/score_histogram.png")
        print(f"  - {args.out_dir}/det_curve.png")


if __name__ == "__main__":
    main()