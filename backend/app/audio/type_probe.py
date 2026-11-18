"""Type probe experiment for ASVspoof 2019 LA.

Uses embeddings to classify spoof type (TTS vs VC). Fit on dev (A01-A06),
test on eval (A07-A19 + hybrid). Includes control probes and leakage checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# Import the attack type mapping from metrics_report
from app.audio.metrics_report import ATTACK_TYPE_MAP, ATTACK_NOTES

RANDOM_STATE = 1234
TSNE_MAX_SAMPLES = 4000

# Attacks that are same algorithm as training attacks (exclude from headline)
SAME_AS_TRAIN_ATTACKS = {"A16", "A19"}  # A16 = A04, A19 = A06


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_scores_csv(csv_path: Path) -> dict:
    """Load scores CSV and return dict with arrays and metadata."""
    scores = []
    labels = []
    attack_ids = []
    durations = []
    utt_ids = []
    speaker_ids = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["score"]))
            labels.append(int(row["label"]))
            attack_ids.append(row["attack_id"])
            durations.append(float(row["duration_s"]))
            utt_ids.append(row["utt_id"])
            speaker_ids.append(row.get("speaker_id", ""))
    return {
        "scores": np.array(scores, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int32),
        "attack_ids": attack_ids,
        "durations": np.array(durations, dtype=np.float32),
        "utt_ids": utt_ids,
        "speaker_ids": speaker_ids,
    }


def load_embeddings(emb_path: Path, expected_rows: int, dim: int = 160) -> np.ndarray:
    """Load embeddings from .emb.f32 sidecar file."""
    data = np.fromfile(emb_path, dtype=np.float32)
    if len(data) != expected_rows * dim:
        raise ValueError(
            f"Embeddings file {emb_path} has {len(data)} values, "
            f"expected {expected_rows * dim} ({expected_rows} rows * {dim} dim)"
        )
    return data.reshape(expected_rows, dim).astype(np.float32)


def get_type_label(attack_id: str) -> str | None:
    """Map attack_id to type: 'TTS', 'VC', 'hybrid', or None for bonafide."""
    if attack_id == "-":
        return None
    return ATTACK_TYPE_MAP.get(attack_id, "unknown")


def prepare_data(
    scores_data: dict,
    embeddings: np.ndarray,
    include_bonafide: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Prepare features and labels for training/testing.

    Returns:
        X: features (embeddings)
        y: labels (0=TTS, 1=VC, 2=bonafide if include_bonafide)
        attack_ids: list of attack_ids for each sample
        durations: duration for each sample
    """
    attack_ids = scores_data["attack_ids"]
    labels = scores_data["labels"]  # 1=spoof, 0=bonafide
    durations = scores_data["durations"]

    X_list = []
    y_list = []
    att_list = []
    dur_list = []

    for i, (att, lbl) in enumerate(zip(attack_ids, labels)):
        if lbl == 0:  # bonafide
            if include_bonafide:
                X_list.append(embeddings[i])
                y_list.append(2)  # bonafide class
                att_list.append(att)
                dur_list.append(durations[i])
            continue

        # Spoof: map attack to type
        typ = get_type_label(att)
        if typ == "TTS":
            y_list.append(0)
        elif typ == "VC":
            y_list.append(1)
        elif typ == "hybrid":
            y_list.append(3)  # hybrid - will be handled separately
        else:
            continue  # skip unknown

        X_list.append(embeddings[i])
        att_list.append(att)
        dur_list.append(durations[i])

    return np.array(X_list), np.array(y_list), att_list, np.array(dur_list)


def train_probe(X_train: np.ndarray, y_train: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    """Train StandardScaler + LogisticRegression probe."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
        multi_class="multinomial",
    )
    clf.fit(X_scaled, y_train)
    return scaler, clf


def evaluate_probe(
    scaler: StandardScaler,
    clf: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    attack_ids: list[str],
) -> dict:
    """Evaluate probe and return detailed metrics."""
    X_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)

    # Overall metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    # Per-attack metrics
    unique_attacks = sorted(set(attack_ids))
    per_attack = {}
    for att in unique_attacks:
        mask = np.array([a == att for a in attack_ids])
        if not mask.any():
            continue
        n = int(mask.sum())
        y_true = y_test[mask]
        y_pred_att = y_pred[mask]
        y_prob_att = y_prob[mask]

        # Accuracy with Wilson interval
        acc_att = accuracy_score(y_true, y_pred_att)
        lo, hi = wilson_interval(acc_att, n)

        # Fraction predicted TTS/VC
        frac_tts = float(np.mean(y_pred_att == 0))
        frac_vc = float(np.mean(y_pred_att == 1))

        per_attack[att] = {
            "n": n,
            "accuracy": acc_att,
            "wilson_95": [lo, hi],
            "frac_pred_tts": frac_tts,
            "frac_pred_vc": frac_vc,
        }

    # Headline accuracy excluding same-as-train attacks (A16, A19)
    headline_mask = np.array([a not in SAME_AS_TRAIN_ATTACKS for a in attack_ids])
    if headline_mask.any():
        y_test_headline = y_test[headline_mask]
        y_pred_headline = y_pred[headline_mask]
        headline_acc = accuracy_score(y_test_headline, y_pred_headline)
        headline_bal_acc = balanced_accuracy_score(y_test_headline, y_pred_headline)
        headline_macro_f1 = f1_score(y_test_headline, y_pred_headline, average="macro")
    else:
        headline_acc = headline_bal_acc = headline_macro_f1 = 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "per_attack": per_attack,
        "headline_excl_same_as_train": {
            "accuracy": float(headline_acc),
            "balanced_accuracy": float(headline_bal_acc),
            "macro_f1": float(headline_macro_f1),
            "excluded_attacks": sorted(SAME_AS_TRAIN_ATTACKS),
        },
    }


def run_control_probes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    attack_ids_test: list[str],
    durations_train: np.ndarray,
    durations_test: np.ndarray,
    scores_train: np.ndarray,
    scores_test: np.ndarray,
    # Filtered arrays for spoof-only
    X_train_spoof: np.ndarray,
    y_train_spoof: np.ndarray,
    X_test_spoof: np.ndarray,
    y_test_spoof: np.ndarray,
    durations_train_spoof: np.ndarray,
    durations_test_spoof: np.ndarray,
    scores_train_spoof: np.ndarray,
    scores_test_spoof: np.ndarray,
) -> dict:
    """Run control probes: majority class, score only, duration only, score+duration, label-shuffle.

    All probes use StandardScaler + LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE).
    """
    results = {}

    # 1. Majority class
    majority_class = int(np.bincount(y_train_spoof).argmax())
    y_pred_maj = np.full_like(y_test_spoof, majority_class)
    results["majority_class"] = {
        "accuracy": float(accuracy_score(y_test_spoof, y_pred_maj)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_spoof, y_pred_maj)),
        "macro_f1": float(f1_score(y_test_spoof, y_pred_maj, average="macro")),
        "metric": "accuracy, balanced_accuracy, macro_f1",
        "class_weighting": "balanced",
    }

    # 2. Score only (single feature)
    scaler_s = StandardScaler()
    X_train_s = scaler_s.fit_transform(scores_train_spoof.reshape(-1, 1))
    X_test_s = scaler_s.transform(scores_test_spoof.reshape(-1, 1))
    clf_s = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    clf_s.fit(X_train_s, y_train_spoof)
    y_pred_s = clf_s.predict(X_test_s)
    results["score_only"] = {
        "accuracy": float(accuracy_score(y_test_spoof, y_pred_s)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_spoof, y_pred_s)),
        "macro_f1": float(f1_score(y_test_spoof, y_pred_s, average="macro")),
        "metric": "accuracy, balanced_accuracy, macro_f1",
        "class_weighting": "balanced",
    }

    # 3. Duration only
    scaler_d = StandardScaler()
    X_train_d = scaler_d.fit_transform(durations_train_spoof.reshape(-1, 1))
    X_test_d = scaler_d.transform(durations_test_spoof.reshape(-1, 1))
    clf_d = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    clf_d.fit(X_train_d, y_train_spoof)
    y_pred_d = clf_d.predict(X_test_d)
    results["duration_only"] = {
        "accuracy": float(accuracy_score(y_test_spoof, y_pred_d)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_spoof, y_pred_d)),
        "macro_f1": float(f1_score(y_test_spoof, y_pred_d, average="macro")),
        "metric": "accuracy, balanced_accuracy, macro_f1",
        "class_weighting": "balanced",
    }

    # 4. Score + Duration
    X_train_sd = np.column_stack([scores_train_spoof, durations_train_spoof])
    X_test_sd = np.column_stack([scores_test_spoof, durations_test_spoof])
    scaler_sd = StandardScaler()
    X_train_sd = scaler_sd.fit_transform(X_train_sd)
    X_test_sd = scaler_sd.transform(X_test_sd)
    clf_sd = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    clf_sd.fit(X_train_sd, y_train_spoof)
    y_pred_sd = clf_sd.predict(X_test_sd)
    results["score_duration"] = {
        "accuracy": float(accuracy_score(y_test_spoof, y_pred_sd)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_spoof, y_pred_sd)),
        "macro_f1": float(f1_score(y_test_spoof, y_pred_sd, average="macro")),
        "metric": "accuracy, balanced_accuracy, macro_f1",
        "class_weighting": "balanced",
    }

    # 5. Label-shuffle control (20 repeats)
    shuffle_accs = []
    shuffle_bal_accs = []
    shuffle_macro_f1s = []
    for i in range(20):
        rng = np.random.default_rng(RANDOM_STATE + i)
        y_train_shuffled = rng.permutation(y_train_spoof)
        scaler_shuf = StandardScaler()
        X_train_shuf = scaler_shuf.fit_transform(X_train_spoof)
        X_test_shuf = scaler_shuf.transform(X_test_spoof)
        clf_shuf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE + i, solver="lbfgs")
        clf_shuf.fit(X_train_shuf, y_train_shuffled)
        y_pred_shuf = clf_shuf.predict(X_test_spoof)
        shuffle_accs.append(accuracy_score(y_test_spoof, y_pred_shuf))
        shuffle_bal_accs.append(balanced_accuracy_score(y_test_spoof, y_pred_shuf))
        shuffle_macro_f1s.append(f1_score(y_test_spoof, y_pred_shuf, average="macro"))
    results["label_shuffle_20x"] = {
        "accuracy_mean": float(np.mean(shuffle_accs)),
        "accuracy_std": float(np.std(shuffle_accs)),
        "balanced_accuracy_mean": float(np.mean(shuffle_bal_accs)),
        "balanced_accuracy_std": float(np.std(shuffle_bal_accs)),
        "macro_f1_mean": float(np.mean(shuffle_macro_f1s)),
        "macro_f1_std": float(np.std(shuffle_macro_f1s)),
        "metric": "accuracy, balanced_accuracy, macro_f1",
        "class_weighting": "balanced",
        "note": "20 repeats with label permutation, mean ± std",
    }

    return results


def leave_one_attack_out(
    X: np.ndarray,
    y: np.ndarray,
    attack_ids: list[str],
) -> dict:
    """Leave-one-attack-out cross-validation on dev data."""
    unique_attacks = sorted(set(a for a in attack_ids if a != "-"))
    results = {}

    for held_out in unique_attacks:
        train_mask = np.array([a != held_out for a in attack_ids])
        test_mask = np.array([a == held_out for a in attack_ids])

        if not test_mask.any():
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Assert held-out attack not in training
        train_attacks = set(a for a, m in zip(attack_ids, train_mask) if m)
        assert held_out not in train_attacks, f"Leakage: {held_out} in training set"

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        n_test = int(test_mask.sum())
        n_correct = int((y_pred == y_test).sum())
        results[held_out] = {
            "n_test": n_test,
            "n_correct": n_correct,
            "accuracy": float(acc),
            "accuracy_pct": f"{n_correct}/{n_test} = {100 * n_correct / n_test:.1f}%",
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
        }

    return results


def predict_hybrid_distribution(
    scaler: StandardScaler,
    clf: LogisticRegression,
    X_hybrid: np.ndarray,
    attack_ids_hybrid: list[str],
) -> dict:
    """Get assignment distribution for hybrid attacks (A13-A15).

    The probe is trained on TTS vs VC only; hybrid attacks are assigned
    to the nearer of the two known classes.
    """
    if len(X_hybrid) == 0:
        return {}

    X_scaled = scaler.transform(X_hybrid)
    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)

    # Overall assignment distribution
    frac_tts = float(np.mean(y_pred == 0))
    frac_vc = float(np.mean(y_pred == 1))

    # Per-attack
    per_attack = {}
    unique_attacks = sorted(set(attack_ids_hybrid))
    for att in unique_attacks:
        mask = np.array([a == att for a in attack_ids_hybrid])
        if not mask.any():
            continue
        n = int(mask.sum())
        pred = y_pred[mask]
        prob = y_prob[mask]
        per_attack[att] = {
            "n": n,
            "frac_assigned_tts": float(np.mean(pred == 0)),
            "frac_assigned_vc": float(np.mean(pred == 1)),
            "mean_prob_tts": float(np.mean(prob[:, 0])),
            "mean_prob_vc": float(np.mean(prob[:, 1])),
        }

    return {
        "overall": {"frac_assigned_tts": frac_tts, "frac_assigned_vc": frac_vc},
        "per_attack": per_attack,
    }


def leakage_checks(
    fit_utt_ids: list[str],
    test_utt_ids: list[str],
    fit_attack_ids: list[str],
    test_attack_ids: list[str],
    fit_speaker_ids: list[str] | None = None,
    test_speaker_ids: list[str] | None = None,
) -> dict:
    """Check for data leakage between fit and test sets."""
    fit_utt_set = set(fit_utt_ids)
    test_utt_set = set(test_utt_ids)

    overlap_utt = fit_utt_set & test_utt_set

    result = {
        "files_in_both_fit_and_test": len(overlap_utt),
        "overlap_files": list(overlap_utt)[:20],
    }

    if fit_speaker_ids and test_speaker_ids:
        fit_spk = set(fit_speaker_ids)
        test_spk = set(test_speaker_ids)
        overlap_spk = fit_spk & test_spk
        result["speakers_in_both_fit_and_test"] = len(overlap_spk)
        result["overlap_speakers"] = list(overlap_spk)
    else:
        result["speakers_in_both_fit_and_test"] = "not_available"

    return result


def write_plots(
    out_dir: Path,
    scaler: StandardScaler,
    clf: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    attack_ids: list[str],
    labels_map: dict[int, str],
) -> None:
    """Write confusion matrix and t-SNE plots."""
    if not MATPLOTLIB_AVAILABLE:
        return

    # Confusion matrix
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = [labels_map.get(i, str(i)) for i in sorted(set(y))]
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title="Confusion Matrix", ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # t-SNE
    if TSNE_AVAILABLE and len(X) > 1:
        n_samples = min(TSNE_MAX_SAMPLES, len(X))
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), n_samples, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
        att_sub = [attack_ids[i] for i in idx]

        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, n_samples - 1))
        X_tsne = tsne.fit_transform(X_sub)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = {0: "blue", 1: "red", 2: "gray", 3: "purple"}
        labels = {0: "TTS", 1: "VC", 2: "Bonafide", 3: "Hybrid"}

        for cls in sorted(set(y_sub)):
            mask = y_sub == cls
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=colors.get(cls, "black"), label=labels.get(cls, str(cls)),
                       alpha=0.6, s=20)

        ax.set_title(f"t-SNE of Embeddings (n={n_samples})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "tsne_embeddings.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Type probe experiment on ASVspoof 2019 LA embeddings")
    parser.add_argument("--fit-scores", type=Path, required=True, help="Dev scores CSV (fit set)")
    parser.add_argument("--fit-emb", type=Path, required=True, help="Dev embeddings .emb.f32")
    parser.add_argument("--test-scores", type=Path, required=True, help="Eval scores CSV (test set)")
    parser.add_argument("--test-emb", type=Path, required=True, help="Eval embeddings .emb.f32")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not installed. Install with: pip install -e .[eval]", file=sys.stderr)
        sys.exit(1)

    for p in [args.fit_scores, args.fit_emb, args.test_scores, args.test_emb]:
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load fit (dev) data
    print("Loading fit (dev) data...")
    fit_scores_data = load_scores_csv(args.fit_scores)
    fit_rows = len(fit_scores_data["scores"])
    fit_emb = load_embeddings(args.fit_emb, fit_rows)
    print(f"  Fit: {fit_rows} files, embeddings shape: {fit_emb.shape}")

    # Load test (eval) data
    print("Loading test (eval) data...")
    test_scores_data = load_scores_csv(args.test_scores)
    test_rows = len(test_scores_data["scores"])
    test_emb = load_embeddings(args.test_emb, test_rows)
    print(f"  Test: {test_rows} files, embeddings shape: {test_emb.shape}")

    # Prepare data for TTS vs VC (spoof only)
    print("Preparing data...")
    X_fit, y_fit, fit_attacks, fit_durations = prepare_data(fit_scores_data, fit_emb, include_bonafide=False)
    X_test, y_test, test_attacks, test_durations = prepare_data(test_scores_data, test_emb, include_bonafide=False)

    # Also prepare with bonafide for 3-class experiment
    X_fit_3, y_fit_3, fit_attacks_3, fit_durations_3 = prepare_data(fit_scores_data, fit_emb, include_bonafide=True)
    X_test_3, y_test_3, test_attacks_3, test_durations_3 = prepare_data(test_scores_data, test_emb, include_bonafide=True)

    # Separate hybrid attacks from test
    hybrid_mask = np.array([get_type_label(a) == "hybrid" for a in test_attacks])
    X_test_main = X_test[~hybrid_mask]
    y_test_main = y_test[~hybrid_mask]
    test_attacks_main = [a for a, h in zip(test_attacks, hybrid_mask) if not h]
    test_durations_main = test_durations[~hybrid_mask]

    X_hybrid = X_test[hybrid_mask]
    hybrid_attacks = [a for a, h in zip(test_attacks, hybrid_mask) if h]

    print(f"  Fit (dev): {len(X_fit)} spoof samples (TTS/VC)")
    print(f"  Test (eval): {len(X_test_main)} spoof samples (TTS/VC, excl hybrid)")
    print(f"  Hybrid (eval): {len(X_hybrid)} samples")

    # Train probe on dev (TTS vs VC)
    print("Training TTS vs VC probe on dev...")
    scaler, clf = train_probe(X_fit, y_fit)

    # Evaluate on eval (TTS vs VC, excluding hybrid)
    print("Evaluating on eval (TTS vs VC)...")
    eval_results = evaluate_probe(scaler, clf, X_test_main, y_test_main, test_attacks_main)

    # Predict hybrid distribution
    print("Predicting hybrid distribution...")
    hybrid_dist = predict_hybrid_distribution(scaler, clf, X_hybrid, hybrid_attacks)

    # Control probes
    print("Running control probes...")
    fit_scores = fit_scores_data["scores"]
    test_scores = test_scores_data["scores"]
    fit_durations_full = fit_scores_data["durations"]
    test_durations_full = test_scores_data["durations"]
    fit_spoof_mask = np.array([get_type_label(a) in ("TTS", "VC") for a in fit_scores_data["attack_ids"]])
    test_spoof_mask = np.array([get_type_label(a) in ("TTS", "VC") for a in test_scores_data["attack_ids"]])
    control_results = run_control_probes(
        X_fit, y_fit, X_test_main, y_test_main, test_attacks_main,
        fit_durations, test_durations_main, fit_scores, test_scores,
        X_fit, y_fit, X_test_main, y_test_main,
        fit_durations_full[fit_spoof_mask], test_durations_full[test_spoof_mask],
        fit_scores[fit_spoof_mask], test_scores[test_spoof_mask],
    )

    # Leave-one-attack-out on dev
    print("Running leave-one-attack-out on dev...")
    loo_results = leave_one_attack_out(X_fit, y_fit, fit_attacks)

    # Leakage checks
    print("Running leakage checks...")
    leak = leakage_checks(
        fit_scores_data["utt_ids"],
        test_scores_data["utt_ids"],
        fit_attacks,
        test_attacks,
        fit_scores_data.get("speaker_ids"),
        test_scores_data.get("speaker_ids"),
    )

    # Hybrid prediction distribution
    print("Hybrid prediction distribution:", hybrid_dist.get("overall", {}))

    # Train 3-class probe (TTS, VC, Bonafide) for confusion matrix and t-SNE
    print("Training 3-class probe for visualization...")
    scaler_3, clf_3 = train_probe(X_fit_3, y_fit_3)

    # Build report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fit_scores_file": str(args.fit_scores),
        "test_scores_file": str(args.test_scores),
        "fit_embeddings_file": str(args.fit_emb),
        "test_embeddings_file": str(args.test_emb),
        "n_fit": len(X_fit),
        "n_test": len(X_test_main),
        "n_hybrid": len(X_hybrid),
        "leakage_checks": leak,
        "tts_vc_probe": {
            "eval_results": eval_results,
            "hybrid_predictions": hybrid_dist,
            "controls": control_results,
            "leave_one_attack_out": loo_results,
        },
        "versions": {
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__ if SKLEARN_AVAILABLE else None,
            "joblib": joblib.__version__ if JOBLIB_AVAILABLE else None,
            "python": sys.version,
        },
        "note": "Evidence from ASVspoof 2019 LA attacks only. Says nothing about modern generators, replay or human impersonation.",
    }

    # Save JSON report
    json_path = args.out_dir / "type_probe_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write Markdown report
    md_path = args.out_dir / "type_probe_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ASVspoof 2019 LA Type Probe Experiment\n\n")
        f.write(f"Generated: {report['generated_at']}\n")
        f.write(f"Fit (dev) scores: {args.fit_scores}\n")
        f.write(f"Test (eval) scores: {args.test_scores}\n\n")

        f.write("## Leakage Checks\n\n")
        leak = report["leakage_checks"]
        f.write(f"- Files in both fit and test: {leak['files_in_both_fit_and_test']}\n")
        if leak['overlap_files']:
            f.write(f"  Overlap (first 20): {leak['overlap_files']}\n")
        f.write(f"- Speakers in both fit and test: {leak['speakers_in_both_fit_and_test']}\n")
        if isinstance(leak.get('overlap_speakers'), list) and leak['overlap_speakers']:
            f.write(f"  Overlap: {leak['overlap_speakers']}\n")
        f.write("\n")

        f.write("## Primary Task: TTS vs VC on Eval (excl. Hybrid)\n\n")
        res = report["tts_vc_probe"]["eval_results"]
        f.write(f"- Accuracy: {res['accuracy']:.4f}\n")
        f.write(f"- Balanced Accuracy: {res['balanced_accuracy']:.4f}\n")
        f.write(f"- Macro F1: {res['macro_f1']:.4f}\n")
        f.write(f"- Confusion Matrix:\n")
        f.write("```\n")
        f.write(f"  TTS VC\n")
        cm = res['confusion_matrix']
        f.write(f"TTS {cm[0][0]} {cm[0][1]}\n")
        f.write(f"VC  {cm[1][0]} {cm[1][1]}\n")
        f.write("```\n\n")

        f.write("### Per-Attack Results\n\n")
        f.write("| Attack | n | Accuracy | Wilson 95% | Pred TTS | Pred VC |\n")
        f.write("|--------|---|----------|------------|----------|---------|\n")
        for att in sorted(res["per_attack"]):
            a = res["per_attack"][att]
            f.write(f"| {att} | {a['n']} | {a['accuracy']:.4f} | "
                    f"[{a['wilson_95'][0]:.4f}, {a['wilson_95'][1]:.4f}] | "
                    f"{a['frac_pred_tts']:.4f} | {a['frac_pred_vc']:.4f} |\n")
        f.write("\n")

        f.write("## Hybrid Attack Assignment Distribution\n\n")
        hyb = report["tts_vc_probe"]["hybrid_predictions"]
        if hyb:
            overall = hyb.get("overall", {})
            f.write("- Overall: assigned to TTS={overall.get('frac_assigned_tts', 0):.4f}, assigned to VC={overall.get('frac_assigned_vc', 0):.4f}\n\n")
            f.write("(Hybrid attacks A13-A15 are assigned to the nearer of the two known classes TTS/VC, "
                    "since the probe was trained on TTS vs VC only.)\n\n")
            f.write("| Attack | n | Assigned TTS | Assigned VC | Mean Prob TTS | Mean Prob VC |\n")
            f.write("|--------|---|--------------|-------------|---------------|--------------|\n")
            for att, vals in hyb.get("per_attack", {}).items():
                f.write(f"| {att} | {vals['n']} | {vals['frac_assigned_tts']:.4f} | {vals['frac_assigned_vc']:.4f} | "
                        f"{vals.get('mean_prob_tts', 0):.4f} | {vals.get('mean_prob_vc', 0):.4f} |\n")
        f.write("\n")

        f.write("## Control Probes\n\n")
        f.write("| Probe | Accuracy | Balanced Acc | Macro F1 | Metric | Class Weighting |\n")
        f.write("|-------|----------|--------------|----------|--------|-----------------|\n")
        for name, res in report["tts_vc_probe"]["controls"].items():
            if name == "label_shuffle_20x":
                f.write(f"| {name} | {res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f} | "
                        f"{res['balanced_accuracy_mean']:.4f}±{res['balanced_accuracy_std']:.4f} | "
                        f"{res['macro_f1_mean']:.4f}±{res['macro_f1_std']:.4f} | "
                        f"{res.get('metric', '')} | {res.get('class_weighting', '')} |\n")
            else:
                f.write(f"| {name} | {res['accuracy']:.4f} | {res['balanced_accuracy']:.4f} | {res['macro_f1']:.4f} | "
                        f"{res.get('metric', '')} | {res.get('class_weighting', '')} |\n")
        f.write("\n")

        f.write("## Leave-One-Attack-Out (on Dev)\n\n")
        f.write("| Held-out Attack | n_test | Accuracy | Balanced Acc | Macro F1 |\n")
        f.write("|-----------------|--------|----------|--------------|----------|\n")
        for att, res in report["tts_vc_probe"]["leave_one_attack_out"].items():
            f.write(f"| {att} | {res['n_test']} | {res['accuracy']:.4f} | {res['balanced_accuracy']:.4f} | {res['macro_f1']:.4f} |\n")
        f.write("\n")

        f.write("## Note\n\n")
        f.write(report["note"] + "\n")

    # Save fitted probe
    if JOBLIB_AVAILABLE:
        probe_path = args.out_dir / "type_probe.joblib"
        joblib.dump({"scaler": scaler, "clf": clf}, probe_path)
        print(f"Saved probe to {probe_path}")

    # Plots
    if MATPLOTLIB_AVAILABLE:
        print("Generating plots...")
        labels_map = {0: "TTS", 1: "VC"}
        write_plots(args.out_dir, scaler, clf, X_test_main, y_test_main, test_attacks_main, labels_map)

    # Print summary to stdout
    print(json.dumps({
        "accuracy": eval_results["accuracy"],
        "balanced_accuracy": eval_results["balanced_accuracy"],
        "macro_f1": eval_results["macro_f1"],
        "headline_excl_same_as_train": eval_results.get("headline_excl_same_as_train", {}),
        "hybrid_assigned_tts": hybrid_dist.get("overall", {}).get("frac_assigned_tts", 0),
        "hybrid_assigned_vc": hybrid_dist.get("overall", {}).get("frac_assigned_vc", 0),
        "controls": {k: v.get("accuracy", v.get("accuracy_mean")) for k, v in control_results.items()},
        "leakage": leak,
    }, indent=2))

    print(f"\nReports written to: {args.out_dir}")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    if MATPLOTLIB_AVAILABLE:
        print(f"  - {args.out_dir}/confusion_matrix.png")
        print(f"  - {args.out_dir}/tsne_embeddings.png")


if __name__ == "__main__":
    main()