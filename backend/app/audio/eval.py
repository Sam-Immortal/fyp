"""Evaluation script for ASVspoof dataset using official AASIST EER computation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from app.audio.inference import get_inference_service
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_settings = get_settings()


def compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Detection Error Tradeoff (DET) curve.
    
    Official AASIST implementation (pure numpy, no sklearn).
    
    Args:
        target_scores: Scores for bonafide (genuine) samples
        nontarget_scores: Scores for spoof (attack) samples
        
    Returns:
        frr: False rejection rates
        far: False acceptance rates
        thresholds: Decision thresholds
    """
    # Combine scores and labels
    all_scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([
        np.ones(len(target_scores)),      # bonafide = 1 (target)
        np.zeros(len(nontarget_scores))   # spoof = 0 (nontarget)
    ])
    
    # Sort by score descending (high score = more likely spoof)
    # Note: AASIST outputs spoof probability, so higher = more spoof-like
    sorted_indices = np.argsort(all_scores)[::-1]
    all_scores = all_scores[sorted_indices]
    labels = labels[sorted_indices]
    
    # Compute cumulative sums
    tar_cumsum = np.cumsum(labels)           # bonafide count up to threshold
    nontar_cumsum = np.cumsum(1 - labels)    # spoof count up to threshold
    
    # Total counts
    n_target = len(target_scores)
    n_nontarget = len(nontarget_scores)
    
    # FRR (False Rejection Rate) = P(reject | bonafide) = 1 - P(accept | bonafide)
    # At threshold t, we accept if score <= t (lower spoof prob = more bonafide)
    # So FRR = (bonafide rejected) / total_bonafide = tar_cumsum / n_target
    frr = tar_cumsum / n_target
    
    # FAR (False Acceptance Rate) = P(accept | spoof)
    # FAR = (spoof accepted) / total_spoof = (total_spoof - spoof_rejected) / total_spoof
    # spoof_rejected = nontar_cumsum (spoof with score > threshold)
    # spoof_accepted = n_nontarget - nontar_cumsum
    far = (n_nontarget - nontar_cumsum) / n_nontarget
    
    # Thresholds are the score values
    thresholds = all_scores
    
    return frr, far, thresholds


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) using official AASIST method.
    
    EER is the point where FRR = FAR on the DET curve.
    
    Args:
        target_scores: Scores for bonafide (genuine) samples
        nontarget_scores: Scores for spoof (attack) samples
        
    Returns:
        EER as percentage (0-100)
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    
    # Find threshold where FRR ≈ FAR
    # EER = (FRR + FAR) / 2 at the crossing point
    diff = np.abs(frr - far)
    eer_idx = np.argmin(diff)
    
    eer = (frr[eer_idx] + far[eer_idx]) / 2
    
    return float(eer * 100)  # Percentage


def _write_checkpoint(
    output_path: Path,
    scores: list[float],
    paired_data: list[tuple[float, int]],
    file_names: list[str],
    processed_count: int,
    excluded_count: int,
) -> None:
    """Write partial progress to a checkpoint file."""
    if output_path is None:
        return
    checkpoint_path = output_path.with_suffix(".partial.json")
    try:
        checkpoint_data = {
            "scores": scores,
            "paired_data": paired_data,
            "file_names": file_names,
            "processed_count": processed_count,
            "excluded_count": excluded_count,
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)
        logger.debug("Checkpoint written to %s (%d files processed)", checkpoint_path, processed_count)
    except Exception as e:
        logger.warning("Failed to write checkpoint: %s", e)


def evaluate_on_asvspoof(
    data_dir: Path,
    protocol_file: Path | None = None,
    max_samples: int | None = None,
    checkpoint_every: int = 5000,
    output_path: Path | None = None,
) -> dict:
    """Evaluate model on ASVspoof dataset.

    Args:
        data_dir: Directory containing ASVspoof audio files
        protocol_file: Path to protocol file (optional)
        max_samples: Maximum number of samples to evaluate (for quick testing)
        checkpoint_every: Write checkpoint every N files (default 5000)
        output_path: Optional output path for checkpointing

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Starting evaluation on ASVspoof dataset")
    logger.info("Data directory: %s", data_dir)

    # Initialize inference service
    inference_service = get_inference_service()

    # Find audio files
    audio_files = list(data_dir.glob("*.flac")) + list(data_dir.glob("*.wav"))
    if protocol_file:
        # Parse protocol file to get labels
        labels = {}
        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    utt_id = parts[1]
                    label = parts[-1]
                    labels[utt_id] = 1 if label == "spoof" else 0
    else:
        labels = {}

    if max_samples:
        audio_files = audio_files[:max_samples]

    logger.info("Found %d audio files", len(audio_files))

    scores = []
    paired_data = []  # (score, label) for files with protocol match
    file_names = []
    excluded_count = 0

    for i, audio_file in enumerate(audio_files):
        if i % 100 == 0:
            logger.info("Processing %d/%d", i, len(audio_files))

        try:
            # Load audio
            import librosa

            audio, sr = librosa.load(audio_file, sr=16000)

            # Run inference
            result = get_inference_service().infer(audio, sr=16000)

            score = result.anomaly_score
            scores.append(score)
            file_names.append(audio_file.name)

            # Get true label if available
            utt_id = audio_file.stem
            if utt_id in labels:
                paired_data.append((score, labels[utt_id]))
            else:
                excluded_count += 1

            if i % 100 == 0:
                logger.info(
                    "File: %s, Score: %.4f, High Risk: %s",
                    audio_file.name,
                    score,
                    result.is_high_risk,
                )

            # Checkpoint
            if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
                _write_checkpoint(output_path, scores, paired_data, file_names, i + 1, excluded_count)

        except Exception as e:
            logger.warning("Failed to process %s: %s", audio_file, e)
            continue

    scores_arr = np.array(scores)
    results = {
        "num_samples": len(scores),
        "mean_score": float(scores_arr.mean()) if len(scores) > 0 else 0.0,
        "std_score": float(scores_arr.std()) if len(scores) > 0 else 0.0,
        "min_score": float(scores_arr.min()) if len(scores) > 0 else 0.0,
        "max_score": float(scores_arr.max()) if len(scores) > 0 else 0.0,
    }

    # Log excluded files count
    if excluded_count > 0:
        logger.info("%d/%d files had no protocol entry and were excluded from EER", excluded_count, len(scores))

    try:
        if paired_data:
            paired_arr = np.array(paired_data)
            paired_scores = paired_arr[:, 0]
            paired_labels = paired_arr[:, 1].astype(int)

            # Label 0 = bonafide (target), Label 1 = spoof (nontarget)
            bonafide_scores = paired_scores[paired_labels == 0]
            spoof_scores = paired_scores[paired_labels == 1]

            logger.info("Bonafide samples: %d, Spoof samples: %d", len(bonafide_scores), len(spoof_scores))

            if len(bonafide_scores) > 0 and len(spoof_scores) > 0:
                eer = compute_eer(bonafide_scores, spoof_scores)
                results["eer_percent"] = eer
                results["num_bonafide"] = len(bonafide_scores)
                results["num_spoof"] = len(spoof_scores)
                logger.info("EER: %.2f%%", eer)
            else:
                logger.warning("Not enough samples of both classes for EER computation")
        else:
            logger.warning("No labeled files found for EER computation")

    except Exception as e:
        logger.exception("Error computing final metrics (EER): %s", e)
        # Still return what we have

    logger.info("Evaluation complete: %s", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AASIST model on ASVspoof dataset")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with audio files")
    parser.add_argument("--protocol", type=Path, help="Protocol file path")
    parser.add_argument("--max-samples", type=int, help="Max samples to evaluate")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="Write checkpoint every N files (default 5000)")

    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(1)

    results = evaluate_on_asvspoof(
        data_dir=args.data_dir,
        protocol_file=args.protocol,
        max_samples=args.max_samples,
        checkpoint_every=args.checkpoint_every,
        output_path=args.output,
    )

    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Results saved to %s", args.output)
        except Exception as e:
            logger.exception("Failed to write output file: %s", e)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()