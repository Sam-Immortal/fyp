"""Per-file ASVspoof2019 LA eval scorer.

Scores every eval audio file through the exact same code path as
app.audio.eval.py (pad_audio + AudioInferenceService.infer) and writes
per-file results to a resumable CSV. Produces a provenance manifest.

Run as:
    python -m app.audio.eval_scores --data-dir ... --protocol ... --output-csv ...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio.inference import AASIST_MAX_LEN, get_inference_service, pad_audio
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
THRESHOLD = float(_settings.safety_threshold)

MODEL_PATH = Path("models") / "AASIST.pth"


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path.cwd(),
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, FileNotFoundError):
        pass
    return None


def parse_protocol(protocol_file: Path) -> dict[str, tuple[str, str, int]]:
    """Parse CM protocol lines: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY.

    Returns dict: utt_id (file stem) -> (speaker_id, attack_id, label).
    label: spoof=1, bonafide=0.
    """
    entries: dict[str, tuple[str, str, int]] = {}
    with open(protocol_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            speaker_id = parts[0]
            utt_id = parts[1]
            # parts[2] is the unused "-" column
            attack_id = parts[3] if len(parts) >= 4 else "-"
            key = parts[-1]
            label = 1 if key == "spoof" else 0
            entries[utt_id] = (speaker_id, attack_id, label)
    return entries


def read_audio_mono(path: Path):
    """Read audio as float32 mono. Returns (audio, sample_rate, num_samples).

    Raises on any read error so the caller can record it in failed.csv.
    """
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=0)
    data = data.astype(np.float32)
    return data, int(sr), int(data.shape[0])


def window_mode_for(num_samples: int) -> str:
    if num_samples < AASIST_MAX_LEN:
        return "padded"
    if num_samples > AASIST_MAX_LEN:
        return "truncated"
    return "exact"


def load_existing_utt_ids(output_csv: Path) -> set[str]:
    ids: set[str] = set()
    if not output_csv.exists():
        return ids
    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("utt_id")
                if uid:
                    ids.add(uid)
    except (OSError, csv.Error):
        return ids
    return ids


def write_header_if_needed(output_csv: Path) -> None:
    if output_csv.exists() and output_csv.stat().st_size > 0:
        return
    header = [
        "utt_id", "speaker_id", "attack_id", "label", "score", "duration_s",
        "num_samples", "window_mode", "sr", "file_name",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_rows(output_csv: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "utt_id", "speaker_id", "attack_id", "label", "score", "duration_s",
        "num_samples", "window_mode", "sr", "file_name",
    ]
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


def write_unlabelled(unlabelled_csv: Path, rows: list[dict]) -> None:
    fieldnames = ["name", "duration_s"]
    with open(unlabelled_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_failed(failed_csv: Path, rows: list[dict]) -> None:
    fieldnames = ["file_name", "error"]
    with open(failed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_provenance(provenance_path: Path, info: dict) -> None:
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def select_stratified(
    labelled: list[tuple[Path, str, str, str, int]],
    per_group: int | None,
    seed: int,
) -> list[tuple[Path, str, str, str, int]]:
    """labelled: (file_path, utt_id, speaker_id, attack_id, label).

    If per_group is set, take up to per_group files per group
    (attack_id for spoof, "bonafide" for bonafide), shuffled by seed.
    Returns the selected list sorted by file name.
    """
    if per_group is None:
        return sorted(labelled, key=lambda t: t[0].name)

    rng = random.Random(seed)
    by_group: dict[str, list[tuple[Path, str, str, str, int]]] = {}
    for item in labelled:
        file_path, _utt, speaker, attack, label = item
        group = attack if label == 1 else "bonafide"
        by_group.setdefault(group, []).append(item)

    selected: list[tuple[Path, str, str, str, int]] = []
    for group, items in by_group.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:per_group])

    return sorted(selected, key=lambda t: t[0].name)


def run_scorer(
    data_dir: Path,
    protocol_file: Path,
    output_csv: Path,
    limit: int | None,
    per_group: int | None,
    seed: int,
    flush_every: int,
) -> dict:
    proto_map = parse_protocol(protocol_file)
    logger.info("Protocol entries: %d", len(proto_map))

    audio_files = sorted(
        [p for p in data_dir.iterdir() if p.suffix.lower() in (".flac", ".wav")],
        key=lambda p: p.name,
    )
    logger.info("Audio files found: %d", len(audio_files))

    labelled: list[tuple[Path, str, str, str, int]] = []
    unlabelled: list[Path] = []
    proto_utts = set(proto_map.keys())
    file_stems = {p.stem for p in audio_files}

    for p in audio_files:
        stem = p.stem
        if stem in proto_map:
            speaker, attack, label = proto_map[stem]
            labelled.append((p, stem, speaker, attack, label))
        else:
            unlabelled.append(p)

    protocol_no_file = sorted(proto_utts - file_stems)
    logger.info("Labelled: %d, Unlabelled: %d, Protocol-no-file: %d",
                len(labelled), len(unlabelled), len(protocol_no_file))

    to_process = select_stratified(labelled, per_group, seed)
    if limit is not None:
        to_process = to_process[:limit]
    logger.info("Files to score: %d", len(to_process))

    existing_ids = load_existing_utt_ids(output_csv)
    write_header_if_needed(output_csv)

    inference_service = get_inference_service()

    base = output_csv
    unlabelled_csv = base.with_suffix(base.suffix + ".unlabelled.csv")
    failed_csv = base.with_suffix(base.suffix + ".failed.csv")
    provenance_csv = base.with_suffix(base.suffix + ".provenance.json")

    started = datetime.now(timezone.utc)
    start_ts = time.perf_counter()

    unlabelled_rows: list[dict] = []
    failed_rows: list[dict] = []
    processed = 0
    skipped = 0
    batch: list[dict] = []
    last_eta_log = 0

    def eta_log(force: bool = False) -> None:
        nonlocal last_eta_log
        now_count = processed + skipped
        if not force and (now_count - last_eta_log) < 500 and now_count < len(to_process):
            return
        last_eta_log = now_count
        elapsed = time.perf_counter() - start_ts
        rate = processed / elapsed if elapsed > 0 and processed > 0 else 0.0
        remaining = len(to_process) - now_count
        eta = remaining / rate if rate > 0 else float("inf")
        logger.info(
            "Progress %d/%d (scored %d, skipped %d) | %.1f files/s | ETA %s",
            now_count, len(to_process), processed, skipped, rate,
            f"{eta:.0f}s" if eta != float("inf") else "unknown",
        )

    try:
        for idx, (file_path, utt_id, speaker, attack, label) in enumerate(to_process, 1):
            if utt_id in existing_ids:
                skipped += 1
                eta_log()
                continue

            try:
                audio, sr, num_samples = read_audio_mono(file_path)
            except Exception as exc:  # noqa: BLE001 - record and continue
                logger.warning("Failed to read %s: %s", file_path, exc)
                failed_rows.append({"file_name": file_path.name, "error": str(exc)})
                processed += 1
                eta_log()
                continue

            if sr != 16000:
                logger.warning("File %s has sample rate %d (expected 16000)", file_path.name, sr)

            duration_s = num_samples / sr if sr else 0.0
            mode = window_mode_for(num_samples)

            try:
                result = inference_service.infer(audio, sr=16000)
                score = float(result.anomaly_score)
            except Exception as exc:  # noqa: BLE001 - record and continue
                logger.warning("Failed to score %s: %s", file_path, exc)
                failed_rows.append({"file_name": file_path.name, "error": f"score: {exc}"})
                processed += 1
                eta_log()
                continue

            batch.append({
                "utt_id": utt_id,
                "speaker_id": speaker,
                "attack_id": attack,
                "label": label,
                "score": f"{score:.10f}",
                "duration_s": f"{duration_s:.6f}",
                "num_samples": num_samples,
                "window_mode": mode,
                "sr": sr,
                "file_name": file_path.name,
            })
            processed += 1
            existing_ids.add(utt_id)

            if len(batch) >= flush_every:
                append_rows(output_csv, batch)
                batch.clear()

            eta_log()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; flushing and exiting gracefully.")

    if batch:
        append_rows(output_csv, batch)

    # Unlabelled: record duration only, no scoring.
    for p in unlabelled:
        try:
            _, sr, num_samples = read_audio_mono(p)
            duration_s = num_samples / sr if sr else 0.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read unlabelled %s: %s", p, exc)
            duration_s = 0.0
        unlabelled_rows.append({"name": p.name, "duration_s": f"{duration_s:.6f}"})

    write_unlabelled(unlabelled_csv, unlabelled_rows)
    write_failed(failed_csv, failed_rows)

    ended = datetime.now(timezone.utc)

    # Counts from the CSV we just (re)wrote/appended.
    total_rows = processed  # approximate; recount below for accuracy
    n_bonafide = 0
    n_spoof = 0
    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["label"]) == 1:
                    n_spoof += 1
                else:
                    n_bonafide += 1
        total_rows = n_bonafide + n_spoof
    except OSError:
        total_rows = processed

    provenance = {
        "model_sha256": sha256_file(MODEL_PATH),
        "protocol_sha256": sha256_file(protocol_file),
        "git_commit": git_commit(),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": None,
            "torchaudio": None,
            "soundfile": sf.__version__,
        },
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "threshold": THRESHOLD,
        "command_line": " ".join(sys.argv),
        "start_timestamp": started.isoformat(),
        "end_timestamp": ended.isoformat(),
        "counts": {
            "labelled_total": len(labelled),
            "unlabelled_total": len(unlabelled),
            "protocol_no_file": len(protocol_no_file),
            "scored_in_this_run": processed,
            "skipped_existing": skipped,
            "total_rows_in_csv": total_rows,
            "bonafide": n_bonafide,
            "spoof": n_spoof,
            "failed": len(failed_rows),
        },
    }
    try:
        import torch  # noqa: import-outside-toplevel
        provenance["versions"]["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import torchaudio  # noqa: import-outside-toplevel
        provenance["versions"]["torchaudio"] = torchaudio.__version__
    except ImportError:
        pass

    write_provenance(provenance_csv, provenance)

    # Print unlabelled summary.
    logger.info("Unlabelled files: %d (first 20): %s",
                len(unlabelled_rows), [r["name"] for r in unlabelled_rows[:20]])
    if protocol_no_file:
        logger.info("Protocol entries with no audio file (%d): %s",
                    len(protocol_no_file), protocol_no_file[:20])

    return provenance


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Score ASVspoof2019 LA eval files per-file")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with audio files")
    parser.add_argument("--protocol", type=Path, required=True, help="CM protocol file path")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Max files to score")
    parser.add_argument("--stratified-per-group", type=int, default=None,
                        help="Take up to N files per group (attack id / bonafide)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for stratification")
    parser.add_argument("--flush-every", type=int, default=200, help="Flush CSV every N files")
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(1)
    if not args.protocol.exists():
        logger.error("Protocol file does not exist: %s", args.protocol)
        sys.exit(1)

    run_scorer(
        data_dir=args.data_dir,
        protocol_file=args.protocol,
        output_csv=args.output_csv,
        limit=args.limit,
        per_group=args.stratified_per_group,
        seed=args.seed,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
