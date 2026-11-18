"""ASVspoof 2021 LA eval per-file scorer with stratified sampling.

Samples per (codec, key) cell, scores through the same AASIST path as 2019.
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
from typing import NamedTuple

import numpy as np
import soundfile as sf

from app.audio.embeddings import embed_window
from app.audio.flac_integrity import FlacIntegrityError, verify_and_decode
from app.audio.eval_scores import (
    AASIST_MAX_LEN,
    get_inference_service,
    pad_audio,
    sha256_file,
    window_mode_for,
)
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
THRESHOLD = float(_settings.safety_threshold)

MODEL_PATH = Path("models") / "AASIST.pth"

# Columns for the trial_metadata.txt
TRIAL_FIELDS = [
    "speaker_id", "utt_id", "codec", "transmission",
    "attack", "key", "trim", "subset"
]

CODEC_ORDER = ["none", "alaw", "ulaw", "g722", "gsm", "opus", "pstn"]

EMBEDDING_DIM = 160
EMBEDDING_BYTES_PER_ROW = EMBEDDING_DIM * 4  # float32 = 4 bytes


class TrialRow(NamedTuple):
    speaker_id: str
    utt_id: str
    codec: str
    transmission: str
    attack: str
    key: str
    trim: str
    subset: str


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, cwd=Path.cwd()
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, FileNotFoundError):
        pass
    return None


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=False, cwd=Path.cwd()
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (OSError, FileNotFoundError):
        pass
    return False


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def parse_trial_metadata(path: Path) -> list[TrialRow]:
    rows: list[TrialRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 8:
                logger.warning("Line %d: expected 8 fields, got %d: %s", line_num, len(parts), line)
                continue
            rows.append(TrialRow(*parts))
    return rows


def filter_rows(rows: list[TrialRow], subset: str) -> list[TrialRow]:
    return [r for r in rows if r.subset == subset]


def sample_per_cell(
    rows: list[TrialRow],
    per_cell: int,
    seed: int,
) -> tuple[list[TrialRow], dict[tuple[str, str], int]]:
    """Sample up to per_cell rows per (codec, key) cell."""
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str], list[TrialRow]] = {}
    for r in rows:
        by_cell.setdefault((r.codec, r.key), []).append(r)

    sampled: list[TrialRow] = []
    cell_counts: dict[tuple[str, str], int] = {}
    for cell, cell_rows in by_cell.items():
        shuffled = list(cell_rows)
        rng.shuffle(shuffled)
        take = shuffled[:per_cell] if len(shuffled) > per_cell else shuffled
        sampled.extend(take)
        cell_counts[cell] = len(take)

    return sampled, cell_counts


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
        "codec", "transmission", "trim", "subset", "loader",
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
        "codec", "transmission", "trim", "subset", "loader",
    ]
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


def write_failed(failed_csv: Path, rows: list[dict]) -> None:
    fieldnames = ["file_name", "utt_id", "codec", "key", "reason", "error", "path", "exists", "size_bytes", "first_4_bytes", "loader"]
    with open(failed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def classify_failure(file_path: Path, error_msg: str) -> str:
    """Classify failure reason based on file existence, size, header, and error message."""
    if not file_path.exists():
        return "missing"
    try:
        size = file_path.stat().st_size
        if size == 0:
            return "zero_bytes"
    except OSError:
        return "missing"
    
    is_valid, header, header_err = check_flac_header(file_path)
    if not is_valid:
        return "bad_header"
    
    # Decode error
    error_lower = error_msg.lower()
    if any(kw in error_lower for kw in ["decode", "flac", "sync", "header", "crc", "checksum", "stream"]):
        return "decode_error"
    
    return "unknown"


def write_extract_list(extract_list: Path, archive_prefix: str, rows: list[TrialRow]) -> None:
    with open(extract_list, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{archive_prefix}{r.utt_id}.flac\n")


def write_provenance(provenance_path: Path, info: dict) -> None:
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def read_audio_mono(path: Path):
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=0)
    data = data.astype(np.float32)
    return data, int(sr), int(data.shape[0])


def check_flac_header(path: Path) -> tuple[bool, bytes, str]:
    """Check if file has valid FLAC header.
    
    Returns: (is_valid, first_4_bytes, error_msg)
    """
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        if len(header) < 4:
            return False, header, "file too small (< 4 bytes)"
        if header == b"fLaC":
            return True, header, ""
        return False, header, f"invalid FLAC header: {header!r} (expected b'fLaC')"
    except Exception as e:
        return False, b"", f"header check failed: {e}"


def read_audio_mono_with_fallback(path: Path, accept_md5_mismatch: bool = False):
    """Read audio with soundfile first; fall back to a *verified* strict ffmpeg decode.

    A file that only ffmpeg can read is accepted only if it is provably intact: strict
    decode (any error is fatal), decoded length equals the length declared in the FLAC
    header, and the decoded audio matches the MD5 stored in the header (see
    app.audio.flac_integrity). Anything else raises FlacIntegrityError (a RuntimeError
    with a .status reason code) so the caller records it as failed instead of scoring
    possibly-damaged audio.

    Returns: (audio_data, sample_rate, num_samples, loader_name)
    loader_name is "soundfile", "ffmpeg" (verified strict decode) or, only with
    accept_md5_mismatch=True, "ffmpeg_md5mismatch" (strict decode and length fine, header MD5
    disagrees - flagged so results can be reported with and without these files)
    """
    try:
        data, sr, num_samples = read_audio_mono(path)
        return data, sr, num_samples, "soundfile"
    except Exception as sf_exc:  # noqa: BLE001 - fall through to the verified fallback
        sf_error = str(sf_exc)

    try:
        audio, sr, _info, md5_state = verify_and_decode(path, accept_md5_mismatch)
    except FlacIntegrityError as exc:
        raise FlacIntegrityError(exc.status, f"{exc.detail} [soundfile: {sf_error}]") from exc
    if md5_state == "md5_mismatch":
        logger.warning("%s: header MD5 mismatch accepted (frames CRC-valid, length matches)", Path(path).name)
        return audio, sr, int(audio.shape[0]), "ffmpeg_md5mismatch"
    if md5_state != "verified":
        logger.warning("%s: no MD5 in FLAC header; integrity rests on strict decode + length only",
                       Path(path).name)
    return audio, sr, int(audio.shape[0]), "ffmpeg"


def run_scorer(
    data_dir: Path,
    rows: list[TrialRow],
    output_csv: Path,
    flush_every: int,
    write_extract: Path | None,
    archive_prefix: str | None,
    save_embeddings: bool = False,
    retry_failed: bool = False,
    accept_md5_mismatch: bool = False,
) -> dict:
    existing_ids = load_existing_utt_ids(output_csv)
    write_header_if_needed(output_csv)

    inference_service = get_inference_service()

    base = output_csv
    failed_csv = base.with_suffix(base.suffix + ".failed.csv")
    provenance_path = base.with_suffix(base.suffix + ".provenance.json")
    emb_sidecar = base.with_suffix(base.suffix + ".emb.f32")

    # Handle embeddings sidecar resume logic
    if save_embeddings:
        if emb_sidecar.exists():
            # Check alignment: file_size should equal csv_rows * 640
            try:
                with open(output_csv, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    csv_rows = sum(1 for _ in reader)
                expected_size = csv_rows * EMBEDDING_BYTES_PER_ROW
                actual_size = emb_sidecar.stat().st_size
                if actual_size != expected_size:
                    logger.error(
                        "Embeddings sidecar size mismatch: expected %d bytes (%d rows * 640), "
                        "got %d bytes. Start a new output file.",
                        expected_size, csv_rows, actual_size
                    )
                    sys.exit(1)
                logger.info("Resuming with existing embeddings sidecar (%d rows aligned)", csv_rows)
            except (OSError, csv.Error) as e:
                logger.error("Failed to verify embeddings sidecar alignment: %s", e)
                sys.exit(1)
        elif output_csv.exists() and output_csv.stat().st_size > 0:
            # CSV exists and has content - check if it has data rows (not just header)
            try:
                with open(output_csv, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    csv_rows = sum(1 for _ in reader)
                if csv_rows > 0:
                    # CSV has data rows but no sidecar - refuse to resume
                    logger.error(
                        "CSV has %d data rows but no embeddings sidecar. "
                        "Cannot resume with --save-embeddings. Start a new output file.",
                        csv_rows
                    )
                    sys.exit(1)
            except (OSError, csv.Error):
                # If we can't read it, assume it has data and error
                logger.error(
                    "CSV exists but cannot be read. Cannot resume with --save-embeddings. "
                    "Start a new output file."
                )
                sys.exit(1)

    if write_extract:
        write_extract_list(write_extract, archive_prefix or "", rows)
        logger.info("Extract list written to %s (%d files)", write_extract, len(rows))
        return {"extract_only": True}

    # If retry_failed, load failed utt_ids from previous run
    if retry_failed:
        if not failed_csv.exists():
            logger.error("No failed.csv found to retry. Run without --retry-failed first.")
            sys.exit(1)
        failed_utt_ids = set()
        with open(failed_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                failed_utt_ids.add(row["utt_id"])
        rows = [r for r in rows if r.utt_id in failed_utt_ids]
        logger.info("Retry mode: will re-attempt %d previously failed utt_ids", len(rows))
        # Clear existing failed.csv so we can write fresh results
        failed_csv.unlink(missing_ok=True)

    if write_extract:
        write_extract_list(write_extract, archive_prefix or "", rows)
        logger.info("Extract list written to %s (%d files)", write_extract, len(rows))
        return {"extract_only": True}

    started = datetime.now(timezone.utc)
    start_ts = time.perf_counter()

    # Open embeddings sidecar for appending
    emb_file = None
    if save_embeddings:
        emb_file = open(emb_sidecar, "ab")

    failed_rows: list[dict] = []
    processed = 0
    skipped = 0
    batch: list[dict] = []
    last_eta_log = 0
    sr_counts: dict[int, int] = {}
    sr_by_codec: dict[str, dict[int, int]] = {}
    # Per-cell tracking: (codec, key) -> {sampled, scored, failed, soundfile, ffmpeg}
    cell_stats: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        cell = (row.codec, row.key)
        if cell not in cell_stats:
            cell_stats[cell] = {"sampled": 0, "scored": 0, "failed": 0, "soundfile": 0, "ffmpeg": 0, "ffmpeg_md5mismatch": 0}
        cell_stats[cell]["sampled"] += 1

    def eta_log(force: bool = False) -> None:
        nonlocal last_eta_log
        now_count = processed + skipped
        if not force and (now_count - last_eta_log) < 500 and now_count < len(rows):
            return
        last_eta_log = now_count
        elapsed = time.perf_counter() - start_ts
        rate = processed / elapsed if elapsed > 0 and processed > 0 else 0.0
        remaining = len(rows) - now_count
        eta = remaining / rate if rate > 0 else float("inf")
        logger.info(
            "Progress %d/%d (scored %d, skipped %d) | %.1f files/s | ETA %s",
            now_count, len(rows), processed, skipped, rate,
            f"{eta:.0f}s" if eta != float("inf") else "unknown",
        )

    try:
        for row in rows:
            utt_id = row.utt_id
            if utt_id in existing_ids:
                skipped += 1
                eta_log()
                continue

            file_path = data_dir / f"{utt_id}.flac"
            cell = (row.codec, row.key)
            if not file_path.exists():
                exists = "no"
                size = 0
                first_4 = b""
                reason = "missing"
                error_msg = "file not found"
                logger.warning("Audio file not found: %s", file_path)
            else:
                exists = "yes"
                size = file_path.stat().st_size
                is_valid, first_4, header_err = check_flac_header(file_path)
                if not is_valid:
                    reason = "bad_header"
                    error_msg = header_err
                    logger.warning("Audio file bad header: %s (%s)", file_path, header_err)
                else:
                    reason = None
                    error_msg = None
                    logger.debug("File exists and has valid FLAC header: %s", file_path)
            
            if reason is not None:
                failed_rows.append({
                    "file_name": file_path.name,
                    "utt_id": utt_id,
                    "codec": row.codec,
                    "key": row.key,
                    "reason": reason,
                    "error": error_msg,
                    "path": str(file_path),
                    "exists": exists,
                    "size_bytes": size,
                    "first_4_bytes": first_4.hex() if first_4 else "",
                    "loader": "none",
                })
                cell_stats[cell]["failed"] += 1
                processed += 1
                eta_log()
                continue

            # File exists and has valid header, try to read audio with fallback
            try:
                audio, sr, num_samples, loader = read_audio_mono_with_fallback(
                    file_path, accept_md5_mismatch
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read %s (both loaders): %s", file_path, exc)
                first_4 = b""
                try:
                    with open(file_path, "rb") as f:
                        first_4 = f.read(4)
                except Exception:
                    first_4 = b""
                failed_rows.append({
                    "file_name": file_path.name,
                    "utt_id": utt_id,
                    "codec": row.codec,
                    "key": row.key,
                    "reason": getattr(exc, "status", "decode_error"),
                    "error": str(exc),
                    "path": str(file_path),
                    "exists": "yes",
                    "size_bytes": file_path.stat().st_size,
                    "first_4_bytes": first_4.hex() if first_4 else "",
                    "loader": "both_failed",
                })
                cell_stats[cell]["failed"] += 1
                processed += 1
                eta_log()
                continue

            sr_counts[sr] = sr_counts.get(sr, 0) + 1
            sr_by_codec.setdefault(row.codec, {})
            sr_by_codec[row.codec][sr] = sr_by_codec[row.codec].get(sr, 0) + 1

            if sr != 16000:
                logger.warning("File %s has sample rate %d (expected 16000)", file_path.name, sr)

            duration_s = num_samples / sr if sr else 0.0
            mode = window_mode_for(num_samples)

            try:
                if save_embeddings:
                    score, embedding = embed_window(audio, sr=16000)
                    emb_file.write(embedding.tobytes())
                else:
                    result = inference_service.infer(audio, sr=16000)
                    score = float(result.anomaly_score)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to score %s: %s", file_path, exc)
                first_4 = b""
                try:
                    with open(file_path, "rb") as f:
                        first_4 = f.read(4)
                except Exception:
                    first_4 = b""
                failed_rows.append({
                    "file_name": file_path.name,
                    "utt_id": utt_id,
                    "codec": row.codec,
                    "key": row.key,
                    "reason": "decode_error",
                    "error": f"score: {exc}",
                    "path": str(file_path),
                    "exists": "yes",
                    "size_bytes": file_path.stat().st_size,
                    "first_4_bytes": first_4.hex() if first_4 else "",
                    "loader": loader,
                })
                cell_stats[cell]["failed"] += 1
                processed += 1
                eta_log()
                continue

            # Success!
            cell_stats[cell]["scored"] += 1
            cell_stats[cell][loader] += 1

            label = 1 if row.key == "spoof" else 0
            attack_id = row.attack if row.attack != "bonafide" else "-"

            batch.append({
                "utt_id": utt_id,
                "speaker_id": row.speaker_id,
                "attack_id": attack_id,
                "label": label,
                "score": f"{score:.10f}",
                "duration_s": f"{duration_s:.6f}",
                "num_samples": num_samples,
                "window_mode": mode,
                "sr": sr,
                "file_name": file_path.name,
                "codec": row.codec,
                "transmission": row.transmission,
                "trim": row.trim,
                "subset": row.subset,
                "loader": loader,
            })
            processed += 1
            existing_ids.add(utt_id)

            if len(batch) >= flush_every:
                append_rows(output_csv, batch)
                if save_embeddings and emb_file:
                    emb_file.flush()
                batch.clear()

            eta_log()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; flushing and exiting gracefully.")

    if batch:
        append_rows(output_csv, batch)

    # Close embeddings sidecar
    if emb_file:
        emb_file.flush()
        emb_file.close()

    write_failed(failed_csv, failed_rows)

    ended = datetime.now(timezone.utc)

    # Print per-cell table to console
    logger.info("Per-cell (codec x key) summary:")
    logger.info("  codec    | key      | sampled | scored | failed | sf  | ff  | fail%%")
    logger.info("  ---------|----------|---------|--------|--------|-----|-----|------")
    high_fail_cells = []
    for (codec, key), stats in sorted(cell_stats.items()):
        sampled = stats["sampled"]
        scored = stats["scored"]
        failed = stats["failed"]
        sf = stats.get("soundfile", 0)
        ff = stats.get("ffmpeg", 0)
        fail_pct = (failed / sampled * 100) if sampled > 0 else 0.0
        logger.info("  %-8s | %-8s | %7d | %6d | %6d | %3d | %3d | %4.1f%%",
                    codec, key, sampled, scored, failed, sf, ff, fail_pct)
        if sampled > 0 and fail_pct > 2.0:
            high_fail_cells.append(f"{codec}|{key}: {fail_pct:.1f}%")
    if high_fail_cells:
        logger.warning("HIGH FAILURE RATE (>2%%): %s", ", ".join(high_fail_cells))

    # Counts from CSV
    total_rows = 0
    n_bonafide = 0
    n_spoof = 0
    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if int(r["label"]) == 1:
                    n_spoof += 1
                else:
                    n_bonafide += 1
        total_rows = n_bonafide + n_spoof
    except OSError:
        total_rows = processed

    # Compute embeddings sidecar SHA-256 if created
    emb_sha256 = None
    if save_embeddings and emb_sidecar.exists():
        emb_sha256 = sha256_file(emb_sidecar)

    return {
        "processed": processed,
        "skipped": skipped,
        "total_rows": total_rows,
        "bonafide": n_bonafide,
        "spoof": n_spoof,
        "failed": len(failed_rows),
        "sr_counts": sr_counts,
        "sr_by_codec": sr_by_codec,
        "started": started,
        "ended": ended,
        "embeddings": {
            "enabled": save_embeddings,
            "dim": EMBEDDING_DIM,
            "bytes_per_row": EMBEDDING_BYTES_PER_ROW,
            "sidecar_sha256": emb_sha256,
        },
        "cell_stats": {f"{c}|{k}": v for (c, k), v in cell_stats.items()},
        "loader_counts": {
            "soundfile": sum(v.get("soundfile", 0) for v in cell_stats.values()),
            "ffmpeg": sum(v.get("ffmpeg", 0) for v in cell_stats.values()),
            "ffmpeg_md5mismatch": sum(v.get("ffmpeg_md5mismatch", 0) for v in cell_stats.values()),
        },
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Score ASVspoof 2021 LA eval files with sampling")
    parser.add_argument("--keys", type=Path, required=True, help="trial_metadata.txt path")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with audio files (flac)")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--subset", type=str, default="eval", choices=["eval", "progress", "hidden"], help="Subset to score")
    parser.add_argument("--sample-per-cell", type=int, default=500, help="Max rows per (codec, key) cell")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for sampling")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 10 per cell")
    parser.add_argument("--write-extract-list", type=Path, default=None, help="Write extract list and exit")
    parser.add_argument("--archive-prefix", type=str, default="", help="Prefix for archive paths in extract list")
    parser.add_argument("--flush-every", type=int, default=200, help="Flush CSV every N files")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="Save 160-dim embeddings to sidecar .emb.f32 file")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt only previously failed utt_ids from .failed.csv")
    parser.add_argument("--accept-md5-mismatch", action="store_true",
                        help="Score files whose FLAC header MD5 disagrees with the audio but whose frames "
                             "all pass CRC and whose length matches. They are flagged loader=ffmpeg_md5mismatch.")
    parser.add_argument("--audit-sidecar", type=int, default=0, metavar="N",
                        help="Re-embed N random rows from the CSV (min 10 with loader=ffmpeg if available) and compare with sidecar. Exit non-zero on mismatch.")
    args = parser.parse_args()

    if not args.keys.exists():
        logger.error("Keys file not found: %s", args.keys)
        sys.exit(1)
    if not args.data_dir.exists():
        logger.error("Data directory not found: %s", args.data_dir)
        sys.exit(1)

    all_rows = parse_trial_metadata(args.keys)
    logger.info("Total trial rows: %d", len(all_rows))

    subset_rows = filter_rows(all_rows, args.subset)
    logger.info("Rows in subset '%s': %d", args.subset, len(subset_rows))

    per_cell = 10 if args.smoke else args.sample_per_cell
    sampled_rows, cell_counts = sample_per_cell(subset_rows, per_cell, args.seed)
    logger.info("Sampled rows: %d (per_cell=%d, seed=%d)", len(sampled_rows), per_cell, args.seed)

    # Print per-attack counts in sample
    attack_counts = {}
    for r in sampled_rows:
        attack_counts[r.attack] = attack_counts.get(r.attack, 0) + 1
    logger.info("Per-attack counts in sample:")
    for att in sorted(attack_counts):
        logger.info("  %s: %d", att, attack_counts[att])

    # Print per-cell counts
    logger.info("Per-cell (codec, key) counts:")
    for (codec, key), count in sorted(cell_counts.items()):
        logger.info("  (%s, %s): %d", codec, key, count)

    run_info = run_scorer(
        data_dir=args.data_dir,
        rows=sampled_rows,
        output_csv=args.output_csv,
        flush_every=args.flush_every,
        write_extract=args.write_extract_list,
        archive_prefix=args.archive_prefix,
        save_embeddings=args.save_embeddings,
        retry_failed=args.retry_failed,
        accept_md5_mismatch=args.accept_md5_mismatch,
    )

    if run_info.get("extract_only"):
        return

    # Check sample rates
    sr_counts = run_info.get("sr_counts", {})
    sr_by_codec = run_info.get("sr_by_codec", {})
    non_16k = {sr: c for sr, c in sr_counts.items() if sr != 16000}
    if non_16k:
        logger.warning("Non-16000 sample rates found: %s", non_16k)
        logger.warning("Per-codec sample rate distribution:")
        for codec, dist in sorted(sr_by_codec.items()):
            logger.warning("  %s: %s", codec, dist)
        if not args.smoke:
            logger.error("Non-16kHz files detected in full run. Stopping.")
            sys.exit(1)

    provenance_path = args.output_csv.with_suffix(args.output_csv.suffix + ".provenance.json")

    # Compute cumulative stats from the FINAL CSV (all rows ever written)
    total_rows_final = 0
    n_bonafide_final = 0
    n_spoof_final = 0
    loader_counts_final = {"soundfile": 0, "ffmpeg": 0, "ffmpeg_md5mismatch": 0}
    cell_stats_final = {}
    sr_counts_final = {}
    sr_by_codec_final = {}
    if args.output_csv.exists():
        try:
            with open(args.output_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if int(r["label"]) == 1:
                        n_spoof_final += 1
                    else:
                        n_bonafide_final += 1
                    loader = r.get("loader", "unknown")
                    if loader in loader_counts_final:
                        loader_counts_final[loader] += 1
                    # CSV uses "attack_id" column, not "key"
                    attack_id = r.get("attack_id", "-")
                    cell = (r["codec"], attack_id)
                    if cell not in cell_stats_final:
                        cell_stats_final[cell] = {"sampled": 0, "scored": 0, "failed": 0, "soundfile": 0, "ffmpeg": 0, "ffmpeg_md5mismatch": 0}
                    cell_stats_final[cell]["sampled"] += 1
                    if loader == "soundfile":
                        cell_stats_final[cell]["soundfile"] += 1
                    elif loader == "ffmpeg":
                        cell_stats_final[cell]["ffmpeg"] += 1
                    elif loader == "ffmpeg_md5mismatch":
                        cell_stats_final[cell]["ffmpeg_md5mismatch"] += 1
                    sr = int(r.get("sr", 16000))
                    sr_counts_final[sr] = sr_counts_final.get(sr, 0) + 1
                    sr_by_codec_final.setdefault(r["codec"], {})
                    sr_by_codec_final[r["codec"]][sr] = sr_by_codec_final[r["codec"]].get(sr, 0) + 1
            total_rows_final = n_bonafide_final + n_spoof_final
        except (OSError, csv.Error, ValueError):
            total_rows_final = run_info.get("total_rows", 0)
            n_bonafide_final = run_info.get("bonafide", 0)
            n_spoof_final = run_info.get("spoof", 0)
            loader_counts_final = run_info.get("loader_counts", {"soundfile": 0, "ffmpeg": 0, "ffmpeg_md5mismatch": 0})
            cell_stats_final = run_info.get("cell_stats", {})
            sr_counts_final = run_info.get("sr_counts", {})
            sr_by_codec_final = run_info.get("sr_by_codec", {})

    # Load existing provenance to preserve runs history
    existing_runs = []
    if provenance_path.exists():
        try:
            with open(provenance_path, "r", encoding="utf-8") as f:
                existing_provenance = json.load(f)
            existing_runs = existing_provenance.get("runs", [])
        except (OSError, json.JSONDecodeError):
            existing_runs = []

    # Build run record for this invocation
    run_record = {
        "start_timestamp": run_info["started"].isoformat(),
        "end_timestamp": run_info["ended"].isoformat(),
        "command_line": " ".join(sys.argv),
        "scored_in_this_run": run_info["processed"],
        "skipped_existing": run_info["skipped"],
        "failed_in_this_run": run_info["failed"],
        "options": {
            "subset": args.subset,
            "sample_per_cell": per_cell,
            "seed": args.seed,
            "smoke": args.smoke,
        },
    }

    # Append to runs list (never overwrite)
    all_runs = existing_runs + [run_record]

    # Provenance
    provenance = {
        "model_sha256": sha256_file(MODEL_PATH),
        "keys_sha256": sha256_file(args.keys),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": None,
            "torchaudio": None,
            "soundfile": sf.__version__,
        },
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "threshold": THRESHOLD,
        "runs": all_runs,
        "cell_counts": {f"{c}|{k}": v for (c, k), v in cell_counts.items()},
        "attack_counts": attack_counts,
        "sample_rate_distribution": sr_counts_final,
        "sample_rate_by_codec": sr_by_codec_final,
        "counts": {
            "sampled": len(sampled_rows),
            "total_rows_in_csv": total_rows_final,
            "bonafide": n_bonafide_final,
            "spoof": n_spoof_final,
            "failed": run_info.get("failed", 0),
        },
        "embeddings": run_info.get("embeddings", {}),
        "cell_stats": {f"{c}|{k}": v for (c, k), v in cell_stats_final.items()},
        "loader_counts": loader_counts_final,
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

    write_provenance(provenance_path, provenance)
    logger.info("Provenance written to %s", provenance_path)

    # Run sidecar audit if requested
    if args.audit_sidecar > 0:
        emb_path = args.output_csv.with_suffix(args.output_csv.suffix + ".emb.f32")
        audit_rc = audit_sidecar(
            args.output_csv, emb_path, args.audit_sidecar, args.seed,
            data_dir=args.data_dir, accept_md5_mismatch=args.accept_md5_mismatch,
        )
        if audit_rc != 0:
            sys.exit(audit_rc)


def audit_sidecar(
    csv_path: Path,
    emb_path: Path,
    n: int,
    seed: int = 1234,
    data_dir: Path | None = None,
    accept_md5_mismatch: bool = False,
) -> int:
    """Audit sidecar by re-embedding N random rows and comparing with sidecar.
    
    Returns 0 on success, 1 on mismatch or error.
    """
    import random
    
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return 1
    if not emb_path.exists():
        logger.error("Sidecar file not found: %s", emb_path)
        return 1
    
    # Load CSV
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    if not rows:
        logger.error("CSV has no data rows")
        return 1
    
    # Load embeddings sidecar
    emb_data = np.fromfile(emb_path, dtype=np.float32)
    expected_rows = len(rows)
    if len(emb_data) != expected_rows * EMBEDDING_DIM:
        logger.error("Sidecar size mismatch: %d values, expected %d (%d rows * %d dim)",
                     len(emb_data), expected_rows * EMBEDDING_DIM, expected_rows, EMBEDDING_DIM)
        return 1
    embeddings = emb_data.reshape(expected_rows, EMBEDDING_DIM)
    
    # Select N random rows, preferring at least 10 with loader=ffmpeg
    rng = random.Random(seed)
    ffmpeg_indices = [i for i, r in enumerate(rows) if str(r.get("loader", "")).startswith("ffmpeg")]
    other_indices = [i for i, r in enumerate(rows) if not str(r.get("loader", "")).startswith("ffmpeg")]
    
    selected_indices = []
    if ffmpeg_indices:
        n_ffmpeg = min(10, len(ffmpeg_indices))
        selected_indices.extend(rng.sample(ffmpeg_indices, n_ffmpeg))
    
    remaining = n - len(selected_indices)
    if remaining > 0:
        available_other = [i for i in other_indices if i not in selected_indices]
        if available_other:
            n_other = min(remaining, len(available_other))
            selected_indices.extend(rng.sample(available_other, n_other))
    
    # Fill remaining with any available indices
    if len(selected_indices) < n:
        all_available = [i for i in range(len(rows)) if i not in selected_indices]
        if all_available:
            additional = min(n - len(selected_indices), len(all_available))
            selected_indices.extend(rng.sample(all_available, additional))
    
    selected_indices = selected_indices[:n]
    
    if not selected_indices:
        logger.error("No rows available for audit")
        return 1
    
    logger.info("Auditing %d rows (including %d ffmpeg)...", len(selected_indices),
                sum(1 for i in selected_indices if str(rows[i].get("loader", "")).startswith("ffmpeg")))
    
    if data_dir is None:
        data_dir = Path("data/LA2021/ASVspoof2021_LA_eval/flac")
    max_diff = 0.0
    checked = 0
    for idx in selected_indices:
        row = rows[idx]
        file_path = Path(data_dir) / row["file_name"]
        if not file_path.exists():
            logger.error("Audit cannot find %s (is --data-dir correct?)", file_path)
            return 1
        
        try:
            audio, sr, num_samples, loader = read_audio_mono_with_fallback(
                file_path, accept_md5_mismatch
            )
            score, embedding = embed_window(audio, sr=16000)
        except Exception as e:
            logger.error("Failed to re-embed %s: %s", file_path, e)
            return 1
        
        stored_embedding = embeddings[idx]
        diff = float(np.max(np.abs(embedding - stored_embedding)))
        checked += 1
        if diff > max_diff:
            max_diff = diff
        
        if diff > 1e-5:
            logger.error("MISMATCH row %d (utt_id=%s): max abs diff = %.6f", idx, rows[idx]["utt_id"], diff)
            return 1
    
    logger.info("AUDIT PASSED: %d rows checked, max abs diff = %.10f", checked, max_diff)
    return 0


if __name__ == "__main__":
    main()