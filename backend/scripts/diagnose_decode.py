"""Diagnose FLAC files that libsndfile cannot decode (ASVspoof 2021 LA/DF).

Answers one question per file: is it genuinely damaged, or merely rejected by libsndfile?

Run from the backend folder, e.g.:
    python scripts/diagnose_decode.py --csv eval_results_2021_smoke.csv \
        --data-dir data/LA2021/ASVspoof2021_LA_eval/flac

By default it examines the rows whose `loader` column is "ffmpeg" (the files soundfile could
not read). Use --all to check every row, or --files a.flac b.flac for specific files.
Writes <csv>.decode_diagnosis.csv and prints a summary by outcome, label and codec.

Verdicts:
  intact_sf_rejects   strict decode ok + length ok + MD5 ok  -> safe to score; libsndfile is just stricter
  intact_no_md5       strict decode ok + length ok, header has no MD5 -> probably fine, weaker proof
  damaged:<status>    md5_mismatch / length_mismatch / decode_error / bad_header -> do NOT score; re-extract
  readable_by_sf      libsndfile read it this time (e.g. earlier failure was transient)
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "flac_integrity", ROOT / "app" / "audio" / "flac_integrity.py"
)
fi = importlib.util.module_from_spec(_spec)
sys.modules["flac_integrity"] = fi
_spec.loader.exec_module(fi)

try:
    import soundfile as sf
except ImportError:  # the diagnosis still works without libsndfile
    sf = None


def libsndfile_status(path: Path) -> str:
    if sf is None:
        return "unavailable"
    try:
        sf.read(str(path), dtype="float32")
        return "ok"
    except Exception as exc:
        return f"error: {str(exc)[:60]}"


def diagnose(path: Path) -> tuple[str, str, str]:
    """Return (verdict, libsndfile_status, detail)."""
    if not path.exists():
        return "missing", "n/a", "file not found"
    sf_status = libsndfile_status(path)
    try:
        _, _, _info, md5_state = fi.verify_and_decode(path)
    except fi.FlacIntegrityError as exc:
        return f"damaged:{exc.status}", sf_status, exc.detail[:120]
    if sf_status == "ok":
        return "readable_by_sf", sf_status, ""
    return (
        ("intact_sf_rejects" if md5_state == "verified" else "intact_no_md5"),
        sf_status,
        md5_state,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument(
        "--csv", type=Path, help="scores CSV from eval_scores_2021 (needs file_name, loader)"
    )
    ap.add_argument(
        "--all", action="store_true", help="check every CSV row, not just loader=ffmpeg"
    )
    ap.add_argument("--files", nargs="*", help="explicit file names inside --data-dir")
    args = ap.parse_args(argv)

    meta: dict[str, dict] = {}
    if args.files:
        names = list(args.files)
    elif args.csv:
        with open(args.csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        sel = rows if args.all else [r for r in rows if r.get("loader") == "ffmpeg"]
        names = [r["file_name"] for r in sel]
        meta = {r["file_name"]: r for r in sel}
    else:
        ap.error("give --csv or --files")

    out_rows = []
    for name in names:
        verdict, sf_status, detail = diagnose(args.data_dir / name)
        m = meta.get(name, {})
        out_rows.append(
            {
                "file_name": name,
                "verdict": verdict,
                "libsndfile": sf_status,
                "detail": detail,
                "label": m.get("label", ""),
                "codec": m.get("codec", ""),
                "attack_id": m.get("attack_id", ""),
            }
        )

    out_path = (
        args.csv.with_suffix(args.csv.suffix + ".decode_diagnosis.csv")
        if args.csv
        else Path("decode_diagnosis.csv")
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["file_name"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"Examined {len(out_rows)} files -> {out_path}\n")
    print("Verdict counts:")
    for k, v in collections.Counter(r["verdict"] for r in out_rows).most_common():
        print(f"  {k:28s} {v}")
    for col in ("label", "codec"):
        if any(r[col] for r in out_rows):
            print(f"\nVerdict by {col}:")
            tab = collections.defaultdict(collections.Counter)
            for r in out_rows:
                tab[r[col]][r["verdict"].split(":")[0]] += 1
            for key in sorted(tab):
                print(f"  {key:8s} {dict(tab[key])}")
    damaged = sum(1 for r in out_rows if r["verdict"].startswith(("damaged", "missing")))
    print(f"\nDamaged or missing: {damaged}/{len(out_rows)}")
    return 1 if damaged else 0


if __name__ == "__main__":
    raise SystemExit(main())
