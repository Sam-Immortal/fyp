"""Check that the strict ffmpeg decoder returns the same audio as libsndfile.

The previous version called read_audio_mono_with_fallback(), which tries soundfile first, so on
files soundfile can read it compared soundfile with itself and always passed. This version
calls the ffmpeg path directly. Needs the `soundfile` package (as the reference decoder).

    python scripts/equivalence_check.py --csv eval_results_2021_smoke.csv \
        --data-dir data/LA2021/ASVspoof2021_LA_eval/flac -n 30
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "flac_integrity", ROOT / "app" / "audio" / "flac_integrity.py"
)
fi = importlib.util.module_from_spec(_spec)
sys.modules["flac_integrity"] = fi
_spec.loader.exec_module(fi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args(argv)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("loader") == "soundfile"][: args.n]
    if not rows:
        print("No loader=soundfile rows to compare.")
        return 2

    worst, failures, md5_verified = 0.0, 0, 0
    for r in rows:
        path = args.data_dir / r["file_name"]
        ref, sr_ref = sf.read(str(path), dtype="float32")
        if ref.ndim > 1:
            ref = ref.mean(axis=1)  # (frames, channels) -> average channels
        try:
            alt, sr_alt, _info, state = fi.verify_and_decode(path)  # ffmpeg path, called directly
        except fi.FlacIntegrityError as exc:
            print(
                f"  {r['file_name']}: strict decode REJECTED ({exc.status}) although libsndfile read it"
            )
            failures += 1
            continue
        md5_verified += state == "verified"
        if sr_ref != sr_alt or ref.shape != alt.shape:
            print(
                f"  {r['file_name']}: shape/rate mismatch {ref.shape}@{sr_ref} vs {alt.shape}@{sr_alt}"
            )
            failures += 1
            continue
        diff = float(np.max(np.abs(ref - alt)))
        worst = max(worst, diff)
        if diff > args.tol:
            print(f"  {r['file_name']}: max abs diff {diff:.3e}")
            failures += 1

    print(
        f"Compared {len(rows)} files | MD5 verified on {md5_verified} | worst max-abs-diff {worst:.3e} "
        f"| failures {failures}"
    )
    print("Equivalence check:", "PASS" if failures == 0 else "FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
