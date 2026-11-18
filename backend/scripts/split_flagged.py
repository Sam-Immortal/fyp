"""Write a copy of a 2021 scores CSV without the md5-flagged files (sensitivity analysis).

Files scored with --accept-md5-mismatch carry loader=ffmpeg_md5mismatch. To show that they do not
change your conclusions, compute metrics on the full CSV and on the "clean" copy this script writes:

    python scripts/split_flagged.py --csv eval_results_2021_full.csv
    python -m app.audio.metrics_report --scores eval_results_2021_full.csv        --out-dir metrics_2021_all   --sampled
    python -m app.audio.metrics_report --scores eval_results_2021_full.clean.csv  --out-dir metrics_2021_clean --sampled

If an embeddings sidecar (<csv>.emb.f32) exists, a row-aligned filtered copy is written too.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

FLAG = "ffmpeg_md5mismatch"
DIM = 160


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <csv stem>.clean.csv")
    args = ap.parse_args(argv)

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    keep = [i for i, r in enumerate(rows) if r.get("loader") != FLAG]
    out = args.out or args.csv.with_name(args.csv.stem + ".clean.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows[i] for i in keep)

    emb_in = Path(str(args.csv) + ".emb.f32")
    if emb_in.exists():
        emb = np.fromfile(emb_in, dtype=np.float32)
        if emb.size != len(rows) * DIM:
            print(
                f"Sidecar size mismatch ({emb.size} values for {len(rows)} rows); sidecar not written."
            )
            return 1
        emb.reshape(len(rows), DIM)[keep].tofile(Path(str(out) + ".emb.f32"))

    for label in ("0", "1"):
        tot = sum(r["label"] == label for r in rows)
        dropped = sum(r["label"] == label and r.get("loader") == FLAG for r in rows)
        print(
            f"  label {label} ({'bonafide' if label == '0' else 'spoof'}): {tot} rows, {dropped} flagged"
        )
    print(f"Wrote {out} with {len(keep)} of {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
