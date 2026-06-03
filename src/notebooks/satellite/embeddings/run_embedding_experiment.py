#!/usr/bin/env python3
"""Run the DINO embedding extraction + classifier suite from one place."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent / "drone_phenology_rf_package"
EXTRACTOR = PACKAGE / "python" / "14_extract_embedding_features.py"
MODEL_SUITE = PACKAGE / "python" / "12_run_model_suite.py"
EXPORT_DIR = ROOT / "exports"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_LABELS = [
    "label_acacia",
    "label_deciduous",
    "label_esd",
    "label_showy_flower",
    "label_yellow_strict",
]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--all-default-labels", action="store_true")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--seasons", nargs="+", default=["mar", "apr", "may"])
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--patch-px", type=int, default=56)
    ap.add_argument("--max-cloud", type=float, default=70.0)
    ap.add_argument("--max-items-per-season", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-crowns", type=int, default=None)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--max-holdouts", type=int, default=6)
    ap.add_argument("--random-only", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = DEFAULT_LABELS if args.all_default_labels else (args.label or ["label_acacia"])
    season_tag = "-".join(args.seasons)
    label_tag = "default_labels" if args.all_default_labels else "-".join(labels)
    out_csv = Path(args.out_csv) if args.out_csv else EXPORT_DIR / f"dino_{label_tag}_{args.year}_{season_tag}.csv"
    outdir = Path(args.outdir) if args.outdir else OUTPUT_DIR / out_csv.stem

    if not args.skip_extract:
        extract_cmd = [
            sys.executable,
            str(EXTRACTOR),
            "--year",
            str(args.year),
            "--seasons",
            *args.seasons,
            "--patch-px",
            str(args.patch_px),
            "--model",
            args.model,
            "--max-cloud",
            str(args.max_cloud),
            "--max-items-per-season",
            str(args.max_items_per_season),
            "--batch-size",
            str(args.batch_size),
            "--label-filter",
            *labels,
            "--out-csv",
            str(out_csv),
        ]
        if args.limit_crowns is not None:
            extract_cmd.extend(["--limit-crowns", str(args.limit_crowns)])
        run(extract_cmd)

    if args.extract_only:
        print(f"Wrote embeddings to {out_csv}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    suite_cmd = [
        sys.executable,
        str(MODEL_SUITE),
        "--csv",
        str(out_csv),
        "--outdir",
        str(outdir),
    ]
    for label in labels:
        suite_cmd.extend(["--label", label])
    suite_cmd.extend([
        "--trees",
        str(args.trees),
        "--max-holdouts",
        str(args.max_holdouts),
    ])
    if args.random_only:
        suite_cmd.append("--random-only")
    run(suite_cmd)


if __name__ == "__main__":
    main()
