#!/usr/bin/env python3
"""Run classifier sweeps and train model artifacts for a downloaded GEE feature CSV.

Works for GEE Satellite Embedding exports and other crown-level GEE tables, such
as Planet NICFI features, as long as the label/crown metadata columns are present.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent / "drone_phenology_rf_package"
MODEL_SUITE = PACKAGE / "python" / "12_run_model_suite.py"
TRAIN_BEST = ROOT / "train_best_classifiers.py"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
EXPORT_DIR = ROOT / "exports"

DEFAULT_LABELS = [
    "label_acacia",
    "label_deciduous",
    "label_esd",
    "label_showy_flower",
    "label_yellow_strict",
    "label_yellow_broad",
    "label_red_showy",
]

GEE_COLUMN_RENAMES = {
    "lab_acac": "label_acacia",
    "lab_decid": "label_deciduous",
    "lab_esd": "label_esd",
    "lab_red": "label_red_showy",
    "lab_showy": "label_showy_flower",
    "lab_yel_b": "label_yellow_broad",
    "lab_yel_s": "label_yellow_strict",
    "sp_clean": "species_clean",
    "sp_raw": "species_raw",
    "sp_status": "species_status",
    "src_file": "source_file",
    "src_idx": "source_index",
    "orig_id": "orig_crown_id",
    "fld_desc": "field_description",
    "fld_stat": "field_status",
    "tree_type": "tree_type_raw",
}


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def normalize_gee_csv(csv: Path) -> Path:
    """Map shapefile-shortened GEE columns back to the package schema."""
    df = pd.read_csv(csv)
    df = df.rename(columns={k: v for k, v in GEE_COLUMN_RENAMES.items() if k in df.columns})
    for label in DEFAULT_LABELS:
        if label in df.columns:
            df[label] = df[label].fillna(-1).astype(int)

    # Keep raw Earth Engine export intact; write a model-ready copy next to other embedding exports.
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / f"normalized_{csv.stem}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote normalized GEE CSV: {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Downloaded GEE feature CSV from Drive.")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--max-holdouts", type=int, default=6)
    ap.add_argument("--random-only", action="store_true")
    ap.add_argument("--normalize-only", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    csv = normalize_gee_csv(Path(args.csv))
    if args.normalize_only:
        return

    labels = args.label or DEFAULT_LABELS
    outdir = Path(args.outdir) if args.outdir else OUTPUT_DIR / f"gee_{csv.stem}"
    outdir.mkdir(parents=True, exist_ok=True)

    suite_cmd = [
        sys.executable,
        str(MODEL_SUITE),
        "--csv",
        str(csv),
        "--outdir",
        str(outdir),
        "--trees",
        str(args.trees),
        "--max-holdouts",
        str(args.max_holdouts),
    ]
    for label in labels:
        suite_cmd.extend(["--label", label])
    if args.random_only:
        suite_cmd.append("--random-only")
    run(suite_cmd)

    if not args.skip_train:
        run(
            [
                sys.executable,
                str(TRAIN_BEST),
                "--csv",
                str(csv),
                "--sweep-dir",
                str(outdir),
                "--models-dir",
                str(MODELS_DIR),
                "--trees",
                "500",
            ]
        )


if __name__ == "__main__":
    main()
