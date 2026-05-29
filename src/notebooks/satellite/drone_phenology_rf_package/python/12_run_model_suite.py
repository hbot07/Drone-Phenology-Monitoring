#!/usr/bin/env python3
"""Run the tabular model sweep for several labels/splits on a feature CSV."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_SWEEP = ROOT / "python" / "06_model_sweep.py"

DEFAULT_LABELS = [
    "label_acacia",
    "label_deciduous",
    "label_esd",
    "label_showy_flower",
    "label_yellow_strict",
]


def safe_run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def holdout_areas(df: pd.DataFrame, label: str, max_areas: int = 6) -> list[str]:
    usable = df[df[label] != -1]
    rows = []
    for area, group in usable.groupby("area"):
        counts = group[label].value_counts()
        if len(counts) < 2:
            continue
        rows.append((area, len(group), int(counts.min())))
    rows = sorted(rows, key=lambda x: (x[2], x[1]), reverse=True)
    return [area for area, _, _ in rows[:max_areas]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--max-holdouts", type=int, default=6)
    ap.add_argument("--random-only", action="store_true")
    args = ap.parse_args()

    csv = Path(args.csv)
    df = pd.read_csv(csv)
    labels = args.label or [label for label in DEFAULT_LABELS if label in df.columns]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for label in labels:
        if label not in df.columns:
            print(f"Skipping missing label {label}", flush=True)
            continue
        if (df[label] != -1).sum() < 30:
            print(f"Skipping {label}: too few usable rows", flush=True)
            continue
        safe_run(
            [
                sys.executable,
                str(MODEL_SWEEP),
                "--csv",
                str(csv),
                "--label",
                label,
                "--split",
                "random",
                "--trees",
                str(args.trees),
                "--outdir",
                str(outdir),
            ]
        )
        if args.random_only:
            continue
        for area in holdout_areas(df, label, args.max_holdouts):
            safe_run(
                [
                    sys.executable,
                    str(MODEL_SWEEP),
                    "--csv",
                    str(csv),
                    "--label",
                    label,
                    "--split",
                    "leave_area_out",
                    "--holdout",
                    area,
                    "--trees",
                    str(args.trees),
                    "--outdir",
                    str(outdir),
                ]
            )


if __name__ == "__main__":
    main()
