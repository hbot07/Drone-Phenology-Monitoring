#!/usr/bin/env python3
"""Print the best row from each embedding model-sweep CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = ap.parse_args()

    rows = []
    root = Path(args.outdir)
    for path in sorted(root.rglob("*_model_sweep.csv")):
        df = pd.read_csv(path)
        if df.empty or "balanced_accuracy" not in df.columns:
            continue
        best = df.sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "file": path.name,
                "run": str(path.parent.relative_to(root)),
                "label": best.get("label", ""),
                "split": best.get("split", ""),
                "holdout": best.get("holdout", ""),
                "model": best.get("model", ""),
                "decision": best.get("decision", ""),
                "balanced_accuracy": round(float(best.get("balanced_accuracy", 0.0)), 3),
                "macro_f1": round(float(best.get("macro_f1", 0.0)), 3),
                "n_train": best.get("n_train", ""),
                "n_test": best.get("n_test", ""),
            }
        )

    if not rows:
        print(f"No model sweep CSVs found in {args.outdir}")
        return
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
