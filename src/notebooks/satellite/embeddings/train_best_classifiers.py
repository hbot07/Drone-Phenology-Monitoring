#!/usr/bin/env python3
"""Train practical full-data classifiers from the best embedding sweep rows."""
from __future__ import annotations

import argparse
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent / "drone_phenology_rf_package"
MODEL_SWEEP = SourceFileLoader(
    "model_sweep", str(PACKAGE / "python" / "06_model_sweep.py")
).load_module()
RF_EVAL = MODEL_SWEEP.rf_eval


def best_random_rows(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(sweep_dir.rglob("*_random_model_sweep.csv")):
        df = pd.read_csv(path)
        df = df[df["decision"] != "error"].copy()
        if df.empty:
            continue
        best = df.sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False]).iloc[0].copy()
        best["sweep_file"] = str(path)
        rows.append(best)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument(
        "--run-name",
        default=None,
        help="Optional model artifact subdirectory name. Defaults to the source CSV stem.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trees", type=int, default=500)
    args = ap.parse_args()

    csv = Path(args.csv)
    sweep_dir = Path(args.sweep_dir)
    models_dir = Path(args.models_dir) / (args.run_name or csv.stem)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    trained = []
    for _, row in best_random_rows(sweep_dir).iterrows():
        label = row["label"]
        model_name = row["model"]
        usable = df[df[label] != -1].copy()
        if usable.empty:
            continue

        feature_cols = RF_EVAL.infer_feature_cols(df, label)
        x = usable[feature_cols]
        y = usable[label].astype(int)
        model = MODEL_SWEEP.make_models(args.seed, args.trees, len(feature_cols))[model_name]
        model.fit(x, y)

        artifact = {
            "model": model,
            "label": label,
            "model_name": model_name,
            "decision": row.get("decision", "default"),
            "decision_threshold": (
                None
                if pd.isna(row.get("threshold", None))
                else float(row.get("threshold"))
            ),
            "feature_cols": feature_cols,
            "classes": sorted(y.unique().tolist()),
            "train_counts": y.value_counts().sort_index().to_dict(),
            "source_csv": str(csv),
            "best_random_balanced_accuracy": float(row["balanced_accuracy"]),
            "best_random_macro_f1": float(row["macro_f1"]),
            "sweep_file": row["sweep_file"],
        }
        out_path = models_dir / f"{label}_{model_name}.joblib"
        joblib.dump(artifact, out_path)
        trained.append(
            {
                "label": label,
                "model": model_name,
                "decision": row.get("decision", "default"),
                "decision_threshold": (
                    None
                    if pd.isna(row.get("threshold", None))
                    else float(row.get("threshold"))
                ),
                "artifact": str(out_path),
                "n_train": int(len(usable)),
                "n_features": len(feature_cols),
                "random_balanced_accuracy": float(row["balanced_accuracy"]),
                "random_macro_f1": float(row["macro_f1"]),
            }
        )

    summary_path = models_dir / "trained_models_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(trained, f, indent=2)
    print(pd.DataFrame(trained).to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
