#!/usr/bin/env python3
"""
Train a binary Random Forest and sweep probability thresholds.

This is useful when the default 0.5 hard decision is too conservative for the
positive class, which happened in the first Acacia leave-area-out runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from importlib.machinery import SourceFileLoader


rf_eval = SourceFileLoader("rf_eval", str(Path(__file__).with_name("02_local_rf_from_gee_export.py"))).load_module()


def train_model(x_train: pd.DataFrame, y_train: pd.Series, trees: int, seed: int) -> Pipeline:
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=trees,
                    random_state=seed,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def metrics_for_threshold(y_true: np.ndarray, proba_pos: np.ndarray, threshold: float) -> dict:
    pred = (proba_pos >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "positive_recall": float(((pred == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())),
        "positive_precision": float(((pred == 1) & (y_true == 1)).sum() / max(1, (pred == 1).sum())),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def choose_threshold_cv(x_train: pd.DataFrame, y_train: pd.Series, trees: int, seed: int) -> tuple[float, pd.DataFrame]:
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(x_train, y_train), start=1):
        model = train_model(x_train.iloc[tr_idx], y_train.iloc[tr_idx], trees, seed + fold)
        prob = model.predict_proba(x_train.iloc[val_idx])[:, 1]
        yt = y_train.iloc[val_idx].to_numpy()
        for threshold in thresholds:
            row = metrics_for_threshold(yt, prob, float(threshold))
            row["fold"] = fold
            rows.append(row)
    sweep = pd.DataFrame(rows)
    mean_scores = sweep.groupby("threshold", as_index=False)["balanced_accuracy"].mean()
    best = mean_scores.sort_values(["balanced_accuracy", "threshold"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--split", default="random", choices=["random", "leave_area_out", "leave_species_out"])
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--outdir", default="outputs/local_rf_threshold")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    train, test = rf_eval.make_split(df, args.label, args.split, args.holdout, args.seed)
    feature_cols = rf_eval.infer_feature_cols(df, args.label)
    y_train = train[args.label].astype(int)
    y_test = test[args.label].astype(int)
    if sorted(y_train.unique().tolist()) != [0, 1]:
        raise ValueError("This script expects binary training labels 0/1")

    if args.threshold is None:
        threshold, cv_sweep = choose_threshold_cv(train[feature_cols], y_train, args.trees, args.seed)
    else:
        threshold = float(args.threshold)
        cv_sweep = pd.DataFrame()

    model = train_model(train[feature_cols], y_train, args.trees, args.seed)
    proba = model.predict_proba(test[feature_cols])[:, 1]
    report = metrics_for_threshold(y_test.to_numpy(), proba, threshold)
    report.update(
        {
            "csv": args.csv,
            "label": args.label,
            "split": args.split,
            "holdout": args.holdout,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_label_counts": y_train.value_counts().sort_index().to_dict(),
            "test_label_counts": y_test.value_counts().sort_index().to_dict(),
            "threshold_source": "cv" if args.threshold is None else "manual",
        }
    )

    stem = f"{args.label}_{args.split}" + (f"_{args.holdout}" if args.holdout else "")
    with (outdir / f"{stem}_threshold_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if not cv_sweep.empty:
        cv_sweep.to_csv(outdir / f"{stem}_cv_threshold_sweep.csv", index=False)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
