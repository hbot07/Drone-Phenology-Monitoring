#!/usr/bin/env python3
"""
Compare several tabular classifiers on an extracted satellite feature CSV.

This is meant for quick iteration after feature extraction. It reuses the same
feature-column inference and split logic as 02_local_rf_from_gee_export.py.
"""
from __future__ import annotations

import argparse
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight


rf_eval = SourceFileLoader("rf_eval", str(Path(__file__).with_name("02_local_rf_from_gee_export.py"))).load_module()


def make_models(seed: int, trees: int, n_features: int) -> dict[str, Pipeline]:
    k = min(40, n_features)
    return {
        "rf_balanced": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=trees,
                        random_state=seed,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "rf_deeper": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=trees,
                        random_state=seed,
                        class_weight="balanced_subsample",
                        min_samples_leaf=1,
                        max_features=0.5,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=trees,
                        random_state=seed,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_kbest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=trees,
                        random_state=seed,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "logistic_l2": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=seed,
                        class_weight="balanced",
                        C=0.5,
                    ),
                ),
            ]
        ),
        "svc_rbf": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gradient": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.04,
                        l2_regularization=0.1,
                        min_samples_leaf=8,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def score(y_true, pred) -> dict:
    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(pred).tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
        "labels": labels,
    }


def threshold_scores(model, x_train, y_train, x_test, y_test, seed: int) -> tuple[float | None, dict | None]:
    classes = sorted(pd.Series(y_train).unique().tolist())
    if classes != [0, 1] or not hasattr(model, "predict_proba"):
        return None, None
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(x_train, y_train), start=1):
        cloned = model
        # Recreate the model by round-tripping through sklearn clone without importing another name.
        from sklearn.base import clone

        cloned = clone(model)
        try:
            sample_weight = compute_sample_weight("balanced", y_train.iloc[tr_idx])
            if "hist_gradient" in str(cloned):
                cloned.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx], model__sample_weight=sample_weight)
            else:
                cloned.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
        except TypeError:
            cloned.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
        proba = cloned.predict_proba(x_train.iloc[val_idx])[:, 1]
        yt = y_train.iloc[val_idx].to_numpy()
        for threshold in thresholds:
            pred = (proba >= threshold).astype(int)
            rows.append({"threshold": float(threshold), "balanced_accuracy": balanced_accuracy_score(yt, pred)})
    cv_scores = pd.DataFrame(rows).groupby("threshold", as_index=False)["balanced_accuracy"].mean()
    best_threshold = float(cv_scores.sort_values(["balanced_accuracy", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
    proba_test = model.predict_proba(x_test)[:, 1]
    tuned_pred = (proba_test >= best_threshold).astype(int)
    tuned_score = score(y_test, tuned_pred)
    tuned_score["threshold"] = best_threshold
    return best_threshold, tuned_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--split", default="random", choices=["random", "leave_area_out", "leave_species_out"])
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--outdir", default="outputs/model_sweeps")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    train, test = rf_eval.make_split(df, args.label, args.split, args.holdout, args.seed)
    feature_cols = rf_eval.infer_feature_cols(df, args.label)
    x_train = train[feature_cols]
    y_train = train[args.label].astype(int)
    x_test = test[feature_cols]
    y_test = test[args.label].astype(int)

    rows = []
    for name, model in make_models(args.seed, args.trees, len(feature_cols)).items():
        try:
            if name == "hist_gradient":
                sample_weight = compute_sample_weight("balanced", y_train)
                model.fit(x_train, y_train, model__sample_weight=sample_weight)
            else:
                model.fit(x_train, y_train)
            pred = model.predict(x_test)
            base = score(y_test, pred)
            rows.append(
                {
                    "model": name,
                    "decision": "default",
                    "csv": args.csv,
                    "label": args.label,
                    "split": args.split,
                    "holdout": args.holdout or "",
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_counts": y_train.value_counts().sort_index().to_dict(),
                    "test_counts": y_test.value_counts().sort_index().to_dict(),
                    **base,
                }
            )
            _, tuned = threshold_scores(model, x_train, y_train, x_test, y_test, args.seed)
            if tuned is not None:
                rows.append(
                    {
                        "model": name,
                        "decision": "threshold_cv",
                        "csv": args.csv,
                        "label": args.label,
                        "split": args.split,
                        "holdout": args.holdout or "",
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "train_counts": y_train.value_counts().sort_index().to_dict(),
                        "test_counts": y_test.value_counts().sort_index().to_dict(),
                        **tuned,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "model": name,
                    "decision": "error",
                    "csv": args.csv,
                    "label": args.label,
                    "split": args.split,
                    "holdout": args.holdout or "",
                    "error": repr(exc),
                }
            )

    out = pd.DataFrame(rows).sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False], na_position="last")
    if "threshold" not in out.columns:
        out["threshold"] = np.nan
    stem = f"{args.label}_{args.split}" + (f"_{args.holdout}" if args.holdout else "")
    out.to_csv(outdir / f"{stem}_model_sweep.csv", index=False)
    print(out[["model", "decision", "holdout", "accuracy", "balanced_accuracy", "macro_f1", "threshold"]].to_string(index=False))
    print(f"Wrote {outdir / f'{stem}_model_sweep.csv'}")


if __name__ == "__main__":
    main()
