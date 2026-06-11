#!/usr/bin/env python3
"""Original-geometry GEE embedding experiments with stricter area diagnostics.

The standard suite is useful, but for small crowns we want to avoid rewarding
models that mostly learn site/context differences. This script keeps only the
GEE embedding bands (A00-A63) and compares:

  - full: all usable rows for the label
  - mixed_area_only: only areas containing both classes for binary labels

It writes compact metrics plus best-model prediction files for error analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight


LABELS = [
    "label_acacia",
    "label_deciduous",
    "label_showy_flower",
    "label_yellow_strict",
    "label_yellow_broad",
    "label_red_showy",
]


def embedding_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("A") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def make_models(seed: int, n_features: int, fast: bool = False) -> dict[str, Pipeline]:
    k_features = max(1, min(n_features, 64))
    _ = k_features  # kept for easy extension without changing output schema
    tree_count = 150 if fast else 500
    models = {
        "logistic_c0.1": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced", random_state=seed)),
            ]
        ),
        "logistic_c0.5": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced", random_state=seed)),
            ]
        ),
        "logistic_c2": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, C=2.0, class_weight="balanced", random_state=seed)),
            ]
        ),
        "svc_c0.5": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", SVC(C=0.5, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
            ]
        ),
        "svc_c1": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
            ]
        ),
        "svc_c2": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
            ]
        ),
        "rf_leaf1": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=tree_count,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "rf_leaf3": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=tree_count,
                        min_samples_leaf=3,
                        max_features=0.5,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_leaf1": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=tree_count,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_leaf3": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=tree_count,
                        min_samples_leaf=3,
                        max_features=0.5,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_leaf5": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=350,
                        learning_rate=0.03,
                        l2_regularization=0.1,
                        min_samples_leaf=5,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_leaf10": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        l2_regularization=0.2,
                        min_samples_leaf=10,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    if fast:
        return {
            key: models[key]
            for key in [
                "logistic_c0.5",
                "rf_leaf3",
                "extra_leaf1",
                "hist_leaf5",
            ]
        }
    return models


def fit_model(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> Pipeline:
    if "HistGradientBoostingClassifier" in repr(model):
        weights = compute_sample_weight("balanced", y)
        model.fit(x, y, model__sample_weight=weights)
    else:
        model.fit(x, y)
    return model


def score_row(y_true: pd.Series, pred: np.ndarray, labels: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
    }


def choose_threshold(model: Pipeline, x: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int) -> float | None:
    if sorted(y.unique().tolist()) != [0, 1] or not hasattr(model, "predict_proba"):
        return None

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows: list[dict] = []
    group_counts = groups.nunique()
    use_group_cv = group_counts >= 3

    if use_group_cv:
        splitter = GroupKFold(n_splits=min(5, group_counts))
        split_iter = splitter.split(x, y, groups)
    else:
        min_class = int(y.value_counts().min())
        if min_class < 3:
            return 0.5
        splitter = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=seed)
        split_iter = splitter.split(x, y)

    for train_idx, val_idx in split_iter:
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        if y_train.nunique() < 2 or y_val.nunique() < 2:
            continue
        m = clone(model)
        fit_model(m, x.iloc[train_idx], y_train)
        prob = m.predict_proba(x.iloc[val_idx])[:, 1]
        for threshold in thresholds:
            pred = (prob >= threshold).astype(int)
            rows.append({"threshold": float(threshold), "balanced_accuracy": balanced_accuracy_score(y_val, pred)})

    if not rows:
        return 0.5
    cv = pd.DataFrame(rows).groupby("threshold", as_index=False)["balanced_accuracy"].mean()
    return float(cv.sort_values(["balanced_accuracy", "threshold"], ascending=[False, True]).iloc[0]["threshold"])


def label_frame(df: pd.DataFrame, label: str, mode: str) -> pd.DataFrame:
    usable = df[df[label] != -1].copy()
    usable[label] = usable[label].astype(int)
    if mode == "mixed_area_only" and sorted(usable[label].unique().tolist()) == [0, 1]:
        mixed = []
        for area, group in usable.groupby("area"):
            if group[label].nunique() == 2:
                mixed.append(area)
        usable = usable[usable["area"].isin(mixed)].copy()
    return usable


def evaluate_split(
    df: pd.DataFrame,
    label: str,
    mode: str,
    split: str,
    holdout: str | None,
    features: list[str],
    seed: int,
    fast: bool,
    no_threshold: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    usable = label_frame(df, label, mode)
    if len(usable) < 20 or usable[label].nunique() < 2:
        return pd.DataFrame(), None

    if split == "random":
        train, test = train_test_split(usable, test_size=0.30, random_state=seed, stratify=usable[label])
    else:
        if holdout is None:
            raise ValueError("holdout is required for leave_area_out")
        train = usable[usable["area"] != holdout]
        test = usable[usable["area"] == holdout]
        if len(test) == 0 or train[label].nunique() < 2 or test[label].nunique() < 2:
            return pd.DataFrame(), None

    x_train, y_train = train[features], train[label].astype(int)
    x_test, y_test = test[features], test[label].astype(int)
    labels = sorted(usable[label].unique().tolist())
    rows = []
    prediction_frames = []

    for model_name, model in make_models(seed, len(features), fast=fast).items():
        m = fit_model(clone(model), x_train, y_train)
        pred = m.predict(x_test)
        base = score_row(y_test, pred, labels)
        rows.append(
            {
                "label": label,
                "mode": mode,
                "split": split,
                "holdout": holdout or "",
                "model": model_name,
                "decision": "default",
                "seed": seed,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_counts": json.dumps(y_train.value_counts().sort_index().to_dict()),
                "test_counts": json.dumps(y_test.value_counts().sort_index().to_dict()),
                **base,
            }
        )

        threshold = None if no_threshold else choose_threshold(m, x_train, y_train, train["area"], seed)
        if threshold is not None and hasattr(m, "predict_proba"):
            prob = m.predict_proba(x_test)[:, 1]
            tuned = (prob >= threshold).astype(int)
            tuned_score = score_row(y_test, tuned, labels)
            rows.append(
                {
                    "label": label,
                    "mode": mode,
                    "split": split,
                    "holdout": holdout or "",
                    "model": model_name,
                    "decision": "group_threshold",
                    "threshold": threshold,
                    "seed": seed,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_counts": json.dumps(y_train.value_counts().sort_index().to_dict()),
                    "test_counts": json.dumps(y_test.value_counts().sort_index().to_dict()),
                    **tuned_score,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        best = result.sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False]).iloc[0]
        best_model = make_models(seed, len(features), fast=fast)[best["model"]]
        best_model = fit_model(best_model, x_train, y_train)
        if best["decision"] == "group_threshold" and hasattr(best_model, "predict_proba"):
            prob = best_model.predict_proba(x_test)[:, 1]
            best_pred = (prob >= float(best["threshold"])).astype(int)
        else:
            prob = best_model.predict_proba(x_test)[:, 1] if hasattr(best_model, "predict_proba") and len(labels) == 2 else np.nan
            best_pred = best_model.predict(x_test)
        pred_df = test[
            [
                c
                for c in [
                    "crown_uid",
                    "area",
                    "species_clean",
                    "species_raw",
                    "source_file",
                    "source_index",
                    "orig_crown_id",
                ]
                if c in test.columns
            ]
        ].copy()
        pred_df["label"] = label
        pred_df["mode"] = mode
        pred_df["split"] = split
        pred_df["holdout"] = holdout or ""
        pred_df["model"] = best["model"]
        pred_df["decision"] = best["decision"]
        pred_df["y_true"] = y_test.to_numpy()
        pred_df["y_pred"] = best_pred
        if np.ndim(prob) == 1:
            pred_df["prob_positive"] = prob
        prediction_frames.append(pred_df)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else None
    return result, predictions


def holdouts_for(df: pd.DataFrame, label: str, mode: str) -> list[str]:
    usable = label_frame(df, label, mode)
    rows = []
    for area, group in usable.groupby("area"):
        counts = group[label].value_counts()
        if len(counts) >= 2:
            rows.append((area, int(counts.min()), len(group)))
    return [area for area, _, _ in sorted(rows, key=lambda x: (x[1], x[2]), reverse=True)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="src/notebooks/satellite/embeddings/exports/normalized_iitd_sv_gee_embeddings_2024_original.csv",
    )
    parser.add_argument("--outdir", default="src/notebooks/satellite/embeddings/outputs/original_only_refined")
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Use a small model set for quick iteration.")
    parser.add_argument("--no-threshold", action="store_true", help="Skip inner CV threshold tuning.")
    args = parser.parse_args()

    csv = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    features = embedding_cols(df)
    labels = args.label or [label for label in LABELS if label in df.columns]
    all_rows = []
    all_predictions = []

    for label in labels:
        for mode in ["full", "mixed_area_only"]:
            for split, holdout in [("random", None)] + [("leave_area_out", area) for area in holdouts_for(df, label, mode)]:
                rows, predictions = evaluate_split(
                    df,
                    label,
                    mode,
                    split,
                    holdout,
                    features,
                    args.seed,
                    args.fast,
                    args.no_threshold,
                )
                if not rows.empty:
                    all_rows.append(rows)
                if predictions is not None:
                    all_predictions.append(predictions)

    metrics = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    metrics_path = outdir / "metrics.csv"
    predictions_path = outdir / "best_predictions.csv"
    summary_path = outdir / "best_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    if not metrics.empty:
        best = (
            metrics.sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False])
            .groupby(["label", "mode", "split", "holdout"], as_index=False)
            .head(1)
        )
        best.to_csv(summary_path, index=False)
        print(best[["label", "mode", "split", "holdout", "model", "decision", "balanced_accuracy", "macro_f1", "confusion_matrix"]].to_string(index=False))

    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
