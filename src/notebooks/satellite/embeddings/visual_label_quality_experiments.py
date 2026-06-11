#!/usr/bin/env python3
"""Fast label-quality experiments for GEE original visual Acacia classifiers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def embedding_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("A") and c[1:].isdigit()], key=lambda c: int(c[1:]))


def load_with_crown_area(csv_path: Path, crowns_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    crowns = gpd.read_file(crowns_path).to_crs("EPSG:32643")
    crowns["crown_area_m2"] = crowns.geometry.area
    return df.merge(crowns[["crown_uid", "crown_area_m2"]], on="crown_uid", how="left")


def models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_c0.1": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced", random_state=seed)),
            ]
        ),
        "logistic_c0.5": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=seed)),
            ]
        ),
        "svc_linearish": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", SVC(C=0.5, kernel="linear", class_weight="balanced", random_state=seed)),
            ]
        ),
        "svc_rbf_c0.5": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", SVC(C=0.5, gamma="scale", class_weight="balanced", random_state=seed)),
            ]
        ),
        "rf_leaf3": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
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
                        n_estimators=250,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def score(y_true: pd.Series, pred: np.ndarray) -> dict:
    labels = sorted(set(y_true.tolist()) | set(pred.tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": json.dumps(confusion_matrix(y_true, pred, labels=labels).tolist()),
    }


def subset_frame(df: pd.DataFrame, label: str, experiment: str, min_area: float | None) -> pd.DataFrame:
    usable = df[df[label] != -1].copy()
    usable[label] = usable[label].astype(int)
    if min_area is not None:
        usable = usable[usable["crown_area_m2"] >= min_area].copy()
    if experiment == "visual_cluster_agree":
        usable = usable[usable["label_acacia_visual"] == usable["label_acacia_clustering"]].copy()
    return usable


def eval_one(
    data: pd.DataFrame,
    label: str,
    feature_cols: list[str],
    split: str,
    holdout: str | None,
    seed: int,
    experiment: str,
) -> pd.DataFrame:
    if len(data) < 30 or data[label].nunique() < 2:
        return pd.DataFrame()
    if split == "random":
        train, test = train_test_split(data, test_size=0.30, random_state=seed, stratify=data[label])
    else:
        train = data[data["area"] != holdout]
        test = data[data["area"] == holdout]
        if len(test) == 0 or train[label].nunique() < 2 or test[label].nunique() < 2:
            return pd.DataFrame()
    rows = []
    x_train, y_train = train[feature_cols], train[label].astype(int)
    x_test, y_test = test[feature_cols], test[label].astype(int)
    for name, model in models(seed).items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append(
            {
                "experiment": experiment,
                "label": label,
                "feature_set": "embeddings_plus_area" if "crown_area_m2" in feature_cols else "embeddings",
                "split": split,
                "holdout": holdout or "",
                "model": name,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_counts": json.dumps(y_train.value_counts().sort_index().to_dict()),
                "test_counts": json.dumps(y_test.value_counts().sort_index().to_dict()),
                **score(y_test, pred),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="src/notebooks/satellite/embeddings/exports/gee_original_acacia_label_configs.csv",
    )
    parser.add_argument(
        "--crowns",
        default="src/notebooks/satellite/drone_phenology_rf_package/data/sv_crowns_clustering_acacia_labeled.geojson",
    )
    parser.add_argument(
        "--outdir",
        default="src/notebooks/satellite/embeddings/outputs/gee_original_acacia_visual_quality",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_with_crown_area(Path(args.csv), Path(args.crowns))
    embed = embedding_cols(df)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    visual = df[df["label_acacia_visual"] != -1].copy()
    q25, q50 = visual["crown_area_m2"].quantile([0.25, 0.50]).tolist()
    experiments = [
        ("visual_all", "label_acacia_visual", None),
        ("visual_drop_smallest_quartile", "label_acacia_visual", q25),
        ("visual_drop_smallest_half", "label_acacia_visual", q50),
        ("visual_cluster_agree", "label_acacia_visual", None),
        ("visual_or_species", "label_acacia_visual_or_species", None),
    ]
    all_rows = []
    for experiment, label, min_area in experiments:
        data = subset_frame(df, label, experiment, min_area)
        holdouts = [
            area
            for area, group in data.groupby("area")
            if group[label].nunique() == 2 and len(group) >= 10
        ]
        for feature_cols in [embed, embed + ["crown_area_m2"]]:
            for split, holdout in [("random", None)] + [("leave_area_out", h) for h in holdouts]:
                result = eval_one(data, label, feature_cols, split, holdout, args.seed, experiment)
                if not result.empty:
                    all_rows.append(result)

    metrics = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    metrics_path = outdir / "metrics.csv"
    summary_path = outdir / "best_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    if not metrics.empty:
        summary = (
            metrics.sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False])
            .groupby(["experiment", "label", "feature_set", "split", "holdout"], as_index=False)
            .head(1)
        )
        summary.to_csv(summary_path, index=False)
        print(
            summary[
                [
                    "experiment",
                    "feature_set",
                    "split",
                    "holdout",
                    "model",
                    "balanced_accuracy",
                    "macro_f1",
                    "confusion_matrix",
                    "n_train",
                    "n_test",
                ]
            ].to_string(index=False)
        )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
