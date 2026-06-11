#!/usr/bin/env python3
"""Write held-out-site prediction diagnostics for GEE original Acacia labels."""
from __future__ import annotations

import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent / "drone_phenology_rf_package"
MODEL_SWEEP = SourceFileLoader(
    "model_sweep", str(PACKAGE / "python" / "06_model_sweep.py")
).load_module()
RF_EVAL = MODEL_SWEEP.rf_eval


def best_leave_site_rows(sweep_dir: Path, label: str) -> pd.DataFrame:
    rows = []
    for path in sorted(sweep_dir.glob(f"{label}_leave_area_out_*_model_sweep.csv")):
        df = pd.read_csv(path)
        df = df[(df["decision"] != "error") & (df["label"] == label)].copy()
        if df.empty:
            continue
        best = df.sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False]).iloc[0].copy()
        rows.append(best)
    return pd.DataFrame(rows)


def fit_predict(df: pd.DataFrame, label: str, row: pd.Series, seed: int, trees: int) -> pd.DataFrame:
    train, test = RF_EVAL.make_split(df, label, "leave_area_out", row["holdout"], seed)
    feature_cols = RF_EVAL.infer_feature_cols(df, label)
    model = MODEL_SWEEP.make_models(seed, trees, len(feature_cols))[row["model"]]
    x_train = train[feature_cols]
    y_train = train[label].astype(int)
    x_test = test[feature_cols]

    if row["model"] == "hist_gradient":
        sample_weight = compute_sample_weight("balanced", y_train)
        model.fit(x_train, y_train, model__sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)

    if row["decision"] == "threshold_cv" and hasattr(model, "predict_proba"):
        threshold = float(row["threshold"])
        pred = (model.predict_proba(x_test)[:, 1] >= threshold).astype(int)
    else:
        threshold = np.nan
        pred = model.predict(x_test)

    base_cols = [
        "crown_uid",
        "area",
        "species_clean",
        "label_acacia_species",
        "label_acacia_clustering",
        "label_acacia_visual",
        label,
    ]
    out = test[list(dict.fromkeys(base_cols))].copy()
    out["eval_label"] = label
    out["holdout"] = row["holdout"]
    out["model"] = row["model"]
    out["decision"] = row["decision"]
    out["threshold"] = threshold
    out["y_true"] = out[label].astype(int)
    out["y_pred"] = pred.astype(int)
    out["correct"] = out["y_true"].eq(out["y_pred"])
    return out


def add_crown_area(preds: pd.DataFrame, crowns_geojson: Path) -> pd.DataFrame:
    crowns = gpd.read_file(crowns_geojson).to_crs("EPSG:32643")
    crowns["crown_area_m2"] = crowns.geometry.area
    area = crowns[["crown_uid", "crown_area_m2"]]
    return preds.merge(area, on="crown_uid", how="left")


def write_summaries(preds: pd.DataFrame, out_dir: Path, label: str) -> None:
    site_summary = (
        preds.groupby(["eval_label", "holdout", "y_true"], observed=True)["correct"]
        .agg(["count", "mean"])
        .reset_index()
    )
    site_summary.to_csv(out_dir / f"{label}_leave_site_error_by_site_and_class.csv", index=False)

    preds = preds.copy()
    preds["crown_area_quartile"] = pd.qcut(preds["crown_area_m2"], 4, duplicates="drop")
    area_summary = (
        preds.groupby(["eval_label", "crown_area_quartile"], observed=True)["correct"]
        .agg(["count", "mean"])
        .reset_index()
    )
    area_summary.to_csv(out_dir / f"{label}_leave_site_error_by_crown_area.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="src/notebooks/satellite/embeddings/exports/gee_original_acacia_label_configs.csv",
    )
    parser.add_argument("--label", default="label_acacia_visual")
    parser.add_argument(
        "--sweep-dir",
        default="src/notebooks/satellite/embeddings/outputs/gee_original_acacia_label_configs_full",
    )
    parser.add_argument(
        "--crowns-geojson",
        default="src/notebooks/satellite/drone_phenology_rf_package/data/sv_crowns_clustering_acacia_labeled.geojson",
    )
    parser.add_argument(
        "--out-dir",
        default="src/notebooks/satellite/embeddings/outputs/gee_original_acacia_label_configs_full/error_analysis",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = best_leave_site_rows(Path(args.sweep_dir), args.label)
    if rows.empty:
        raise ValueError(f"No leave-site sweep rows found for {args.label}")

    preds = pd.concat(
        [fit_predict(df, args.label, row, args.seed, args.trees) for _, row in rows.iterrows()],
        ignore_index=True,
    )
    preds = add_crown_area(preds, Path(args.crowns_geojson))
    pred_path = out_dir / f"{args.label}_leave_site_predictions.csv"
    preds.to_csv(pred_path, index=False)
    write_summaries(preds, out_dir, args.label)

    print(f"Wrote {pred_path}")
    print("\nCorrect by site and class:")
    print(
        preds.groupby(["holdout", "y_true"], observed=True)["correct"]
        .agg(["count", "mean"])
        .to_string()
    )
    preds["crown_area_quartile"] = pd.qcut(preds["crown_area_m2"], 4, duplicates="drop")
    print("\nCorrect by crown area quartile:")
    print(
        preds.groupby("crown_area_quartile", observed=True)["correct"]
        .agg(["count", "mean"])
        .to_string()
    )


if __name__ == "__main__":
    main()
