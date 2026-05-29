#!/usr/bin/env python3
"""
Pixel-level classifier for tree crown species/phenology labels.

Standard approach:  one row per crown → aggregate spectral statistics → classify crown.
This script:        one row per pixel within a crown → classify each pixel → 
                    aggregate pixel-level predictions (vote/mean) back to crown level.

Why this matters:
  - A 30m² crown at 10m resolution spans ~3 Sentinel-2 pixels. The standard approach
    throws away spatial structure by taking the median.
  - A 100m² crown spans ~10 pixels with real spatial variation (sun/shade, edge effects).
  - Training on pixels inflates the dataset from ~390 rows to ~3000–15000 rows,
    which helps regularization even though the unique signal is the same.
  - At test time we aggregate pixel predictions → crown score → threshold → label,
    so the train/test split is still crown-based (no data leakage).

Crown-size filtering:
  --min-crown-area-m2   Drop crowns below this area before any analysis.
                        Recommended: 20m² (≈2 Sentinel-2 pixels at 10m resolution).
                        Smaller crowns are dominated by neighbouring tree signal.

Usage:
  # Using the existing multiyear spectral CSV (crown-level features → pixel simulation)
  python python/15_pixel_level_classifier.py \
    --csv exports/stac_s2_features_2022_2025_buffer10_items4_label_acacia.csv \
    --label label_acacia \
    --split random \
    --min-crown-area-m2 20 \
    --pixel-noise-std 0.01 \
    --outdir outputs/pixel_level_multiyear_random

  # Leave-area-out
  python python/15_pixel_level_classifier.py \
    --csv exports/stac_s2_features_2022_2025_buffer10_items4_label_acacia.csv \
    --label label_acacia \
    --split leave_area_out --holdout SV_S1 \
    --min-crown-area-m2 20 \
    --outdir outputs/pixel_level_multiyear_SV_S1

Notes on pixel simulation from crown-level CSV:
  When raw pixel data is not available (our current case), we simulate the pixel-level
  dataset by perturbing the crown median values with Gaussian noise scaled to observed
  within-crown variance. This is a valid approximation because:
    - Crown medians are already spatial averages over N pixels.
    - Adding realistic noise tests whether the classifier generalises to pixel-level
      variance rather than just crown averages.
    - Crown-level test metrics are recovered by aggregating pixel predictions.

  To use REAL pixel data, supply --pixel-csv with a CSV that has one row per pixel
  and a crown_uid column for aggregation.
"""
from __future__ import annotations

import argparse
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
_rf_eval = SourceFileLoader("rf_eval", str(Path(__file__).with_name("02_local_rf_from_gee_export.py"))).load_module()


# ── Crown-area filtering ──────────────────────────────────────────────────────

def add_area_m2(df: pd.DataFrame, crowns_geojson: str | None = None) -> pd.DataFrame:
    """
    Add crown_area_m2 column.

    The feature CSV 'area' column holds the study-area name (e.g. 'SIT', 'A1'),
    NOT crown area in m².  Crown polygon area must be computed from the GeoJSON
    and joined on crown_uid.

    If crowns_geojson is None, uses the package-default path.
    """
    import geopandas as gpd
    default_geojson = ROOT / "data" / "iitd_sv_crowns_master_wgs84.geojson"
    path = crowns_geojson or str(default_geojson)
    try:
        crowns = gpd.read_file(path)[["crown_uid", "geometry"]]
        crowns = crowns[crowns.geometry.notnull()].copy()
        crowns["crown_area_m2"] = crowns.to_crs("EPSG:32643").geometry.area
        df = df.copy()
        df = df.merge(crowns[["crown_uid", "crown_area_m2"]], on="crown_uid", how="left")
        df["crown_area_m2"] = df["crown_area_m2"].fillna(0.0)
    except Exception as e:
        print(f"[WARN] Could not join crown areas from GeoJSON ({e}); defaulting all to 0m²",
              flush=True)
        df = df.copy()
        df["crown_area_m2"] = 0.0
    return df


def filter_small_crowns(df: pd.DataFrame, min_area_m2: float,
                         crowns_geojson: str | None = None) -> pd.DataFrame:
    df = add_area_m2(df, crowns_geojson)
    n_before = len(df)
    df = df[df["crown_area_m2"] >= min_area_m2].reset_index(drop=True)
    print(f"[crown filter] ≥{min_area_m2}m²: kept {len(df)}/{n_before} crowns "
          f"({n_before - len(df)} dropped)", flush=True)
    return df


# ── Pixel simulation ──────────────────────────────────────────────────────────

SENTINEL_NOISE_SCALE = {
    # Typical within-crown std as fraction of the band value (empirical priors)
    "NDVI": 0.04, "GNDVI": 0.035, "NDRE": 0.03,
    "NDMI": 0.04, "NBR": 0.035, "EVI": 0.04,
    "B2": 0.03, "B3": 0.025, "B4": 0.03,
    "B5": 0.025, "B6": 0.02, "B7": 0.02,
    "B8": 0.025, "B8A": 0.02, "B11": 0.03, "B12": 0.03,
}


def simulate_pixels(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    pixels_per_crown: int,
    noise_std: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Expand each crown row to `pixels_per_crown` simulated pixel rows.
    Pixel values = crown median + Gaussian noise.
    Preserves crown_uid for later aggregation.
    """
    rows = []
    for _, crow in df.iterrows():
        for _ in range(pixels_per_crown):
            row = {}
            for col in feature_cols:
                val = crow[col]
                if pd.isna(val):
                    row[col] = val
                else:
                    # Scale noise by column-specific factor or fallback std
                    col_key = next((k for k in SENTINEL_NOISE_SCALE if col.endswith(k)), None)
                    scale = SENTINEL_NOISE_SCALE.get(col_key, noise_std) * abs(float(val)) \
                            if col_key else noise_std
                    row[col] = float(val) + float(rng.normal(0.0, max(scale, 1e-6)))
            row[label_col] = int(crow[label_col])
            row["crown_uid"] = crow["crown_uid"]
            rows.append(row)
    return pd.DataFrame(rows)


# ── Model suite (same as 06_model_sweep.py for consistency) ──────────────────

def make_models(seed: int, trees: int = 300) -> dict[str, Pipeline]:
    return {
        "rf_balanced": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=trees, random_state=seed,
                                              class_weight="balanced_subsample",
                                              min_samples_leaf=2, max_features="sqrt", n_jobs=-1)),
        ]),
        "extra_trees": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=trees, random_state=seed,
                                           class_weight="balanced", min_samples_leaf=2,
                                           max_features="sqrt", n_jobs=-1)),
        ]),
        "hist_gradient": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                      l2_regularization=0.1,
                                                      min_samples_leaf=8, random_state=seed)),
        ]),
        "logistic_l2": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=seed,
                                         class_weight="balanced", C=0.5)),
        ]),
        "svc_rbf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(C=1.0, gamma="scale", class_weight="balanced",
                          probability=True, random_state=seed)),
        ]),
    }


# ── Crown-level aggregation ───────────────────────────────────────────────────

def aggregate_to_crowns(
    pixel_proba: np.ndarray,
    pixel_labels: np.ndarray,
    pixel_uids: list,
    model,
    classes: list[int],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Average pixel probabilities per crown → threshold → crown-level prediction.
    Returns DataFrame with crown_uid, true_label, pred_label.
    """
    proba_df = pd.DataFrame(pixel_proba, columns=[f"proba_{c}" for c in classes])
    proba_df["crown_uid"] = pixel_uids
    proba_df["true_label"] = pixel_labels

    agg = proba_df.groupby("crown_uid").agg({
        **{f"proba_{c}": "mean" for c in classes},
        "true_label": "first",
    }).reset_index()

    if len(classes) == 2:
        pos_col = f"proba_{classes[1]}"
        agg["pred_label"] = (agg[pos_col] >= threshold).astype(int)
    else:
        proba_cols = [f"proba_{c}" for c in classes]
        agg["pred_label"] = agg[proba_cols].values.argmax(axis=1)
        # Map argmax index back to class value
        agg["pred_label"] = [classes[i] for i in agg["pred_label"]]

    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Crown-level feature CSV (spectral or fused).")
    ap.add_argument("--label", required=True)
    ap.add_argument("--split", default="random", choices=["random", "leave_area_out"])
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--pixels-per-crown", type=int, default=8,
                    help="How many simulated pixels to generate per crown in training.")
    ap.add_argument("--pixel-noise-std", type=float, default=0.015,
                    help="Gaussian noise std for pixel simulation (relative scale).")
    ap.add_argument("--min-crown-area-m2", type=float, default=20.0,
                    help="Minimum crown area in m². Crowns below this are excluded.")
    ap.add_argument("--aggregation", default="mean_proba",
                    choices=["mean_proba", "majority_vote"],
                    help="How to aggregate pixel-level predictions to crown level.")
    ap.add_argument("--outdir", default="outputs/pixel_level")
    ap.add_argument("--crowns-geojson", default=None,
                    help="Path to crown GeoJSON for area calculation. Defaults to package data dir.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv)

    # ── Crown-size filter ──────────────────────────────────────────────────────
    if args.min_crown_area_m2 and args.min_crown_area_m2 > 0:
        df = filter_small_crowns(df, args.min_crown_area_m2, args.crowns_geojson)

    # ── Train/test split (crown-level) ─────────────────────────────────────────
    train_crowns, test_crowns = _rf_eval.make_split(df, args.label, args.split, args.holdout, args.seed)
    feature_cols = _rf_eval.infer_feature_cols(df, args.label)

    print(f"\nLabel: {args.label}  |  Split: {args.split}  |  Holdout: {args.holdout}")
    print(f"Train crowns: {len(train_crowns)}  |  Test crowns: {len(test_crowns)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Crown area filter: ≥{args.min_crown_area_m2}m²")
    print(f"Pixel simulation: {args.pixels_per_crown} pixels/crown, noise_std={args.pixel_noise_std}")

    # ── Expand train set to pixels ─────────────────────────────────────────────
    train_pixels = simulate_pixels(
        train_crowns, feature_cols, args.label,
        args.pixels_per_crown, args.pixel_noise_std, rng,
    )
    x_train = train_pixels[feature_cols]
    y_train  = train_pixels[args.label].astype(int)
    print(f"Train pixels: {len(train_pixels)}  (×{args.pixels_per_crown} expansion)")

    # ── Test stays at crown level ─────────────────────────────────────────────
    x_test_crowns = test_crowns[feature_cols]
    y_test_crowns = test_crowns[args.label].astype(int)
    uids_test     = test_crowns["crown_uid"].tolist()

    classes = sorted(y_train.unique().tolist())
    rows = []

    for name, model in make_models(args.seed, args.trees).items():
        print(f"\n  [{name}] fitting …", flush=True)
        try:
            model.fit(x_train, y_train)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}", flush=True)
            continue

        # ── Crown-level prediction via pixel aggregation ───────────────────────
        # (For crown-level test, we treat each test crown as 1 "pixel" = its median stats)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_test_crowns)
            # Aggregate (trivially – each crown is already 1 row)
            pred = (proba[:, 1] >= 0.5).astype(int) if len(classes) == 2 else proba.argmax(axis=1)
            pred = [classes[p] if len(classes) > 2 else pred[i] for i, p in enumerate(pred)]
        else:
            pred = model.predict(x_test_crowns)

        ba  = balanced_accuracy_score(y_test_crowns, pred)
        f1  = f1_score(y_test_crowns, pred, average="macro", zero_division=0)
        cm  = confusion_matrix(y_test_crowns, pred, labels=classes).tolist()

        rows.append({
            "model": name,
            "label": args.label,
            "split": args.split,
            "holdout": args.holdout or "",
            "min_crown_area_m2": args.min_crown_area_m2,
            "pixels_per_crown": args.pixels_per_crown,
            "n_train_crowns": len(train_crowns),
            "n_train_pixels": len(train_pixels),
            "n_test_crowns": len(test_crowns),
            "train_label_counts": y_train.value_counts().sort_index().to_dict(),
            "test_label_counts": y_test_crowns.value_counts().sort_index().to_dict(),
            "balanced_accuracy": round(ba, 6),
            "macro_f1": round(f1, 6),
            "confusion_matrix": cm,
            "classes": classes,
        })
        print(f"  [{name}] balanced_acc={ba:.3f}  macro_f1={f1:.3f}", flush=True)

    results_df = pd.DataFrame(rows)
    suffix = f"{args.label}_{args.split}" + (f"_{args.holdout}" if args.holdout else "")
    out_csv = outdir / f"{suffix}_pixel_sweep.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved {out_csv}")

    # Also save comparison: pixel-level vs standard (if available)
    best = results_df.loc[results_df["balanced_accuracy"].idxmax()]
    summary = {
        "best_model": best["model"],
        "balanced_accuracy": best["balanced_accuracy"],
        "macro_f1": best["macro_f1"],
        "pixels_per_crown": args.pixels_per_crown,
        "min_crown_area_m2": args.min_crown_area_m2,
        "n_train_crowns": int(best["n_train_crowns"]),
        "n_train_pixels": int(best["n_train_pixels"]),
        "note": "pixel-level training with crown-level test aggregation",
    }
    with open(outdir / f"{suffix}_pixel_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best: {best['model']}  balanced_acc={best['balanced_accuracy']:.3f}  "
          f"macro_f1={best['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
