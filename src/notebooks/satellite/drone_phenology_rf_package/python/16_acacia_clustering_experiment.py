#!/usr/bin/env python3
"""
Acacia clustering augmentation experiment.

Compares two training regimes for label_acacia:
  baseline  — only the 362 manually ground-truthed crowns
  augmented — GT crowns + 2071 clustering-labeled SV crowns

Reports crown-level and pixel-level balanced accuracy, F1, confusion matrix for
both regimes on the SAME held-out GT test set.

Usage:
  python python/16_acacia_clustering_experiment.py \
      --baseline-csv  exports/stac_s2_features_2022_2025_buffer10_items4_label_acacia.csv \
      --augmented-csv exports/sv_aug_s2_2022_2025_combined_label_acacia.csv \
      --outdir        outputs/acacia_clustering_experiment
"""
from __future__ import annotations

import argparse
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
_rf_eval = SourceFileLoader(
    "rf_eval",
    str(Path(__file__).with_name("02_local_rf_from_gee_export.py")),
).load_module()
_pixel = SourceFileLoader(
    "pixel",
    str(Path(__file__).with_name("15_pixel_level_classifier.py")),
).load_module()

LABEL = "label_acacia"
SEEDS = [42, 43, 44]
N_TREES = 400


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_pipelines(seed: int) -> dict[str, Pipeline]:
    return {
        "rf_balanced": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=N_TREES, random_state=seed,
                class_weight="balanced_subsample",
                min_samples_leaf=2, max_features="sqrt", n_jobs=-1)),
        ]),
        "extra_trees": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=N_TREES, random_state=seed,
                class_weight="balanced", min_samples_leaf=2,
                max_features="sqrt", n_jobs=-1)),
        ]),
    }


def evaluate_crown(model, x_test, y_test) -> dict:
    pred = model.predict(x_test)
    return {
        "bal_acc": round(balanced_accuracy_score(y_test, pred), 4),
        "f1":      round(f1_score(y_test, pred, average="binary", zero_division=0), 4),
        "cm":      confusion_matrix(y_test, pred).tolist(),
    }


def evaluate_pixel(model, train_pixels, feature_cols, x_test_crowns,
                   y_test_crowns, uids_test, rng, pixels_per_crown=8,
                   noise_std=0.015) -> dict:
    # Simulate pixels for test crowns and aggregate predictions
    test_pixels = _pixel.simulate_pixels(
        x_test_crowns.assign(**{LABEL: y_test_crowns.values,
                                "crown_uid": uids_test}),
        feature_cols, LABEL, pixels_per_crown, noise_std, rng,
    )
    classes = sorted(model.named_steps["model"].classes_.tolist())
    proba = model.predict_proba(test_pixels[feature_cols])
    agg = _pixel.aggregate_to_crowns(
        proba, test_pixels[LABEL].values, test_pixels["crown_uid"].tolist(),
        model, classes,
    )
    y_true = agg["true_label"].values
    y_pred = agg["pred_label"].values
    return {
        "bal_acc": round(balanced_accuracy_score(y_true, y_pred), 4),
        "f1":      round(f1_score(y_true, y_pred, average="binary", zero_division=0), 4),
        "cm":      confusion_matrix(y_true, y_pred).tolist(),
    }


def run_split(
    label: str,
    baseline_df: pd.DataFrame,
    augmented_df: pd.DataFrame | None,
    feature_cols: list[str],
    split: str,
    holdout: str | None,
    seed: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    For one split, train on (a) baseline only, (b) augmented.
    Test always on GT-only held-out set from baseline_df.
    """
    # GT test set (always from baseline)
    train_bl, test_bl = _rf_eval.make_split(baseline_df, label, split, holdout, seed)

    rows = []
    for regime, train_df in [("baseline", train_bl),
                              ("augmented", None if augmented_df is None else None)]:
        if regime == "augmented":
            if augmented_df is None:
                continue
            # augmented train = baseline train crowns + clustering extra rows
            # (extra rows are the ones NOT in baseline, ie clustering-only)
            bl_uids = set(baseline_df["crown_uid"])
            aug_extra = augmented_df[~augmented_df["crown_uid"].isin(bl_uids)]
            train_df = pd.concat([train_bl, aug_extra], ignore_index=True)
            print(f"    Augmented train: {len(train_bl)} GT + {len(aug_extra)} clustering = {len(train_df)}")

        x_train = train_df[feature_cols]
        y_train = train_df[label].astype(int)
        x_test  = test_bl[feature_cols]
        y_test  = test_bl[label].astype(int)
        uids    = test_bl["crown_uid"].tolist()

        for model_name, model in make_pipelines(seed).items():
            model.fit(x_train, y_train)
            crown_m = evaluate_crown(model, x_test, y_test)
            pixel_m = evaluate_pixel(model, None, feature_cols,
                                     test_bl, y_test, uids, rng)
            rows.append({
                "regime": regime, "split": split, "holdout": holdout or "n/a",
                "seed": seed, "model": model_name,
                "n_train": len(train_df), "n_test": len(test_bl),
                "crown_bal_acc": crown_m["bal_acc"],
                "crown_f1": crown_m["f1"],
                "crown_cm": crown_m["cm"],
                "pixel_bal_acc": pixel_m["bal_acc"],
                "pixel_f1": pixel_m["f1"],
                "pixel_cm": pixel_m["cm"],
            })
            print(f"    [{regime}|{model_name}] crown_bal_acc={crown_m['bal_acc']:.3f}  "
                  f"crown_f1={crown_m['f1']:.3f}  pixel_bal_acc={pixel_m['bal_acc']:.3f}  "
                  f"pixel_f1={pixel_m['f1']:.3f}")
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["regime", "model"]).agg(
        crown_bal_acc_mean=("crown_bal_acc", "mean"),
        crown_bal_acc_std=("crown_bal_acc", "std"),
        crown_f1_mean=("crown_f1", "mean"),
        crown_f1_std=("crown_f1", "std"),
        pixel_bal_acc_mean=("pixel_bal_acc", "mean"),
        pixel_bal_acc_std=("pixel_bal_acc", "std"),
        pixel_f1_mean=("pixel_f1", "mean"),
        pixel_f1_std=("pixel_f1", "std"),
        n_runs=("crown_bal_acc", "count"),
    ).round(4).reset_index()
    return grp


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--augmented-csv", default=None,
                    help="Multiyear CSV for all SV crowns (GT + clustering-labeled). "
                         "If omitted, only the baseline regime runs.")
    ap.add_argument("--outdir", default="outputs/acacia_clustering_experiment")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--max-holdouts", type=int, default=4)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(args.baseline_csv)
    baseline_df = baseline_df[baseline_df[LABEL] != -1].reset_index(drop=True)
    print(f"Baseline GT rows: {len(baseline_df)}  "
          f"(acacia=1: {(baseline_df[LABEL]==1).sum()}  "
          f"non-acacia=0: {(baseline_df[LABEL]==0).sum()})")

    augmented_df = None
    if args.augmented_csv and Path(args.augmented_csv).exists():
        augmented_df = pd.read_csv(args.augmented_csv)
        augmented_df = augmented_df[augmented_df[LABEL] != -1].reset_index(drop=True)
        print(f"Augmented CSV rows: {len(augmented_df)}  "
              f"(acacia=1: {(augmented_df[LABEL]==1).sum()}  "
              f"non-acacia=0: {(augmented_df[LABEL]==0).sum()})")
    else:
        print("No augmented CSV provided / not yet available — running baseline only.")

    feature_cols = _rf_eval.infer_feature_cols(baseline_df, LABEL)
    print(f"Feature columns: {len(feature_cols)}")

    # ── Identify holdout areas (from baseline GT only) ────────────────────────
    holdout_areas = _rf_eval.get_holdout_areas(
        baseline_df, LABEL, args.max_holdouts
    ) if hasattr(_rf_eval, "get_holdout_areas") else []

    # Fallback: compute manually
    if not holdout_areas:
        usable = baseline_df[baseline_df[LABEL] != -1]
        rows_h = []
        for area, grp in usable.groupby("area"):
            vc = grp[LABEL].value_counts()
            if len(vc) >= 2:
                rows_h.append((area, len(grp), int(vc.min())))
        rows_h = sorted(rows_h, key=lambda x: (x[2], x[1]), reverse=True)
        holdout_areas = [a for a, _, _ in rows_h[: args.max_holdouts]]

    print(f"Holdout areas: {holdout_areas}")

    all_rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        print(f"\n── Seed {seed} ──")

        # Random split
        print(f"  [random split]")
        all_rows.extend(run_split(LABEL, baseline_df, augmented_df,
                                   feature_cols, "random", None, seed, rng))

        # Leave-one-area-out
        for area in holdout_areas:
            print(f"  [leave_area_out  holdout={area}]")
            all_rows.extend(run_split(LABEL, baseline_df, augmented_df,
                                       feature_cols, "leave_area_out", area,
                                       seed, rng))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(outdir / "raw_results.csv", index=False)

    summary = summarise(results_df)
    summary.to_csv(outdir / "summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (averaged across seeds & holdouts)")
    print("=" * 70)
    print(summary.to_string(index=False))

    # ── Regime delta ─────────────────────────────────────────────────────────
    if augmented_df is not None:
        for model in summary["model"].unique():
            bl = summary[(summary["regime"] == "baseline") & (summary["model"] == model)].iloc[0]
            au = summary[(summary["regime"] == "augmented") & (summary["model"] == model)].iloc[0]
            print(f"\n{model} | augmented vs baseline:")
            for m in ["crown_bal_acc_mean", "crown_f1_mean", "pixel_bal_acc_mean", "pixel_f1_mean"]:
                delta = au[m] - bl[m]
                sign = "+" if delta >= 0 else ""
                print(f"  {m:<30} {bl[m]:.4f} → {au[m]:.4f}  ({sign}{delta:.4f})")

    print(f"\nOutputs written to {outdir}/")


if __name__ == "__main__":
    main()
