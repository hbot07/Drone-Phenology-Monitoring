#!/usr/bin/env python3
"""Rank crowns for Acacia annotation review using the visual-label GEE embedding model."""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd


def add_crown_area(df: pd.DataFrame, crowns_geojson: Path) -> pd.DataFrame:
    crowns = gpd.read_file(crowns_geojson).to_crs("EPSG:32643")
    crowns["crown_area_m2"] = crowns.geometry.area
    return df.merge(crowns[["crown_uid", "crown_area_m2"]], on="crown_uid", how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="src/notebooks/satellite/embeddings/exports/gee_original_acacia_label_configs.csv",
    )
    parser.add_argument(
        "--model",
        default="src/notebooks/satellite/embeddings/models/gee_original_acacia_label_configs/label_acacia_visual_rf_balanced.joblib",
    )
    parser.add_argument(
        "--crowns",
        default="src/notebooks/satellite/drone_phenology_rf_package/data/sv_crowns_clustering_acacia_labeled.geojson",
    )
    parser.add_argument(
        "--outdir",
        default="src/notebooks/satellite/embeddings/outputs/gee_original_acacia_review_candidates",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    artifact = joblib.load(args.model)
    feature_cols = artifact["feature_cols"]
    model = artifact["model"]

    probs = model.predict_proba(df[feature_cols])[:, 1]
    pred = (probs >= 0.5).astype(int)
    scored = add_crown_area(df, Path(args.crowns))
    scored["pred_acacia_visual_model"] = pred
    scored["prob_acacia_visual_model"] = probs
    scored["model_confidence"] = scored["prob_acacia_visual_model"].where(
        scored["pred_acacia_visual_model"].eq(1),
        1.0 - scored["prob_acacia_visual_model"],
    )
    scored["visual_labeled"] = scored["label_acacia_visual"].isin([0, 1])
    scored["cluster_labeled"] = scored["label_acacia_clustering"].isin([0, 1])
    scored["species_labeled"] = scored["label_acacia_species"].isin([0, 1])
    scored["model_cluster_disagree"] = (
        scored["cluster_labeled"]
        & scored["pred_acacia_visual_model"].ne(scored["label_acacia_clustering"])
    )
    scored["uncertain"] = scored["prob_acacia_visual_model"].between(0.4, 0.6)

    keep_cols = [
        "crown_uid",
        "area",
        "crown_area_m2",
        "species_clean",
        "label_acacia_visual",
        "label_acacia_species",
        "label_acacia_clustering",
        "pred_acacia_visual_model",
        "prob_acacia_visual_model",
        "model_confidence",
        "model_cluster_disagree",
        "uncertain",
        "lon",
        "lat",
        "source_file",
        "source_index",
        "orig_crown_id",
    ]
    keep_cols = [c for c in keep_cols if c in scored.columns]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scored[keep_cols].to_csv(outdir / "all_crowns_scored.csv", index=False)

    unlabeled = scored[~scored["visual_labeled"]].copy()
    high_conf_unlabeled = unlabeled.sort_values("model_confidence", ascending=False).head(300)
    uncertain = unlabeled.sort_values("model_confidence", ascending=True).head(300)
    disagreements = unlabeled[unlabeled["model_cluster_disagree"]].sort_values(
        "model_confidence", ascending=False
    ).head(300)

    high_conf_unlabeled[keep_cols].to_csv(outdir / "high_confidence_unlabeled.csv", index=False)
    uncertain[keep_cols].to_csv(outdir / "uncertain_unlabeled.csv", index=False)
    disagreements[keep_cols].to_csv(outdir / "model_cluster_disagreements.csv", index=False)

    summary = {
        "total_crowns": int(len(scored)),
        "visual_labeled": int(scored["visual_labeled"].sum()),
        "unlabeled_by_visual": int((~scored["visual_labeled"]).sum()),
        "model_predicted_acacia_all": int(scored["pred_acacia_visual_model"].sum()),
        "model_predicted_non_acacia_all": int(scored["pred_acacia_visual_model"].eq(0).sum()),
        "unlabeled_predicted_acacia": int(unlabeled["pred_acacia_visual_model"].sum()),
        "unlabeled_predicted_non_acacia": int(unlabeled["pred_acacia_visual_model"].eq(0).sum()),
        "unlabeled_uncertain_40_60": int(unlabeled["uncertain"].sum()),
        "unlabeled_model_cluster_disagreements": int(unlabeled["model_cluster_disagree"].sum()),
        "high_confidence_unlabeled_0_80_or_more": int((unlabeled["model_confidence"] >= 0.8).sum()),
        "high_confidence_unlabeled_0_90_or_more": int((unlabeled["model_confidence"] >= 0.9).sum()),
    }
    pd.Series(summary).to_csv(outdir / "review_candidate_summary.csv", header=["value"])

    print(pd.Series(summary).to_string())
    print("\nUnlabeled predictions by site:")
    print(pd.crosstab(unlabeled["area"], unlabeled["pred_acacia_visual_model"]).to_string())
    print("\nTop model-vs-clustering disagreements:")
    print(
        disagreements[
            [
                "crown_uid",
                "area",
                "label_acacia_clustering",
                "pred_acacia_visual_model",
                "prob_acacia_visual_model",
                "model_confidence",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
