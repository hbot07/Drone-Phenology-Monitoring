#!/usr/bin/env python3
"""Create Acacia label configuration columns for GEE original embedding experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def prioritized(*values: int) -> int:
    for value in values:
        if int(value) != -1:
            return int(value)
    return -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gee-csv",
        default="src/notebooks/satellite/embeddings/exports/normalized_iitd_sv_gee_embeddings_2024_original_visual_acacia.csv",
    )
    parser.add_argument(
        "--cluster-geojson",
        default="src/notebooks/satellite/drone_phenology_rf_package/data/sv_crowns_clustering_acacia_labeled.geojson",
    )
    parser.add_argument(
        "--out-csv",
        default="src/notebooks/satellite/embeddings/exports/gee_original_acacia_label_configs.csv",
    )
    args = parser.parse_args()

    gee = pd.read_csv(args.gee_csv)
    cluster_geo = json.loads(Path(args.cluster_geojson).read_text(encoding="utf-8"))
    cluster = pd.DataFrame([feature["properties"] for feature in cluster_geo["features"]])
    cluster = cluster[["crown_uid", "label_acacia_clustering"]].copy()
    cluster["label_acacia_clustering"] = cluster["label_acacia_clustering"].fillna(-1).astype(int)

    df = gee.drop(columns=["label_acacia_clustering"], errors="ignore").merge(cluster, on="crown_uid", how="left")
    df["label_acacia_clustering"] = df["label_acacia_clustering"].fillna(-1).astype(int)

    df["label_acacia_species"] = df["label_acacia"].fillna(-1).astype(int)
    df["label_acacia_visual"] = df["label_acacia_visual"].fillna(-1).astype(int)
    df["label_acacia_visual_or_species"] = [
        prioritized(v, s) for v, s in zip(df["label_acacia_visual"], df["label_acacia_species"])
    ]
    df["label_acacia_visual_or_clustering"] = [
        prioritized(v, c) for v, c in zip(df["label_acacia_visual"], df["label_acacia_clustering"])
    ]
    df["label_acacia_species_or_clustering"] = [
        prioritized(s, c) for s, c in zip(df["label_acacia_species"], df["label_acacia_clustering"])
    ]
    df["label_acacia_all_priority"] = [
        prioritized(v, s, c)
        for v, s, c in zip(df["label_acacia_visual"], df["label_acacia_species"], df["label_acacia_clustering"])
    ]

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    labels = [
        "label_acacia_species",
        "label_acacia_clustering",
        "label_acacia_visual",
        "label_acacia_visual_or_species",
        "label_acacia_visual_or_clustering",
        "label_acacia_species_or_clustering",
        "label_acacia_all_priority",
    ]
    print(f"Wrote {out}")
    print("Label config counts:")
    for label in labels:
        print(label, df[label].value_counts(dropna=False).sort_index().to_dict())

    print("\nComparable agreement with visual labels:")
    for label in ["label_acacia_species", "label_acacia_clustering"]:
        comparable = df[(df["label_acacia_visual"] != -1) & (df[label] != -1)].copy()
        if comparable.empty:
            continue
        agreement = (comparable["label_acacia_visual"] == comparable[label]).mean()
        print(f"{label}: {agreement:.3f} on n={len(comparable)}")
        print(pd.crosstab(comparable["label_acacia_visual"], comparable[label]).to_string())


if __name__ == "__main__":
    main()
