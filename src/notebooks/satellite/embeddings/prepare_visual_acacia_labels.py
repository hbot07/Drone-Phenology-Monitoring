#!/usr/bin/env python3
"""Prepare visually annotated Sanjay Van Acacia labels for experiments."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


AREA_MAP = {
    "s1": "SV_S1",
    "s2": "SV_S2",
    "s3": "SV_S3",
    "s4": "SV_S4",
}
LABEL_MAP = {
    "non_acacia": 0,
    "acacia": 1,
}


def image_name_to_crown_uid(image_name: str, id_offset: int) -> str | None:
    match = re.fullmatch(r"(s[1-4])_tree_(\d+)\.tif", str(image_name).strip())
    if not match:
        return None
    area = AREA_MAP[match.group(1)]
    crown_num = int(match.group(2)) + id_offset
    return f"{area}:crown_{crown_num:04d}"


def load_visual_labels(path: Path, id_offset: int) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"image_name", "label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    labels = labels.copy()
    labels["label"] = labels["label"].astype(str).str.strip().str.lower()
    unknown = sorted(set(labels["label"]) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"Unknown label values in {path}: {unknown}")
    labels["crown_uid"] = labels["image_name"].map(lambda name: image_name_to_crown_uid(name, id_offset))
    labels["label_acacia_visual"] = labels["label"].map(LABEL_MAP).astype(int)
    labels["visual_label_source"] = labels["image_name"]
    return labels


def write_visual_geojson(crowns_path: Path, labels: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    geo = json.loads(crowns_path.read_text(encoding="utf-8"))
    label_by_uid = labels.set_index("crown_uid").to_dict(orient="index")
    rows = []
    features = []
    missing_in_geo = []

    for feature in geo["features"]:
        props = feature.get("properties") or {}
        uid = props.get("crown_uid")
        if uid not in label_by_uid:
            continue
        label_info = label_by_uid[uid]
        props = dict(props)
        props["label_acacia_visual"] = int(label_info["label_acacia_visual"])
        props["visual_label"] = label_info["label"]
        props["visual_label_source"] = label_info["visual_label_source"]
        feature = dict(feature)
        feature["properties"] = props
        features.append(feature)
        rows.append(props)

    geo_uids = {feature.get("properties", {}).get("crown_uid") for feature in geo["features"]}
    for uid in labels["crown_uid"]:
        if uid not in geo_uids:
            missing_in_geo.append(uid)

    out_geo = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_geo), encoding="utf-8")

    summary = pd.DataFrame(rows)
    if missing_in_geo:
        print(f"[WARN] {len(missing_in_geo)} visual labels did not match crown GeoJSON: {missing_in_geo[:10]}")
    print(f"Wrote {out_path} with {len(features)} matched visual labels")
    return summary


def write_visual_gee_csv(gee_path: Path, labels: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    gee = pd.read_csv(gee_path)
    labels_small = labels[["crown_uid", "label_acacia_visual", "visual_label_source"]].copy()
    merged = gee.merge(labels_small, on="crown_uid", how="left")
    merged["label_acacia_visual"] = merged["label_acacia_visual"].fillna(-1).astype(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {(merged['label_acacia_visual'] != -1).sum()} matched visual labels")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        default="src/notebooks/satellite/drone_phenology_rf_package/data/labeling_sheet.csv",
    )
    parser.add_argument(
        "--crowns",
        default="src/notebooks/satellite/drone_phenology_rf_package/data/sv_crowns_clustering_acacia_labeled.geojson",
    )
    parser.add_argument(
        "--gee-original",
        default="src/notebooks/satellite/embeddings/exports/normalized_iitd_sv_gee_embeddings_2024_original.csv",
    )
    parser.add_argument(
        "--out-geojson",
        default="src/notebooks/satellite/drone_phenology_rf_package/data/sv_crowns_visual_acacia_labeled.geojson",
    )
    parser.add_argument(
        "--out-csv",
        default="src/notebooks/satellite/embeddings/exports/normalized_iitd_sv_gee_embeddings_2024_original_visual_acacia.csv",
    )
    parser.add_argument(
        "--id-offset",
        type=int,
        default=1,
        help="Offset added to tree number from image_name before building crown_XXXX IDs. Use 1 for zero-based image names.",
    )
    args = parser.parse_args()

    labels = load_visual_labels(Path(args.labels), args.id_offset)
    print(f"Using image-name to crown UID offset: {args.id_offset}")
    print("Visual label counts:")
    print(labels["label"].value_counts().to_string())
    print("Visual labels by area:")
    print(pd.crosstab(labels["crown_uid"].str.split(":").str[0], labels["label"]).to_string())

    summary = write_visual_geojson(Path(args.crowns), labels, Path(args.out_geojson))
    write_visual_gee_csv(Path(args.gee_original), labels, Path(args.out_csv))

    if not summary.empty:
        summary["label_acacia_visual"] = summary["label_acacia_visual"].astype(int)
        for compare_col in ["label_acacia", "label_acacia_clustering"]:
            if compare_col not in summary.columns:
                continue
            comparable = summary[summary[compare_col].isin([0, 1])].copy()
            if comparable.empty:
                continue
            agreement = (comparable["label_acacia_visual"] == comparable[compare_col].astype(int)).mean()
            print(f"Agreement with {compare_col}: {agreement:.3f} on n={len(comparable)}")
            print(pd.crosstab(comparable["label_acacia_visual"], comparable[compare_col].astype(int)).to_string())


if __name__ == "__main__":
    main()
