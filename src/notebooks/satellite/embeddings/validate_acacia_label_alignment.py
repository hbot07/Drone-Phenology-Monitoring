#!/usr/bin/env python3
"""Validate Sanjay Van Acacia label/crown ID alignment.

This is a guard against the easy off-by-one mistake:
visual chip names use zero-based tree indices, while crown IDs are one-based.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
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
SOURCE_LABELS = {
    "field_raw": "field raw Tree type",
    "species_first": "species-first master label_acacia",
    "clustering": "clustering label",
    "visual": "visual chip label",
}


def norm_tree_type(value: object) -> pd.NA | int:
    if pd.isna(value) or value is None:
        return pd.NA
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if "non" in text and "acacia" in text:
        return 0
    if text == "acacia" or ("acacia" in text and "non" not in text):
        return 1
    return pd.NA


def image_name_to_uid(image_name: str, offset: int) -> str | None:
    match = re.fullmatch(r"(s[1-4])_tree_(\d+)\.tif", str(image_name).strip())
    if not match:
        return None
    area = AREA_MAP[match.group(1)]
    crown_num = int(match.group(2)) + offset
    return f"{area}:crown_{crown_num:04d}"


def load_field_labels(data_dir: Path) -> pd.DataFrame:
    rows = []
    field_dir = data_dir / "field mapped crowns sanjay van"
    for stem, area in AREA_MAP.items():
        raw = gpd.read_file(field_dir / f"{stem}.geojson").drop(columns="geometry")
        raw["area"] = area
        raw["crown_num"] = raw["fid"].astype(int)
        raw["crown_uid"] = raw["area"] + ":crown_" + raw["crown_num"].map(lambda n: f"{n:04d}")
        raw["field_raw"] = raw["Tree type"].map(norm_tree_type)
        rows.append(raw[["crown_uid", "area", "crown_num", "fid", "species", "Tree type", "field_raw"]])
    return pd.concat(rows, ignore_index=True)


def load_reference_labels(data_dir: Path) -> pd.DataFrame:
    master = gpd.read_file(data_dir / "iitd_sv_crowns_master_wgs84.geojson").drop(columns="geometry")
    master = master[master["area"].astype(str).str.startswith("SV_")].copy()
    master = master[
        ["crown_uid", "species_raw", "species_clean", "tree_type_raw", "label_acacia"]
    ].rename(columns={"label_acacia": "species_first"})

    cluster = gpd.read_file(data_dir / "sv_crowns_clustering_acacia_labeled.geojson").drop(columns="geometry")
    cluster = cluster[["crown_uid", "label_acacia_clustering", "clustering_label"]].rename(
        columns={"label_acacia_clustering": "clustering"}
    )

    return master.merge(cluster, on="crown_uid", how="left")


def load_visual_labels(labels_csv: Path, offset: int) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv).copy()
    labels["visual"] = labels["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    labels["visual_source"] = labels["image_name"]
    labels["crown_uid"] = labels["image_name"].map(lambda name: image_name_to_uid(name, offset))
    return labels[["crown_uid", "visual", "visual_source"]]


def agreement(df: pd.DataFrame, a: str, b: str) -> dict[str, float | int]:
    comparable = df[df[a].isin([0, 1]) & df[b].isin([0, 1])].copy()
    if comparable.empty:
        return {"n": 0, "agree": 0, "disagree": 0, "agreement": float("nan")}
    agree = int((comparable[a].astype(int) == comparable[b].astype(int)).sum())
    n = int(len(comparable))
    return {"n": n, "agree": agree, "disagree": n - agree, "agreement": agree / n}


def known_values(row: pd.Series) -> dict[str, int]:
    vals = {}
    for col in ["field_raw", "species_first", "clustering", "visual"]:
        if pd.notna(row.get(col)) and int(row[col]) in (0, 1):
            vals[col] = int(row[col])
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="src/notebooks/satellite/drone_phenology_rf_package/data",
    )
    parser.add_argument(
        "--labels",
        default="src/notebooks/satellite/drone_phenology_rf_package/data/labeling_sheet.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="src/notebooks/satellite/drone_phenology_rf_package/outputs",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    field = load_field_labels(data_dir)
    refs = load_reference_labels(data_dir)
    base = field.merge(refs, on="crown_uid", how="left")
    for col in ["field_raw", "species_first", "clustering"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
        base.loc[base[col] == -1, col] = pd.NA

    offset_rows = []
    for offset in [0, 1]:
        visual = load_visual_labels(Path(args.labels), offset)
        merged = visual.merge(base, on="crown_uid", how="left")
        row = {
            "offset": offset,
            "visual_labels": len(visual),
            "matched_crowns": int(merged["area"].notna().sum()),
            "unmatched_crowns": int(merged["area"].isna().sum()),
        }
        for other in ["field_raw", "species_first", "clustering"]:
            stats = agreement(merged, "visual", other)
            row[f"visual_vs_{other}_n"] = stats["n"]
            row[f"visual_vs_{other}_agreement"] = stats["agreement"]
            row[f"visual_vs_{other}_disagree"] = stats["disagree"]
        offset_rows.append(row)

    offset_summary = pd.DataFrame(offset_rows)
    offset_summary.to_csv(out_dir / "acacia_visual_offset_validation.csv", index=False)

    visual = load_visual_labels(Path(args.labels), 1)
    full = base.merge(visual, on="crown_uid", how="left")
    full["visual"] = pd.to_numeric(full["visual"], errors="coerce")

    full["known_label_values"] = full.apply(
        lambda row: "; ".join(f"{SOURCE_LABELS[k]}={v}" for k, v in known_values(row).items()),
        axis=1,
    )
    full["has_disagreement"] = full.apply(
        lambda row: len(set(known_values(row).values())) > 1,
        axis=1,
    )
    disagreements = full[full["has_disagreement"]].copy().sort_values(["area", "crown_num"])
    disagreements.to_csv(out_dir / "acacia_label_disagreements.csv", index=False)

    raw_geo_checks = []
    for stem, area in AREA_MAP.items():
        raw = gpd.read_file(data_dir / "field mapped crowns sanjay van" / f"{stem}.geojson").to_crs("EPSG:4326")
        master = gpd.read_file(data_dir / "iitd_sv_crowns_master_wgs84.geojson")
        master = master[master["area"].eq(area)].copy().to_crs("EPSG:4326")
        joined = raw[["fid", "geometry"]].merge(
            master[["crown_num", "crown_uid", "geometry"]],
            left_on="fid",
            right_on="crown_num",
            suffixes=("_raw", "_master"),
        )
        raw_geo_checks.append(
            {
                "area": area,
                "raw_features": len(raw),
                "master_features": len(master),
                "fid_to_crown_num_matches": len(joined),
                "missing_matches": len(raw) - len(joined),
            }
        )
    raw_alignment = pd.DataFrame(raw_geo_checks)
    raw_alignment.to_csv(out_dir / "acacia_raw_field_alignment_validation.csv", index=False)

    report = {
        "visual_offset_validation": offset_rows,
        "raw_field_alignment": raw_geo_checks,
        "disagreement_count_offset_1": int(len(disagreements)),
        "disagreements_by_area_offset_1": disagreements.groupby("area").size().to_dict(),
    }
    (out_dir / "acacia_label_alignment_validation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Visual offset validation")
    print(offset_summary.to_string(index=False))
    print("\nRaw field alignment")
    print(raw_alignment.to_string(index=False))
    print(f"\nDisagreements with offset 1: {len(disagreements)}")
    print(f"Wrote validation outputs to {out_dir}")


if __name__ == "__main__":
    main()
