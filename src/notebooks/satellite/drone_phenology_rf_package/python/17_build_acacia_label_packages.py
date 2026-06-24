#!/usr/bin/env python3
"""Build Acacia label GeoJSON packages from species, field, desk, and clustering labels.

Outputs:
  data/acacia_clean_confident_labels.geojson
  data/acacia_clean_plus_clustering_labels.geojson
  outputs/acacia_label_package_area_summary.csv
  outputs/acacia_label_package_conflicts.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_CONFIG = ROOT / "configs" / "exp01_species_review" / "acacia_vs_non_acacia.json"
DEFAULT_MASTER = DATA_DIR / "iitd_sv_crowns_master_wgs84.geojson"
DEFAULT_FIELD_DIR = DATA_DIR / "field mapped crowns sanjay van"
DEFAULT_DESK_CSV = DATA_DIR / "labeling_sheet.csv"
DEFAULT_CLUSTER = DATA_DIR / "sv_crowns_clustering_acacia_labeled.geojson"

AREA_MAP = {
    "s1": "SV_S1",
    "s2": "SV_S2",
    "s3": "SV_S3",
    "s4": "SV_S4",
}

TREE_TYPE_MAP = {
    "acacia": 1,
    "non-acacia": 0,
    "non acacia": 0,
    "non_acacia": 0,
}

DESK_LABEL_MAP = {
    "acacia": 1,
    "non_acacia": 0,
    "non-acacia": 0,
    "non acacia": 0,
}


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def int_or_ignore(value: Any) -> int:
    try:
        if pd.isna(value):
            return -1
        return int(value)
    except (TypeError, ValueError):
        return -1


def desk_image_to_uid(image_name: str, id_offset: int) -> str | None:
    match = re.fullmatch(r"(s[1-4])_tree_(\d+)\.tif", norm_text(image_name))
    if not match:
        return None
    area = AREA_MAP[match.group(1)]
    crown_num = int(match.group(2)) + id_offset
    return f"{area}:crown_{crown_num:04d}"


def load_species_labels(master_rows: pd.DataFrame, config_path: Path) -> dict[str, int]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    positive = set(cfg.get("positive_species", []))
    negative = set(cfg.get("negative_species", []))
    labels: dict[str, int] = {}
    for row in master_rows.itertuples(index=False):
        species = norm_text(getattr(row, "species_clean", ""))
        uid = norm_text(getattr(row, "crown_uid", ""))
        if not uid:
            continue
        if species in positive:
            labels[uid] = 1
        elif species in negative:
            labels[uid] = 0
        else:
            labels[uid] = -1
    return labels


def load_field_labels(field_dir: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for stem, area in AREA_MAP.items():
        path = field_dir / f"{stem}.geojson"
        if not path.exists():
            new_path = field_dir / f"{stem}_new.geojson"
            path = new_path if new_path.exists() else path
        if not path.exists():
            continue
        geo = json.loads(path.read_text(encoding="utf-8"))
        for feature in geo.get("features", []):
            props = feature.get("properties") or {}
            raw = norm_text(props.get("Tree type")).lower()
            if raw not in TREE_TYPE_MAP:
                continue
            try:
                crown_num = int(props.get("fid"))
            except (TypeError, ValueError):
                continue
            uid = f"{area}:crown_{crown_num:04d}"
            labels[uid] = TREE_TYPE_MAP[raw]
    return labels


def load_desk_labels(labels_csv: Path, id_offset: int) -> dict[str, int]:
    labels: dict[str, int] = {}
    table = pd.read_csv(labels_csv)
    for row in table.itertuples(index=False):
        uid = desk_image_to_uid(getattr(row, "image_name", ""), id_offset)
        raw = norm_text(getattr(row, "label", "")).lower()
        if uid and raw in DESK_LABEL_MAP:
            labels[uid] = DESK_LABEL_MAP[raw]
    return labels


def load_cluster_labels(cluster_geojson: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    geo = json.loads(cluster_geojson.read_text(encoding="utf-8"))
    for feature in geo.get("features", []):
        props = feature.get("properties") or {}
        uid = norm_text(props.get("crown_uid"))
        label = int_or_ignore(props.get("label_acacia_clustering"))
        if uid and label in (0, 1):
            labels[uid] = label
    return labels


def choose_label(uid: str, sources: list[tuple[str, dict[str, int]]]) -> tuple[int, str]:
    for source_name, labels in sources:
        label = labels.get(uid, -1)
        if label in (0, 1):
            return label, source_name
    return -1, "none"


def conflict_record(
    uid: str,
    area: str,
    source_values: dict[str, int],
    conflict_sources: list[str],
) -> dict[str, Any] | None:
    known = {k: source_values.get(k, -1) for k in conflict_sources if source_values.get(k, -1) in (0, 1)}
    if len(set(known.values())) <= 1:
        return None
    return {
        "crown_uid": uid,
        "area": area,
        **{f"label_acacia_{k}": source_values.get(k, -1) for k in ["species", "field", "desk", "clustering"]},
        "known_values": json.dumps(known, sort_keys=True),
    }


def add_labels(
    master_geojson: dict,
    source_labels: dict[str, dict[str, int]],
    include_clustering: bool,
) -> tuple[dict, list[dict[str, Any]]]:
    priority = [
        ("species", source_labels["species"]),
        ("field", source_labels["field"]),
        ("desk", source_labels["desk"]),
    ]
    if include_clustering:
        priority.append(("clustering", source_labels["clustering"]))
    conflict_sources = [name for name, _ in priority]

    out_features = []
    conflicts = []
    final_col = "label_acacia_clean_plus_clustering" if include_clustering else "label_acacia_clean_confident"
    source_col = f"{final_col}_source"

    for feature in master_geojson.get("features", []):
        props = dict(feature.get("properties") or {})
        uid = norm_text(props.get("crown_uid"))
        area = norm_text(props.get("area"))
        values = {
            "species": source_labels["species"].get(uid, -1),
            "field": source_labels["field"].get(uid, -1),
            "desk": source_labels["desk"].get(uid, -1),
            "clustering": source_labels["clustering"].get(uid, -1),
        }
        final_label, source = choose_label(uid, priority)
        props.update(
            {
                "label_acacia_species_only": values["species"],
                "label_acacia_field": values["field"],
                "label_acacia_desk": values["desk"],
                "label_acacia_clustering": values["clustering"],
                final_col: final_label,
                source_col: source,
            }
        )
        conflict = conflict_record(uid, area, values, conflict_sources)
        if conflict is not None:
            conflict["package"] = "clean_plus_clustering" if include_clustering else "clean_confident"
            conflicts.append(conflict)
        out_feature = dict(feature)
        out_feature["properties"] = props
        out_features.append(out_feature)

    out_geojson = {k: v for k, v in master_geojson.items() if k != "features"}
    out_geojson["features"] = out_features
    return out_geojson, conflicts


def summarize_package(geojson: dict, label_col: str, source_col: str, package: str) -> pd.DataFrame:
    rows = [feature.get("properties") or {} for feature in geojson.get("features", [])]
    table = pd.DataFrame(rows)
    grouped = (
        table.groupby("area", dropna=False)
        .agg(
            crowns=("crown_uid", "count"),
            labelled=(label_col, lambda s: int(s.isin([0, 1]).sum())),
            acacia=(label_col, lambda s: int((s == 1).sum())),
            non_acacia=(label_col, lambda s: int((s == 0).sum())),
            species_source=(source_col, lambda s: int((s == "species").sum())),
            field_source=(source_col, lambda s: int((s == "field").sum())),
            desk_source=(source_col, lambda s: int((s == "desk").sum())),
            clustering_source=(source_col, lambda s: int((s == "clustering").sum())),
        )
        .reset_index()
    )
    grouped.insert(0, "package", package)
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(DEFAULT_MASTER))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--field-dir", default=str(DEFAULT_FIELD_DIR))
    ap.add_argument("--desk-csv", default=str(DEFAULT_DESK_CSV))
    ap.add_argument("--desk-id-offset", type=int, default=1)
    ap.add_argument("--cluster-geojson", default=str(DEFAULT_CLUSTER))
    ap.add_argument("--clean-out", default=str(DATA_DIR / "acacia_clean_confident_labels.geojson"))
    ap.add_argument("--cluster-out", default=str(DATA_DIR / "acacia_clean_plus_clustering_labels.geojson"))
    args = ap.parse_args()

    master_geojson = json.loads(Path(args.master).read_text(encoding="utf-8"))
    master_rows = pd.DataFrame([f.get("properties") or {} for f in master_geojson.get("features", [])])

    source_labels = {
        "species": load_species_labels(master_rows, Path(args.config)),
        "field": load_field_labels(Path(args.field_dir)),
        "desk": load_desk_labels(Path(args.desk_csv), args.desk_id_offset),
        "clustering": load_cluster_labels(Path(args.cluster_geojson)),
    }

    clean_geojson, clean_conflicts = add_labels(master_geojson, source_labels, include_clustering=False)
    clustered_geojson, clustered_conflicts = add_labels(master_geojson, source_labels, include_clustering=True)

    clean_path = Path(args.clean_out)
    cluster_path = Path(args.cluster_out)
    clean_path.write_text(json.dumps(clean_geojson, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    cluster_path.write_text(json.dumps(clustered_geojson, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_summary = summarize_package(
        clean_geojson,
        "label_acacia_clean_confident",
        "label_acacia_clean_confident_source",
        "clean_confident",
    )
    cluster_summary = summarize_package(
        clustered_geojson,
        "label_acacia_clean_plus_clustering",
        "label_acacia_clean_plus_clustering_source",
        "clean_plus_clustering",
    )
    summary = pd.concat([clean_summary, cluster_summary], ignore_index=True)
    summary_path = OUTPUT_DIR / "acacia_label_package_area_summary.csv"
    summary.to_csv(summary_path, index=False)

    conflicts = pd.DataFrame(clean_conflicts + clustered_conflicts).drop_duplicates()
    conflicts_path = OUTPUT_DIR / "acacia_label_package_conflicts.csv"
    conflicts.to_csv(conflicts_path, index=False)

    print(f"Wrote {clean_path}")
    print(f"Wrote {cluster_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {conflicts_path}")
    print("\nSource counts:")
    for source, labels in source_labels.items():
        counts = pd.Series(labels).value_counts().sort_index().to_dict()
        print(f"  {source:<10} {counts}")
    print("\nArea summary:")
    print(summary.to_string(index=False))
    print("\nAreas:", ", ".join(sorted(master_rows["area"].dropna().unique())))


if __name__ == "__main__":
    main()
