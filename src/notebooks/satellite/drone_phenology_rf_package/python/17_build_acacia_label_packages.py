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
from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_CONFIG = ROOT / "configs" / "exp01_species_review" / "acacia_vs_non_acacia.json"
DEFAULT_MASTER = DATA_DIR / "iitd_sv_crowns_master_wgs84.geojson"
DEFAULT_FIELD_DIR = DATA_DIR / "field mapped crowns sanjay van" / "updated"
DEFAULT_DESK_CSV = DATA_DIR / "labeling_sheet.csv"
DEFAULT_CLUSTER = DATA_DIR / "sv_crowns_clustering_acacia_labeled.geojson"
DEFAULT_EXTRA_MASTER = DATA_DIR / "LHC_master.geojson"
SPECIES_CONFIG_FILENAMES = [
    "acacia_vs_non_acacia.json",
    "deciduous_vs_rest.json",
    "esd_multiclass.json",
    "yellow_showy_strict.json",
    "yellow_broad.json",
    "red_showy.json",
    "showy_flower_vs_rest.json",
]
LABEL_COLUMNS = [
    "label_esd",
    "label_deciduous",
    "label_acacia",
    "label_yellow_strict",
    "label_yellow_broad",
    "label_red_showy",
    "label_showy_flower",
]
ESD_CLASS_VALUES = {
    "evergreen": 0,
    "semi_evergreen": 1,
    "deciduous": 2,
}

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


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def species_label_from_config(species: str | None, config: dict[str, Any]) -> int:
    if not species:
        return -1
    if "positive_species_by_class" in config:
        for class_name, class_species in config.get("positive_species_by_class", {}).items():
            if species in set(class_species):
                return ESD_CLASS_VALUES.get(class_name, -1)
        return -1
    if species in set(config.get("positive_species", [])):
        return 1
    if species in set(config.get("negative_species", [])):
        return 0
    return -1


def load_species_trait_labels(species: str | None, config_dir: Path) -> dict[str, int]:
    labels = {column: -1 for column in LABEL_COLUMNS}
    for filename in SPECIES_CONFIG_FILENAMES:
        path = config_dir / filename
        if not path.exists():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        label_property = norm_text(config.get("label_property"))
        if label_property in labels:
            labels[label_property] = species_label_from_config(species, config)
    return labels


def source_crs_from_geojson(geojson: dict[str, Any]) -> CRS:
    crs_name = get_nested(geojson, "crs", "properties", "name")
    if crs_name:
        try:
            return CRS.from_user_input(crs_name)
        except Exception:
            pass
    return CRS.from_epsg(4326)


def flatten_extra_master(path: Path, config_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Flatten a raw master-format area file into the package's WGS84 schema."""
    geojson = json.loads(path.read_text(encoding="utf-8"))
    src_crs = source_crs_from_geojson(geojson)
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    flattened = []
    skipped_labels: dict[str, int] = {}
    skipped_null_geom = 0
    area_from_name = path.stem.removesuffix("_master").upper()

    for idx, feature in enumerate(geojson.get("features", [])):
        props = feature.get("properties") or {}
        geom_data = feature.get("geometry")
        raw_species = norm_text(props.get("species") or get_nested(props, "field_data", "species")) or None
        species_clean = raw_species
        species_status = "clean" if species_clean else "missing"
        orig_crown_id = norm_text(props.get("crown_id") or get_nested(props, "ids", "crown_id")) or f"feature_{idx:05d}"
        area = norm_text(get_nested(props, "ids", "dataset_id")) or area_from_name
        crown_uid = f"{area}:{orig_crown_id}"
        tree_type = get_nested(props, "field_data", "tree_type")

        trait_labels = load_species_trait_labels(species_clean, config_path.parent)
        species_label = trait_labels["label_acacia"]
        skipped_labels[crown_uid] = species_label

        if geom_data is None:
            skipped_null_geom += 1
            continue

        geom = shapely_transform(transformer.transform, shape(geom_data))
        centroid = geom.centroid
        flat_props = {
            "crown_uid": crown_uid,
            "area": area,
            "source_file": path.name,
            "source_index": idx,
            "orig_crown_id": orig_crown_id,
            "crown_num": props.get("crown_num"),
            "species_raw": raw_species,
            "species_clean": species_clean,
            "species_status": species_status,
            "tree_type_raw": tree_type,
            "health_class": get_nested(props, "field_data", "health_class"),
            "field_status": get_nested(props, "field_data", "status"),
            "field_description": get_nested(props, "field_data", "description"),
            "lon": centroid.x,
            "lat": centroid.y,
            **trait_labels,
        }
        flattened.append({"type": "Feature", "geometry": mapping(geom), "properties": flat_props})

    if skipped_null_geom:
        print(f"[extra-master] {path.name}: skipped {skipped_null_geom} features with null geometry")
    print(f"[extra-master] {path.name}: added {len(flattened)} features")
    return flattened, skipped_labels


def append_extra_master_files(
    master_geojson: dict[str, Any],
    extra_paths: list[Path],
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    features = list(master_geojson.get("features", []))
    existing_uids = {norm_text((f.get("properties") or {}).get("crown_uid")) for f in features}
    extra_species_labels: dict[str, int] = {}

    for path in extra_paths:
        if not path.exists():
            continue
        flattened, labels = flatten_extra_master(path, config_path)
        for feature in flattened:
            uid = norm_text((feature.get("properties") or {}).get("crown_uid"))
            if uid in existing_uids:
                raise ValueError(f"Duplicate crown_uid while adding {path}: {uid}")
            existing_uids.add(uid)
            features.append(feature)
        extra_species_labels.update({uid: label for uid, label in labels.items() if uid in existing_uids})

    out = dict(master_geojson)
    out["features"] = features
    return out, extra_species_labels


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
    ap.add_argument(
        "--extra-master",
        action="append",
        default=[str(DEFAULT_EXTRA_MASTER)],
        help="Additional raw master-format GeoJSON area to flatten and append. Can be repeated.",
    )
    ap.add_argument("--clean-out", default=str(DATA_DIR / "acacia_clean_confident_labels.geojson"))
    ap.add_argument("--cluster-out", default=str(DATA_DIR / "acacia_clean_plus_clustering_labels.geojson"))
    args = ap.parse_args()

    config_path = Path(args.config)
    extra_paths = [Path(path) for path in args.extra_master or []]
    master_geojson = json.loads(Path(args.master).read_text(encoding="utf-8"))
    master_geojson, extra_species_labels = append_extra_master_files(master_geojson, extra_paths, config_path)
    master_rows = pd.DataFrame([f.get("properties") or {} for f in master_geojson.get("features", [])])

    source_labels = {
        "species": load_species_labels(master_rows, config_path),
        "field": load_field_labels(Path(args.field_dir)),
        "desk": load_desk_labels(Path(args.desk_csv), args.desk_id_offset),
        "clustering": load_cluster_labels(Path(args.cluster_geojson)),
    }
    source_labels["species"].update(extra_species_labels)

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
