#!/usr/bin/env python3
"""
Prepare one flat, GEE-ready crown GeoJSON from IITD + Sanjay Van master-format GeoJSONs.

Outputs:
  data/iitd_sv_crowns_master_wgs84.geojson
  data/crown_label_table.csv
  outputs/species_counts_by_area.csv
  outputs/classifier_label_summary.csv

Run from repository root:
  python python/01_prepare_crowns.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "raw_inputs"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# If you run this outside the prepared package, update these paths.
DEFAULT_INPUT_FILES = [
    "A1_master.geojson",
    "A2_master.geojson",
    "A3_master.geojson",
    "A4_master.geojson",
    "A5_master.geojson",
    "mittal_master.geojson",
    "sit_master(1).geojson",
    "s1_master_format.geojson",
    "s2_master_format.geojson",
    "s3_master_format.geojson",
    "s4_master_format.geojson",
]
TRAIT_CSV_NAME = "Tree Species Phenology Mapping - LLM FIlled (1).csv"

# Canonical species aliases. Ambiguous/mixed field labels are intentionally not forced.
ALIASES = {
    "franjipani": "Frangipani",
    "caribbean trumpet": "Caribbean trumpet",
    "caribbean trumpet tree": "Caribbean trumpet",
    "bamboo": "Bamboo clump",
    "bananas": "Banana",
    "banana": "Banana",
    "mandphali": "Marodphali",
    "marodphali": "Marodphali",
    "mandphali / marodphali": "Marodphali",
    "shisham": "Sheesham",
    "sheesham": "Sheesham",
    "sheesham / shisham": "Sheesham",
    "silk cotton": "Semal",
    "semal": "Semal",
    "semal / silk cotton": "Semal",
    "maha neem": "Mahneem",
    "mahneem": "Mahneem",
    "maha neem / mahneem": "Mahneem",
    "bakain tree": "Bakain",
    "bamboo clump": "Bamboo clump",
    "prosopis juliflora": "Prosopis Juliflora",
    "native acacia": "Native acacia",
    "subabool": "Subabool",
    "palm tree": "Palm tree",
    "buddha's coconut": "Buddha's Coconut",
}

# ── Label configuration ─────────────────────────────────────────────────────
# Species-to-label mappings live in configs/<experiment_subfolder>/.
# Each JSON file in that folder defines one label task via label_property,
# positive_species, negative_species (and optionally positive_species_by_class
# for multiclass tasks).
#
# To run a different experiment: create a new subfolder under configs/ (e.g.
# configs/no_prosopis/) with modified JSONs, then pass --config-dir to this
# script or to python/01b_relabel.py.

DEFAULT_CONFIGS_DIR = ROOT / "configs" / "baseline"


def load_label_configs(configs_dir: Path = DEFAULT_CONFIGS_DIR) -> dict:
    """
    Load all *.json files from configs_dir.
    Returns dict keyed by label_property value.
    """
    configs: dict = {}
    for cfg_path in sorted(configs_dir.glob("*.json")):
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        label_prop = cfg.get("label_property")
        if label_prop:
            configs[label_prop] = cfg
    return configs


def apply_label_from_config(
    species_clean: Optional[str],
    label_cfg: dict,
    extra_props: Optional[dict] = None,
) -> int:
    """
    Apply one label config to one crown.

    Binary tasks (positive_species / negative_species):
      - species in positive_species  → 1
      - species in negative_species  → 0
      - species in neither           → -1  (excluded)
      - species_clean is None        → try fallback_field/fallback_map, else -1

    Multiclass tasks (positive_species_by_class):
      - class name → int via classes dict (e.g. "evergreen" → 0)
      - species found in exactly one class list → that class int
      - species not found in any class          → -1
    """
    # ── multiclass ──────────────────────────────────────────────────────────
    pbc = label_cfg.get("positive_species_by_class")
    if pbc is not None:
        classes = label_cfg.get("classes", {})
        cls_name_to_int = {
            v: int(k) for k, v in classes.items()
            if k.lstrip("-").isdigit() and v != "ignore"
        }
        if species_clean is not None:
            for cls_name, members in pbc.items():
                if species_clean in members:
                    return cls_name_to_int.get(cls_name, -1)
        return -1

    # ── binary ───────────────────────────────────────────────────────────────
    positive = set(label_cfg.get("positive_species", []))
    negative = set(label_cfg.get("negative_species", []))

    if species_clean is not None:
        if species_clean in positive:
            return 1
        if species_clean in negative:
            return 0
        return -1

    # species_clean is None — try fallback field (e.g. tree_type_raw for acacia)
    if extra_props and "fallback_field" in label_cfg:
        ff_val = str(extra_props.get(label_cfg["fallback_field"], "")).strip().lower()
        fmap = {k.lower(): int(v) for k, v in label_cfg.get("fallback_map", {}).items()}
        if ff_val in fmap:
            return fmap[ff_val]

    return -1


_LABEL_CONFIGS: Optional[dict] = None


def _get_label_configs() -> dict:
    global _LABEL_CONFIGS
    if _LABEL_CONFIGS is None:
        _LABEL_CONFIGS = load_label_configs()
    return _LABEL_CONFIGS

AMBIGUOUS_PATTERNS = [
    r",", r"/", r"\bor\b", r"\band\b", r"unknown", r"others", r"other", r"like",
    r"next to", r"oblong", r"fig$",
]
# Known names containing slash or apostrophe that are not ambiguous once canonicalized.
NON_AMBIGUOUS_CANONICALS = {
    "Marodphali", "Sheesham", "Semal", "Mahneem", "Buddha's Coconut"
}


def infer_area(filename: str) -> str:
    lower = filename.lower()
    if lower.startswith("s1_"):
        return "SV_S1"
    if lower.startswith("s2_"):
        return "SV_S2"
    if lower.startswith("s3_"):
        return "SV_S3"
    if lower.startswith("s4_"):
        return "SV_S4"
    if lower.startswith("sit"):
        return "SIT"
    if lower.startswith("mittal"):
        return "MITTAL"
    m = re.match(r"(A\d+)_", filename, re.I)
    if m:
        return m.group(1).upper()
    return Path(filename).stem


def get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def clean_species(raw: Any, tree_type: Any = None) -> Tuple[Optional[str], str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, "missing"
    s = str(raw).strip()
    if not s:
        return None, "missing"
    s_norm = re.sub(r"\s+", " ", s).strip()
    key = s_norm.lower()

    # Special case: QField sometimes says only Acacia. Keep as acacia positive, but do not pretend species is known.
    if key == "acacia":
        return "Acacia unknown", "acacia_unknown"

    if key in ALIASES:
        return ALIASES[key], "clean"

    # Exact title-case-ish match to trait sheet species names is allowed.
    # Leave clean single-token names as they are.
    candidate = s_norm
    if candidate in NON_AMBIGUOUS_CANONICALS:
        return candidate, "clean"

    low = candidate.lower()
    if any(re.search(p, low) for p in AMBIGUOUS_PATTERNS):
        return None, "ambiguous_or_unknown"

    return candidate, "clean"


# Individual label helpers kept for backward-compat imports; all delegate to config.

def label_esd(species_clean: Optional[str]) -> int:
    return apply_label_from_config(species_clean, _get_label_configs()["label_esd"])


def label_deciduous_binary(species_clean: Optional[str]) -> int:
    return apply_label_from_config(species_clean, _get_label_configs()["label_deciduous"])


def label_acacia(species_clean: Optional[str], tree_type: Any) -> int:
    return apply_label_from_config(
        species_clean,
        _get_label_configs()["label_acacia"],
        extra_props={"tree_type_raw": tree_type},
    )


def label_yellow_strict(species_clean: Optional[str]) -> int:
    return apply_label_from_config(species_clean, _get_label_configs()["label_yellow_strict"])


def label_yellow_broad(species_clean: Optional[str]) -> int:
    return apply_label_from_config(species_clean, _get_label_configs()["label_yellow_broad"])


def label_red_showy(species_clean: Optional[str]) -> int:
    return apply_label_from_config(species_clean, _get_label_configs()["label_red_showy"])


def label_showy_flower(species_clean: Optional[str]) -> int:
    return apply_label_from_config(species_clean, _get_label_configs()["label_showy_flower"])


def flatten_feature(feat: Dict[str, Any], src_file: str, src_index: int, transformer: Transformer) -> Dict[str, Any]:
    props = feat.get("properties", {}) or {}
    area = infer_area(src_file)
    orig_crown_id = props.get("crown_id") or get_nested(props, "ids", "crown_id") or f"feature_{src_index:05d}"
    tree_type = get_nested(props, "field_data", "tree_type")
    raw_species = props.get("species") or get_nested(props, "field_data", "species")
    species_clean, status = clean_species(raw_species, tree_type)

    geom = shape(feat["geometry"])
    geom4326 = shapely_transform(transformer.transform, geom)
    centroid = geom4326.centroid

    out_props = {
        "crown_uid": f"{area}:{orig_crown_id}",
        "area": area,
        "source_file": src_file,
        "source_index": src_index,
        "orig_crown_id": orig_crown_id,
        "crown_num": props.get("crown_num"),
        "species_raw": raw_species,
        "species_clean": species_clean,
        "species_status": status,
        "tree_type_raw": tree_type,
        "health_class": get_nested(props, "field_data", "health_class"),
        "field_status": get_nested(props, "field_data", "status"),
        "field_description": get_nested(props, "field_data", "description"),
        "lon": centroid.x,
        "lat": centroid.y,
        "label_esd": label_esd(species_clean),
        "label_deciduous": label_deciduous_binary(species_clean),
        "label_acacia": label_acacia(species_clean, tree_type),
        "label_yellow_strict": label_yellow_strict(species_clean),
        "label_yellow_broad": label_yellow_broad(species_clean),
        "label_red_showy": label_red_showy(species_clean),
        "label_showy_flower": label_showy_flower(species_clean),
        # ↑ To add a new label task: add a JSON file to configs/baseline/ with
        #   the appropriate label_property, positive_species, negative_species
        #   and add a line here: "label_xxx": apply_label_from_config(species_clean, _get_label_configs()["label_xxx"])
    }
    return {"type": "Feature", "geometry": mapping(geom4326), "properties": out_props}


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Prefer package raw_inputs, otherwise fall back to /mnt/data for this ChatGPT run.
    search_roots = [INPUT_ROOT, Path("/mnt/data")]
    input_paths = []
    for name in DEFAULT_INPUT_FILES:
        for root in search_roots:
            p = root / name
            if p.exists():
                input_paths.append(p)
                break
        else:
            raise FileNotFoundError(f"Could not find {name} in {search_roots}")

    transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    all_features = []

    for path in input_paths:
        with path.open("r", encoding="utf-8") as f:
            gj = json.load(f)
        for i, feat in enumerate(gj.get("features", [])):
            all_features.append(flatten_feature(feat, path.name, i, transformer))

    out_geojson = {"type": "FeatureCollection", "features": all_features}
    out_geojson_path = DATA_DIR / "iitd_sv_crowns_master_wgs84.geojson"
    with out_geojson_path.open("w", encoding="utf-8") as f:
        json.dump(out_geojson, f, ensure_ascii=False, separators=(",", ":"))

    rows = [f["properties"] for f in all_features]
    table = pd.DataFrame(rows)
    table.to_csv(DATA_DIR / "crown_label_table.csv", index=False)

    species_counts = (
        table.assign(species_for_count=table["species_clean"].fillna("<unlabelled_or_ambiguous>"))
        .groupby(["area", "species_for_count"])
        .size()
        .reset_index(name="count")
        .sort_values(["area", "count"], ascending=[True, False])
    )
    species_counts.to_csv(OUTPUT_DIR / "species_counts_by_area.csv", index=False)

    summaries = []
    label_cols = [
        "label_esd", "label_deciduous", "label_acacia", "label_yellow_strict",
        "label_yellow_broad", "label_red_showy", "label_showy_flower",
    ]
    for col in label_cols:
        vc = table[col].value_counts(dropna=False).sort_index()
        for label, count in vc.items():
            summaries.append({"classifier_label": col, "label_value": label, "count": int(count)})
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "classifier_label_summary.csv", index=False)

    print(f"Wrote {out_geojson_path} with {len(all_features)} crowns")
    print(table[label_cols].apply(lambda s: s.value_counts().sort_index()).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
