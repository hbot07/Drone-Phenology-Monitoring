#!/usr/bin/env python3
"""
Generate configs/baseline/*.json from the trait CSV.

Reads:
  raw_inputs/Tree Species Phenology Mapping - LLM FIlled (1).csv

Writes (overwrites):
  configs/baseline/acacia_vs_non_acacia.json
  configs/baseline/deciduous_vs_rest.json
  configs/baseline/esd_multiclass.json
  configs/baseline/yellow_showy_strict.json
  configs/baseline/yellow_broad.json
  configs/baseline/red_showy.json
  configs/baseline/showy_flower_vs_rest.json

Run before 01b_relabel.py whenever the trait CSV is updated:
  python python/00_generate_configs_from_traits.py [--dry-run] [--out-dir configs/baseline]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIT_CSV = ROOT / "raw_inputs" / "Tree Species Phenology Mapping - LLM FIlled (1).csv"
DEFAULT_OUT_DIR = ROOT / "configs" / "baseline"

# ── Leaf-shed classification rules ──────────────────────────────────────────
# Map free-text values in "Leaf-shed extent" to ESD class.
# Anything not matching any pattern → unknown (-1 / excluded).
EVERGREEN_PATTERNS = [
    "evergreen",
    "persistent foliage",
    "herbaceous",   # Banana: technically evergreen perennial
]
SEMI_EVERGREEN_PATTERNS = [
    "semi-evergreen",
    "partly leaf",
    "partial seasonal",
    "partly leafless",
    "partially leafless",
    "partial leaf",
    "mostly evergreen",
    "observed leaf shed",   # Mandphali note says "Leafless phase observed" → treat as semi
]
DECIDUOUS_PATTERNS = [
    "deciduous",
    "leafless",
    "leaf shed",
]

# Patches for species whose leaf-shed text is ambiguous or missing but we know
# the phenology. These override the text-based rules.
ESD_OVERRIDES: dict[str, int] = {
    "Mahneem": 0,              # Evergreen (Melia dubia)
    "Buddha's Coconut": 0,     # Evergreen palm relative
    "Chamrod": -1,             # Unknown, keep excluded
    "Anjan": 2,                # Deciduous (Hardwickia binata)
    "Cassia": 2,               # Deciduous flowering cassia
    "Semal fig": -1,           # Rare/unclear, keep excluded
    "Kanju": 2,                # Pterocarpus marsupium — deciduous
    "Marodphali": 1,           # Semi-evergreen patches observed
}

# ── Showy-flower classification rules ───────────────────────────────────────
SHOWY_KEYWORDS = {"showy", "visible", "showy / visible"}


def classify_esd(leaf_shed: str) -> int:
    """Return 0/1/2/-1 from free-text leaf-shed column."""
    if not leaf_shed or pd.isna(leaf_shed):
        return -1
    s = str(leaf_shed).lower()
    if any(p in s for p in DECIDUOUS_PATTERNS):
        return 2
    if any(p in s for p in SEMI_EVERGREEN_PATTERNS):
        return 1
    if any(p in s for p in EVERGREEN_PATTERNS):
        return 0
    return -1


def is_showy(visibility: str) -> bool:
    if not visibility or pd.isna(visibility):
        return False
    s = str(visibility).strip().lower()
    return any(kw in s for kw in SHOWY_KEYWORDS)


def build_configs(df: pd.DataFrame) -> dict[str, dict]:
    """
    Return a dict: config_filename_stem -> config_dict
    """
    configs: dict[str, dict] = {}

    # ── label_acacia ────────────────────────────────────────────────────────
    pos_acacia, neg_acacia = [], []
    for _, row in df.iterrows():
        sp = row["Species"]
        cat = str(row.get("Acacia category", "")).strip()
        if cat == "Acacia":
            pos_acacia.append(sp)
        elif cat == "Non-acacia":
            neg_acacia.append(sp)
        # "Unknown" → excluded (-1)
    configs["acacia_vs_non_acacia"] = {
        "name": "acacia_vs_non_acacia_v1",
        "description": (
            "Binary classifier: Acacia-type crowns (1) vs clean non-acacia crowns (0). "
            "Species not in either list get label -1 (excluded). "
            "Sanjay Van field tree_type is used as a secondary positive signal "
            "when species is unavailable."
        ),
        "label_property": "label_acacia",
        "classes": {"1": "acacia", "0": "non_acacia", "-1": "ignore"},
        "positive_species": sorted(pos_acacia),
        "negative_species": sorted(neg_acacia),
        "fallback_field": "tree_type_raw",
        "fallback_map": {"acacia": 1, "non-acacia": 0, "non acacia": 0},
        "recommended_first_run": True,
        "notes": (
            "Positives are concentrated in Sanjay Van areas; validate with "
            "leave-one-area-out. fallback_field is used only when species_clean is null."
        ),
    }

    # ── label_esd ───────────────────────────────────────────────────────────
    evergreen, semi, deciduous = [], [], []
    for _, row in df.iterrows():
        sp = row["Species"]
        if sp in ESD_OVERRIDES:
            cls = ESD_OVERRIDES[sp]
        else:
            cls = classify_esd(row.get("Leaf-shed extent", ""))
        if cls == 0:
            evergreen.append(sp)
        elif cls == 1:
            semi.append(sp)
        elif cls == 2:
            deciduous.append(sp)
        # -1 → excluded

    configs["esd_multiclass"] = {
        "name": "esd_multiclass_v1",
        "description": (
            "Three-class phenology classifier: Evergreen (0) / Semi-evergreen (1) / "
            "Deciduous (2). Species not listed get label -1 (excluded)."
        ),
        "label_property": "label_esd",
        "classes": {
            "0": "evergreen", "1": "semi_evergreen", "2": "deciduous", "-1": "ignore"
        },
        "positive_species_by_class": {
            "evergreen": sorted(evergreen),
            "semi_evergreen": sorted(semi),
            "deciduous": sorted(deciduous),
        },
        "recommended_first_run": True,
        "validation": ["random_split", "leave_one_area_out", "leave_one_species_out"],
        "notes": "Primary spatial phenology classifier. Report balanced accuracy, not overall accuracy.",
    }

    # ── label_deciduous ─────────────────────────────────────────────────────
    configs["deciduous_vs_rest"] = {
        "name": "deciduous_vs_rest_v1",
        "description": (
            "Binary classifier: deciduous species (1) vs evergreen/semi-evergreen species (0). "
            "Species not in either list get label -1 (excluded)."
        ),
        "label_property": "label_deciduous",
        "classes": {"1": "deciduous", "0": "not_deciduous", "-1": "ignore"},
        "positive_species": sorted(deciduous),
        "negative_species": sorted(evergreen + semi),
        "recommended_first_run": True,
        "notes": "More stable than the 3-class ESD classifier. Good for first model sanity check.",
    }

    # ── flower label tasks ───────────────────────────────────────────────────
    yellow_strict, yellow_broad_extra = [], []
    red_showy_sp = []
    white_pink_showy_sp = []
    all_known = [row["Species"] for _, row in df.iterrows()]

    for _, row in df.iterrows():
        sp = row["Species"]
        colour = str(row.get("Flower colour label", "")).strip().lower()
        visibility = str(row.get("Flower visibility / usefulness", "")).strip().lower()
        showy = is_showy(visibility)

        # Yellow strict: must be yellow AND showy
        if "yellow" in colour and showy:
            yellow_strict.append(sp)
        # Yellow broad: yellow AND at least somewhat visible (not explicitly "not useful")
        elif "yellow" in colour and "not useful" not in visibility and visibility != "nan":
            yellow_broad_extra.append(sp)

        # Red/orange/maroon showy
        if any(c in colour for c in ["red", "orange", "maroon"]) and showy:
            red_showy_sp.append(sp)

        # White/pink showy
        if any(c in colour for c in ["white", "pink"]) and showy:
            white_pink_showy_sp.append(sp)

    yellow_broad = sorted(set(yellow_strict + yellow_broad_extra))
    yellow_strict = sorted(set(yellow_strict))
    showy_flower = sorted(set(yellow_strict + yellow_broad_extra + red_showy_sp + white_pink_showy_sp))

    # Negatives for flower tasks = all known species NOT in positive
    def neg_for(pos_list):
        pos_set = set(pos_list)
        return sorted(sp for sp in all_known if sp not in pos_set)

    configs["yellow_showy_strict"] = {
        "name": "yellow_showy_strict_v1",
        "description": (
            "Binary flower-colour classifier: strictly yellow and showy flowering species (1) "
            "vs everything else with a known identity (0). "
            "Species not in either list get label -1 (excluded)."
        ),
        "label_property": "label_yellow_strict",
        "classes": {"1": "yellow_showy", "0": "not_yellow_showy", "-1": "ignore"},
        "positive_species": yellow_strict,
        "negative_species": neg_for(yellow_strict),
        "recommended_first_run": True,
        "notes": "Best flower-colour starter. Prosopis excluded — its yellow flowers are small.",
    }

    configs["yellow_broad"] = {
        "name": "yellow_broad_v1",
        "description": (
            "Exploratory yellow-flower classifier: includes moderately visible yellow-flowering "
            "species in addition to strictly showy ones. "
            "Species not in either list get label -1 (excluded)."
        ),
        "label_property": "label_yellow_broad",
        "classes": {"1": "yellow", "0": "not_yellow", "-1": "ignore"},
        "positive_species": yellow_broad,
        "negative_species": neg_for(yellow_broad),
        "recommended_first_run": False,
        "notes": "Broad sensitivity experiment. May learn Acacia/Prosopis structure instead of flower colour.",
    }

    configs["red_showy"] = {
        "name": "red_showy_v1",
        "description": (
            "Binary red/orange/maroon showy-flower classifier. "
            "Species not in either list get label -1 (excluded)."
        ),
        "label_property": "label_red_showy",
        "classes": {"1": "red_showy", "0": "not_red_showy", "-1": "ignore"},
        "positive_species": sorted(set(red_showy_sp)),
        "negative_species": neg_for(sorted(set(red_showy_sp))),
        "recommended_first_run": False,
        "notes": "Currently underpowered (~12 positives). Run but treat as exploratory.",
    }

    configs["showy_flower_vs_rest"] = {
        "name": "showy_flower_vs_rest_v1",
        "description": (
            "Binary classifier: any visibly showy-flowering species (yellow/red/white/pink) (1) "
            "vs non-showy identified species (0). "
            "Species not in either list get label -1 (excluded)."
        ),
        "label_property": "label_showy_flower",
        "classes": {"1": "showy_flower", "0": "not_showy_flower", "-1": "ignore"},
        "positive_species": showy_flower,
        "negative_species": neg_for(showy_flower),
        "recommended_first_run": True,
        "notes": "More stable than exact red/yellow/white multiclass.",
    }

    return configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print derived configs without writing files.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="Output configs subfolder (default: configs/baseline).")
    ap.add_argument("--trait-csv", default=str(TRAIT_CSV),
                    help="Path to the trait CSV.")
    args = ap.parse_args()

    trait_csv = Path(args.trait_csv)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    df = pd.read_csv(trait_csv)
    print(f"Loaded {len(df)} species from {trait_csv.name}")

    configs = build_configs(df)

    for stem, cfg in configs.items():
        out_path = out_dir / f"{stem}.json"
        label_prop = cfg["label_property"]
        pos_count = (
            sum(len(v) for v in cfg["positive_species_by_class"].values())
            if "positive_species_by_class" in cfg
            else len(cfg.get("positive_species", []))
        )
        print(f"  {label_prop:<25} → {out_path.name}  ({pos_count} positives)")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print(f"    Written: {out_path}")

    if args.dry_run:
        print("\n(dry-run: no files written)")
        print("\nSample — showy_flower_vs_rest positives:")
        print(" ", configs["showy_flower_vs_rest"]["positive_species"])
    else:
        print(f"\nAll configs written to {out_dir}")
        print("Next step: python python/01b_relabel.py --config-dir", args.out_dir)


if __name__ == "__main__":
    main()
