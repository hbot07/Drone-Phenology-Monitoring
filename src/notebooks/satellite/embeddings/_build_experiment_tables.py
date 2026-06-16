#!/usr/bin/env python3
"""Build merged feature tables (keyed by crown_uid) for the round-2 experiments.

Label/metadata columns come from the centroid label-config CSV (has all labels +
area + species). Feature columns are merged in from each source. Fusions are
inner-joined so there are no all-NaN feature rows.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "exports"
PKG = ROOT.parent / "drone_phenology_rf_package"

LABELS = ["label_acacia","label_deciduous","label_esd","label_showy_flower",
          "label_yellow_strict","label_yellow_broad","label_red_showy",
          "label_acacia_visual","label_acacia_clustering","label_acacia_species",
          "label_acacia_visual_or_species","label_acacia_visual_or_clustering",
          "label_acacia_species_or_clustering","label_acacia_all_priority"]
META = ["crown_uid","area","species_clean","species_raw","species_status",
        "tree_type_raw","orig_crown_id","source_file","source_index","lon","lat"]


def feature_cols(df: pd.DataFrame) -> list[str]:
    skip = set(LABELS) | set(META) | {"system:index",".geo","geo","geometry",
        "crown_num","health_class","field_status","field_description","classification",
        "random","stac_year","stac_item_count","buffer_meters","s1_item_count"}
    return [c for c in df.columns if c not in skip and not c.startswith("label_")
            and pd.api.types.is_numeric_dtype(df[c])]


def main() -> None:
    base = pd.read_csv(EXP / "gee_centroid_2024_acacia_label_configs.csv")
    keep_meta = [c for c in META if c in base.columns]
    label_meta = base[keep_meta + LABELS].copy()

    # feature-only frames (crown_uid + features, prefixed to avoid collisions)
    def feats(path: str, prefix: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        fc = feature_cols(df)
        out = df[["crown_uid"] + fc].copy()
        out = out.rename(columns={c: f"{prefix}{c}" for c in fc})
        return out

    gee   = feats(str(EXP / "gee_centroid_2024_features.csv"), "gee_")
    dino  = feats(str(EXP / "dino_default_labels_2025_mar.csv"), "")          # already dino_*
    s2t   = feats(str(PKG / "exports/stac_s2_features_2022_2025_buffer10_items4_temporal_label_acacia.csv"), "s2t_")

    print(f"feature counts: gee={gee.shape[1]-1} dino={dino.shape[1]-1} s2t={s2t.shape[1]-1}")

    tables = {
        "exp_s2temporal":     [s2t],
        "exp_gee_dino":       [gee, dino],
        "exp_gee_s2temporal": [gee, s2t],
        "exp_dino_s2temporal":[dino, s2t],
        "exp_all3":           [gee, dino, s2t],
    }
    for name, parts in tables.items():
        m = label_meta.copy()
        for p in parts:
            m = m.merge(p, on="crown_uid", how="inner")
        # ensure no all-NaN feature columns from the join
        out = EXP / f"{name}.csv"
        m.to_csv(out, index=False)
        nfeat = sum(p.shape[1]-1 for p in parts)
        print(f"{name:22s} rows={len(m):5d} feat={nfeat:5d} -> {out.name}")


if __name__ == "__main__":
    main()
