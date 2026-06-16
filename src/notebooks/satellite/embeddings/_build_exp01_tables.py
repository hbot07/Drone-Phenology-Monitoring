#!/usr/bin/env python3
"""Re-label all feature tables with the canonical exp01 labels (from the master
geojson) and write corrected feature CSVs for re-running the sweeps."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "exports"
PKG = ROOT.parent / "drone_phenology_rf_package"

LABELS = ["label_acacia","label_deciduous","label_esd","label_showy_flower",
          "label_yellow_strict","label_yellow_broad","label_red_showy"]
META = ["crown_uid","area","species_clean"]


def master_labels() -> pd.DataFrame:
    geo = json.loads((PKG / "data/iitd_sv_crowns_master_wgs84.geojson").read_text())
    df = pd.DataFrame([f["properties"] for f in geo["features"]])
    for l in LABELS:
        df[l] = df[l].fillna(-1).astype(int)
    return df[META + LABELS].copy()


def feat_only(path: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    skip = set(LABELS) | set(META) | {"system:index",".geo","geo","geometry","crown_num",
        "health_class","field_status","field_description","classification","random","lon","lat",
        "species_raw","species_status","tree_type_raw","orig_crown_id","source_file","source_index",
        "label_acacia_visual","label_acacia_clustering","label_acacia_species",
        "label_acacia_visual_or_species","label_acacia_visual_or_clustering",
        "label_acacia_species_or_clustering","label_acacia_all_priority",
        "stac_year","stac_item_count","buffer_meters","s1_item_count"}
    fc = [c for c in df.columns if c not in skip and not c.startswith("label_")
          and pd.api.types.is_numeric_dtype(df[c])]
    out = df[["crown_uid"] + fc].copy().rename(columns={c: f"{prefix}{c}" for c in fc})
    return out


def main() -> None:
    lab = master_labels()
    print("exp01 master label counts:")
    for l in LABELS:
        print(f"  {l:22s} {lab[l].value_counts().sort_index().to_dict()}")

    gee = feat_only(str(EXP / "gee_centroid_2024_features.csv"), "gee_")
    dino = feat_only(str(EXP / "dino_default_labels_2025_mar.csv"), "")
    s2t = feat_only(str(PKG / "exports/stac_s2_features_2022_2025_buffer10_items4_temporal_label_acacia.csv"), "s2t_")
    print(f"\nfeatures: gee={gee.shape[1]-1} dino={dino.shape[1]-1} s2t={s2t.shape[1]-1}")

    tables = {
        "exp01_gee_centroid": [gee],
        "exp01_s2temporal": [s2t],
        "exp01_all3": [gee, dino, s2t],
    }
    for name, parts in tables.items():
        m = lab.copy()
        for p in parts:
            m = m.merge(p, on="crown_uid", how="inner")
        out = EXP / f"{name}.csv"
        m.to_csv(out, index=False)
        print(f"{name:20s} rows={len(m):5d} feat={sum(p.shape[1]-1 for p in parts):5d} -> {out.name}")


if __name__ == "__main__":
    main()
