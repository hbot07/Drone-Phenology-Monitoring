#!/usr/bin/env python3
"""Combine one-year Sentinel-2 feature exports into one multi-year table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "exports"

PROP_COLS = {
    "crown_uid",
    "area",
    "source_file",
    "source_index",
    "orig_crown_id",
    "crown_num",
    "species_raw",
    "species_clean",
    "species_status",
    "tree_type_raw",
    "health_class",
    "field_status",
    "field_description",
    "lon",
    "lat",
    "label_esd",
    "label_deciduous",
    "label_acacia",
    "label_yellow_strict",
    "label_yellow_broad",
    "label_red_showy",
    "label_showy_flower",
    "geometry_mode",
}


def load_year(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {c: f"y{year}_{c}" for c in df.columns if c not in PROP_COLS}
    return df.rename(columns=rename)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--year-csv",
        action="append",
        required=True,
        help="YEAR=CSV, for example 2025=exports/stac_s2_features_2025_buffer20_items4_label_acacia.csv",
    )
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    year_tables = []
    for item in args.year_csv:
        year_str, path_str = item.split("=", 1)
        year_tables.append((int(year_str), Path(path_str)))
    year_tables = sorted(year_tables)

    merged = None
    for year, path in year_tables:
        df = load_year(path, year)
        if merged is None:
            merged = df
        else:
            feature_cols = [c for c in df.columns if c not in PROP_COLS and c != "crown_uid"]
            merged = merged.merge(df[["crown_uid", *feature_cols]], on="crown_uid", how="inner")

    if merged is None:
        raise SystemExit("No inputs")
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote {out} with {len(merged)} rows and {len(merged.columns)} columns")


if __name__ == "__main__":
    main()
