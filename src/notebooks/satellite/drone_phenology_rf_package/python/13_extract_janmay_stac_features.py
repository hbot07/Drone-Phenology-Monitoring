#!/usr/bin/env python3
"""
Extract crown-level Sentinel-2 features with fine-grained Jan–May season windows.

Why:  The standard 4-season split (winter/premonsoon/monsoon/postmonsoon) misses the
      phenologically critical Jan–May window when:
        - Deciduous species drop leaves fully (Jan–Feb)
        - Amaltas, Cassia, Flame-of-the-Forest, Siris flower (Mar–Apr)
        - Acacia Prosopis flushes new foliage (Mar–May)
        - Semi-evergreen crowns show partial flush vs full flush differences

This script adds MONTHLY sub-windows for Jan–May ON TOP of the 4 standard seasons,
giving the classifier 10 windows per year instead of 4.

  Standard seasons (kept for multi-year amplitude features):
    winter       : Dec–Feb  (from prior year Dec)
    premonsoon   : Mar–May
    monsoon      : Jun–Sep
    postmonsoon  : Oct–Nov

  Fine-grained Jan–May additions:
    jan          : Jan 01 – Jan 31
    feb          : Feb 01 – Feb 28/29
    mar          : Mar 01 – Mar 31
    apr          : Apr 01 – Apr 30
    may          : May 01 – May 31

Usage:
  python python/13_extract_janmay_stac_features.py \
    --year 2025 \
    --years 2022 2023 2024 2025 \
    --geometry-mode buffer \
    --buffer-meters 10 \
    --max-items-per-season 4 \
    --out-csv exports/stac_s2_janmay_2022_2025_buffer10.csv

For a single year:
  python python/13_extract_janmay_stac_features.py \
    --year 2025 \
    --out-csv exports/stac_s2_janmay_2025_buffer10.csv
"""
from __future__ import annotations

import argparse
import calendar
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

# Reuse all internals from the standard extractor — no code duplication
_BASE_PATH = Path(__file__).with_name("03_extract_sentinel2_stac_features.py")
_s2 = SourceFileLoader("s2extract", str(_BASE_PATH)).load_module()

Season = _s2.Season


def extended_seasons_for_year(year: int) -> list[Season]:
    """4 standard seasons + 5 monthly Jan–May windows."""
    standard = _s2.seasons_for_year(year)
    feb_end_day = 29 if calendar.isleap(year) else 28
    fine = [
        Season("jan",  f"{year}-01-01", f"{year}-02-01"),
        Season("feb",  f"{year}-02-01", f"{year}-03-01"),
        Season("mar",  f"{year}-03-01", f"{year}-04-01"),
        Season("apr",  f"{year}-04-01", f"{year}-05-01"),
        Season("may",  f"{year}-05-01", f"{year}-06-01"),
    ]
    return standard + fine


def extract_one_year(
    crowns,
    catalog,
    year: int,
    geometry_mode: str,
    buffer_meters: float,
    max_cloud: float,
    max_items: int,
    pad_pixels: int,
    min_pixels: int,
    all_touched: bool,
    prefix: bool = False,
) -> "pd.DataFrame":
    """Run the full item-stack pipeline for one year and return a features DataFrame."""
    import numpy as np
    import pandas as pd

    bbox = list(map(float, crowns.total_bounds))
    fallback_lat = float((bbox[1] + bbox[3]) / 2.0)

    seasons = extended_seasons_for_year(year)
    epsg_votes = []
    for s in seasons:
        items = _s2.query_items(catalog, bbox, s, max_cloud, max_items)
        for it in items:
            epsg_votes.append(_s2.sentinel_epsg_from_item(it, fallback_lat))
    import statistics
    epsg = statistics.mode(epsg_votes) if epsg_votes else 32643
    dst_crs = f"EPSG:{epsg}"
    bounds = _s2.transform_bbox_wgs84(bbox, dst_crs)
    geoms = _s2.extraction_geometries(crowns, dst_crs, geometry_mode, buffer_meters)

    prop_cols = [c for c in crowns.columns if c != "geometry"]
    features = crowns[prop_cols].copy()

    for season in seasons:
        items = _s2.query_items(catalog, bbox, season, max_cloud, max_items)
        if not items:
            print(f"  {year}/{season.name}: no items, filling NaN", flush=True)
            for band in _s2.ALL_BANDS:
                sname = f"y{year}_{season.name}_{band}" if prefix else f"{season.name}_{band}"
                features[sname] = float("nan")
            continue

        per_item, masks, mask_sig = [], None, None
        for idx, item in enumerate(items, 1):
            print(f"  [{year}/{season.name} {idx}/{len(items)}] {item.id}", flush=True)
            import rasterio
            with rasterio.Env(GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_CONNECTTIMEOUT="20",
                              CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                arrays, valid, transform = _s2.read_item_stack(item, bounds, pad_pixels)
            sig = (transform.a, transform.c, transform.d, transform.f, valid.shape)
            if masks is None or sig != mask_sig:
                masks = _s2.build_masks(geoms, transform, valid.shape, all_touched)
                mask_sig = sig
            stats = _s2.summarize_item(arrays, valid, masks, min_pixels)
            stats["item_id"] = item.id
            per_item.append(stats)

        stacked = pd.concat(per_item, keys=range(len(per_item)), names=["item_no", "row_no"]).reset_index(level=0)
        grouped = stacked.groupby(stacked.index)
        for band in _s2.ALL_BANDS:
            sname = f"y{year}_{season.name}_{band}" if prefix else f"{season.name}_{band}"
            features[sname] = grouped[band].median().to_numpy()

    # Add standard amplitude features (only over the 4 standard seasons)
    std_seasons = ["winter", "premonsoon", "monsoon", "postmonsoon"]
    for band in ["NDVI", "GNDVI", "NDRE", "NDMI", "NBR", "EVI",
                 "green_ratio", "red_ratio", "yellow_proxy"]:
        cols_std = [f"y{year}_{s}_{band}" if prefix else f"{s}_{band}" for s in std_seasons]
        existing = [c for c in cols_std if c in features.columns]
        if existing:
            amp_col = f"y{year}_{band}_amp" if prefix else f"{band}_amp"
            features[amp_col] = features[existing].max(axis=1, skipna=True) - \
                                  features[existing].min(axis=1, skipna=True)
    return features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year",  type=int, default=None,
                    help="Single year to extract. Omit if using --years.")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="Multiple years to extract and join into one multi-year table.")
    ap.add_argument("--geometry-mode", default="buffer", choices=["polygon", "point", "buffer"])
    ap.add_argument("--buffer-meters", type=float, default=10.0)
    ap.add_argument("--max-cloud", type=float, default=70.0)
    ap.add_argument("--max-items-per-season", type=int, default=4)
    ap.add_argument("--pad-pixels", type=int, default=40)
    ap.add_argument("--min-pixels", type=int, default=1)
    ap.add_argument("--all-touched", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--crowns", default=None)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    import geopandas as gpd
    import pandas as pd
    import planetary_computer as pc
    from pystac_client import Client

    ROOT     = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"

    crowns_path = args.crowns or str(DATA_DIR / "iitd_sv_crowns_master_wgs84.geojson")
    crowns = gpd.read_file(crowns_path)
    crowns = crowns[crowns.geometry.notnull() & ~crowns.geometry.is_empty].reset_index(drop=True)

    catalog = Client.open(_s2.STAC_API)

    years = args.years or ([args.year] if args.year else None)
    if not years:
        ap.error("Provide --year or --years")

    if len(years) == 1:
        df = extract_one_year(
            crowns, catalog, years[0],
            args.geometry_mode, args.buffer_meters,
            args.max_cloud, args.max_items_per_season,
            args.pad_pixels, args.min_pixels, args.all_touched,
            prefix=False,
        )
    else:
        # Multi-year: prefix all feature columns with y{year}_
        merged = None
        for yr in sorted(years):
            df_yr = extract_one_year(
                crowns, catalog, yr,
                args.geometry_mode, args.buffer_meters,
                args.max_cloud, args.max_items_per_season,
                args.pad_pixels, args.min_pixels, args.all_touched,
                prefix=True,
            )
            if merged is None:
                merged = df_yr
            else:
                prop_cols = set(merged.columns) & set(df_yr.columns) - \
                            {c for c in df_yr.columns if c.startswith("y")}
                feat_cols_yr = [c for c in df_yr.columns if c.startswith(f"y{yr}_")]
                merged = merged.merge(df_yr[["crown_uid"] + feat_cols_yr], on="crown_uid", how="inner")
        df = merged

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
