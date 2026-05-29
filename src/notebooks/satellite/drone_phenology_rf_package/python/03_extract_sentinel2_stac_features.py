#!/usr/bin/env python3
"""
Extract crown-level Sentinel-2 seasonal features locally from Microsoft Planetary
Computer STAC, without Google Earth Engine.

The output CSV is intentionally shaped like the GEE export expected by
02_local_rf_from_gee_export.py: crown properties plus numeric predictor columns.

Typical use:
  python python/03_extract_sentinel2_stac_features.py \
    --year 2025 \
    --geometry-mode buffer \
    --buffer-meters 20 \
    --out-csv exports/stac_s2_features_2025_buffer20.csv
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPORT_DIR = ROOT / "exports"
META_DIR = ROOT / "outputs" / "stac_meta"

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
OPTICAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
STAC_ASSET_KEYS = {
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B5": "B05",
    "B6": "B06",
    "B7": "B07",
    "B8": "B08",
    "B8A": "B8A",
    "B11": "B11",
    "B12": "B12",
}
INDEX_BANDS = [
    "NDVI",
    "GNDVI",
    "NDRE",
    "NDMI",
    "NBR",
    "EVI",
    "blue_ratio",
    "green_ratio",
    "red_ratio",
    "yellow_proxy",
]
ALL_BANDS = OPTICAL_BANDS + INDEX_BANDS
SCL_INVALID = {0, 1, 3, 6, 8, 9, 10, 11}


@dataclass(frozen=True)
class Season:
    name: str
    start: str
    end: str


def seasons_for_year(year: int) -> list[Season]:
    return [
        Season("winter", f"{year - 1}-12-01", f"{year}-03-01"),
        Season("premonsoon", f"{year}-03-01", f"{year}-06-01"),
        Season("monsoon", f"{year}-06-01", f"{year}-10-01"),
        Season("postmonsoon", f"{year}-10-01", f"{year}-12-01"),
    ]


def sentinel_epsg_from_item(item, fallback_lat: float) -> int:
    epsg = item.properties.get("proj:epsg")
    if epsg is not None:
        return int(epsg)
    tile = item.properties.get("mgrs:tile")
    zone = None
    if isinstance(tile, str) and len(tile) >= 2 and tile[:2].isdigit():
        zone = int(tile[:2])
    if zone is None:
        match = re.search(r"_T(\d{2})[A-Z]{3}_", item.id)
        if match:
            zone = int(match.group(1))
    if zone is None:
        raise RuntimeError(f"Could not derive Sentinel UTM zone for {item.id}")
    return 32600 + zone if fallback_lat >= 0 else 32700 + zone


def signed_href(item, asset_key: str) -> str:
    asset = item.assets.get(asset_key)
    if asset is None:
        raise KeyError(f"Missing asset {asset_key!r} on {item.id}")
    return pc.sign(asset.href)


def query_items(catalog: Client, bbox: list[float], season: Season, max_cloud: float | None, limit: int | None):
    query = {"eo:cloud_cover": {"lt": float(max_cloud)}} if max_cloud is not None else None
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{season.start}/{season.end}",
        query=query,
    )
    items = list(search.items())
    items = sorted(
        items,
        key=lambda it: (
            float(it.properties.get("eo:cloud_cover") or 999.0),
            it.datetime.isoformat() if it.datetime else "",
            it.id,
        ),
    )
    if limit is not None:
        items = items[: int(limit)]
    return items


def transform_bbox_wgs84(bbox: Iterable[float], dst_crs: str) -> tuple[float, float, float, float]:
    minlon, minlat, maxlon, maxlat = bbox
    tx = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs, ys = tx.transform(
        [minlon, minlon, maxlon, maxlon],
        [minlat, maxlat, minlat, maxlat],
    )
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def padded_window(src: rasterio.io.DatasetReader, bounds: tuple[float, float, float, float], pad_pixels: int):
    win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
    col_off = max(0, int(win.col_off) - pad_pixels)
    row_off = max(0, int(win.row_off) - pad_pixels)
    right = min(src.width, int(math.ceil(win.col_off + win.width)) + pad_pixels)
    bottom = min(src.height, int(math.ceil(win.row_off + win.height)) + pad_pixels)
    return rasterio.windows.Window(col_off, row_off, max(1, right - col_off), max(1, bottom - row_off))


def read_asset_to_shape(item, asset_key: str, bounds: tuple[float, float, float, float], shape: tuple[int, int] | None, pad_pixels: int):
    stac_key = STAC_ASSET_KEYS.get(asset_key, asset_key)
    with rasterio.open(signed_href(item, stac_key)) as src:
        win = padded_window(src, bounds, pad_pixels)
        if shape is None:
            data = src.read(1, window=win, masked=True).astype("float32")
        else:
            data = src.read(
                1,
                window=win,
                out_shape=shape,
                resampling=Resampling.nearest if asset_key == "SCL" else Resampling.bilinear,
                masked=True,
            ).astype("float32")
        transform = rasterio.windows.transform(win, src.transform)
        if shape is not None:
            transform = transform * rasterio.Affine.scale(win.width / shape[1], win.height / shape[0])
    return data, transform


def read_item_stack(item, bounds: tuple[float, float, float, float], pad_pixels: int) -> tuple[dict[str, np.ndarray], np.ndarray, object]:
    red, transform = read_asset_to_shape(item, "B4", bounds, None, pad_pixels)
    shape = red.shape
    raw = {"B4": red}
    for band in ["B2", "B3", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]:
        raw[band], _ = read_asset_to_shape(item, band, bounds, shape, pad_pixels)
    scl, _ = read_asset_to_shape(item, "SCL", bounds, shape, pad_pixels)

    arrays = {band: np.asarray(raw[band], dtype="float32") / 10000.0 for band in OPTICAL_BANDS}
    valid = np.ones(shape, dtype=bool)
    for arr in raw.values():
        valid &= ~np.ma.getmaskarray(arr)
    valid &= ~np.ma.getmaskarray(scl)
    valid &= ~np.isin(np.asarray(scl, dtype="int16"), list(SCL_INVALID))

    blue = arrays["B2"]
    green = arrays["B3"]
    red_arr = arrays["B4"]
    nir = arrays["B8"]
    nir_narrow = arrays["B8A"]
    red_edge = arrays["B5"]
    swir1 = arrays["B11"]
    swir2 = arrays["B12"]
    rgb_sum = np.maximum(blue + green + red_arr, 0.0001)
    denom_evi = nir + 6.0 * red_arr - 7.5 * blue + 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        arrays["NDVI"] = (nir - red_arr) / (nir + red_arr)
        arrays["GNDVI"] = (nir - green) / (nir + green)
        arrays["NDRE"] = (nir_narrow - red_edge) / (nir_narrow + red_edge)
        arrays["NDMI"] = (nir - swir1) / (nir + swir1)
        arrays["NBR"] = (nir - swir2) / (nir + swir2)
        arrays["EVI"] = 2.5 * ((nir - red_arr) / denom_evi)
        arrays["blue_ratio"] = blue / rgb_sum
        arrays["green_ratio"] = green / rgb_sum
        arrays["red_ratio"] = red_arr / rgb_sum
        arrays["yellow_proxy"] = (green + red_arr) / rgb_sum

    return arrays, valid, transform


def extraction_geometries(crowns: gpd.GeoDataFrame, dst_crs: str, mode: str, buffer_meters: float) -> list[dict]:
    gdf = crowns.to_crs(dst_crs)
    if mode == "point":
        geoms = gdf.geometry.centroid
    elif mode == "buffer":
        geoms = gdf.geometry.centroid.buffer(float(buffer_meters))
    elif mode == "polygon":
        geoms = gdf.geometry
    else:
        raise ValueError(f"Unknown geometry mode: {mode}")
    return [geom.__geo_interface__ for geom in geoms]


def build_masks(geometries: list[dict], transform, shape: tuple[int, int], all_touched: bool) -> list[np.ndarray]:
    return [
        geometry_mask([geom], out_shape=shape, transform=transform, invert=True, all_touched=all_touched)
        for geom in geometries
    ]


def summarize_item(arrays: dict[str, np.ndarray], valid: np.ndarray, masks: list[np.ndarray], min_pixels: int) -> pd.DataFrame:
    rows = []
    for mask in masks:
        use = mask & valid
        n = int(use.sum())
        row = {"valid_pixels": n}
        for band in ALL_BANDS:
            if n >= min_pixels:
                vals = arrays[band][use]
                vals = vals[np.isfinite(vals)]
                row[band] = float(np.nanmedian(vals)) if len(vals) else np.nan
            else:
                row[band] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_amplitude_features(df: pd.DataFrame) -> pd.DataFrame:
    for band in ["NDVI", "GNDVI", "NDRE", "NDMI", "NBR", "EVI", "green_ratio", "red_ratio", "yellow_proxy"]:
        cols = [f"{season}_{band}" for season in ["winter", "premonsoon", "monsoon", "postmonsoon"]]
        df[f"{band}_amp"] = df[cols].max(axis=1, skipna=True) - df[cols].min(axis=1, skipna=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crowns", default=str(DATA_DIR / "iitd_sv_crowns_master_wgs84.geojson"))
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--geometry-mode", default="buffer", choices=["polygon", "point", "buffer"])
    ap.add_argument("--buffer-meters", type=float, default=20.0)
    ap.add_argument("--max-cloud", type=float, default=70.0)
    ap.add_argument("--max-items-per-season", type=int, default=12)
    ap.add_argument("--pad-pixels", type=int, default=40)
    ap.add_argument("--min-pixels", type=int, default=1)
    ap.add_argument("--all-touched", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--label-filter",
        default=None,
        help="Optional label column; keep only rows where this label is not -1 before extracting features.",
    )
    ap.add_argument("--out-csv", default=str(EXPORT_DIR / "stac_s2_features_2025_buffer20.csv"))
    args = ap.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    crowns = gpd.read_file(args.crowns)
    crowns = crowns[crowns.geometry.notnull() & ~crowns.geometry.is_empty].reset_index(drop=True)
    if args.label_filter:
        if args.label_filter not in crowns.columns:
            raise KeyError(f"--label-filter {args.label_filter!r} not found in crowns columns")
        before = len(crowns)
        crowns = crowns[crowns[args.label_filter].astype(int) != -1].reset_index(drop=True)
        print(f"Label filter {args.label_filter}: kept {len(crowns)} of {before} crowns", flush=True)
    bbox = list(map(float, crowns.total_bounds))
    fallback_lat = float((bbox[1] + bbox[3]) / 2.0)

    catalog = Client.open(STAC_API)
    season_items = {}
    item_records = []
    for season in seasons_for_year(args.year):
        items = query_items(catalog, bbox, season, args.max_cloud, args.max_items_per_season)
        if not items:
            raise RuntimeError(f"No Sentinel-2 items found for {season.name} ({season.start}/{season.end})")
        season_items[season.name] = items
        for item in items:
            item_records.append(
                {
                    "season": season.name,
                    "id": item.id,
                    "datetime": item.datetime.isoformat() if item.datetime else None,
                    "cloud_cover": item.properties.get("eo:cloud_cover"),
                    "mgrs_tile": item.properties.get("mgrs:tile"),
                    "epsg": sentinel_epsg_from_item(item, fallback_lat),
                }
            )
        print(f"{season.name}: {len(items)} items", flush=True)

    item_table = pd.DataFrame(item_records)
    item_table.to_csv(META_DIR / f"stac_items_{args.year}_{args.geometry_mode}.csv", index=False)
    epsg = int(item_table["epsg"].mode().iloc[0])
    dst_crs = f"EPSG:{epsg}"
    bounds = transform_bbox_wgs84(bbox, dst_crs)
    geoms = extraction_geometries(crowns, dst_crs, args.geometry_mode, args.buffer_meters)

    prop_cols = [c for c in crowns.columns if c != "geometry"]
    features = crowns[prop_cols].copy()
    coverage_cols = []

    for season_name, items in season_items.items():
        per_item = []
        masks = None
        mask_signature = None
        for idx, item in enumerate(items, start=1):
            print(f"[{season_name} {idx:02d}/{len(items):02d}] {item.id} cloud={item.properties.get('eo:cloud_cover')}", flush=True)
            with rasterio.Env(
                GDAL_HTTP_TIMEOUT="60",
                GDAL_HTTP_CONNECTTIMEOUT="20",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
            ):
                arrays, valid, transform = read_item_stack(item, bounds, args.pad_pixels)
            signature = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f, valid.shape)
            if masks is None or signature != mask_signature:
                print(f"  Building masks for {len(geoms)} crowns over raster shape {valid.shape}", flush=True)
                masks = build_masks(geoms, transform, valid.shape, args.all_touched)
                mask_signature = signature
            stats = summarize_item(arrays, valid, masks, args.min_pixels)
            stats["item_id"] = item.id
            per_item.append(stats)

        stacked = pd.concat(per_item, keys=range(len(per_item)), names=["item_no", "row_no"]).reset_index(level=0)
        grouped = stacked.groupby(stacked.index)
        for band in ALL_BANDS:
            features[f"{season_name}_{band}"] = grouped[band].median().to_numpy()
        cov = grouped["valid_pixels"].agg(["median", "max"]).rename(
            columns={"median": f"{season_name}_valid_pixels_median", "max": f"{season_name}_valid_pixels_max"}
        )
        for col in cov.columns:
            features[col] = cov[col].to_numpy()
            coverage_cols.append(col)

    features = add_amplitude_features(features)
    features["stac_year"] = args.year
    features["geometry_mode"] = args.geometry_mode
    features["buffer_meters"] = args.buffer_meters if args.geometry_mode == "buffer" else 0
    features["stac_item_count"] = int(len(item_table))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv, index=False)

    meta = {
        "out_csv": str(out_csv),
        "n_crowns": int(len(features)),
        "year": args.year,
        "geometry_mode": args.geometry_mode,
        "buffer_meters": args.buffer_meters,
        "max_cloud": args.max_cloud,
        "max_items_per_season": args.max_items_per_season,
        "min_pixels": args.min_pixels,
        "label_filter": args.label_filter,
        "epsg": epsg,
        "feature_columns": [c for c in features.columns if c not in prop_cols],
        "coverage_columns": coverage_cols,
        "items_by_season": {name: len(items) for name, items in season_items.items()},
    }
    with (META_DIR / f"stac_run_{args.year}_{args.geometry_mode}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_csv} with {len(features)} rows and {len(features.columns)} columns")
    print(features[coverage_cols].describe().to_string())


if __name__ == "__main__":
    main()
