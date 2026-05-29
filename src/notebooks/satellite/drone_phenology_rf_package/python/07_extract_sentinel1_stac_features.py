#!/usr/bin/env python3
"""
Extract crown-level Sentinel-1 RTC radar features from Microsoft Planetary
Computer STAC.

This scales the sat_data_sentinel1.ipynb prototype from 20 SIT crowns to the
prepared IITD + Sanjay Van package.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPORT_DIR = ROOT / "exports"
META_DIR = ROOT / "outputs" / "stac_meta"
STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"


def season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "premonsoon"
    if month in {6, 7, 8, 9}:
        return "monsoon"
    return "postmonsoon"


def signed_href(item, keys: list[str]) -> str:
    for key in keys:
        asset = item.assets.get(key)
        if asset is not None:
            return pc.sign(asset.href)
    raise KeyError(f"Missing assets {keys} on {item.id}")


def query_s1_items(bbox: list[float], years: list[int], orbit_state: str | None, relative_orbit: int | None, one_per_date: bool):
    catalog = Client.open(STAC_API)
    items = []
    for year in years:
        search = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
        )
        items.extend(list(search.items()))

    rows = []
    lookup = {}
    for item in items:
        dt = item.datetime
        if dt is None:
            continue
        props = item.properties
        pols = props.get("sar:polarizations", [])
        pol_text = ",".join(pols) if isinstance(pols, list) else str(pols)
        if "VV" not in pol_text or "VH" not in pol_text:
            continue
        if props.get("sar:instrument_mode", "IW") != "IW":
            continue
        row = {
            "id": item.id,
            "datetime": dt.isoformat(),
            "date": pd.Timestamp(dt).date().isoformat(),
            "year": int(dt.year),
            "month": int(dt.month),
            "season": season_for_month(int(dt.month)),
            "orbit_state": props.get("sat:orbit_state"),
            "relative_orbit": props.get("sat:relative_orbit"),
            "platform": props.get("platform"),
        }
        rows.append(row)
        lookup[item.id] = item
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable Sentinel-1 RTC VV/VH items found")

    orbit_counts = (
        df.groupby(["orbit_state", "relative_orbit"], dropna=False)
        .size()
        .reset_index(name="n_items")
        .sort_values("n_items", ascending=False)
    )
    if orbit_state is None and relative_orbit is None:
        chosen = orbit_counts.iloc[0]
        orbit_state = chosen["orbit_state"]
        relative_orbit = int(chosen["relative_orbit"])
    if orbit_state is not None:
        df = df[df["orbit_state"] == orbit_state].copy()
    if relative_orbit is not None:
        df = df[df["relative_orbit"] == int(relative_orbit)].copy()
    if one_per_date:
        df = df.sort_values(["date", "id"]).groupby("date", as_index=False).first()
    df = df.sort_values("date").reset_index(drop=True)
    return df, {item_id: lookup[item_id] for item_id in df["id"].tolist()}, orbit_counts


def padded_window(src: rasterio.io.DatasetReader, bounds: tuple[float, float, float, float], pad_pixels: int):
    win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
    col_off = max(0, int(win.col_off) - pad_pixels)
    row_off = max(0, int(win.row_off) - pad_pixels)
    right = min(src.width, int(math.ceil(win.col_off + win.width)) + pad_pixels)
    bottom = min(src.height, int(math.ceil(win.row_off + win.height)) + pad_pixels)
    return rasterio.windows.Window(col_off, row_off, max(1, right - col_off), max(1, bottom - row_off))


def transform_bbox(bbox_wgs84: list[float], dst_crs) -> tuple[float, float, float, float]:
    minlon, minlat, maxlon, maxlat = bbox_wgs84
    tx = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs, ys = tx.transform([minlon, minlon, maxlon, maxlon], [minlat, maxlat, minlat, maxlat])
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def read_item(item, bbox_wgs84: list[float], pad_pixels: int):
    href_vv = signed_href(item, ["vv", "VV"])
    href_vh = signed_href(item, ["vh", "VH"])
    with rasterio.Env(
        GDAL_HTTP_TIMEOUT="60",
        GDAL_HTTP_CONNECTTIMEOUT="20",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        CPL_VSIL_CURL_USE_HEAD="NO",
    ):
        with rasterio.open(href_vv) as src_vv:
            crs = src_vv.crs
            bounds = transform_bbox(bbox_wgs84, crs)
            win = padded_window(src_vv, bounds, pad_pixels)
            vv = src_vv.read(1, window=win, masked=True).astype("float32")
            transform = rasterio.windows.transform(win, src_vv.transform)
        with rasterio.open(href_vh) as src_vh:
            vh = src_vh.read(1, window=win, masked=True).astype("float32")
    return vv, vh, transform, crs


def compute_features(vv, vh):
    vv_arr = np.asarray(vv, dtype="float32")
    vh_arr = np.asarray(vh, dtype="float32")
    invalid = np.ma.getmaskarray(vv) | np.ma.getmaskarray(vh) | ~np.isfinite(vv_arr) | ~np.isfinite(vh_arr)
    vv_arr = np.where(invalid, np.nan, vv_arr)
    vh_arr = np.where(invalid, np.nan, vh_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        vv_db = np.where(vv_arr > 0, 10.0 * np.log10(vv_arr), np.nan).astype("float32")
        vh_db = np.where(vh_arr > 0, 10.0 * np.log10(vh_arr), np.nan).astype("float32")
        vh_vv_ratio = np.where((vv_arr > 0) & (vh_arr > 0), vh_arr / vv_arr, np.nan).astype("float32")
        vh_minus_vv_db = (vh_db - vv_db).astype("float32")
    valid = np.isfinite(vv_db) & np.isfinite(vh_db)
    return {
        "vv_db": vv_db,
        "vh_db": vh_db,
        "vh_vv_ratio": vh_vv_ratio,
        "vh_minus_vv_db": vh_minus_vv_db,
    }, valid


def extraction_geometries(crowns: gpd.GeoDataFrame, dst_crs, mode: str, buffer_meters: float):
    gdf = crowns.to_crs(dst_crs)
    if mode == "point":
        geoms = gdf.geometry.centroid
    elif mode == "buffer":
        geoms = gdf.geometry.centroid.buffer(buffer_meters)
    elif mode == "polygon":
        geoms = gdf.geometry
    else:
        raise ValueError(mode)
    return [geom.__geo_interface__ for geom in geoms]


def build_masks(geoms, transform, shape, all_touched: bool):
    return [geometry_mask([geom], out_shape=shape, transform=transform, invert=True, all_touched=all_touched) for geom in geoms]


def summarize(arrays: dict[str, np.ndarray], valid: np.ndarray, masks, min_pixels: int) -> pd.DataFrame:
    rows = []
    for mask in masks:
        use = mask & valid
        n = int(use.sum())
        row = {"valid_pixels": n}
        for name, arr in arrays.items():
            if n >= min_pixels:
                vals = arr[use]
                vals = vals[np.isfinite(vals)]
                row[name] = float(np.nanmedian(vals)) if len(vals) else np.nan
            else:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_features(obs: pd.DataFrame, prop_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["vv_db", "vh_db", "vh_vv_ratio", "vh_minus_vv_db"]
    rows = []
    for row_no, group in obs.groupby("row_no"):
        row = {"row_no": int(row_no), "s1_n_obs": int(len(group)), "s1_n_years": int(group["year"].nunique())}
        for metric in metrics:
            s = group[metric].astype(float)
            row[f"s1_{metric}_median"] = float(s.median())
            row[f"s1_{metric}_mean"] = float(s.mean())
            row[f"s1_{metric}_std"] = float(s.std(ddof=0))
            row[f"s1_{metric}_amp"] = float(s.max() - s.min())
            for season, season_group in group.groupby("season"):
                ss = season_group[metric].astype(float)
                row[f"s1_{season}_{metric}_median"] = float(ss.median())
        rows.append(row)
    features = pd.DataFrame(rows)
    features = prop_df.reset_index(names="row_no").merge(features, on="row_no", how="left").drop(columns=["row_no"])
    return features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crowns", default=str(DATA_DIR / "iitd_sv_crowns_master_wgs84.geojson"))
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--geometry-mode", default="buffer", choices=["polygon", "point", "buffer"])
    ap.add_argument("--buffer-meters", type=float, default=20.0)
    ap.add_argument("--label-filter", default=None)
    ap.add_argument("--orbit-state", default=None)
    ap.add_argument("--relative-orbit", type=int, default=None)
    ap.add_argument("--one-per-date", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pad-pixels", type=int, default=20)
    ap.add_argument("--min-pixels", type=int, default=1)
    ap.add_argument("--all-touched", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-csv", default=str(EXPORT_DIR / "stac_s1_features_2022_2025_buffer20.csv"))
    args = ap.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    crowns = gpd.read_file(args.crowns)
    crowns = crowns[crowns.geometry.notnull() & ~crowns.geometry.is_empty].reset_index(drop=True)
    if args.label_filter:
        before = len(crowns)
        crowns = crowns[crowns[args.label_filter].astype(int) != -1].reset_index(drop=True)
        print(f"Label filter {args.label_filter}: kept {len(crowns)} of {before}", flush=True)
    bbox = list(map(float, crowns.total_bounds))
    prop_cols = [c for c in crowns.columns if c != "geometry"]
    prop_df = crowns[prop_cols].copy()

    items_df, items_lookup, orbit_counts = query_s1_items(bbox, years, args.orbit_state, args.relative_orbit, args.one_per_date)
    print(f"Sentinel-1 items: {len(items_df)}", flush=True)
    print(items_df[["orbit_state", "relative_orbit"]].drop_duplicates().to_string(index=False), flush=True)
    orbit_counts.to_csv(META_DIR / "s1_orbit_counts.csv", index=False)
    items_df.to_csv(META_DIR / "s1_items_filtered.csv", index=False)

    records = []
    masks = None
    signature = None
    for idx, item_row in enumerate(items_df.itertuples(index=False), start=1):
        item = items_lookup[item_row.id]
        print(f"[{idx:03d}/{len(items_df):03d}] {item_row.id} {item_row.date}", flush=True)
        vv, vh, transform, crs = read_item(item, bbox, args.pad_pixels)
        arrays, valid = compute_features(vv, vh)
        cur_sig = (str(crs), transform.a, transform.b, transform.c, transform.d, transform.e, transform.f, valid.shape)
        if masks is None or cur_sig != signature:
            geoms = extraction_geometries(crowns, crs, args.geometry_mode, args.buffer_meters)
            print(f"  Building masks for {len(geoms)} crowns over {valid.shape}", flush=True)
            masks = build_masks(geoms, transform, valid.shape, args.all_touched)
            signature = cur_sig
        stats = summarize(arrays, valid, masks, args.min_pixels)
        for row_no, rec in stats.iterrows():
            if rec["valid_pixels"] < args.min_pixels:
                continue
            payload = {
                "row_no": int(row_no),
                "date": item_row.date,
                "year": int(item_row.year),
                "month": int(item_row.month),
                "season": item_row.season,
                "item_id": item_row.id,
                "valid_pixels": int(rec["valid_pixels"]),
            }
            for key in arrays:
                payload[key] = float(rec[key]) if pd.notna(rec[key]) else np.nan
            records.append(payload)

    obs = pd.DataFrame(records)
    obs.to_csv(META_DIR / "s1_observations_last_run.csv", index=False)
    features = aggregate_features(obs, prop_df)
    features["s1_years"] = ",".join(map(str, years))
    features["s1_item_count"] = int(len(items_df))
    features["geometry_mode"] = args.geometry_mode
    features["buffer_meters"] = args.buffer_meters if args.geometry_mode == "buffer" else 0
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv, index=False)
    meta = {
        "out_csv": str(out_csv),
        "n_crowns": int(len(features)),
        "n_observations": int(len(obs)),
        "years": years,
        "items": int(len(items_df)),
        "label_filter": args.label_filter,
        "geometry_mode": args.geometry_mode,
        "buffer_meters": args.buffer_meters,
    }
    (META_DIR / "s1_run_last.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv} with {features.shape[0]} rows and {features.shape[1]} columns", flush=True)


if __name__ == "__main__":
    main()
