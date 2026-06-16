#!/usr/bin/env python3
"""Extract GEE Satellite-Embedding-V1 2024 features over ORIGINAL crown geometry,
pulled client-side in chunks (no Drive export).

Two prof-endorsed small-crown reductions:
  - weighted mean over the original crown polygon (reduceRegions mean is
    area-weighted by pixel coverage fraction by default)
  - centroid pixel value (geometry set to crown centroid)

Writes A00..A63 + crown_uid to exports/, one CSV per mode.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd
import ee

PROJECT = "adept-vigil-418410"
ASSET = "projects/adept-vigil-418410/assets/iitd_sv_crowns_master_shapefile_for_gee"
ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"


def init():
    ee.Initialize(project=PROJECT)


def emb_image(year: int) -> ee.Image:
    coll = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(f"{year}-01-01", f"{year+1}-01-01"))
    return coll.mosaic()


def reduce_fc(mode: str, year: int) -> tuple[ee.FeatureCollection, list[str]]:
    img = emb_image(year)
    bands = img.bandNames()
    crowns = ee.FeatureCollection(ASSET)
    if mode == "centroid":
        crowns = crowns.map(lambda f: f.setGeometry(f.geometry().centroid(maxError=1)))
    # mean reducer = area-weighted over the polygon by default
    fc = img.reduceRegions(collection=crowns, reducer=ee.Reducer.mean(), scale=10, tileScale=4)
    band_list = bands.getInfo()
    keep = ["crown_uid"] + band_list
    fc = fc.select(keep, retainGeometry=False)
    return fc, band_list


def pull(fc: ee.FeatureCollection, bands: list[str], chunk: int) -> pd.DataFrame:
    n = fc.size().getInfo()
    lst = fc.toList(n)
    rows = []
    for off in range(0, n, chunk):
        sub = ee.FeatureCollection(lst.slice(off, off + chunk)).getInfo()
        for feat in sub["features"]:
            rows.append(feat["properties"])
        print(f"  pulled {min(off+chunk, n)}/{n}", flush=True)
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--mode", choices=["mean", "centroid"], required=True)
    ap.add_argument("--chunk", type=int, default=400)
    args = ap.parse_args()
    init()
    fc, bands = reduce_fc(args.mode, args.year)
    print(f"mode={args.mode} bands={len(bands)} ({bands[0]}..{bands[-1]})", flush=True)
    df = pull(fc, bands, args.chunk)
    # only crowns with a value (small crowns over no-data become NaN)
    before = len(df)
    df = df.dropna(subset=bands, how="all")
    EXPORTS.mkdir(parents=True, exist_ok=True)
    out = EXPORTS / f"gee_{args.mode}_{args.year}_features.csv"
    df.to_csv(out, index=False)
    print(f"rows={len(df)} (dropped {before-len(df)} all-NaN) -> {out}")


if __name__ == "__main__":
    main()
