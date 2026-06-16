#!/usr/bin/env python3
"""Export Planet NICFI Basemap crown features from Earth Engine.

The output is one row per crown, with median band values over the original crown
geometry or centroid buffer. Use this as a finer-than-10 m comparison against
GEE Satellite Embedding V1.
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

import ee


NICFI_COLLECTIONS = {
    "asia": "projects/planet-nicfi/assets/basemaps/asia",
    "africa": "projects/planet-nicfi/assets/basemaps/africa",
    "americas": "projects/planet-nicfi/assets/basemaps/americas",
}


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT_ID") or os.getenv("EE_PROJECT"),
        help="Google Cloud / Earth Engine project ID. Can also use GOOGLE_CLOUD_PROJECT_ID in .env.",
    )
    parser.add_argument(
        "--crowns-asset",
        default=os.getenv("EE_CROWNS_ASSET"),
        help="Earth Engine FeatureCollection asset path. Can also use EE_CROWNS_ASSET in .env.",
    )
    parser.add_argument("--region", default="asia", choices=sorted(NICFI_COLLECTIONS))
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-01-01")
    parser.add_argument(
        "--cadence",
        default="monthly",
        choices=["monthly", "biannual", "all"],
        help="NICFI mosaic cadence to use. Monthly is best for seasonal/phenology experiments.",
    )
    parser.add_argument(
        "--geometry-mode",
        default="original",
        choices=["original", "polygon", "centroid", "centroid_buffer"],
        help="Geometry used for reduceRegions. 'polygon' is accepted as an alias for 'original'.",
    )
    parser.add_argument("--buffer-meters", type=float, default=0.0)
    parser.add_argument("--drive-folder", default="drone_phenology_gee_exports")
    parser.add_argument("--description", default=None)
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Download the sampled FeatureCollection directly to this local CSV instead of starting a Drive export.",
    )
    parser.add_argument("--scale", type=float, default=4.77, help="Nominal NICFI basemap scale in meters")
    parser.add_argument("--tile-scale", type=float, default=4)
    parser.add_argument(
        "--no-indices",
        action="store_true",
        help="Export only B/G/R/N band statistics, without derived vegetation indices.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start the Drive export and exit immediately instead of polling task status.",
    )
    args = parser.parse_args()
    if not args.project:
        parser.error("--project is required, or set GOOGLE_CLOUD_PROJECT_ID / EE_PROJECT in .env")
    if not args.crowns_asset:
        parser.error("--crowns-asset is required, or set EE_CROWNS_ASSET in .env")
    return args


def centroid_buffer(feature: ee.Feature, buffer_meters: float) -> ee.Feature:
    geom = feature.geometry()
    buffered = geom.centroid(maxError=1).buffer(distance=buffer_meters, maxError=1)
    return feature.setGeometry(buffered)


def centroid(feature: ee.Feature) -> ee.Feature:
    return feature.setGeometry(feature.geometry().centroid(maxError=1))


def add_indices(image: ee.Image) -> ee.Image:
    """Add simple NICFI vegetation/color indices from scaled reflectance bands."""
    scaled = image.select(["B", "G", "R", "N"]).multiply(0.0001)
    ndvi = scaled.normalizedDifference(["N", "R"]).rename("NDVI")
    gndvi = scaled.normalizedDifference(["N", "G"]).rename("GNDVI")
    ndwi = scaled.normalizedDifference(["G", "N"]).rename("NDWI")
    evi = scaled.expression(
        "2.5 * ((N - R) / (N + 6 * R - 7.5 * B + 1))",
        {
            "N": scaled.select("N"),
            "R": scaled.select("R"),
            "B": scaled.select("B"),
        },
    ).rename("EVI")
    return image.addBands([ndvi, gndvi, ndwi, evi])


def main() -> None:
    args = parse_args()
    ee.Initialize(project=args.project)

    crowns = ee.FeatureCollection(args.crowns_asset)
    if args.geometry_mode == "centroid_buffer":
        if args.buffer_meters <= 0:
            raise ValueError("--buffer-meters must be positive when using centroid_buffer")
        crowns = crowns.map(lambda f: centroid_buffer(f, args.buffer_meters))
    elif args.geometry_mode == "centroid":
        crowns = crowns.map(centroid)

    collection_id = NICFI_COLLECTIONS[args.region]
    collection = (
        ee.ImageCollection(collection_id)
        .filterDate(args.start_date, args.end_date)
        .filterBounds(crowns.geometry())
    )
    if args.cadence != "all":
        collection = collection.filter(ee.Filter.eq("cadence", args.cadence))

    image = ee.Image(collection.median()).select(["B", "G", "R", "N"])
    if not args.no_indices:
        image = add_indices(image)
    band_names = image.bandNames()
    required_sample_properties = band_names.map(lambda name: ee.String(name).cat("_median"))

    print("Crown count:", crowns.size().getInfo())
    print("NICFI collection:", collection_id)
    print("Cadence:", args.cadence)
    print("Image count:", collection.size().getInfo())
    print("Bands:", band_names.getInfo())

    reducer = ee.Reducer.median().combine(
        reducer2=ee.Reducer.stdDev(),
        sharedInputs=True,
    )
    samples = image.reduceRegions(
        collection=crowns,
        reducer=reducer,
        scale=args.scale,
        tileScale=args.tile_scale,
    )
    samples = samples.filter(ee.Filter.notNull(required_sample_properties))

    print("Sample count after notNull:", samples.size().getInfo())
    print("Sample feature:", samples.first().toDictionary().getInfo())

    if args.geometry_mode == "centroid_buffer":
        mode = f"buffer{int(args.buffer_meters)}m"
    elif args.geometry_mode == "polygon":
        mode = "original"
    else:
        mode = args.geometry_mode
    date_tag = f"{args.start_date}_{args.end_date}".replace("-", "")
    description = args.description or f"iitd_sv_nicfi_{date_tag}_{args.cadence}_{mode}"

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        url = samples.getDownloadURL(filetype="CSV", filename=description)
        print("Downloading local CSV:", out_csv)
        urllib.request.urlretrieve(url, out_csv)
        print("Wrote:", out_csv)
        return

    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=description,
        folder=args.drive_folder,
        fileNamePrefix=description,
        fileFormat="CSV",
    )
    task.start()

    print(f"Started export task: {description}")
    print("Task ID:", task.id)
    print("Check Google Drive folder:", args.drive_folder)
    if args.no_wait:
        return

    while task.active():
        status = task.status()
        print("Task state:", status.get("state"))
        time.sleep(30)

    print("Final status:", task.status())


if __name__ == "__main__":
    main()
