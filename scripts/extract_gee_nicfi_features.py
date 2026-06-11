#!/usr/bin/env python3
"""Export Planet NICFI Basemap crown features from Earth Engine.

The output is one row per crown, with median band values over the original crown
geometry or centroid buffer. Use this as a finer-than-10 m comparison against
GEE Satellite Embedding V1.
"""
from __future__ import annotations

import argparse
import time

import ee


NICFI_COLLECTIONS = {
    "asia": "projects/planet-nicfi/assets/basemaps/asia",
    "africa": "projects/planet-nicfi/assets/basemaps/africa",
    "americas": "projects/planet-nicfi/assets/basemaps/americas",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Google Cloud / Earth Engine project ID")
    parser.add_argument("--crowns-asset", required=True, help="Earth Engine FeatureCollection asset path")
    parser.add_argument("--region", default="asia", choices=sorted(NICFI_COLLECTIONS))
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-01-01")
    parser.add_argument("--geometry-mode", default="polygon", choices=["polygon", "centroid_buffer"])
    parser.add_argument("--buffer-meters", type=float, default=0.0)
    parser.add_argument("--drive-folder", default="drone_phenology_gee_exports")
    parser.add_argument("--description", default=None)
    parser.add_argument("--scale", type=float, default=4.77, help="Nominal NICFI basemap scale in meters")
    parser.add_argument("--tile-scale", type=float, default=4)
    return parser.parse_args()


def centroid_buffer(feature: ee.Feature, buffer_meters: float) -> ee.Feature:
    geom = feature.geometry()
    buffered = geom.centroid(maxError=1).buffer(distance=buffer_meters, maxError=1)
    return feature.setGeometry(buffered)


def main() -> None:
    args = parse_args()
    ee.Initialize(project=args.project)

    crowns = ee.FeatureCollection(args.crowns_asset)
    if args.geometry_mode == "centroid_buffer":
        if args.buffer_meters <= 0:
            raise ValueError("--buffer-meters must be positive when using centroid_buffer")
        crowns = crowns.map(lambda f: centroid_buffer(f, args.buffer_meters))

    collection_id = NICFI_COLLECTIONS[args.region]
    collection = ee.ImageCollection(collection_id).filterDate(args.start_date, args.end_date)
    image = ee.Image(collection.median())
    band_names = image.bandNames()

    print("Crown count:", crowns.size().getInfo())
    print("NICFI collection:", collection_id)
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
    samples = samples.filter(ee.Filter.notNull(band_names))

    print("Sample count after notNull:", samples.size().getInfo())
    print("Sample feature:", samples.first().toDictionary().getInfo())

    mode = args.geometry_mode if args.geometry_mode == "polygon" else f"buffer{int(args.buffer_meters)}m"
    description = args.description or f"iitd_sv_nicfi_{args.start_date}_{args.end_date}_{mode}"

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

    while task.active():
        status = task.status()
        print("Task state:", status.get("state"))
        time.sleep(30)

    print("Final status:", task.status())


if __name__ == "__main__":
    main()
