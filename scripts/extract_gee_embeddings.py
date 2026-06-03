#!/usr/bin/env python3
"""Export crown-level Google Satellite Embedding vectors from Earth Engine."""
from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv

import ee


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
    parser.add_argument("--year", type=int, default=int(os.getenv("EE_EMBEDDING_YEAR", "2024")))
    parser.add_argument("--buffer-meters", type=float, default=float(os.getenv("EE_BUFFER_METERS", "20")))
    parser.add_argument(
        "--geometry-mode",
        default=os.getenv("EE_GEOMETRY_MODE", "centroid_buffer"),
        choices=["original", "centroid", "centroid_buffer"],
        help="Geometry used for reduceRegions. Use original for uploaded crown geometry without buffering.",
    )
    parser.add_argument("--drive-folder", default=os.getenv("EE_DRIVE_FOLDER", "drone_phenology_gee_exports"))
    parser.add_argument("--description", default=None)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--tile-scale", type=float, default=4.0)
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


def main() -> None:
    args = parse_args()

    ee.Initialize(project=args.project)

    crowns = ee.FeatureCollection(args.crowns_asset)
    print("Crown count:", crowns.size().getInfo())
    print("First feature:", crowns.first().toDictionary().getInfo())

    embeddings = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{args.year}-01-01", f"{args.year + 1}-01-01")
        .filterBounds(crowns.geometry())
    )
    print("Embedding image count over crowns:", embeddings.size().getInfo())
    emb_img = embeddings.mosaic()

    band_names = emb_img.bandNames()
    print("Embedding bands:", band_names.getInfo())

    if args.geometry_mode == "original":
        sampling_crowns = crowns
    elif args.geometry_mode == "centroid":
        sampling_crowns = crowns.map(centroid)
    else:
        sampling_crowns = crowns.map(lambda f: centroid_buffer(f, args.buffer_meters))

    samples = emb_img.reduceRegions(
        collection=sampling_crowns,
        reducer=ee.Reducer.median(),
        scale=args.scale,
        tileScale=args.tile_scale,
    )
    samples = samples.filter(ee.Filter.notNull(band_names))

    sample_count = samples.size().getInfo()
    print("Sample count after notNull:", sample_count)
    if sample_count == 0:
        raise RuntimeError(
            "No embedding samples were found. Check the crowns geometry, year, and embedding collection coverage."
        )
    print("Sample feature:", samples.first().toDictionary().getInfo())

    if args.description:
        description = args.description
    elif args.geometry_mode == "centroid_buffer":
        description = f"iitd_sv_gee_embeddings_{args.year}_buffer{int(args.buffer_meters)}m"
    else:
        description = f"iitd_sv_gee_embeddings_{args.year}_{args.geometry_mode}"
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
