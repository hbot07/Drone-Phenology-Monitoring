#!/usr/bin/env python3
"""Upload the local crown GeoJSON to Google Cloud Storage for Earth Engine ingestion."""
from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--blob", default=None)
    parser.add_argument("--location", default="US")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    client = storage.Client(project=args.project)
    bucket = client.lookup_bucket(args.bucket)
    if bucket is None:
        bucket = client.bucket(args.bucket)
        bucket = client.create_bucket(bucket, project=args.project, location=args.location)
        print(f"Created bucket gs://{bucket.name} in {args.location}")
    else:
        print(f"Using existing bucket gs://{bucket.name}")

    blob_name = args.blob or source.name
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(source))
    print(f"Uploaded {source} to gs://{bucket.name}/{blob_name}")


if __name__ == "__main__":
    main()
