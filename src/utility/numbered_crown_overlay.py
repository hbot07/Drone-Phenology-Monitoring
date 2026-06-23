#!/usr/bin/env python3
"""Create a printable orthomosaic overlay with numbered crown polygons."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling


def _read_rgb_preview(orthomosaic: Path, max_size: int) -> tuple[np.ndarray, list[float]]:
    with rasterio.open(orthomosaic) as src:
        scale = max(src.width, src.height) / max_size if max(src.width, src.height) > max_size else 1.0
        out_width = max(1, int(round(src.width / scale)))
        out_height = max(1, int(round(src.height / scale)))
        indexes = [1, 2, 3] if src.count >= 3 else [1]
        data = src.read(indexes, out_shape=(len(indexes), out_height, out_width), resampling=Resampling.bilinear)
        img = np.moveaxis(data, 0, -1).astype(np.float32)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        lo, hi = np.nanpercentile(img, [2, 98])
        if hi > lo:
            img = np.clip((img - lo) / (hi - lo), 0, 1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        return img, extent


def _label_point(geometry):
    point = geometry.representative_point()
    return point.x, point.y


def create_overlay(
    orthomosaic: Path,
    crowns: Path,
    output: Path,
    layer: str | None,
    id_field: str | None,
    max_size: int,
    dpi: int,
) -> None:
    image, extent = _read_rgb_preview(orthomosaic, max_size=max_size)
    crowns_gdf = gpd.read_file(crowns, layer=layer) if layer else gpd.read_file(crowns)
    crowns_gdf = crowns_gdf[crowns_gdf.geometry.notna() & ~crowns_gdf.geometry.is_empty].copy()

    if crowns_gdf.empty:
        raise ValueError(f"No valid crown geometries found in {crowns}")

    if id_field and id_field not in crowns_gdf.columns:
        raise ValueError(f"id field '{id_field}' not found. Available fields: {list(crowns_gdf.columns)}")

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(image, extent=extent, origin="upper", aspect="equal")
    crowns_gdf.boundary.plot(ax=ax, color="#ffcc00", linewidth=0.7)

    for index, row in crowns_gdf.reset_index(drop=True).iterrows():
        label = row[id_field] if id_field else index + 1
        x, y = _label_point(row.geometry)
        ax.text(
            x,
            y,
            str(label),
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    ax.set_axis_off()
    ax.set_title(f"{orthomosaic.stem}: {len(crowns_gdf)} numbered crowns")
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a printable numbered crown overlay.")
    parser.add_argument("--orthomosaic", required=True, help="Path to orthomosaic GeoTIFF.")
    parser.add_argument("--crowns", required=True, help="Path to crown polygons (.gpkg, .geojson, etc.).")
    parser.add_argument("--output", required=True, help="Output image path, usually .png.")
    parser.add_argument("--layer", default=None, help="Optional GPKG layer name, such as conf_0p45.")
    parser.add_argument("--id-field", default=None, help="Optional existing field to use as label text.")
    parser.add_argument("--max-size", type=int, default=4000, help="Max preview dimension in pixels.")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    args = parser.parse_args()

    create_overlay(
        orthomosaic=Path(args.orthomosaic),
        crowns=Path(args.crowns),
        output=Path(args.output),
        layer=args.layer,
        id_field=args.id_field,
        max_size=args.max_size,
        dpi=args.dpi,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
