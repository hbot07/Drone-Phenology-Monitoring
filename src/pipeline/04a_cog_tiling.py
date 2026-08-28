#!/usr/bin/env python3
"""
Pipeline Step 4a: Cloud-Optimised GeoTIFF Generation + XYZ Tile Pyramid

Converts all raw orthomosaics to Cloud-Optimised GeoTIFFs (COGs) and
generates local XYZ tile pyramids for efficient browser streaming.

What it does:
  - Reprojects each OM to EPSG:3857 (Web Mercator)
  - Stretches to uint8 RGB
  - Writes COG with LZW compression and internal overviews
  - Generates XYZ tile pyramid: tiles/{OM_stem}/{z}/{x}/{y}.png
  - Writes tile_manifest.json consumed by Step 4b

Output layout:
    04_viewer/
    ├── cogs/
    │   ├── OM01_<stem>.tif
    │   └── ...
    ├── tiles/
    │   ├── OM01_<stem>/
    │   │   ├── 14/582/375.png  (only tiles covering the OM extent)
    │   │   └── ...
    │   └── ...
    └── tile_manifest.json      ← consumed by 04b_interactive_viz.py

Requires: dpm-tracking conda environment
    rasterio, numpy, mercantile (or gdal2tiles on PATH)

Usage:
    python 04a_cog_tiling.py --config /path/to/pipeline_config.json
        [--underlay-om last|first|N]
        [--tile-size 256]
        [--max-cog-zoom 22]
        [--force-regen-cogs]
        [--skip-if-done]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def save_config(config: dict, config_path: Path) -> None:
    config_path.write_text(json.dumps(config, indent=2))


def build_pairs_and_om_stems(config: dict) -> Tuple[List[Tuple[str, str, str]], Dict[int, str]]:
    crowns_dir = Path(config["crowns_dir"])
    om_dir     = Path(config["om_dir"])
    pairs, om_stems = [], {}
    for i, (gpkg_raw, tif_raw, stem) in enumerate(config["pairs"], 1):
        gpkg_from_config = Path(gpkg_raw)
        tif_from_config  = Path(tif_raw)
        gpkg = str(gpkg_from_config) if gpkg_from_config.exists() else str(crowns_dir / f"{stem}_multithreshold.gpkg")
        tif  = str(tif_from_config)  if tif_from_config.exists()  else str(om_dir     / f"{stem}.tif")
        pairs.append((gpkg, tif, stem))
        om_stems[i] = stem
    return pairs, om_stems


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError("Expected HxWxC array")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype == np.uint8:
        return arr
    arr  = arr.astype(np.float32)
    lo   = float(np.nanpercentile(arr, 2))
    hi   = float(np.nanpercentile(arr, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
        hi = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
        if hi <= lo:
            hi = lo + 1.0
    return (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# COG generation
# ---------------------------------------------------------------------------

def generate_cog(src_path: str, dst_path: str, tile_size: int = 256) -> None:
    """
    Convert a raw GeoTIFF to a Cloud-Optimised GeoTIFF (COG).

    Steps:
      1. Reproject to EPSG:3857 (Web Mercator) if needed
      2. Stretch bands to uint8
      3. Build internal overviews (2x, 4x, 8x, …)
      4. Write tiled + LZW-compressed COG with COPY_SRC_OVERVIEWS=YES
    """
    import tempfile, os
    print(f"    COG: {Path(src_path).name} → {Path(dst_path).name}")

    import rasterio
    target_crs = rasterio.crs.CRS.from_epsg(3857)

    with rasterio.open(src_path) as src:
        already_3857 = (src.crs == target_crs)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix="_prep.tif")
    os.close(tmp_fd)
    try:
        if already_3857:
            _normalise_to_uint8_tif(src_path, tmp_path)
        else:
            _reproject_to_file(src_path, tmp_path, target_crs, tile_size)
        _build_cog_from_file(tmp_path, dst_path, tile_size)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _reproject_to_file(src_path: str, dst_path: str,
                        target_crs, tile_size: int) -> None:
    """Reproject + stretch to uint8, write intermediate GeoTIFF."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject as warp_reproject

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        n_bands = min(src.count, 3)
        meta = {
            "driver": "GTiff", "dtype": "uint8",
            "crs": target_crs, "transform": transform,
            "width": width, "height": height, "count": n_bands,
        }
        with rasterio.open(dst_path, "w", **meta) as dst:
            for b in range(1, n_bands + 1):
                raw = src.read(b)
                if raw.dtype != np.uint8:
                    mask = raw > 0
                    if mask.any():
                        p2  = float(np.nanpercentile(raw[mask], 2))
                        p98 = float(np.nanpercentile(raw[mask], 98))
                    else:
                        p2, p98 = 0.0, 1.0
                    raw = np.clip((raw.astype(np.float32) - p2) /
                                   max(p98 - p2, 1.0), 0.0, 1.0)
                    raw = (raw * 255).astype(np.uint8)
                dest = np.zeros((height, width), dtype=np.uint8)
                warp_reproject(
                    source=raw, destination=dest,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
                dst.write(dest, b)


def _normalise_to_uint8_tif(src_path: str, dst_path: str) -> None:
    """Stretch bands to uint8 RGB if needed, write plain GeoTIFF."""
    import rasterio
    with rasterio.open(src_path) as src:
        n_bands = min(src.count, 3)
        meta = src.meta.copy()
        meta.update({"driver": "GTiff", "dtype": "uint8", "count": n_bands})
        with rasterio.open(dst_path, "w", **meta) as dst:
            for b in range(1, n_bands + 1):
                raw = src.read(b)
                if raw.dtype != np.uint8:
                    mask = raw > 0
                    if mask.any():
                        p2  = float(np.nanpercentile(raw[mask], 2))
                        p98 = float(np.nanpercentile(raw[mask], 98))
                    else:
                        p2, p98 = 0.0, 1.0
                    raw = np.clip((raw.astype(np.float32) - p2) /
                                   max(p98 - p2, 1.0), 0.0, 1.0)
                    raw = (raw * 255).astype(np.uint8)
                dst.write(raw, b)


def _build_cog_from_file(src_path: str, dst_path: str, tile_size: int) -> None:
    """Add internal overviews to a uint8 GeoTIFF and write as COG."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.shutil import copy as rio_copy
    import tempfile, os, shutil

    tmp_fd, ov_path = tempfile.mkstemp(suffix="_ov.tif")
    os.close(tmp_fd)
    try:
        shutil.copy2(src_path, ov_path)
        with rasterio.open(ov_path, "r+") as ds:
            max_dim = max(ds.width, ds.height)
            factors, f = [], 2
            while max_dim // f > tile_size:
                factors.append(f); f *= 2
            if factors:
                ds.build_overviews(factors, Resampling.average)
                ds.update_tags(ns="rio_overview", resampling="average")
        rio_copy(
            ov_path, dst_path,
            driver="GTiff",
            copy_src_overviews=True,
            compress="LZW",
            predictor=2,
            tiled=True,
            blockxsize=tile_size,
            blockysize=tile_size,
            interleave="band",
            bigtiff="IF_SAFER",
        )
    finally:
        if os.path.exists(ov_path):
            os.unlink(ov_path)


# ---------------------------------------------------------------------------
# XYZ tile pyramid generation
# ---------------------------------------------------------------------------

def generate_xyz_tiles(cog_path: str, tiles_dir: Path,
                        tile_size: int = 256,
                        max_zoom: int = 22) -> dict:
    """
    Generate a local XYZ tile pyramid from a COG.

    Tries gdal2tiles first (fastest). Falls back to pure-rasterio + mercantile.
    Returns dict with bounds and zoom range for the manifest.
    """
    import subprocess, shutil
    tiles_dir.mkdir(parents=True, exist_ok=True)

    gdal2tiles = shutil.which("gdal2tiles") or shutil.which("gdal2tiles.py")
    if gdal2tiles:
        result = subprocess.run([
            gdal2tiles,
            "--zoom", f"0-{max_zoom}",
            "--tile-size", str(tile_size),
            "--processes", "4",
            "--webviewer", "none",
            "--resampling", "average",
            str(cog_path),
            str(tiles_dir),
        ], capture_output=True, text=True)
        if result.returncode == 0:
            return _read_tile_bounds(tiles_dir)
        else:
            print(f"    gdal2tiles failed, using pure-rasterio tiler")

    return _pure_rasterio_tile(cog_path, tiles_dir, tile_size, max_zoom)


def _read_tile_bounds(tiles_dir: Path) -> dict:
    """Scan generated tile dirs to infer zoom range."""
    tilemapresource = tiles_dir / "tilemapresource.xml"
    if tilemapresource.exists():
        import xml.etree.ElementTree as ET
        root = ET.parse(str(tilemapresource)).getroot()
        bb = root.find(".//BoundingBox")
        if bb is not None:
            return {
                "min_lon": float(bb.get("minx", -180)),
                "min_lat": float(bb.get("miny", -90)),
                "max_lon": float(bb.get("maxx",  180)),
                "max_lat": float(bb.get("maxy",   90)),
            }
    zoom_dirs = sorted([int(p.name) for p in tiles_dir.iterdir()
                         if p.is_dir() and p.name.isdigit()])
    return {
        "min_zoom": zoom_dirs[0]  if zoom_dirs else 0,
        "max_zoom": zoom_dirs[-1] if zoom_dirs else 18,
    }


def _pure_rasterio_tile(cog_path: str, tiles_dir: Path,
                         tile_size: int, max_zoom: int) -> dict:
    """
    Pure-Python XYZ tiler using rasterio + mercantile.
    Reads the COG window for each tile bbox, resamples to tile_size×tile_size,
    writes PNG. Only tiles covering the raster extent are written.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    import imageio.v2 as imageio

    try:
        import mercantile
    except ImportError:
        print("    ERROR: mercantile not found. Install with:")
        print("      conda install -c conda-forge mercantile")
        raise

    with rasterio.open(cog_path) as src:
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        w_lon, s_lat, e_lon, n_lat = bounds_wgs84

        min_zoom   = 10  # fixed minimum (entire site visible)
        actual_max = max_zoom

        print(f"    Zoom range: {min_zoom} – {actual_max}")
        print(f"    Bounds: {w_lon:.4f},{s_lat:.4f} → {e_lon:.4f},{n_lat:.4f}")

        n_written = 0
        for zoom in range(min_zoom, actual_max + 1):
            for tile in mercantile.tiles(w_lon, s_lat, e_lon, n_lat, zooms=zoom):
                tb = mercantile.bounds(tile)
                from rasterio.warp import transform_bounds as tb_transform
                try:
                    dst_bounds = tb_transform("EPSG:4326", src.crs,
                                              tb.west, tb.south, tb.east, tb.north)
                    window = src.window(*dst_bounds)
                    win    = window.round_shape().round_offsets()
                    if win.width < 1 or win.height < 1:
                        continue
                    # Read RGB + build alpha mask so OM edges are transparent not black
                    n_bands = min(src.count, 3)
                    data = src.read(
                        list(range(1, n_bands + 1)),
                        window=win,
                        out_shape=(n_bands, tile_size, tile_size),
                        resampling=Resampling.bilinear,
                        boundless=True, fill_value=0,
                    )
                    # Alpha = 0 wherever all bands are 0 (outside raster / nodata)
                    alpha = np.where(np.all(data == 0, axis=0), 0, 255).astype(np.uint8)
                except Exception:
                    continue

                img_rgb = np.moveaxis(data, 0, -1)           # (H,W,3)
                if img_rgb.dtype != np.uint8:
                    img_rgb = _to_uint8_rgb(img_rgb)
                # Write RGBA PNG — transparent outside OM, opaque inside
                img_rgba = np.dstack([img_rgb, alpha])        # (H,W,4)

                out_dir = tiles_dir / str(zoom) / str(tile.x)
                out_dir.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(str(out_dir / f"{tile.y}.png"), img_rgba)
                n_written += 1

        print(f"    Wrote {n_written} tiles")
        return {
            "min_lon": w_lon, "min_lat": s_lat,
            "max_lon": e_lon, "max_lat": n_lat,
            "min_zoom": min_zoom, "max_zoom": actual_max,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Info file generation — comprehensive report of all OMs and processing
# ---------------------------------------------------------------------------

def generate_info_file(config, om_tile_entries, underlay_om_id):
    """
    Generate a comprehensive txt report of all OMs, COG files, tile stats,
    per-zoom quality, before/after comparison, and processing details.
    """
    import datetime, math, os

    viewer_dir = Path(config["viewer_dir"])
    cogs_dir   = viewer_dir / "cogs"
    tiles_dir  = viewer_dir / "tiles"
    info_path  = viewer_dir / "04a_OM_INFO.txt"

    # ── helpers ───────────────────────────────────────────────────────────────
    WEB_MERCATOR_C = 156543.03392  # metres at equator per pixel at zoom 0

    def zoom_gsd(zoom, lat_deg):
        """Ground resolution (m/px) of a Web Mercator tile at this zoom + lat."""
        return WEB_MERCATOR_C * math.cos(math.radians(lat_deg)) / (2 ** zoom)

    def tile_count_at_zoom(bounds, zoom):
        """Number of XYZ tiles that cover a lat/lon bounding box at this zoom."""
        w, s, e, n = bounds["w"], bounds["s"], bounds["e"], bounds["n"]
        def lon_to_tile_x(lon, z): return int((lon + 180) / 360 * 2**z)
        def lat_to_tile_y(lat, z): return int((1 - math.log(math.tan(math.radians(lat)) +
                                    1 / math.cos(math.radians(lat))) / math.pi) / 2 * 2**z)
        x0 = lon_to_tile_x(w, zoom);  x1 = lon_to_tile_x(e, zoom)
        y0 = lat_to_tile_y(n, zoom);  y1 = lat_to_tile_y(s, zoom)
        return max(1, (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1))

    def actual_tile_count(om_tile_dir):
        """Count .png tiles actually written to disk."""
        if not om_tile_dir.exists():
            return 0
        return sum(1 for _ in om_tile_dir.rglob("*.png"))

    def tiles_per_zoom(om_tile_dir):
        """Dict of zoom -> tile count from actual files on disk."""
        counts = {}
        if not om_tile_dir.exists():
            return counts
        for z_dir in sorted(om_tile_dir.iterdir()):
            if z_dir.is_dir() and z_dir.name.isdigit():
                counts[int(z_dir.name)] = sum(1 for _ in z_dir.rglob("*.png"))
        return counts

    def dir_size_bytes(path):
        """Total bytes of all files under path."""
        total = 0
        if Path(path).exists():
            for f in Path(path).rglob("*"):
                if f.is_file():
                    try: total += f.stat().st_size
                    except OSError: pass
        return total

    def fmt_bytes(n):
        for unit in ("B","KB","MB","GB","TB"):
            if n < 1024 or unit == "TB":
                return f"{n:.1f} {unit}"
            n /= 1024

    def quality_label(screen_gsd_m):
        """Human-readable quality description based on GSD visible on screen."""
        if screen_gsd_m < 0.02:  return "Ultra-high detail — sub-2cm (better than native)"
        if screen_gsd_m < 0.06:  return "Native resolution — 2–6cm (individual leaves visible)"
        if screen_gsd_m < 0.15:  return "High detail — 6–15cm (crown texture clearly visible)"
        if screen_gsd_m < 0.40:  return "Good detail — 15–40cm (individual crowns clear)"
        if screen_gsd_m < 1.00:  return "Site overview — 40–100cm (crowns distinguishable)"
        if screen_gsd_m < 5.00:  return "Regional — 1–5m (canopy patches visible)"
        return                          "Wide overview — >5m (site footprint visible)"

    def tiles_visible_on_screen(zoom, screen_w_px=1920, screen_h_px=1080, tile_size=256):
        """How many tiles fit in a typical 1920×1080 viewport at this zoom."""
        cols = math.ceil(screen_w_px / tile_size) + 1
        rows = math.ceil(screen_h_px / tile_size) + 1
        return cols * rows

    # ── global totals ─────────────────────────────────────────────────────────
    total_tiles_all_oms   = 0
    total_cog_size_bytes  = 0
    total_tile_size_bytes = 0

    for entry in om_tile_entries:
        om_id = entry["om_id"]
        stem  = entry["stem"]
        cog_path      = cogs_dir / f"OM{om_id:02d}_{stem}.tif"
        om_tile_dir   = tiles_dir / f"OM{om_id:02d}_{stem}"
        total_cog_size_bytes  += cog_path.stat().st_size  if cog_path.exists()  else 0
        total_tile_size_bytes += dir_size_bytes(om_tile_dir)
        total_tiles_all_oms   += actual_tile_count(om_tile_dir)

    # ── begin writing ─────────────────────────────────────────────────────────
    lines = []
    SEP  = "=" * 80
    SEP2 = "-" * 80

    lines += [SEP,
              "TREE-CROWN PHENOLOGY PIPELINE — STEP 04a: COG TILING & MANIFEST",
              SEP, ""]

    lines += [f"Generated          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              f"Pipeline run       : {config.get('run_name', 'unknown')}",
              f"Tracking directory : {config.get('tracking_dir', 'unknown')}",
              f"Viewer directory   : {viewer_dir}", ""]

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    lines += [SEP, "SUMMARY", SEP, ""]
    lines += [f"  OMs processed           : {len(om_tile_entries)}",
              f"  Default underlay OM     : OM{underlay_om_id:02d}",
              f"  Tile size               : 256 × 256 px (Web Mercator standard)",
              f"  Compression             : LZW (lossless)",
              f"  Output CRS              : EPSG:3857 (Web Mercator)",
              ""]
    lines += [f"  Total COG storage       : {fmt_bytes(total_cog_size_bytes)}",
              f"  Total tile storage      : {fmt_bytes(total_tile_size_bytes)}",
              f"  Total tiles on disk     : {total_tiles_all_oms:,}",
              f"  Combined output size    : {fmt_bytes(total_cog_size_bytes + total_tile_size_bytes)}",
              ""]

    # ── PER-OM DETAIL ─────────────────────────────────────────────────────────
    lines += [SEP, "PER-OM DETAIL", SEP]

    for entry in om_tile_entries:
        om_id    = entry["om_id"]
        stem     = entry["stem"]
        bounds   = entry["bounds"]
        min_zoom = entry["min_zoom"]
        max_zoom = entry["max_zoom"]
        tile_url = entry["tile_url"]

        cog_path    = cogs_dir / f"OM{om_id:02d}_{stem}.tif"
        om_tile_dir = tiles_dir / f"OM{om_id:02d}_{stem}"

        # COG file stats
        cog_size    = cog_path.stat().st_size if cog_path.exists() else 0
        tile_size_b = dir_size_bytes(om_tile_dir)
        n_tiles     = actual_tile_count(om_tile_dir)
        zoom_counts = tiles_per_zoom(om_tile_dir)

        # Raster intrinsics from COG
        src_path = str(cog_path) if cog_path.exists() else None
        cog_width = cog_height = cog_bands = cog_dtype = cog_nodata = None
        src_gsd = None; src_crs_str = "unknown"
        original_size_bytes = 0

        if src_path:
            try:
                import rasterio
                with rasterio.open(src_path) as src:
                    cog_width  = src.width
                    cog_height = src.height
                    cog_bands  = src.count
                    cog_dtype  = src.dtypes[0]
                    cog_nodata = src.nodata
                    src_gsd    = abs(src.transform.a)
                    src_crs_str = src.crs.to_string() if src.crs else "unknown"
            except Exception:
                pass

        # Centroid lat for GSD calculations
        mid_lat = (bounds["n"] + bounds["s"]) / 2

        # Geographic extent
        width_deg  = bounds["e"] - bounds["w"]
        height_deg = bounds["n"] - bounds["s"]
        width_m    = width_deg  * 111320 * math.cos(math.radians(mid_lat))
        height_m   = height_deg * 110540

        lines += ["", f"OM{om_id:02d} — {stem}", SEP2]

        # ── Identity & Paths ──────────────────────────────────────────────────
        lines += ["  IDENTITY",
                  f"    OM ID         : {om_id}",
                  f"    Stem          : {stem}",
                  f"    Tile URL      : {tile_url}",
                  ""]

        # ── Geographic Info ───────────────────────────────────────────────────
        lines += ["  GEOGRAPHIC COVERAGE",
                  f"    West  (lon)   : {bounds['w']:.6f}°",
                  f"    South (lat)   : {bounds['s']:.6f}°",
                  f"    East  (lon)   : {bounds['e']:.6f}°",
                  f"    North (lat)   : {bounds['n']:.6f}°",
                  f"    Centre lat    : {mid_lat:.4f}°",
                  f"    Width         : {width_deg:.6f}° ≈ {width_m:.0f} m",
                  f"    Height        : {height_deg:.6f}° ≈ {height_m:.0f} m",
                  f"    Area          : ≈ {width_m * height_m / 1e6:.3f} km²",
                  ""]

        # ── COG File ──────────────────────────────────────────────────────────
        lines += ["  COG FILE"]
        lines += [f"    Path          : {cog_path}"]
        lines += [f"    Size on disk  : {fmt_bytes(cog_size)}"]
        if cog_width:
            pixels_total = cog_width * cog_height
            lines += [f"    Dimensions    : {cog_width:,} × {cog_height:,} px  ({pixels_total/1e6:.1f} Mpx)",
                      f"    Bands         : {cog_bands} (RGB)",
                      f"    Data type     : {cog_dtype}",
                      f"    Nodata        : {cog_nodata}",
                      f"    CRS           : {src_crs_str}"]
            if src_gsd:
                lines += [f"    Native GSD    : {src_gsd*100:.2f} cm/px  ({src_gsd:.4f} m/px)"]
                raw_size = cog_width * cog_height * cog_bands  # uint8 uncompressed
                ratio    = raw_size / cog_size if cog_size else 0
                lines += [f"    Raw (uncomp.) : {fmt_bytes(raw_size)}",
                          f"    Compression   : {ratio:.1f}× (LZW lossless)"]
        lines += [""]

        # ── Tile Pyramid ──────────────────────────────────────────────────────
        lines += ["  TILE PYRAMID",
                  f"    Directory     : {om_tile_dir}",
                  f"    Zoom range    : z{min_zoom} → z{max_zoom}  ({max_zoom - min_zoom + 1} levels)",
                  f"    Total tiles   : {n_tiles:,}",
                  f"    Total size    : {fmt_bytes(tile_size_b)}",
                  f"    Avg tile size : {fmt_bytes(tile_size_b / n_tiles) if n_tiles else 'N/A'}",
                  ""]

        # ── Per-Zoom Table ────────────────────────────────────────────────────
        lines += ["  PER-ZOOM BREAKDOWN",
                  f"    {'Zoom':<6} {'Tiles on disk':>14} {'Tiles in viewport':>18} "
                  f"{'GSD (cm/px)':>12} {'Tile res (px/m)':>15} {'Tile covers (m)':>16} {'Quality / Use':<40}"]
        lines += [f"    {SEP2}"]

        viewport_tiles = tiles_visible_on_screen  # function ref
        for z in range(min_zoom, max_zoom + 1):
            actual   = zoom_counts.get(z, 0)
            in_view  = viewport_tiles(z)
            gsd_m    = zoom_gsd(z, mid_lat)
            gsd_cm   = gsd_m * 100
            # Tile resolution: pixels per metre (inverse of GSD)
            px_per_m = 1.0 / gsd_m if gsd_m > 0 else 0
            # Each 256×256 tile covers this many metres on the ground
            tile_m   = gsd_m * 256
            label    = quality_label(gsd_m)
            lines += [f"    z{z:<5} {actual:>14,} {in_view:>18,} "
                      f"{gsd_cm:>11.2f}  {px_per_m:>14.1f}  {tile_m:>14.1f}m  {label}"]

        lines += [""]

        # ── Tile Size vs Coverage ─────────────────────────────────────────────
        lines += ["  TILE SIZE ANALYSIS",
                  f"    Each tile     : 256 × 256 px (PNG, RGBA with alpha)"]
        for z in [min_zoom, (min_zoom + max_zoom) // 2, max_zoom]:
            gsd_m  = zoom_gsd(z, mid_lat)
            tile_m = gsd_m * 256
            lines += [f"    z{z}  tile    : covers {tile_m:.1f} m × {tile_m:.1f} m on ground  "
                      f"({gsd_m*100:.2f} cm/px)"]
        lines += [""]

        # ── Before / After Comparison ─────────────────────────────────────────
        # Estimate original raster size (raw uncompressed uint8 RGB)
        if cog_width and src_gsd:
            raw_bytes   = cog_width * cog_height * 3  # 3 bands uint8
            ratio_cog   = raw_bytes / cog_size if cog_size else 0
            ratio_tiles = raw_bytes / tile_size_b if tile_size_b else 0
            lines += ["  BEFORE → AFTER COMPARISON",
                      f"    Raw input (est. uncomp. uint8) : {fmt_bytes(raw_bytes)}",
                      f"    COG output                     : {fmt_bytes(cog_size)}"
                      f"  ({100*(1 - cog_size/raw_bytes):.0f}% smaller)" if cog_size else "",
                      f"    Tile pyramid output            : {fmt_bytes(tile_size_b)}"
                      f"  ({100*(1 - tile_size_b/raw_bytes):.0f}% smaller)" if tile_size_b else "",
                      f"    COG compression ratio          : {ratio_cog:.1f}×",
                      f"    Largest single file (COG)      : {fmt_bytes(cog_size)}",
                      f"    Largest single file (tile)     : {fmt_bytes(tile_size_b // n_tiles * 3) if n_tiles else 'N/A'}"
                      f"  (est. largest tile ≈ 3× avg)",
                      f"    Zoom levels saved by adaptive  : {22 - max_zoom} level(s) vs hardcoded z22"
                      f" ({sum(tile_count_at_zoom(bounds, z) for z in range(max_zoom+1, 23)):,} tiles avoided)",
                      f"    Format change                  : monolithic GeoTIFF → {n_tiles:,} small PNGs",
                      f"    Max file delivered to browser  : {fmt_bytes(tile_size_b // n_tiles * 3) if n_tiles else 'N/A'}"
                      f" per tile (not {fmt_bytes(raw_bytes)} for full raster)",
                      ""]

    # ── GLOBAL BEFORE / AFTER ─────────────────────────────────────────────────
    lines += [SEP, "GLOBAL BEFORE → AFTER (ALL OMs COMBINED)", SEP, ""]

    n_oms = len(om_tile_entries)
    # Estimate total raw (uncompressed uint8 RGB) across all OMs from COG dims
    total_raw = 0
    for entry in om_tile_entries:
        om_id = entry["om_id"]; stem = entry["stem"]
        cp = cogs_dir / f"OM{om_id:02d}_{stem}.tif"
        try:
            import rasterio
            with rasterio.open(str(cp)) as s:
                total_raw += s.width * s.height * 3
        except Exception:
            pass

    lines += [f"  {'Metric':<40} {'Value':>20}",
              f"  {'-'*60}",
              f"  {'OMs processed':<40} {n_oms:>20}",
              f"  {'Raw input size (est.)':<40} {fmt_bytes(total_raw):>20}",
              f"  {'Total COG size':<40} {fmt_bytes(total_cog_size_bytes):>20}",
              f"  {'Total tile size':<40} {fmt_bytes(total_tile_size_bytes):>20}",
              f"  {'Total output size (COG + tiles)':<40} {fmt_bytes(total_cog_size_bytes + total_tile_size_bytes):>20}",
              f"  {'Total tiles on disk':<40} {total_tiles_all_oms:>20,}",
              f"  {'Avg tiles per OM':<40} {total_tiles_all_oms // n_oms if n_oms else 0:>20,}",
              ""]

    if total_raw:
        saved_vs_raw = total_raw - total_cog_size_bytes
        pct_saved    = 100 * saved_vs_raw / total_raw
        lines += [f"  Storage saved by COG (vs raw)  : {fmt_bytes(saved_vs_raw)}  ({pct_saved:.0f}% reduction)",
                  f"  Largest file a browser ever    ",
                  f"    fetches at once              : ~50–100 KB (one tile PNG)",
                  f"    vs without tiling            : {fmt_bytes(total_raw // n_oms if n_oms else 0)} (entire OM!)",
                  f"  Adaptive zoom saved approx.    : {22 - sum(e['max_zoom'] for e in om_tile_entries) // n_oms if n_oms else 0} zoom level(s) on average vs hardcoded z22",
                  ""]

    # ── PROCESSING DETAILS ────────────────────────────────────────────────────
    lines += [SEP, "PROCESSING DETAILS — WHAT STEP 04a DID", SEP, ""]
    lines += [
        "  1. COG GENERATION",
        "     Input  : Raw ortho GeoTIFF (original projection, float32 or uint16)",
        "     Output : Cloud-Optimised GeoTIFF (EPSG:3857, uint8, LZW compressed)",
        "     Steps :",
        "       a) Reproject to EPSG:3857 (Web Mercator) if not already",
        "       b) Stretch each band to uint8 (p2–p98 percentile normalisation)",
        "       c) Build internal overviews (2×, 4×, 8×, 16×, 32×, …)",
        "       d) Write tiled COG with COPY_SRC_OVERVIEWS=YES, LZW+predictor2,",
        "          blockxsize=256, blockysize=256, interleave=band, bigtiff=IF_SAFER",
        "     Result: Single self-contained file with embedded multi-scale pyramid.",
        "             Efficient reads at any zoom without decoding full raster.",
        "",
        "  2. ADAPTIVE ZOOM CALCULATION",
        "     Input  : Native GSD (from raster metadata), centre latitude",
        "     Result : max_zoom set by --max-cog-zoom argument (default: 22).",
        "",
        "  3. XYZ TILE PYRAMID GENERATION",
        "     Input  : COG (EPSG:3857, uint8 RGB)",
        "     Tool   : gdal2tiles (primary) or pure-rasterio+mercantile (fallback)",
        "     Output : 256×256 RGBA PNG tiles at z{min} → z{max} per OM",
        "              Alpha=0 outside raster (transparent edges, not black bars).",
        "              One directory per OM: tiles/OM{N}_{stem}/{z}/{x}/{y}.png",
        "",
        "  4. TILE MANIFEST",
        "     Output : tile_manifest.json (consumed by Step 04b viewer)",
        "              Per-OM: bounds, min_zoom, max_zoom, tile_url pattern",
        "",
        "  5. INFO FILE",
        "     Output : 04a_OM_INFO.txt (this file)",
        "              Auto-generated with per-OM stats, tile counts,",
        "              zoom quality table, and before/after comparison.",
        ""]

    # ── ZOOM REFERENCE TABLE ──────────────────────────────────────────────────
    lines += [SEP, "ZOOM LEVEL REFERENCE (at your site latitude)", SEP, ""]
    avg_lat = sum((e["bounds"]["n"] + e["bounds"]["s"]) / 2
                  for e in om_tile_entries) / len(om_tile_entries) if om_tile_entries else 28.5
    lines += [f"  Reference latitude: {avg_lat:.2f}°", ""]
    lines += [f"  {'Zoom':<6} {'GSD (cm/px)':>12} {'Tile res (px/m)':>15} {'Tile covers (m)':>16} "
              f"{'Tiles in 1080p viewport':>24} {'Typical use':<35}"]
    lines += [f"  {'-'*110}"]
    for z in range(10, 24):
        gsd_m  = zoom_gsd(z, avg_lat)
        px_per_m = 1.0 / gsd_m if gsd_m > 0 else 0
        tile_m = gsd_m * 256
        in_vp  = tiles_visible_on_screen(z)
        use = (
            "Entire site footprint"       if z <= 13 else
            "Site overview (all crowns)"  if z <= 15 else
            "Crown level (individual)"    if z <= 18 else
            "Native drone resolution"     if z <= 21 else
            "Over-zoom (tile stretched)"
        )
        lines += [f"  z{z:<5} {gsd_m*100:>12.2f} {px_per_m:>14.1f} {tile_m:>14.1f}m {in_vp:>24,}  {use}"]
    lines += [""]

    # ── NEXT STEPS ────────────────────────────────────────────────────────────
    lines += [SEP, "NEXT STEP", SEP, "",
              "  python src/pipeline/04b_interactive_viz.py --config pipeline_config.json",
              "  cd output/my_sit_run/04_viewer && python -m http.server 8000",
              "  Open: http://localhost:8000/index.html", ""]

    lines += [SEP, "END OF INFO", SEP]

    content = "\n".join(lines)
    info_path.write_text(content, encoding="utf-8")
    print(f"\nInfo file written: {info_path}")
    return info_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline Step 4a: COG generation + XYZ tile pyramid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--underlay-om", default="last",
                        help="Which OM to use as default map underlay: 'first', 'last', or N")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--max-cog-zoom", type=int, default=22)
    parser.add_argument("--force-regen-cogs", action="store_true",
                        help="Rebuild COGs and tiles even if they already exist")
    parser.add_argument("--skip-if-done", action="store_true",
                        help="Skip if tile_manifest.json already exists")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    config     = load_config(config_path)
    viewer_dir = Path(config["viewer_dir"])
    cogs_dir   = viewer_dir / "cogs"
    tiles_dir  = viewer_dir / "tiles"
    manifest_path = viewer_dir / "tile_manifest.json"

    if args.skip_if_done and manifest_path.exists():
        print(f"[SKIP] tile_manifest.json already exists: {manifest_path}")
        return 0

    viewer_dir.mkdir(parents=True, exist_ok=True)
    cogs_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    import rasterio
    from rasterio.warp import transform_bounds

    pairs, om_stems = build_pairs_and_om_stems(config)
    num_oms = len(pairs)

    # Resolve underlay OM index
    if args.underlay_om == "last":
        underlay_om_id = num_oms
    elif args.underlay_om == "first":
        underlay_om_id = 1
    else:
        underlay_om_id = int(args.underlay_om)
    underlay_om_id = max(1, min(underlay_om_id, num_oms))

    print(f"Generating COGs and XYZ tiles for {num_oms} OMs \u2026")
    om_tile_entries = []

    for om_id, (_gpkg, ortho_path, stem) in enumerate(pairs, start=1):
        cog_path    = cogs_dir  / f"OM{om_id:02d}_{stem}.tif"
        om_tile_dir = tiles_dir / f"OM{om_id:02d}_{stem}"

        needs_cog   = args.force_regen_cogs or not cog_path.exists()
        needs_tiles = args.force_regen_cogs or not om_tile_dir.exists() \
                      or not any(om_tile_dir.iterdir())

        print(f"  OM{om_id:02d} \u2014 {stem}")

        if needs_cog:
            print(f"    Building COG \u2026")
            try:
                generate_cog(str(ortho_path), str(cog_path), tile_size=args.tile_size)
            except Exception as e:
                print(f"    WARNING: COG generation failed: {e}")
        else:
            print(f"    COG exists \u2014 skipping")

        if needs_tiles:
            print(f"    Building XYZ tile pyramid \u2026")
            try:
                src_for_tiles = str(cog_path) if cog_path.exists() else str(ortho_path)
                tile_bounds = generate_xyz_tiles(
                    src_for_tiles, om_tile_dir,
                    tile_size=args.tile_size,
                    max_zoom=args.max_cog_zoom,
                )
            except Exception as e:
                print(f"    WARNING: tile generation failed: {e}")
                import traceback; traceback.print_exc()
                tile_bounds = {}
        else:
            print(f"    Tiles exist \u2014 skipping")
            tile_bounds = _read_tile_bounds(om_tile_dir)

        # Relative tile URL (served from viewer_dir root)
        tile_url = f"tiles/OM{om_id:02d}_{stem}/{{z}}/{{x}}/{{y}}.png"

        # Geographic bounds for JS
        try:
            src_bounds = str(cog_path) if cog_path.exists() else str(ortho_path)
            with rasterio.open(src_bounds) as src:
                w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        except Exception:
            w = tile_bounds.get("min_lon", -180)
            s = tile_bounds.get("min_lat", -90)
            e = tile_bounds.get("max_lon",  180)
            n = tile_bounds.get("max_lat",   90)

        om_tile_entries.append({
            "om_id":    int(om_id),
            "stem":     stem,
            "tile_url": tile_url,
            "bounds":   {"w": w, "s": s, "e": e, "n": n},
            "min_zoom": int(tile_bounds.get("min_zoom", 10)),
            "max_zoom": int(tile_bounds.get("max_zoom", 20)),
        })

    # Write tile manifest consumed by 04b
    manifest = {
        "underlay_om_id": int(underlay_om_id),
        "num_oms":        int(num_oms),
        "oms":            om_tile_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote tile manifest: {manifest_path}")

    # Update pipeline config
    config["tile_manifest"] = str(manifest_path)
    config["underlay_om_id"] = int(underlay_om_id)
    if "04a_cog_tiling" not in config.get("steps_completed", []):
        config["steps_completed"].append("04a_cog_tiling")
    save_config(config, config_path)
    print(f"Config updated: {config_path}")

    print(f"\nStep 4a complete. Run Step 4b next:")
    print(f"  python src/pipeline/04b_interactive_viz.py --config {config_path}")
    
    # Generate comprehensive info file
    generate_info_file(config, om_tile_entries, underlay_om_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
