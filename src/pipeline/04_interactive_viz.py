#!/usr/bin/env python3
"""
Pipeline Step 4: Interactive Viewer Generation

Generates a standalone HTML viewer (Leaflet-based) showing:
- Base orthomosaic as raster underlay
- Consensus crown outlines in pixel coordinates
- Per-crown crop image strips across all orthomosaics
- Click-to-explore interface

Also generates a static matplotlib overview of phenology results (deciduous score
distribution and phenophase timelines for a sample of trees).

Requires: dpm-tracking conda environment

Usage:
    python 04_interactive_viz.py --config /path/to/pipeline_config.json \\
        [--underlay-om last|first|N] \\
        [--max-base-px 2600] \\
        [--force-regen-crops] \\
        [--skip-if-done]

Reads:
    <output_dir>/pipeline_config.json
    <tracking_dir>/consensus_crowns_complete_all.gpkg   (deduped)
    <phenology_dir>/leafshed_tree_scores.csv  (optional, for overview plot)
    <om_dir>/{stem}.tif

Writes:
    <viewer_dir>/index.html
    <viewer_dir>/base_underlay_OM{N}_{stem}.png
    <viewer_dir>/crowns_underlay_OM{N}_{stem}_pixels.geojson
    <viewer_dir>/manifest.json
    <viewer_dir>/crops/crown_{NNNN}/OM{N}_{stem}.png
    <viewer_dir>/phenology_overview.png   (if phenology data available)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# On Windows, GDAL (via geopandas/rasterio) loads libexpat.dll which conflicts
# with Python's pyexpat.pyd when imported first.  Force-import matplotlib
# (and pyexpat through its dependency chain) before any geo-stack import.
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; no display required
import matplotlib.pyplot as plt  # noqa: F401 – side-effect import

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def save_config(config: dict, config_path: Path) -> None:
    config_path.write_text(json.dumps(config, indent=2))


def setup_app_dir(project_root: Path) -> None:
    app_dir = str(project_root / "src" / "flask_app_tracking")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)


def build_pairs_and_om_stems(config: dict) -> Tuple[List[Tuple[str, str, str]], Dict[int, str]]:
    crowns_dir = Path(config["crowns_dir"])
    om_dir = Path(config["om_dir"])
    pairs = []
    om_stems = {}
    for i, (gpkg_raw, tif_raw, stem) in enumerate(config["pairs"], 1):
        gpkg_from_config = Path(gpkg_raw)
        tif_from_config = Path(tif_raw)
        if gpkg_from_config.exists():
            gpkg = str(gpkg_from_config)
        else:
            gpkg = str(crowns_dir / f"{stem}_multithreshold.gpkg")
        if tif_from_config.exists():
            tif = str(tif_from_config)
        else:
            tif = str(om_dir / f"{stem}.tif")
        pairs.append((gpkg, tif, stem))
        om_stems[i] = stem
    return pairs, om_stems


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError("Expected HxWxC array")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo = float(np.nanpercentile(arr, 2))
    hi = float(np.nanpercentile(arr, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
        hi = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
        if hi <= lo:
            hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _largest_polygon(geom):
    from shapely.geometry import Polygon, MultiPolygon
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        parts = list(geom.geoms)
        if not parts:
            return None
        return max(parts, key=lambda p: p.area)
    return None


def _poly_to_leaflet_coords(poly, base_transform, img_h: int) -> List[List[float]]:
    import rasterio
    simple = poly.simplify(0.5)
    xs, ys = simple.exterior.xy
    rows, cols = rasterio.transform.rowcol(base_transform, xs, ys, op=float)
    coords = [[float(c), float(img_h - r)] for c, r in zip(cols, rows)]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _expand_bounds(bounds: Tuple[float, float, float, float], pad_fraction: float = 0.25) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    pad = max(width, height) * pad_fraction
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)


def generate_phenology_overview(
    scores_csv: Path,
    phases_csv: Path,
    om_stems: Dict[int, str],
    out_png: Path,
) -> None:
    """Generate static matplotlib overview of phenology results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        scores_df = pd.read_csv(scores_csv)
        scores_df["deciduous_score"] = pd.to_numeric(scores_df.get("deciduous_score"), errors="coerce")
        scores_df = scores_df[np.isfinite(scores_df["deciduous_score"])].copy()

        threshold = 0.85
        n_decid = int((scores_df["deciduous_score"] >= threshold).sum())
        n_total = len(scores_df)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: histogram
        axes[0].hist(scores_df["deciduous_score"].values, bins=30, color="#4e79a7", alpha=0.85)
        axes[0].axvline(threshold, color="#d62728", lw=2, alpha=0.9, label=f"thr={threshold}")
        axes[0].set_title(f"Deciduous score | n={n_total} | deciduous={n_decid} ({n_decid/max(n_total,1):.1%})")
        axes[0].set_xlabel("Deciduous score")
        axes[0].set_ylabel("# crowns")
        axes[0].legend()

        # Right: amplitude scatter
        xcol = "A_veg" if "A_veg" in scores_df.columns else None
        ycol = "A_gcc" if "A_gcc" in scores_df.columns else None
        if xcol and ycol:
            is_decid = scores_df["deciduous_score"] >= threshold
            axes[1].scatter(
                pd.to_numeric(scores_df[xcol], errors="coerce"),
                pd.to_numeric(scores_df[ycol], errors="coerce"),
                s=10, alpha=0.5,
                c=is_decid.map({True: "#d62728", False: "#2ca02c"}),
            )
            axes[1].set_xlabel(xcol)
            axes[1].set_ylabel(ycol)
            axes[1].set_title("Amplitude scatter (red=deciduous, green=evergreen)")
        else:
            axes[1].axis("off")

        fig.tight_layout()
        plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote phenology overview: {out_png}")
    except Exception as e:
        print(f"Warning: could not generate phenology overview: {e}")


def build_html(
    dataset_name: str,
    run_tag: str,
    base_png_name: str,
    geojson_path_name: str,
    underlay_om_id: int,
    underlay_stem: str,
    img_h: int,
    img_w: int,
    geojson_inline: str,
    manifest_inline: str,
) -> str:
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{dataset_name} consensus crown time series</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
        crossorigin=""/>
  <style>
    html, body {{ height: 100%; margin: 0; }}
    #root {{ height: 100%; display: flex; flex-direction: row; }}
    #map {{ flex: 1 1 auto; }}
    #panel {{ width: 480px; border-left: 1px solid #ddd; padding: 12px; overflow: auto;
              font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
    #panel h2 {{ margin: 0 0 6px 0; font-size: 16px; }}
    .hint {{ color: #555; font-size: 13px; margin-bottom: 10px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
    .item {{ border: 1px solid #eee; padding: 8px; border-radius: 6px; }}
    .meta {{ font-size: 12px; color: #333; margin-bottom: 6px; font-weight: bold; }}
    img {{ width: 100%; height: auto; display: block; background: #f6f6f6; border-radius: 3px; }}
    .no-data {{ color: #999; font-size: 12px; font-style: italic; padding: 8px; }}
    .crown-highlight {{ stroke: #ff4444; stroke-width: 3; }}
  </style>
</head>
<body>
  <div id="root">
    <div id="map"></div>
    <div id="panel">
      <h2>{dataset_name} &mdash; crown time series</h2>
      <div class="hint">Underlay: OM{underlay_om_id:02d} {underlay_stem}. Click a crown outline to view its patches across all orthomosaics.</div>
      <div id="content" class="grid"><div class="hint">Select a crown on the map.</div></div>
    </div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossorigin=""></script>
  <script>
    const BASE_IMAGE = '{base_png_name}';
    const GEOJSON = {geojson_inline};
    const MANIFEST = {manifest_inline};

    const map = L.map('map', {{
      crs: L.CRS.Simple,
      minZoom: -4,
      maxZoom: 2,
      zoomSnap: 0.25,
      attributionControl: false
    }});

    const imgW = {img_w};
    const imgH = {img_h};
    const bounds = [[0, 0], [imgH, imgW]];
    const imageOverlay = L.imageOverlay(BASE_IMAGE, bounds).addTo(map);
    map.fitBounds(bounds);

    let selectedLayer = null;

    const defaultStyle = {{ color: '#ffff00', weight: 1.5, fillOpacity: 0.05, fillColor: '#ffff00' }};
    const hoverStyle   = {{ color: '#ff8800', weight: 2.5, fillOpacity: 0.15, fillColor: '#ff8800' }};
    const selectedStyle = {{ color: '#ff2222', weight: 3, fillOpacity: 0.20, fillColor: '#ff2222' }};

    function showCrownPanel(feature) {{
      const panel = document.getElementById('content');
      const crownId = feature.properties.crown_label;
      const crownIndex = feature.properties.crown_index;
      const entry = MANIFEST.crowns && MANIFEST.crowns[crownIndex];
      if (!entry) {{
        panel.innerHTML = '<div class="hint">No crop data for this crown.</div>';
        return;
      }}
      let html = '';
      for (const item of entry.items) {{
        if (item.status === 'ok') {{
          html += `<div class="item"><div class="meta">OM${{String(item.om_id).padStart(2,'0')}} &mdash; ${{item.stem}}</div><img src="${{item.path}}" loading="lazy" /></div>`;
        }} else {{
          html += `<div class="item"><div class="meta">OM${{String(item.om_id).padStart(2,'0')}} &mdash; ${{item.stem}}</div><div class="no-data">No overlap / error</div></div>`;
        }}
      }}
      panel.innerHTML = `<div style="font-size:13px;margin-bottom:8px;color:#333;">Crown <b>${{crownId}}</b> (${{entry.items.filter(i=>i.status==='ok').length}}/${{entry.num_oms}} patches available)</div>` + html;
    }}

    const geoLayer = L.geoJSON(GEOJSON, {{
      style: defaultStyle,
      onEachFeature: function(feature, layer) {{
        layer.on({{
          mouseover: function(e) {{
            if (layer !== selectedLayer) layer.setStyle(hoverStyle);
            layer.bringToFront();
          }},
          mouseout: function(e) {{
            if (layer !== selectedLayer) layer.setStyle(defaultStyle);
          }},
          click: function(e) {{
            if (selectedLayer && selectedLayer !== layer) selectedLayer.setStyle(defaultStyle);
            selectedLayer = layer;
            layer.setStyle(selectedStyle);
            layer.bringToFront();
            showCrownPanel(feature);
          }}
        }});
      }}
    }}).addTo(map);
  </script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Step 4: Interactive Viewer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-threshold-tag", default="conf_0p45")
    parser.add_argument("--align-threshold-tag", default="conf_0p65")
    parser.add_argument("--align-method", default="pcc_tiled")
    parser.add_argument(
        "--underlay-om",
        default="last",
        help="Which OM to use as the map underlay: 'first', 'last', or a 1-based index (default: last)",
    )
    parser.add_argument("--max-base-px", type=int, default=2600,
                        help="Max dimension for base underlay PNG (default: 2600)")
    parser.add_argument("--force-regen-crops", action="store_true",
                        help="Regenerate crop PNGs even if they already exist")
    parser.add_argument("--skip-if-done", action="store_true",
                        help="Skip if index.html already exists")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    project_root = Path(config["project_root"])
    viewer_dir = Path(config["viewer_dir"])
    tracking_dir = Path(config["tracking_dir"])
    phenology_dir = Path(config["phenology_dir"])
    run_name = config.get("run_name", "pipeline")

    viewer_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = viewer_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    html_path = viewer_dir / "index.html"
    if args.skip_if_done and html_path.exists():
        print(f"[SKIP] Viewer already exists: {html_path}")
        return 0

    # --- Imports ---
    import geopandas as gpd
    import imageio.v2 as imageio
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    from shapely.affinity import translate
    from shapely.geometry import Polygon, MultiPolygon, box, mapping

    setup_app_dir(project_root)
    from tree_tracking import TreeTrackingGraph

    # --- Build pairs ---
    pairs, om_stems = build_pairs_and_om_stems(config)
    crowns_dir_path = Path(config["crowns_dir"])
    om_dir_path = Path(config["om_dir"])
    num_oms = len(pairs)

    # --- Load consensus crowns (always from consensus_gpkg, not tree-only) ---
    consensus_gpkg = Path(config.get("consensus_gpkg",
                                      tracking_dir / "consensus_crowns_complete_all.gpkg"))
    print(f"Using consensus crowns: {consensus_gpkg.name}")

    if not consensus_gpkg.exists():
        print(f"ERROR: consensus crowns not found: {consensus_gpkg}", file=sys.stderr)
        print("Run step 2 first (crown tracking).", file=sys.stderr)
        return 1

    crowns = gpd.read_file(str(consensus_gpkg))
    crowns = crowns[crowns.geometry.notnull() & ~crowns.geometry.is_empty].reset_index(drop=True)
    print(f"Loaded {len(crowns)} consensus crowns")

    # --- Initialize tracker (for alignment_shifts) ---
    print(f"\nInitializing tracker (alignment) for {num_oms} OMs...")
    tracker = TreeTrackingGraph(
        auto_discover=False,
        multithresh_dir=str(crowns_dir_path),
        ortho_dir=str(om_dir_path),
        output_dir=str(viewer_dir),
        simplify_tol=1.0,
        resize_factor=0.1,
        max_crowns_preview=200,
    )
    tracker.file_pairs = [(gpkg, tif) for gpkg, tif, _ in pairs]
    tracker.om_ids = list(range(1, num_oms + 1))
    tracker.base_threshold_tag = None

    # Load saved alignment shifts from step 2 config (ensures identical registration)
    saved_shifts_raw = config.get("alignment_shifts", {})
    saved_shifts = {int(k): (float(v[0]), float(v[1])) for k, v in saved_shifts_raw.items()} if saved_shifts_raw else {}

    if saved_shifts:
        print(f"  Using saved alignment shifts from step 2 ({len(saved_shifts)} OMs)")
        tracker.load_multithreshold_data(
            base_threshold_tag=args.base_threshold_tag,
            load_images=False,
            align=False,
        )
        # Inject saved shifts and apply to crown geometries
        tracker.alignment_shifts = saved_shifts
        from shapely.affinity import affine_transform as shapely_affine
        for om_id in tracker.om_ids:
            dx, dy = saved_shifts.get(om_id, (0.0, 0.0))
            if om_id == tracker.om_ids[0] or (dx == 0.0 and dy == 0.0):
                continue
            gdf = tracker.crowns_gdfs.get(om_id)
            if gdf is None or gdf.empty:
                continue
            params = (1.0, 0.0, 0.0, 1.0, dx, dy)
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].apply(
                lambda g: shapely_affine(g, params) if g is not None else g
            )
            tracker.crowns_gdfs[om_id] = gdf
            tracker.crown_attrs[om_id] = [
                tracker._compute_crown_attributes(row.geometry)
                for _, row in gdf.iterrows()
            ]
    else:
        print("  WARNING: No saved alignment shifts found in config, recomputing alignment")
        tracker.load_multithreshold_data(
            base_threshold_tag=args.base_threshold_tag,
            load_images=False,
            align=True,
            align_method=args.align_method,
            align_threshold_tag=args.align_threshold_tag,
        )
    print("Alignment ready.")

    # --- Determine underlay OM ---
    if args.underlay_om == "last":
        underlay_om_id = num_oms
    elif args.underlay_om == "first":
        underlay_om_id = 1
    else:
        underlay_om_id = int(args.underlay_om)
    underlay_om_id = max(1, min(underlay_om_id, num_oms))

    _underlay_gpkg, underlay_ortho_path, underlay_stem = pairs[underlay_om_id - 1]
    print(f"\nUnderlay: OM{underlay_om_id:02d} {underlay_stem}")

    # Derive a clean dataset name for display
    dataset_name = config.get("run_name", run_name)

    # --- 1) Generate base PNG ---
    base_png = viewer_dir / f"base_underlay_OM{underlay_om_id:02d}_{underlay_stem}.png"
    with rasterio.open(underlay_ortho_path) as src:
        scale = (max(src.width, src.height) / args.max_base_px
                 if max(src.width, src.height) > args.max_base_px else 1.0)
        out_w = int(round(src.width / scale))
        out_h = int(round(src.height / scale))
        data = src.read([1, 2, 3], out_shape=(3, out_h, out_w),
                        resampling=rasterio.enums.Resampling.bilinear)
        base_img = np.moveaxis(data, 0, -1)
        base_img = _to_uint8_rgb(base_img)
        base_transform = src.transform * rasterio.transform.Affine.scale(
            src.width / out_w, src.height / out_h
        )
        raster_crs = src.crs

    imageio.imwrite(str(base_png), base_img)
    img_h, img_w = base_img.shape[0], base_img.shape[1]
    print(f"Base image: {img_w}×{img_h} → {base_png.name}")

    # --- 2) Build crowns GeoJSON in pixel coordinates ---
    crowns_display = crowns.copy()
    if crowns_display.crs is None and raster_crs:
        crowns_display = crowns_display.set_crs(raster_crs, allow_override=True)
    elif raster_crs and crowns_display.crs != raster_crs:
        crowns_display = crowns_display.to_crs(raster_crs)

    # Shift aligned geoms back to underlay OM's raw coordinate space
    dx_u, dy_u = tracker.alignment_shifts.get(underlay_om_id, (0.0, 0.0))
    geoms_underlay_raw = [translate(g, xoff=-dx_u, yoff=-dy_u)
                          for g in crowns_display.geometry.tolist()]

    features = []
    for i, geom in enumerate(geoms_underlay_raw):
        poly = _largest_polygon(geom)
        if poly is None:
            continue
        try:
            ring = _poly_to_leaflet_coords(poly, base_transform, img_h)
        except Exception:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "crown_index": int(i),
                "crown_label": f"crown_{i:04d}",
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    crowns_geojson = {"type": "FeatureCollection", "features": features}
    geojson_path = (viewer_dir /
                    f"crowns_underlay_OM{underlay_om_id:02d}_{underlay_stem}_pixels.geojson")
    with open(geojson_path, "w") as f:
        json.dump(crowns_geojson, f)
    print(f"Crowns GeoJSON: {geojson_path.name} ({len(features)} features)")

    # --- 3) Export crop PNGs + build manifest ---
    manifest_path = viewer_dir / "manifest.json"
    aligned_geoms = crowns_display.geometry.tolist()
    aligned_crop_bounds = [_expand_bounds(geom.bounds) for geom in aligned_geoms]

    if manifest_path.exists() and not args.force_regen_crops:
        with open(manifest_path) as f:
            manifest_blob = json.load(f)
        manifest = manifest_blob.get("crowns", [])
        print(f"Manifest exists ({len(manifest)} crowns), skipping crop export. "
              f"Use --force-regen-crops to regenerate.")
    else:
        print(f"\nExporting crops for {len(aligned_geoms)} crowns × {num_oms} OMs...")
        manifest: List[Dict] = []
        for crown_index in range(len(aligned_geoms)):
            crown_id = f"crown_{crown_index:04d}"
            (crops_dir / crown_id).mkdir(parents=True, exist_ok=True)
            manifest.append({
                "crown_index": int(crown_index),
                "crown_id": crown_id,
                "num_oms": int(num_oms),
                "items": [],
            })

        for om_id, (_gpkg_path, ortho_path, stem) in enumerate(pairs, start=1):
            print(f"  OM{om_id:02d}: {stem}")
            dx, dy = tracker.alignment_shifts.get(om_id, (0.0, 0.0))
            with rasterio.open(ortho_path) as src:
                for crown_index, geom_aligned in enumerate(aligned_geoms):
                    crown_id = f"crown_{crown_index:04d}"
                    out_png = crops_dir / crown_id / f"OM{om_id:02d}_{stem}.png"
                    rel_png = out_png.relative_to(viewer_dir).as_posix()

                    if out_png.exists() and not args.force_regen_crops:
                        status = "exists"
                    else:
                        minx, miny, maxx, maxy = aligned_crop_bounds[crown_index]
                        geom_raw = translate(box(minx, miny, maxx, maxy), xoff=-dx, yoff=-dy)
                        try:
                            out_image, _ = rasterio_mask(
                                src, [mapping(geom_raw)], crop=True, filled=True, nodata=0
                            )
                            crop_rgb = np.moveaxis(out_image, 0, -1)
                            crop_uint8 = _to_uint8_rgb(crop_rgb)
                            imageio.imwrite(str(out_png), crop_uint8)
                            status = "ok"
                        except (ValueError, Exception):
                            status = "no_overlap_or_error"

                    manifest[crown_index]["items"].append({
                        "om_id": int(om_id),
                        "stem": stem,
                        "status": status,
                        "path": rel_png,
                    })

        n_ok = sum(
            1 for entry in manifest for item in entry["items"] if item["status"] in ("ok", "exists")
        )
        print(f"  Crops: {n_ok}/{len(aligned_geoms)*num_oms} available")

        manifest_blob = {
            "dataset": dataset_name,
            "run_tag": run_name,
            "num_oms": int(num_oms),
            "num_crowns": int(len(aligned_geoms)),
            "base_image": base_png.name,
            "crowns_geojson": geojson_path.name,
            "underlay_om_id": int(underlay_om_id),
            "underlay_stem": underlay_stem,
            "image_h": int(img_h),
            "image_w": int(img_w),
            "crowns": manifest,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_blob, f)
        print(f"Manifest: {manifest_path}")

    # Ensure manifest_blob is always defined
    if "manifest_blob" not in locals():
        with open(manifest_path) as f:
            manifest_blob = json.load(f)
        # Update underlay metadata in case it changed
        manifest_blob.update({
            "base_image": base_png.name,
            "crowns_geojson": geojson_path.name,
            "underlay_om_id": int(underlay_om_id),
            "underlay_stem": underlay_stem,
            "image_h": int(img_h),
            "image_w": int(img_w),
        })
        with open(manifest_path, "w") as f:
            json.dump(manifest_blob, f)

    # --- 4) Write index.html ---
    geojson_inline = json.dumps(crowns_geojson)
    manifest_inline = json.dumps(manifest_blob)

    html = build_html(
        dataset_name=dataset_name,
        run_tag=run_name,
        base_png_name=base_png.name,
        geojson_path_name=geojson_path.name,
        underlay_om_id=underlay_om_id,
        underlay_stem=underlay_stem,
        img_h=img_h,
        img_w=img_w,
        geojson_inline=geojson_inline,
        manifest_inline=manifest_inline,
    )
    html_path.write_text(html, encoding="utf-8")
    print(f"\nWrote: {html_path}")

    # --- 5) Phenology overview plot (optional) ---
    scores_csv = Path(config.get("phenology_scores_csv",
                                  phenology_dir / "leafshed_tree_scores.csv"))
    phases_csv = phenology_dir / "leafshed_phenophase_by_om.csv"
    if scores_csv.exists() and phases_csv.exists():
        overview_png = viewer_dir / "phenology_overview.png"
        generate_phenology_overview(scores_csv, phases_csv, om_stems, overview_png)

    # --- 6) Enrich tree_master_geojson with viewer / pixel info ---
    master_geojson_path = Path(config.get("tree_master_geojson",
                                           phenology_dir / "tree_master_geojson.geojson"))
    if master_geojson_path.exists():
        print(f"\nEnriching tree_master_geojson: {master_geojson_path.name}")
        with open(master_geojson_path) as f:
            master_gj = json.load(f)

        # Update top-level viewer block
        master_gj["viewer"] = {
            "default_base_image": base_png.name,
            "default_crowns_geojson": geojson_path.name,
            "underlay_om_id": int(underlay_om_id),
            "underlay_stem": underlay_stem,
            "image_h": int(img_h),
            "image_w": int(img_w),
        }

        # Build per-crown pixel-coord lookup from crowns_geojson
        pixel_by_index = {
            feat["properties"]["crown_index"]: feat["geometry"]
            for feat in crowns_geojson.get("features", [])
        }

        # Per-crown: fill assets + pixel_underlays
        base_img_name = base_png.name
        geojson_name = geojson_path.name
        om_key = f"OM{underlay_om_id:02d}_{underlay_stem}"

        for feat in master_gj.get("features", []):
            props = feat["properties"]
            crown_index = props["ids"]["crown_index"]
            crown_label = props["ids"]["crown_label"]

            # assets
            props["assets"]["viewer_base_image"] = base_img_name
            props["assets"]["viewer_pixel_geojson"] = geojson_name

            # pixel_underlays: add pixel-coord geometry for this underlay OM
            pixel_geom = pixel_by_index.get(crown_index)
            if pixel_geom is not None:
                props["alternate_geometries"]["pixel_underlays"][om_key] = {
                    "crown_index": crown_index,
                    "crown_label": crown_label,
                    "geometry": pixel_geom,
                }

        master_geojson_path.write_text(json.dumps(master_gj, indent=2, default=str))
        print(f"  Updated: {master_geojson_path.name}")
    else:
        print(f"  [SKIP] tree_master_geojson not found: {master_geojson_path.name}")

    # --- Update config ---
    config["viewer_dir"] = str(viewer_dir)
    config["viewer_html"] = str(html_path)
    if "04_interactive_viz" not in config["steps_completed"]:
        config["steps_completed"].append("04_interactive_viz")
    save_config(config, config_path)
    print(f"Config updated: {config_path}")

    print(f"\nStep 4 complete.")
    print(f"To view: open {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
