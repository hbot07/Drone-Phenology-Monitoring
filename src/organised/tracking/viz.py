from __future__ import annotations

"""Visualization writers (save-to-disk).

Design rule: no implicit display.
- All functions save figures to paths and close the matplotlib figure.

This module mirrors the notebook figures:
- per-chain panel: polygons + extracted crown patch
- per-consensus panel: consensus polygon overlay + consensus patch per OM
- spatial map of consensus crowns
"""

from pathlib import Path
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.mask import mask as rio_mask
from shapely.affinity import translate
from shapely.geometry import Polygon, mapping

from .chains import normalize_chain_data
from .models import NodeId
from .state import TrackingState


def _find_ortho_path(state: TrackingState, om_id: int) -> Optional[Path]:
    return state.ortho_paths.get(om_id)


def extract_patch_for_aligned_polygon(
    state: TrackingState,
    om_id: int,
    polygon_aligned,
    open_rasters: Optional[Dict[int, DatasetReader]] = None,
) -> Optional[np.ndarray]:
    """Extract a raster patch for an *aligned* polygon.

    The polygon is assumed to be in the aligned crown-coordinate frame.
    We invert the alignment shift to map back to the raster's native CRS.
    """

    if polygon_aligned is None or polygon_aligned.is_empty:
        return None

    dx, dy = state.alignment_shifts.get(om_id, (0.0, 0.0))
    polygon_original = translate(polygon_aligned, xoff=-dx, yoff=-dy)

    try:
        if open_rasters is not None and om_id in open_rasters:
            src = open_rasters[om_id]
            out_image, _ = rio_mask(src, [mapping(polygon_original)], crop=True)
            return np.moveaxis(out_image, 0, -1)

        ortho_path = _find_ortho_path(state, om_id)
        if ortho_path is None or not ortho_path.exists():
            return None

        with rasterio.open(ortho_path) as src:
            out_image, _ = rio_mask(src, [mapping(polygon_original)], crop=True)
            return np.moveaxis(out_image, 0, -1)
    except Exception:
        return None


def _open_ortho_rasters(state: TrackingState, stack: ExitStack) -> Dict[int, DatasetReader]:
    opened: Dict[int, DatasetReader] = {}
    for om_id in state.om_ids:
        p = _find_ortho_path(state, om_id)
        if p is None or not p.exists():
            continue
        try:
            opened[om_id] = stack.enter_context(rasterio.open(p))
        except Exception:
            continue
    return opened


def save_chain_panel(
    state: TrackingState,
    chain: List[NodeId],
    title: str,
    out_path: Path,
    dpi: int = 150,
    open_rasters: Optional[Dict[int, DatasetReader]] = None,
) -> Path:
    chain_length = len(chain)

    fig = plt.figure(figsize=(5 * chain_length, 10))

    for idx, (om_id, crown_idx) in enumerate(chain):
        gdf = state.crowns_gdfs[om_id]
        crown = gdf.iloc[crown_idx]
        in_deg = state.G.in_degree((om_id, crown_idx)) if (om_id, crown_idx) in state.G else 0
        out_deg = state.G.out_degree((om_id, crown_idx)) if (om_id, crown_idx) in state.G else 0

        ax_poly = plt.subplot(2, chain_length, idx + 1)
        minx, miny, maxx, maxy = crown.geometry.bounds
        margin = max((maxx - minx), (maxy - miny)) * 0.3
        gdf.plot(ax=ax_poly, color="lightgray", edgecolor="gray", alpha=0.3)
        gpd.GeoSeries([crown.geometry]).plot(
            ax=ax_poly,
            facecolor=plt.cm.tab10((om_id - 1) % 10),
            edgecolor="black",
            linewidth=2,
            alpha=0.7,
        )
        centroid = crown.geometry.centroid
        ax_poly.plot(
            centroid.x,
            centroid.y,
            "o",
            color="yellow",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        ax_poly.set_xlim(minx - margin, maxx + margin)
        ax_poly.set_ylim(miny - margin, maxy + margin)
        ax_poly.set_aspect("equal")
        ax_poly.set_title(f"OM{om_id} - Crown {crown_idx}\nArea: {crown.geometry.area:.1f}m2\nIn:{in_deg} Out:{out_deg}", fontsize=10)
        ax_poly.grid(True, alpha=0.3)

        ax_img = plt.subplot(2, chain_length, chain_length + idx + 1)
        patch = extract_patch_for_aligned_polygon(state, om_id, crown.geometry, open_rasters=open_rasters)
        if patch is not None and patch.size > 0:
            patch_display = np.clip(patch[:, :, :3], 0, 255).astype(np.uint8)
            ax_img.imshow(patch_display)
            ax_img.set_title(f"Extracted Patch\n{patch_display.shape[0]}x{patch_display.shape[1]} px", fontsize=9)
        else:
            ax_img.text(0.5, 0.5, "Image not available", ha="center", va="center", fontsize=10, color="red")
            ax_img.set_title("No Image", fontsize=9)
        ax_img.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path


def save_chain_panels(
    state: TrackingState,
    all_extracted_chains: List[Any],
    output_dir: Path,
    max_panels: Optional[int] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        rasters = _open_ortho_rasters(state, stack)
        for idx, chain_data in enumerate(all_extracted_chains, 1):
            if max_panels is not None and idx > max_panels:
                break
            chain, _, quality = normalize_chain_data(chain_data)
            title = f"Chain {idx}/{len(all_extracted_chains)} - {quality}"
            save_chain_panel(
                state,
                chain,
                title=title,
                out_path=output_dir / f"chain_{idx:03d}.png",
                open_rasters=rasters,
            )

    return output_dir


def save_consensus_panel(
    state: TrackingState,
    chain: List[NodeId],
    consensus_poly,
    chain_id: int,
    quality: str,
    avg_sim: float,
    out_path: Path,
    dpi: int = 150,
    open_rasters: Optional[Dict[int, DatasetReader]] = None,
) -> Optional[Path]:
    if consensus_poly is None or consensus_poly.is_empty:
        return None

    chain_length = len(chain)
    fig = plt.figure(figsize=(5 * chain_length, 10))
    consensus_centroid = consensus_poly.centroid

    for idx, (om_id, crown_idx) in enumerate(chain):
        gdf = state.crowns_gdfs[om_id]
        crown = gdf.iloc[crown_idx]

        ax_poly = plt.subplot(2, chain_length, idx + 1)
        gdf.plot(ax=ax_poly, color="lightgray", edgecolor="gray", alpha=0.3, linewidth=0.5)
        gpd.GeoSeries([consensus_poly]).plot(ax=ax_poly, facecolor="red", edgecolor="darkred", linewidth=2.5, alpha=0.4)
        gpd.GeoSeries([crown.geometry]).plot(ax=ax_poly, facecolor="none", edgecolor="black", linewidth=2.0, alpha=0.9)
        ax_poly.plot(consensus_centroid.x, consensus_centroid.y, "o", color="yellow", markersize=10, markeredgecolor="black", markeredgewidth=2)
        ax_poly.set_aspect("equal")
        ax_poly.set_title(f"OM{om_id}", fontsize=11, fontweight="bold")
        ax_poly.grid(True, alpha=0.3)

        ax_img = plt.subplot(2, chain_length, chain_length + idx + 1)
        patch = extract_patch_for_aligned_polygon(state, om_id, consensus_poly, open_rasters=open_rasters)
        if patch is not None and patch.size > 0:
            patch_display = np.clip(patch[:, :, :3], 0, 255).astype(np.uint8)
            ax_img.imshow(patch_display)
            ax_img.set_title(f"Consensus Patch\n{patch_display.shape[0]}x{patch_display.shape[1]} px", fontsize=10)
        else:
            ax_img.text(0.5, 0.5, "Image not available", ha="center", va="center", fontsize=10, color="red")
            ax_img.set_title("No Image", fontsize=9)
        ax_img.axis("off")

    fig.suptitle(f"Consensus Chain {chain_id} | {quality} | Avg Sim: {avg_sim:.3f}", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path


def save_consensus_panels(
    state: TrackingState,
    all_extracted_chains: List[Any],
    consensus_polys: List,
    output_dir: Path,
    max_panels: Optional[int] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        rasters = _open_ortho_rasters(state, stack)
        for idx, (chain_data, poly) in enumerate(zip(all_extracted_chains, consensus_polys), 1):
            if max_panels is not None and idx > max_panels:
                break
            chain, avg_sim, quality = normalize_chain_data(chain_data)
            save_consensus_panel(
                state,
                chain,
                consensus_poly=poly,
                chain_id=idx,
                quality=quality,
                avg_sim=avg_sim,
                out_path=output_dir / f"consensus_chain_{idx:03d}.png",
                open_rasters=rasters,
            )

    return output_dir


def save_consensus_spatial_map(consensus_gdf: gpd.GeoDataFrame, out_path: Path) -> Optional[Path]:
    if consensus_gdf is None or consensus_gdf.empty:
        return None

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    consensus_gdf.plot(ax=ax, column="chain_length" if "chain_length" in consensus_gdf.columns else None, legend=True, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title("Consensus Crowns Spatial Map")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path
