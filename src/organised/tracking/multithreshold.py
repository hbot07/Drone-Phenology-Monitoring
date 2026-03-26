from __future__ import annotations

"""Multi-threshold handling.

Each OM has a GeoPackage with multiple layers named like `conf_0p45`.
We load:
- reference OM (min om_id): a single base layer (usually the highest threshold)
- other OMs: concatenation of layers (optionally including base) and WKT dedup

Additionally, we can build a cache of aligned lower-threshold layers for gap-fill.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd

from shapely.affinity import translate

from . import io
from .alignment import AlignmentResult
from .geometry import compute_crown_attributes
from .models import FilePair
from .state import TrackingState


@dataclass(frozen=True)
class MultiThresholdLoadResult:
    base_threshold_tag: str
    layers_by_om: Dict[int, List[str]]


def apply_alignment_to_gdf(gdf: gpd.GeoDataFrame, shift: Tuple[float, float]) -> gpd.GeoDataFrame:
    dx, dy = shift
    out = gdf.copy()
    out["geometry"] = out["geometry"].apply(lambda g: translate(g, xoff=dx, yoff=dy))
    return out


def load_multithreshold_state(
    pairs: List[Tuple[int, Path, Optional[Path]]],
    base_threshold_tag: Optional[str],
    include_base_for_non_reference: bool,
) -> TrackingState:
    file_pairs: List[FilePair] = []
    om_ids: List[int] = []
    ortho_paths: Dict[int, Path] = {}

    for om_id, gpkg, ortho in pairs:
        file_pairs.append(FilePair(om_id=om_id, crowns_path=str(gpkg), ortho_path=str(ortho) if ortho else None))
        om_ids.append(om_id)
        if ortho:
            ortho_paths[om_id] = ortho

    state = TrackingState(file_pairs=file_pairs, om_ids=om_ids)
    state.ortho_paths = ortho_paths

    layers_by_om: Dict[int, List[str]] = {}
    chosen_base: Optional[str] = None

    for om_id, gpkg, _ in pairs:
        layers = io.list_threshold_layers(gpkg)
        layers_by_om[om_id] = layers

        if base_threshold_tag and base_threshold_tag in layers:
            base_layer = base_threshold_tag
        else:
            base_layer = max(layers, key=io.threshold_tag_to_float)

        if chosen_base is None:
            chosen_base = base_layer

        if om_id == min(om_ids):
            # Reference OM: keep only the base layer to control the node count.
            gdf = io.load_layer_gdf(gpkg, base_layer)
        else:
            # Other OMs: allow multiple thresholds (typically base + lower) to enable gap-filling.
            chosen_layers = [l for l in layers if include_base_for_non_reference or l != base_layer]
            pieces = [io.load_layer_gdf(gpkg, lyr) for lyr in chosen_layers]
            if pieces:
                gdf = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs=pieces[0].crs)
                gdf = io.dedup_by_wkt(gdf)
            else:
                gdf = io.load_layer_gdf(gpkg, base_layer)

        state.crowns_gdfs[om_id] = gdf.reset_index(drop=True)
        state.crown_crs[om_id] = gdf.crs
        state.crown_attrs[om_id] = [compute_crown_attributes(row.geometry) for _, row in state.crowns_gdfs[om_id].iterrows()]
        state.multithreshold_layers[om_id] = layers

    state.base_threshold_tag = chosen_base
    return state


def build_multithreshold_cache(
    state: TrackingState,
    multithresh_dir: Path,
    alignment: AlignmentResult,
    include_base: bool,
    min_threshold_tag: Optional[str],
) -> Tuple[Dict[int, Dict[str, gpd.GeoDataFrame]], List[str]]:
    """Cache: om_id -> threshold_tag -> aligned GeoDataFrame."""

    cache: Dict[int, Dict[str, gpd.GeoDataFrame]] = {}
    all_thresholds = set()

    base_tag = state.base_threshold_tag
    base_val = io.threshold_tag_to_float(base_tag) if base_tag else None
    min_val = io.threshold_tag_to_float(min_threshold_tag) if min_threshold_tag else None

    for pair in state.file_pairs:
        om_id = pair.om_id
        gpkg = Path(pair.crowns_path)
        layers = state.multithreshold_layers.get(om_id) or io.list_threshold_layers(gpkg)

        filtered: List[str] = []
        for layer in layers:
            if not include_base and base_tag and layer == base_tag:
                continue
            val = io.threshold_tag_to_float(layer)
            if min_val is not None and (val != val or val < min_val):
                continue
            if base_val is not None and not include_base and val > base_val:
                continue
            filtered.append(layer)

        cache[om_id] = {}
        for layer in filtered:
            gdf = io.load_layer_gdf(gpkg, layer)
            shift = alignment.shifts.get(om_id, (0.0, 0.0))
            cache[om_id][layer] = apply_alignment_to_gdf(gdf, shift)
            all_thresholds.add(layer)

    thresholds_desc = sorted(all_thresholds, key=io.threshold_tag_to_float, reverse=True)
    return cache, thresholds_desc
