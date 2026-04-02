from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

try:
    import fiona

    FIONA_AVAILABLE = True
except Exception:
    FIONA_AVAILABLE = False


_NUM_RE = re.compile(r"(\d+)")


def extract_numeric_id(name: str) -> Optional[int]:
    m = _NUM_RE.search(os.path.basename(name))
    return int(m.group(1)) if m else None


def discover_sit_pairs(multithresh_dir: Path, ortho_dir: Path) -> List[Tuple[int, Path, Optional[Path]]]:
    """Discover pairs (om_id, gpkg_path, ortho_path) for SIT-style naming.

    Expected:
    - crowns: OM{n}_multithreshold.gpkg
    - orthos: sit_om{n}.tif
    """

    gpkg_files = sorted(multithresh_dir.glob("OM*_multithreshold.gpkg"))
    ortho_files = sorted(ortho_dir.glob("*.tif"))
    ortho_by_id: Dict[int, Path] = {}
    for p in ortho_files:
        oid = extract_numeric_id(p.name)
        if oid is not None:
            ortho_by_id[oid] = p

    pairs: List[Tuple[int, Path, Optional[Path]]] = []
    for gpkg in gpkg_files:
        oid = extract_numeric_id(gpkg.name)
        if oid is None:
            continue
        pairs.append((oid, gpkg, ortho_by_id.get(oid)))

    return sorted(pairs, key=lambda t: t[0])


def threshold_tag_to_float(tag: str) -> float:
    try:
        return float(tag.replace("conf_", "").replace("p", "."))
    except Exception:
        return float("nan")


def float_to_threshold_tag(val: float) -> str:
    return f"conf_{val:.2f}".replace(".", "p")


def list_threshold_layers(gpkg_path: Path) -> List[str]:
    """Return threshold layer tags sorted ascending by threshold."""

    meta_path = gpkg_path.with_suffix(".meta.json")
    meta_layers: List[str] = []
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            for t in meta.get("thresholds", []):
                if isinstance(t, str) and t.startswith("conf_"):
                    meta_layers.append(t)
                elif isinstance(t, (int, float)):
                    meta_layers.append(float_to_threshold_tag(float(t)))
            meta_layers = list(dict.fromkeys(meta_layers))
        except Exception:
            meta_layers = []

    actual_layers: List[str] = []
    if FIONA_AVAILABLE:
        try:
            actual_layers = list(fiona.listlayers(str(gpkg_path)))
        except Exception:
            actual_layers = []

    if actual_layers:
        layers = [l for l in meta_layers if l in actual_layers] if meta_layers else actual_layers
    else:
        layers = meta_layers

    if not layers:
        raise RuntimeError(f"No layers found in {gpkg_path}")

    return sorted(layers, key=threshold_tag_to_float)


def load_layer_gdf(gpkg_path: Path, layer_tag: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path, layer=layer_tag).reset_index(drop=True)
    if "threshold_tag" not in gdf.columns:
        gdf["threshold_tag"] = layer_tag
    if "confidence" not in gdf.columns:
        gdf["confidence"] = np.nan
    if "orthomosaic" not in gdf.columns:
        gdf["orthomosaic"] = gpkg_path.stem
    if "is_augmented" not in gdf.columns:
        gdf["is_augmented"] = False
    return gdf


def dedup_by_wkt(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    out = gdf.copy()
    out["_geom_wkt"] = out.geometry.apply(lambda g: g.buffer(0).wkt)
    out = out.drop_duplicates(subset="_geom_wkt").drop(columns=["_geom_wkt"]).reset_index(drop=True)
    return out


def read_patch(ortho_path: Path, polygon) -> Optional[np.ndarray]:
    if ortho_path is None or not ortho_path.exists():
        return None
    try:
        with rasterio.open(ortho_path) as src:
            out_image, _ = rio_mask(src, [mapping(polygon)], crop=True)
            return np.moveaxis(out_image, 0, -1)
    except Exception:
        return None
