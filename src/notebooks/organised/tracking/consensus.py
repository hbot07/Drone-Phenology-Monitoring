from __future__ import annotations

"""Consensus crown computation.

Given a chain of polygons across OMs, we want a single representative polygon.

Implemented methods (from notebooks):
- medoid: choose one observed polygon that minimizes a weighted distance to others
- intersection_core: robust intersection with a small buffer
- union_shrink: union then shrink by a data-driven erosion

Definitions:
- For polygons $p_i$ with centroids $c_i$ and areas $a_i$.
- Normalized centroid distance between i and j:
  $d_{ij} = \frac{\|c_i - c_j\|}{\max_{k,\ell}\|c_k - c_\ell\|}$.
- IoU: $\mathrm{IoU}(p_i,p_j)$.
- Area ratio similarity: $r_{ij}=\frac{\min(a_i,a_j)}{\max(a_i,a_j)}$.

Medoid score for candidate i:
$$
S_i = \sum_{j\neq i} w_c d_{ij} + w_{\mathrm{iou}}(1-\mathrm{IoU}(p_i,p_j)) + w_a(1-r_{ij}).
$$
Pick i with minimal $S_i$.
"""

from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry.base import BaseGeometry

from .geometry import compute_iou
from .models import NodeId
from .state import TrackingState


def _chain_polys(state: TrackingState, chain: List[NodeId]) -> List[BaseGeometry]:
    polys: List[BaseGeometry] = []
    for om_id, crown_idx in chain:
        poly = state.crowns_gdfs[om_id].iloc[crown_idx].geometry
        if poly is not None and not poly.is_empty:
            polys.append(poly.buffer(0))
    return polys


def consensus_medoid(
    state: TrackingState,
    chain: List[NodeId],
    w_centroid: float = 0.5,
    w_iou: float = 0.4,
    w_area: float = 0.1,
) -> Optional[BaseGeometry]:
    polys = _chain_polys(state, chain)
    if not polys:
        return None

    centroids = [p.centroid for p in polys]
    areas = [p.area for p in polys]

    max_dist = max((c1.distance(c2) for c1 in centroids for c2 in centroids), default=1.0) or 1.0

    best_idx = 0
    best_score = float("inf")
    for i, p in enumerate(polys):
        score = 0.0
        for j, q in enumerate(polys):
            if i == j:
                continue
            dist = centroids[i].distance(centroids[j]) / max_dist
            iou = compute_iou(p, q)
            area_ratio = min(areas[i], areas[j]) / max(areas[i], areas[j]) if max(areas[i], areas[j]) > 0 else 0.0
            score += (w_centroid * dist) + (w_iou * (1.0 - iou)) + (w_area * (1.0 - area_ratio))
        if score < best_score:
            best_score = score
            best_idx = i

    return polys[best_idx]


def consensus_intersection_core(
    state: TrackingState,
    chain: List[NodeId],
    buffer_dist: float = 0.5,
    min_area: float = 1e-3,
) -> Optional[BaseGeometry]:
    polys = _chain_polys(state, chain)
    if not polys:
        return None

    core = polys[0].buffer(buffer_dist)
    for p in polys[1:]:
        core = core.intersection(p.buffer(buffer_dist))
        if core.is_empty:
            break

    if core.is_empty or core.area < min_area:
        return None

    return core.buffer(-buffer_dist)


def consensus_union_shrink(state: TrackingState, chain: List[NodeId]) -> Optional[BaseGeometry]:
    polys = _chain_polys(state, chain)
    if not polys:
        return None

    union = polys[0]
    for p in polys[1:]:
        union = union.union(p)

    if union.is_empty:
        return None

    # Shrink by a fraction of the median equivalent-radius to reduce spurious protrusions.
    areas = np.array([p.area for p in polys], dtype=float)
    radii = np.sqrt(np.maximum(areas, 1e-12) / np.pi)
    shrink = float(np.median(radii) * 0.15)

    try:
        shrunk = union.buffer(-shrink)
    except Exception:
        shrunk = union

    if shrunk.is_empty:
        return None

    return shrunk


def compute_consensus_polygon(state: TrackingState, chain: List[NodeId], method: str) -> Optional[BaseGeometry]:
    method = method.lower().strip()
    if method == "medoid":
        return consensus_medoid(state, chain)
    if method == "intersection_core":
        return consensus_intersection_core(state, chain)
    if method == "union_shrink":
        return consensus_union_shrink(state, chain)
    raise ValueError(f"Unknown consensus method: {method}")
