from __future__ import annotations

"""Gap-fill augmentation using multi-threshold detections.

Goal:
- Extend partial chains by adding missing crowns in intermediate OMs.

We search in progressively lower-confidence layers (descending thresholds),
selecting the first layer that offers a geometrically consistent candidate.

Constraints (matching notebooks):
- centroid distance to neighbor(s) <= `max_centroid_dist`
- IoU to neighbor(s) >= `min_iou`
- suppress duplicates: IoU(candidate, any existing crown in OM) < `duplicate_iou`

Augmentation adds:
- a new node (om_id, new_index) flagged `is_augmented=True` and annotated with `threshold_tag`
- a directed edge labelled case='gap_fill', method='gap_fill'

Mathematically, the candidate score is a simple heuristic:
$$
S = \sum_k \mathrm{IoU}(g, g_k) - \frac{1}{d_{\max}}\sum_k d(\mu(g),\mu(g_k))
$$
where $k$ indexes the available neighbor constraints (prev/next), $\mu(\cdot)$ is centroid.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from .geometry import compute_crown_attributes, compute_iou
from .models import NodeId
from .state import TrackingState


def append_crown_to_state(state: TrackingState, om_id: int, row: Dict[str, Any], threshold_tag: str) -> NodeId:
    gdf = state.crowns_gdfs[om_id]

    new_row = dict(row)
    new_row["threshold_tag"] = threshold_tag
    new_row["is_augmented"] = True
    if "confidence" not in new_row:
        new_row["confidence"] = np.nan

    gdf_new = gpd.GeoDataFrame([new_row], crs=gdf.crs)
    state.crowns_gdfs[om_id] = gpd.GeoDataFrame(
        pd.concat([gdf, gdf_new], ignore_index=True),
        crs=gdf.crs,
    )

    new_idx = len(state.crowns_gdfs[om_id]) - 1

    attrs = compute_crown_attributes(new_row["geometry"])
    state.crown_attrs[om_id].append(attrs)
    state.G.add_node((om_id, new_idx), **attrs)

    return (om_id, new_idx)


def add_gap_edge(state: TrackingState, src: NodeId, dst: NodeId, similarity: float, features: Dict[str, float], parts: Dict[str, float]) -> None:
    state.G.add_edge(
        src,
        dst,
        similarity=float(similarity),
        method="gap_fill",
        case="gap_fill",
        overlap_prev=float(features.get("overlap_prev", 0.0)),
        overlap_curr=float(features.get("overlap_curr", 0.0)),
        iou=float(features.get("iou", 0.0)),
        centroid_distance=float(features.get("centroid_dist", 0.0)),
        base_similarity=float(similarity),
        spatial_similarity=float(parts.get("spatial", 0.0)),
        area_similarity=float(parts.get("area", 0.0)),
        shape_similarity=float(parts.get("shape", 0.0)),
    )


def best_gap_candidate(
    state: TrackingState,
    om_id: int,
    prev_node: Optional[NodeId],
    next_node: Optional[NodeId],
    mt_cache: Dict[int, Dict[str, gpd.GeoDataFrame]],
    thresholds_desc: List[str],
    max_centroid_dist: float,
    min_iou: float,
    duplicate_iou: float,
):
    prev_geom = state.crown_attrs[prev_node[0]][prev_node[1]]["geometry"] if prev_node else None
    next_geom = state.crown_attrs[next_node[0]][next_node[1]]["geometry"] if next_node else None

    existing_geoms = list(state.crowns_gdfs[om_id].geometry)

    best_row = None
    best_score = -1e9
    best_tag = None

    for tag in thresholds_desc:
        gdf = mt_cache.get(om_id, {}).get(tag)
        if gdf is None or gdf.empty:
            continue

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if any(compute_iou(geom, eg) >= duplicate_iou for eg in existing_geoms):
                continue

            scores = []
            if prev_geom is not None:
                dist_prev = geom.centroid.distance(prev_geom.centroid)
                if dist_prev > max_centroid_dist:
                    continue
                iou_prev = compute_iou(geom, prev_geom)
                if iou_prev < min_iou:
                    continue
                scores.append((iou_prev, dist_prev))

            if next_geom is not None:
                dist_next = geom.centroid.distance(next_geom.centroid)
                if dist_next > max_centroid_dist:
                    continue
                iou_next = compute_iou(geom, next_geom)
                if iou_next < min_iou:
                    continue
                scores.append((iou_next, dist_next))

            if not scores:
                continue

            score = sum(iou for iou, _ in scores) - (sum(dist for _, dist in scores) / max_centroid_dist)
            if score > best_score:
                best_score = score
                best_row = row
                best_tag = tag

        # Early stop once we found something at this threshold.
        if best_row is not None:
            break

    return best_row, best_tag


def augment_partial_chains_with_multithreshold(
    state: TrackingState,
    base_chain_categories: Dict[str, List[List[NodeId]]],
    mt_cache: Dict[int, Dict[str, gpd.GeoDataFrame]],
    thresholds_desc: List[str],
    max_centroid_dist: float = 25.0,
    min_iou: float = 0.20,
    duplicate_iou: float = 0.70,
) -> Dict[str, Any]:
    augmented = 0
    augmented_by_threshold = defaultdict(int)

    chains_to_extend = base_chain_categories.get("partial_long", []) + base_chain_categories.get("partial_short", [])

    if not state.om_ids:
        return {"augmented_nodes": 0, "by_threshold": {}}

    min_om = min(state.om_ids)
    max_om = max(state.om_ids)

    for chain in chains_to_extend:
        # Extend forwards.
        while True:
            last_node = chain[-1]
            last_om = last_node[0]
            if last_om >= max_om or state.G.out_degree(last_node) > 0:
                break
            target_om = last_om + 1
            if target_om not in state.om_ids:
                break

            cand, tag = best_gap_candidate(
                state,
                target_om,
                prev_node=last_node,
                next_node=None,
                mt_cache=mt_cache,
                thresholds_desc=thresholds_desc,
                max_centroid_dist=max_centroid_dist,
                min_iou=min_iou,
                duplicate_iou=duplicate_iou,
            )
            if cand is None:
                break

            new_node = append_crown_to_state(state, target_om, cand.to_dict(), tag)

            # Minimal similarity features for edge metadata.
            src_geom = state.crown_attrs[last_node[0]][last_node[1]]["geometry"]
            dst_geom = state.crown_attrs[new_node[0]][new_node[1]]["geometry"]
            iou_val = compute_iou(src_geom, dst_geom)
            dist_val = src_geom.centroid.distance(dst_geom.centroid)
            features = {"iou": iou_val, "centroid_dist": dist_val, "overlap_prev": 0.0, "overlap_curr": 0.0}
            add_gap_edge(state, last_node, new_node, similarity=float(iou_val), features=features, parts={})

            chain.append(new_node)
            augmented += 1
            augmented_by_threshold[tag] += 1

        # Extend backwards.
        while True:
            first_node = chain[0]
            first_om = first_node[0]
            if first_om <= min_om or state.G.in_degree(first_node) > 0:
                break
            target_om = first_om - 1
            if target_om not in state.om_ids:
                break

            cand, tag = best_gap_candidate(
                state,
                target_om,
                prev_node=None,
                next_node=first_node,
                mt_cache=mt_cache,
                thresholds_desc=thresholds_desc,
                max_centroid_dist=max_centroid_dist,
                min_iou=min_iou,
                duplicate_iou=duplicate_iou,
            )
            if cand is None:
                break

            new_node = append_crown_to_state(state, target_om, cand.to_dict(), tag)

            src_geom = state.crown_attrs[new_node[0]][new_node[1]]["geometry"]
            dst_geom = state.crown_attrs[first_node[0]][first_node[1]]["geometry"]
            iou_val = compute_iou(src_geom, dst_geom)
            dist_val = src_geom.centroid.distance(dst_geom.centroid)
            features = {"iou": iou_val, "centroid_dist": dist_val, "overlap_prev": 0.0, "overlap_curr": 0.0}
            add_gap_edge(state, new_node, first_node, similarity=float(iou_val), features=features, parts={})

            chain.insert(0, new_node)
            augmented += 1
            augmented_by_threshold[tag] += 1

    return {"augmented_nodes": int(augmented), "by_threshold": dict(augmented_by_threshold)}
