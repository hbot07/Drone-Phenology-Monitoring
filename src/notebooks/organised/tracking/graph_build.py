from __future__ import annotations

"""Graph construction.

We build a directed graph $G=(V,E)$ with:
- Nodes: $v=(t,i)$ where $t$ is orthomosaic index (OM id) and $i$ is crown index.
- Edges: between consecutive OMs only (unless augmentation later adds gap edges).

Each candidate edge is assigned a match *case* in {one_to_one, containment, nearby}
(or none) and then selected by a case-priority policy.

This module does not do any plotting.
"""

from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional

import networkx as nx

from .matching import (
    classify_match_case,
    compute_pair_metrics,
    select_candidates_by_case,
)
from .models import MatchCaseConfig
from .state import TrackingState


def reset_graph(state: TrackingState) -> None:
    state.G = nx.DiGraph()


def add_nodes_for_all_oms(state: TrackingState) -> None:
    for om_id in state.om_ids:
        gdf = state.crowns_gdfs[om_id]
        for crown_id, _ in gdf.iterrows():
            attrs = state.crown_attrs[om_id][crown_id]
            state.G.add_node((om_id, crown_id), **attrs)


def build_graph_conditional(
    state: TrackingState,
    case_configs: Dict[str, MatchCaseConfig],
    case_order: List[str],
    base_max_dist: float,
    overlap_gate: float,
    min_base_similarity: float,
    classify_mode: str,
    max_candidates_per_prev: Optional[int] = None,
    max_candidates_per_curr: Optional[int] = None,
) -> None:
    """Build edges between consecutive OMs.

    Parameters mirror the notebooks, but are explicit inputs.
    """

    reset_graph(state)
    add_nodes_for_all_oms(state)

    configs = {name: replace(cfg) for name, cfg in case_configs.items()}
    order = list(case_order)

    state.last_case_counts = {}
    state.last_selected_counts = {name: 0 for name in configs.keys()}

    for idx in range(1, len(state.om_ids)):
        prev_om = state.om_ids[idx - 1]
        om_id = state.om_ids[idx]

        prev_nodes = [(prev_om, i) for i in range(len(state.crowns_gdfs[prev_om]))]
        curr_nodes = [(om_id, j) for j in range(len(state.crowns_gdfs[om_id]))]

        candidates: List[Dict[str, Any]] = []
        overlap_counts_prev = defaultdict(int)
        overlap_counts_curr = defaultdict(int)

        for prev_node in prev_nodes:
            prev_attrs = state.crown_attrs[prev_om][prev_node[1]]
            for curr_node in curr_nodes:
                curr_attrs = state.crown_attrs[om_id][curr_node[1]]
                features = compute_pair_metrics(prev_attrs, curr_attrs, max_dist=base_max_dist)
                if features["centroid_dist"] > base_max_dist:
                    continue
                candidates.append(
                    {
                        "prev_node": prev_node,
                        "curr_node": curr_node,
                        "prev_attrs": prev_attrs,
                        "curr_attrs": curr_attrs,
                        "features": features,
                    }
                )
                if features["overlap_prev"] >= overlap_gate:
                    overlap_counts_prev[prev_node] += 1
                if features["overlap_curr"] >= overlap_gate:
                    overlap_counts_curr[curr_node] += 1

        for cand in candidates:
            cand["case"] = classify_match_case(
                cand["prev_node"],
                cand["curr_node"],
                cand["features"],
                overlap_counts_prev,
                overlap_counts_curr,
                overlap_gate,
                mode=classify_mode,
            )

        candidates = [c for c in candidates if c["case"] != "none"]

        if max_candidates_per_prev is not None:
            grouped_prev = defaultdict(list)
            for cand in candidates:
                grouped_prev[cand["prev_node"]].append(cand)
            trimmed = []
            for group in grouped_prev.values():
                group.sort(key=lambda c: (-c["features"]["iou"], c["features"]["centroid_dist"]))
                trimmed.extend(group[:max_candidates_per_prev])
            candidates = trimmed

        if max_candidates_per_curr is not None:
            grouped_curr = defaultdict(list)
            for cand in candidates:
                grouped_curr[cand["curr_node"]].append(cand)
            trimmed = []
            for group in grouped_curr.values():
                group.sort(key=lambda c: (-c["features"]["iou"], c["features"]["centroid_dist"]))
                trimmed.extend(group[:max_candidates_per_curr])
            candidates = trimmed

        case_counts = defaultdict(int)
        for cand in candidates:
            case_counts[cand["case"]] += 1
        for case_name, count in case_counts.items():
            state.last_case_counts[case_name] = state.last_case_counts.get(case_name, 0) + int(count)

        selected = select_candidates_by_case(
            candidates,
            configs,
            order,
            base_max_dist,
            min_base_similarity=min_base_similarity,
        )

        for cand in selected:
            case_name = cand["case"]
            features = cand["features"]
            similarity_parts = cand.get("similarity_parts", {})

            state.G.add_edge(
                cand["prev_node"],
                cand["curr_node"],
                similarity=float(cand.get("score", features["base_similarity"])),
                method="conditional",
                case=case_name,
                overlap_prev=float(features["overlap_prev"]),
                overlap_curr=float(features["overlap_curr"]),
                iou=float(features["iou"]),
                centroid_distance=float(features["centroid_dist"]),
                base_similarity=float(cand.get("base_similarity", features["base_similarity"])),
                spatial_similarity=float(similarity_parts.get("spatial", features["spatial_similarity"])),
                area_similarity=float(similarity_parts.get("area", features["area_similarity"])),
                shape_similarity=float(similarity_parts.get("shape", features["shape_similarity"])),
            )
            state.last_selected_counts[case_name] = state.last_selected_counts.get(case_name, 0) + 1
