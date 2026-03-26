from __future__ import annotations

"""Chain extraction and categorization.

Given a directed graph, we extract chains by greedy successor selection:
- A chain start is a node with in-degree 0.
- From each start, follow the outgoing edge with maximum similarity.

This matches the notebook behavior and makes chain extraction deterministic.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import NodeId
from .state import TrackingState


def greedy_chain(state: TrackingState, start_node: NodeId) -> List[NodeId]:
    chain = [start_node]
    current = start_node
    while True:
        succ = list(state.G.successors(current))
        if not succ:
            break
        best = max(succ, key=lambda n: state.G[current][n].get("similarity", 0.0))
        chain.append(best)
        current = best
    return chain


def extract_all_chains(state: TrackingState) -> List[List[NodeId]]:
    visited = set()
    chains: List[List[NodeId]] = []

    chain_starts = [n for n in state.G.nodes if not list(state.G.predecessors(n))]
    for start in chain_starts:
        if start in visited:
            continue
        chain = greedy_chain(state, start)
        chains.append(chain)
        visited.update(chain)

    remaining = set(state.G.nodes) - visited
    for node in remaining:
        chains.append([node])

    return chains


def categorize_chains(state: TrackingState) -> Dict[str, List[List[NodeId]]]:
    all_chains = extract_all_chains(state)
    max_chain_length = len(state.om_ids)

    full_width_1 = []
    full_branching = []
    partial_long = []
    partial_short = []

    for chain in all_chains:
        L = len(chain)
        if L == max_chain_length:
            has_branch = any(state.G.in_degree(n) > 1 or state.G.out_degree(n) > 1 for n in chain)
            (full_branching if has_branch else full_width_1).append(chain)
        elif L >= 3:
            partial_long.append(chain)
        elif L == 2:
            partial_short.append(chain)

    return {
        "full_width_1": full_width_1,
        "full_branching": full_branching,
        "partial_long": partial_long,
        "partial_short": partial_short,
        "singleton": [c for c in all_chains if len(c) == 1],
    }


def extract_backbone_mixed(state: TrackingState, chain_nodes: List[NodeId]) -> Optional[Dict[str, Any]]:
    """Extract a backbone path for branching chains.

    At each step we pick the best candidate edge using a lexicographic order:
    (match quality, similarity).

    match quality ordering:
    one_to_one > containment > nearby > other.
    """

    edges_by_step: Dict[int, list[dict]] = {}
    for i in range(len(chain_nodes) - 1):
        node1 = chain_nodes[i]
        node2 = chain_nodes[i + 1]
        step_edges = []

        if state.G.has_edge(node1, node2):
            d = state.G.edges[node1, node2]
            step_edges.append(
                {
                    "node1": node1,
                    "node2": node2,
                    "match_type": d.get("case", "unknown"),
                    "similarity": d.get("similarity", 0.0),
                    "iou": d.get("iou", 0.0),
                }
            )

        for successor in state.G.successors(node1):
            if successor != node2 and successor[0] == node2[0]:
                d = state.G.edges[node1, successor]
                step_edges.append(
                    {
                        "node1": node1,
                        "node2": successor,
                        "match_type": d.get("case", "unknown"),
                        "similarity": d.get("similarity", 0.0),
                        "iou": d.get("iou", 0.0),
                    }
                )

        if not step_edges:
            return None
        edges_by_step[i] = step_edges

    quality_map = {"one_to_one": 3, "containment": 2, "nearby": 1, "gap_fill": 0}
    path = []
    for i in range(len(chain_nodes) - 1):
        best = max(edges_by_step[i], key=lambda e: (quality_map.get(e["match_type"], 0), e["similarity"]))
        path.append(best)

    return {
        "edges": path,
        "crowns": [chain_nodes[0]] + [e["node2"] for e in path],
        "avg_similarity": float(np.mean([e["similarity"] for e in path])) if path else 0.0,
        "edge_types": [e["match_type"] for e in path],
        "one_to_one_count": sum(1 for e in path if e["match_type"] == "one_to_one"),
    }


def assemble_high_quality_chains(state: TrackingState) -> Dict[str, Any]:
    categories = categorize_chains(state)
    clean_chains = categories["full_width_1"]
    branching_chains = categories["full_branching"]

    extracted_backbones = []
    for chain in branching_chains:
        backbone = extract_backbone_mixed(state, chain)
        if backbone:
            extracted_backbones.append(backbone)

    all_extracted = list(clean_chains) + list(extracted_backbones)
    return {
        "categories": categories,
        "clean_chains": clean_chains,
        "branching_chains": branching_chains,
        "extracted_backbones": extracted_backbones,
        "all_extracted_chains": all_extracted,
    }


def _chain_one_to_one_ratio(state: TrackingState, chain: List[NodeId]) -> float:
    if len(chain) < 2:
        return 0.0
    total = len(chain) - 1
    oto = 0
    for i in range(total):
        u, v = chain[i], chain[i + 1]
        if not state.G.has_edge(u, v):
            return 0.0
        if state.G.edges[u, v].get("case") == "one_to_one":
            oto += 1
    return float(oto / total) if total else 0.0


def select_consensus_source_chains(
    state: TrackingState,
    hq_result: Dict[str, Any],
    include_partial: bool,
    min_partial_len: int,
    min_partial_one_to_one_ratio: float,
) -> List[List[NodeId]]:
    """Choose which chains should produce a consensus crown.

    The default behavior matches the notebooks:
    - always include full-width-1 + extracted backbones
    - optionally include partial chains that are long enough and mostly one-to-one
    """

    selected: List[List[NodeId]] = list(hq_result.get("all_extracted_chains", []))
    if not include_partial:
        return selected

    categories = hq_result.get("categories", {})
    partial_candidates = categories.get("partial_long", []) + categories.get("partial_short", [])

    for chain in partial_candidates:
        if len(chain) < min_partial_len:
            continue
        if _chain_one_to_one_ratio(state, chain) < min_partial_one_to_one_ratio:
            continue
        selected.append(chain)

    return selected


def normalize_chain_data(chain_data: Any) -> Tuple[List[NodeId], float, str]:
    """Normalize either raw chain list or extracted backbone dict."""

    if isinstance(chain_data, dict) and "edges" in chain_data:
        chain = [chain_data["edges"][0]["node1"]] + [e["node2"] for e in chain_data["edges"]]
        avg_sim = float(chain_data.get("avg_similarity", 0.0))
        oto_count = int(chain_data.get("one_to_one_count", 0))
        quality = "Pure" if oto_count == len(chain) - 1 else "Mixed"
    else:
        chain = list(chain_data)
        avg_sim = 1.0
        quality = "Clean"

    return chain, avg_sim, quality
