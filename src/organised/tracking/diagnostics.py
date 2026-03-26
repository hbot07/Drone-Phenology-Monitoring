from __future__ import annotations

"""Diagnostics and reporting.

This module produces:
- a human-readable markdown/text report (for notebook display + saving)
- a JSON metrics blob (for programmatic comparisons)
- a set of saved figures under output_dir/figures

We keep diagnostics mathematically explicit and reproducible.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .models import NodeId
from .state import TrackingState


def compute_quality_metrics(state: TrackingState) -> Dict[str, Any]:
    G = state.G

    metrics: Dict[str, Any] = {
        "n_oms": len(state.om_ids),
        "total_nodes": int(G.number_of_nodes()),
        "total_edges": int(G.number_of_edges()),
        "total_possible_matches": 0,
        "successful_matches": 0,
        "overall_match_rate": 0.0,
        "match_rate_by_om_pair": {},
        "chain_length_distribution": {},
        "average_chain_length": 0.0,
        "median_chain_length": 0.0,
        "max_chain_length": 0,
        "edge_selection_by_case": dict(state.last_selected_counts or {}),
        "candidate_counts_by_case": dict(state.last_case_counts or {}),
    }

    chains = _extract_all_chains_for_metrics(state)
    lengths = [len(c) for c in chains]
    if lengths:
        metrics["average_chain_length"] = float(np.mean(lengths))
        metrics["median_chain_length"] = float(np.median(lengths))
        metrics["max_chain_length"] = int(max(lengths))
        for L in lengths:
            metrics["chain_length_distribution"][int(L)] = metrics["chain_length_distribution"].get(int(L), 0) + 1

    for i in range(len(state.om_ids) - 1):
        om1 = state.om_ids[i]
        om2 = state.om_ids[i + 1]

        om1_nodes = [n for n in G.nodes if n[0] == om1]
        om2_nodes = [n for n in G.nodes if n[0] == om2]

        matches = sum(1 for u, v in G.edges() if u[0] == om1 and v[0] == om2)
        possible = min(len(om1_nodes), len(om2_nodes))
        rate = float(matches / possible) if possible > 0 else 0.0

        metrics["match_rate_by_om_pair"][f"{om1}->{om2}"] = {"matches": int(matches), "possible": int(possible), "rate": rate}
        metrics["total_possible_matches"] += int(possible)
        metrics["successful_matches"] += int(matches)

    if metrics["total_possible_matches"] > 0:
        metrics["overall_match_rate"] = float(metrics["successful_matches"] / metrics["total_possible_matches"])

    # Graph complexity summary.
    indeg = np.array([G.in_degree(n) for n in G.nodes], dtype=float) if G.nodes else np.array([], dtype=float)
    outdeg = np.array([G.out_degree(n) for n in G.nodes], dtype=float) if G.nodes else np.array([], dtype=float)
    metrics["graph_complexity"] = {
        "avg_in_degree": float(indeg.mean()) if indeg.size else 0.0,
        "avg_out_degree": float(outdeg.mean()) if outdeg.size else 0.0,
        "branching_nodes": int(sum(1 for n in G.nodes if G.out_degree(n) > 1 or G.in_degree(n) > 1)),
        "max_out_degree": int(outdeg.max()) if outdeg.size else 0,
        "max_in_degree": int(indeg.max()) if indeg.size else 0,
    }

    return metrics


def render_quality_report(metrics: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Tree Tracking Quality Report")
    lines.append(f"n_oms: {metrics.get('n_oms')}")
    lines.append(f"total_nodes: {metrics.get('total_nodes')}")
    lines.append(f"total_edges: {metrics.get('total_edges')}")
    lines.append(f"overall_match_rate: {metrics.get('overall_match_rate'):.3f}")
    lines.append(f"average_chain_length: {metrics.get('average_chain_length'):.2f}")
    lines.append(f"median_chain_length: {metrics.get('median_chain_length'):.2f}")
    lines.append(f"max_chain_length: {metrics.get('max_chain_length')}")

    lines.append("")
    lines.append("## Match Rates by Consecutive OM Pair")
    for pair, data in metrics.get("match_rate_by_om_pair", {}).items():
        lines.append(f"- {pair}: {data['matches']}/{data['possible']} ({data['rate']:.3f})")

    lines.append("")
    lines.append("## Chain Length Distribution")
    for length, count in sorted(metrics.get("chain_length_distribution", {}).items()):
        lines.append(f"- L={length}: {count}")

    lines.append("")
    lines.append("## Edge Selection by Case")
    sel = metrics.get("edge_selection_by_case", {})
    cand = metrics.get("candidate_counts_by_case", {})
    for case_name in sorted(sel.keys() | cand.keys()):
        s = int(sel.get(case_name, 0))
        c = int(cand.get(case_name, 0))
        ratio = (s / c) if c else 0.0
        lines.append(f"- {case_name}: selected {s} / candidates {c} (ratio {ratio:.2f})")

    gc = metrics.get("graph_complexity", {})
    lines.append("")
    lines.append("## Graph Complexity")
    lines.append(f"- avg_in_degree: {gc.get('avg_in_degree', 0.0):.2f}")
    lines.append(f"- avg_out_degree: {gc.get('avg_out_degree', 0.0):.2f}")
    lines.append(f"- branching_nodes: {gc.get('branching_nodes', 0)}")
    lines.append(f"- max_in_degree: {gc.get('max_in_degree', 0)}")
    lines.append(f"- max_out_degree: {gc.get('max_out_degree', 0)}")

    return "\n".join(lines)


def save_report_and_metrics(output_dir: Path, report: str, metrics: Dict[str, Any]) -> Tuple[Path, Path]:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "tracking_quality_report.md"
    report_path.write_text(report)

    metrics_path = reports_dir / "tracking_quality_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return report_path, metrics_path


def save_diagnostic_figures(output_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Path]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Path] = {}

    # Chain length distribution.
    dist = metrics.get("chain_length_distribution", {})
    if dist:
        xs = sorted(dist.keys())
        ys = [dist[x] for x in xs]
        plt.figure(figsize=(7, 4))
        plt.bar(xs, ys)
        plt.xlabel("Chain length")
        plt.ylabel("Count")
        plt.title("Chain Length Distribution")
        p = fig_dir / "chain_length_distribution.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        out["chain_length_distribution"] = p

    # Match rates by pair.
    by_pair = metrics.get("match_rate_by_om_pair", {})
    if by_pair:
        pairs = list(by_pair.keys())
        rates = [by_pair[k]["rate"] for k in pairs]
        plt.figure(figsize=(8, 4))
        plt.bar(pairs, rates)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.0)
        plt.ylabel("Match rate")
        plt.title("Match Rate by OM Pair")
        p = fig_dir / "match_rates_by_pair.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        out["match_rates_by_pair"] = p

    # Edge selection by case.
    sel = metrics.get("edge_selection_by_case", {})
    if sel:
        keys = list(sorted(sel.keys()))
        vals = [sel[k] for k in keys]
        plt.figure(figsize=(7, 4))
        plt.bar(keys, vals)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Selected edges")
        plt.title("Edges Selected by Match Case")
        p = fig_dir / "edge_selection_by_case.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        out["edge_selection_by_case"] = p

    return out


def _extract_all_chains_for_metrics(state: TrackingState) -> List[List[NodeId]]:
    # minimal duplicate of chain logic to avoid circular imports
    visited = set()
    chains: List[List[NodeId]] = []
    starts = [n for n in state.G.nodes if not list(state.G.predecessors(n))]

    def greedy(start: NodeId) -> List[NodeId]:
        chain = [start]
        cur = start
        while True:
            succ = list(state.G.successors(cur))
            if not succ:
                break
            best = max(succ, key=lambda n: state.G[cur][n].get("similarity", 0.0))
            chain.append(best)
            cur = best
        return chain

    for s in starts:
        if s in visited:
            continue
        c = greedy(s)
        chains.append(c)
        visited.update(c)

    for n in set(state.G.nodes) - visited:
        chains.append([n])

    return chains
