from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


NodeId = Tuple[int, int]  # (om_id, crown_idx)


@dataclass(frozen=True)
class MatchCaseConfig:
    name: str
    base_similarity_weights: Dict[str, float]
    scoring_weights: Dict[str, float]
    similarity_threshold: float
    min_iou: float = 0.0
    min_overlap_prev: float = 0.0
    min_overlap_curr: float = 0.0
    max_centroid_dist: Optional[float] = None
    mutual_best: bool = False
    allow_multiple: bool = False
    max_edges_per_prev: Optional[int] = None
    max_edges_per_curr: Optional[int] = None


@dataclass(frozen=True)
class FilePair:
    om_id: int
    crowns_path: str
    ortho_path: Optional[str]


@dataclass
class RunArtifacts:
    """Paths to key outputs produced by a run."""

    output_dir: str
    quality_report_path: str
    metrics_json_path: str
    consensus_gpkg_path: str
    consensus_geojson_path: str
    chain_panels_dir: str
    consensus_panels_dir: str


@dataclass
class RunSummary:
    """High-level scalar results useful for notebooks."""

    n_oms: int
    n_nodes: int
    n_edges: int
    n_chains_total: int
    n_consensus: int
    alignment_shifts: Dict[int, Tuple[float, float]]
    notes: Dict[str, Any]
