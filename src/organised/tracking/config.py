from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TrackingConfig:
    # Inputs
    ortho_dir: Path
    multithresh_dir: Path

    # Outputs
    output_dir: Path

    # Alignment
    use_phase_corr_alignment: bool = True
    phase_corr_max_preview: int = 1200

    # Matching / graph build
    base_max_dist: float = 30.0
    overlap_gate: float = 0.10
    min_base_similarity: float = 0.30
    classify_mode: str = "balanced"  # balanced|strict|lite

    # Multithreshold loading
    base_threshold_tag: Optional[str] = None  # e.g. "conf_0p45"; if None uses max threshold in meta/layers
    include_base_for_non_reference: bool = True

    # Candidate trimming (perf)
    max_candidates_per_prev: Optional[int] = 50
    max_candidates_per_curr: Optional[int] = 50

    # Chain selection
    include_partial_for_consensus: bool = True
    min_partial_len: int = 5
    min_partial_one_to_one_ratio: float = 0.9

    # Gap-fill augmentation
    enable_gap_fill: bool = True
    gapfill_min_threshold_tag: str = "conf_0p15"
    gapfill_max_centroid_dist: float = 25.0
    gapfill_min_iou: float = 0.20
    gapfill_duplicate_iou: float = 0.70

    # Consensus
    consensus_method: str = "medoid"  # medoid|intersection_core|union_shrink

    # Visualizations
    save_chain_panels: bool = True
    save_consensus_panels: bool = True
    chain_panels_mode: str = "consensus"  # consensus|all_extracted
    max_panels: Optional[int] = None  # None = all

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
