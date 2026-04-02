from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx

from .models import FilePair


@dataclass
class TrackingState:
    """In-memory state for a run."""

    file_pairs: List[FilePair]
    om_ids: List[int]

    crowns_gdfs: Dict[int, gpd.GeoDataFrame] = field(default_factory=dict)
    crown_attrs: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)

    ortho_paths: Dict[int, Path] = field(default_factory=dict)

    crown_crs: Dict[int, Optional[Any]] = field(default_factory=dict)
    ortho_crs: Dict[int, Optional[Any]] = field(default_factory=dict)

    alignment_shifts: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    multithreshold_layers: Dict[int, List[str]] = field(default_factory=dict)
    base_threshold_tag: Optional[str] = None

    G: nx.DiGraph = field(default_factory=nx.DiGraph)

    last_case_counts: Dict[str, int] = field(default_factory=dict)
    last_selected_counts: Dict[str, int] = field(default_factory=dict)
