from __future__ import annotations

"""End-to-end pipeline orchestration.

This is the single public entry point intended for notebooks.
It wires together the modular pieces:
- IO discovery
- multithreshold loading
- alignment
- graph construction
- chain extraction + optional gap-fill
- consensus crowns
- diagnostics + figures

All plots are saved to disk; the notebook prints textual summaries.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd

from shapely.affinity import translate

from .alignment import compute_cumulative_shifts
from .augmentation import augment_partial_chains_with_multithreshold
from .cases import strict_aligned_configs, ultra_relaxed_case_configs
from .chains import (
    assemble_high_quality_chains,
    normalize_chain_data,
    select_consensus_source_chains,
)
from .config import TrackingConfig
from .consensus import compute_consensus_polygon
from .diagnostics import (
    compute_quality_metrics,
    render_quality_report,
    save_diagnostic_figures,
    save_report_and_metrics,
)
from .graph_build import build_graph_conditional
from .io import discover_sit_pairs
from .models import RunArtifacts, RunSummary
from .multithreshold import build_multithreshold_cache, load_multithreshold_state
from .state import TrackingState
from .viz import save_chain_panels, save_consensus_panels, save_consensus_spatial_map


def _apply_alignment_to_state(state: TrackingState) -> None:
    """Translate crown geometries in-place according to state.alignment_shifts."""

    from shapely.affinity import translate

    for om_id in state.om_ids:
        dx, dy = state.alignment_shifts.get(om_id, (0.0, 0.0))
        if dx == 0.0 and dy == 0.0:
            continue
        gdf = state.crowns_gdfs[om_id].copy()
        gdf["geometry"] = gdf["geometry"].apply(lambda g: translate(g, xoff=dx, yoff=dy))
        state.crowns_gdfs[om_id] = gdf

        # Update cached attributes.
        from .geometry import compute_crown_attributes

        state.crown_attrs[om_id] = [compute_crown_attributes(row.geometry) for _, row in gdf.iterrows()]


def _export_consensus(consensus_gdf: gpd.GeoDataFrame, output_dir: Path) -> Tuple[Path, Path]:
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    gpkg_path = artifacts_dir / "consensus_crowns.gpkg"
    geojson_path = artifacts_dir / "consensus_crowns.geojson"

    # GPKG
    try:
        consensus_gdf.to_file(gpkg_path, driver="GPKG")
    except Exception:
        # fall back to GeoJSON only
        gpkg_path = artifacts_dir / "consensus_crowns_failed.gpkg"

    consensus_gdf.to_file(geojson_path, driver="GeoJSON")

    return gpkg_path, geojson_path


def run_tracking_pipeline(config: TrackingConfig) -> Tuple[RunSummary, RunArtifacts, str]:
    config.ensure_dirs()

    # 1) Discover SIT-style pairs.
    pairs = discover_sit_pairs(config.multithresh_dir, config.ortho_dir)
    if not pairs:
        raise FileNotFoundError(f"No multithreshold crown files found in {config.multithresh_dir}")

    # 2) Load multithreshold crowns into state (unaligned).
    state = load_multithreshold_state(
        pairs,
        base_threshold_tag=config.base_threshold_tag,
        include_base_for_non_reference=config.include_base_for_non_reference,
    )

    # 3) Alignment shifts.
    ortho_paths_by_id = state.ortho_paths
    alignment = compute_cumulative_shifts(
        ortho_paths_by_id,
        om_ids=state.om_ids,
        reference_om_id=min(state.om_ids) if state.om_ids else None,
        max_preview=config.phase_corr_max_preview,
    )
    state.alignment_shifts = alignment.shifts

    if config.use_phase_corr_alignment:
        _apply_alignment_to_state(state)

    # 4) Case configs.
    if config.classify_mode == "strict":
        case_configs = strict_aligned_configs()
    else:
        case_configs = ultra_relaxed_case_configs()
    case_order = ["one_to_one", "containment", "nearby"]

    # 5) Graph construction.
    build_graph_conditional(
        state,
        case_configs=case_configs,
        case_order=case_order,
        base_max_dist=config.base_max_dist,
        overlap_gate=config.overlap_gate,
        min_base_similarity=config.min_base_similarity,
        classify_mode=config.classify_mode,
        max_candidates_per_prev=config.max_candidates_per_prev,
        max_candidates_per_curr=config.max_candidates_per_curr,
    )

    # 6) Chains.
    hq = assemble_high_quality_chains(state)

    # 7) Optional gap-fill augmentation.
    augmentation_notes: Dict[str, Any] = {}
    if config.enable_gap_fill:
        mt_cache, thresholds_desc = build_multithreshold_cache(
            state,
            multithresh_dir=config.multithresh_dir,
            alignment=alignment,
            include_base=False,
            min_threshold_tag=config.gapfill_min_threshold_tag,
        )
        base_categories = hq["categories"]
        aug = augment_partial_chains_with_multithreshold(
            state,
            base_chain_categories=base_categories,
            mt_cache=mt_cache,
            thresholds_desc=thresholds_desc,
            max_centroid_dist=config.gapfill_max_centroid_dist,
            min_iou=config.gapfill_min_iou,
            duplicate_iou=config.gapfill_duplicate_iou,
        )
        augmentation_notes = {"gap_fill": aug}

        # Re-extract high-quality chains after augmentation.
        hq = assemble_high_quality_chains(state)

    # 8) Choose consensus source chains.
    consensus_source_chains = select_consensus_source_chains(
        state,
        hq_result=hq,
        include_partial=config.include_partial_for_consensus,
        min_partial_len=config.min_partial_len,
        min_partial_one_to_one_ratio=config.min_partial_one_to_one_ratio,
    )

    # 9) Consensus crowns.
    consensus_geoms = []
    consensus_polys_for_panels = []
    chain_ids = []
    chain_lengths = []
    qualities = []
    avg_sims = []

    for idx, chain_data in enumerate(consensus_source_chains, 1):
        chain, avg_sim, quality = normalize_chain_data(chain_data)
        poly = compute_consensus_polygon(state, chain, method=config.consensus_method)
        if poly is None or poly.is_empty:
            continue
        consensus_geoms.append(poly)
        consensus_polys_for_panels.append(poly)
        chain_ids.append(idx)
        chain_lengths.append(len(chain))
        qualities.append(quality)
        avg_sims.append(avg_sim)

    base_crs = state.crowns_gdfs[state.om_ids[0]].crs if state.om_ids else None
    consensus_gdf = gpd.GeoDataFrame(
        {
            "chain_id": chain_ids,
            "chain_length": chain_lengths,
            "quality": qualities,
            "avg_similarity": avg_sims,
            "geometry": consensus_geoms,
        },
        crs=base_crs,
    )

    gpkg_path, geojson_path = _export_consensus(consensus_gdf, config.output_dir)

    # 10) Diagnostics.
    metrics = compute_quality_metrics(state)
    metrics["consensus"] = {"n_consensus": int(len(consensus_gdf))}
    metrics["alignment"] = {"method": alignment.method, "shifts": state.alignment_shifts}
    metrics["notes"] = augmentation_notes

    report = render_quality_report(metrics)
    report_path, metrics_path = save_report_and_metrics(config.output_dir, report, metrics)
    figs = save_diagnostic_figures(config.output_dir, metrics)

    # 11) Visualization panels.
    chain_panels_dir = config.output_dir / "figures" / "chains"
    consensus_panels_dir = config.output_dir / "figures" / "consensus_chains"

    if config.save_chain_panels:
        if config.chain_panels_mode == "all_extracted":
            chains_for_panels = hq.get("all_extracted_chains", [])
        else:
            # Default: only generate panels for the same chains that produce consensus crowns.
            chains_for_panels = consensus_source_chains
        save_chain_panels(state, chains_for_panels, chain_panels_dir, max_panels=config.max_panels)

    if config.save_consensus_panels:
        # Note: consensus_source_chains is a list of chains (no backbones dicts here),
        # but we keep the interface generic.
        save_consensus_panels(state, consensus_source_chains, consensus_polys_for_panels, consensus_panels_dir, max_panels=config.max_panels)

    save_consensus_spatial_map(consensus_gdf, config.output_dir / "figures" / "consensus_spatial_map.png")

    summary = RunSummary(
        n_oms=len(state.om_ids),
        n_nodes=int(state.G.number_of_nodes()),
        n_edges=int(state.G.number_of_edges()),
        n_chains_total=int(sum(metrics.get("chain_length_distribution", {}).values())),
        n_consensus=int(len(consensus_gdf)),
        alignment_shifts=dict(state.alignment_shifts),
        notes={"augmentation": augmentation_notes, "figures": {k: str(v) for k, v in figs.items()}},
    )

    artifacts = RunArtifacts(
        output_dir=str(config.output_dir),
        quality_report_path=str(report_path),
        metrics_json_path=str(metrics_path),
        consensus_gpkg_path=str(gpkg_path),
        consensus_geojson_path=str(geojson_path),
        chain_panels_dir=str(chain_panels_dir),
        consensus_panels_dir=str(consensus_panels_dir),
    )

    return summary, artifacts, report
