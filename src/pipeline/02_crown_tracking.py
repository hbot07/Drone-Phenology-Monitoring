#!/usr/bin/env python3
"""
Pipeline Step 2: Crown Tracking + Consensus Crown Generation

Loads multi-threshold detectree2 crowns for all OMs, aligns them using
PCC-tiled method, builds a tracking graph, assembles high-quality chains,
generates and deduplicates consensus crowns.

Requires: detectree conda environment

Usage:
    python 02_crown_tracking.py --config /path/to/pipeline_config.json \\
        [--base-threshold-tag conf_0p45] \\
        [--align-threshold-tag conf_0p65] \\
        [--align-method pcc_tiled] \\
        [--base-max-dist 30.0] \\
        [--min-partial-len 5]       # default: 5 (matches notebook baseline)
        [--min-partial-ratio 0.9]   # default: 0.9 (matches notebook baseline)
        [--dedup-iou 0.75] \\
        [--skip-chain-viz] \\
        [--skip-if-done]

NOTE: For LHC dataset you MUST exclude the bad Dec-9 OM via --exclude-stems in
step 0 (00_discover_oms.py), or that OM will corrupt tracking by introducing
a grossly misaligned step in the sequence.

Reads:
    <output_dir>/pipeline_config.json
    <crowns_dir>/{stem}_multithreshold.gpkg   (from step 1)
    <om_dir>/{stem}.tif

Writes:
    <tracking_dir>/consensus_crowns_complete_all.gpkg    (deduped)
    <tracking_dir>/consensus_crowns_complete_all_raw.gpkg
    <tracking_dir>/consensus_crowns_om1_phenology.geojson
    <tracking_dir>/tracking_quality_report.txt
    <tracking_dir>/tracking_quality_metrics.json
    <tracking_dir>/consensus_crowns_summary.json
    <tracking_dir>/chain_viz/   (optional - chain strip images)
    <tracking_dir>/consensus_viz/   (optional - consensus strip images)
    <tracking_dir>/consensus_overlay_om1.png
    <tracking_dir>/diagnostics/alignment_shifts.csv
    <tracking_dir>/diagnostics/alignment_shifts.png
    <tracking_dir>/diagnostics/match_rates_by_pair.png
    <tracking_dir>/diagnostics/chain_length_distribution.png
    <tracking_dir>/diagnostics/chain_breakdown.json
    <tracking_dir>/diagnostics/consensus_overlay_om1_raw.png
    <tracking_dir>/diagnostics/tracking_diagnostics_report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def save_config(config: dict, config_path: Path) -> None:
    config_path.write_text(json.dumps(config, indent=2))


def setup_app_dir(project_root: Path) -> None:
    app_dir = str(project_root / "src" / "flask_app_tracking")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)


def build_pairs_and_om_stems(
    config: dict,
) -> Tuple[List[Tuple[str, str, str]], Dict[int, str]]:
    """Return (pairs, om_stems) from config.

    pairs: list of (gpkg_path, ortho_path, stem)
    om_stems: {1: stem1, 2: stem2, ...}
    """
    raw_pairs = config["pairs"]  # [[gpkg, tif, stem], ...]
    crowns_dir = Path(config["crowns_dir"])
    om_dir = Path(config["om_dir"])

    pairs = []
    om_stems = {}
    for i, (gpkg_raw, tif_raw, stem) in enumerate(raw_pairs, 1):
        # Use paths from config directly if they exist; fall back to reconstructed paths
        gpkg_from_config = Path(gpkg_raw)
        tif_from_config = Path(tif_raw)
        if gpkg_from_config.exists():
            gpkg = str(gpkg_from_config)
        else:
            gpkg = str(crowns_dir / f"{stem}_multithreshold.gpkg")
        if tif_from_config.exists():
            tif = str(tif_from_config)
        else:
            tif = str(om_dir / f"{stem}.tif")
        pairs.append((gpkg, tif, stem))
        om_stems[i] = stem

    return pairs, om_stems


def save_overlay_png(
    tracker,
    pairs: List[Tuple[str, str, str]],
    consensus_gdf,
    out_path: Path,
) -> None:
    """Generate a simple matplotlib overlay of consensus crowns on OM1."""
    try:
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from shapely.geometry import mapping
        import numpy as np

        _gpkg, ortho_path, stem = pairs[0]
        with rasterio.open(ortho_path) as src:
            scale = max(src.width, src.height) / 2000 if max(src.width, src.height) > 2000 else 1.0
            out_w = int(round(src.width / scale))
            out_h = int(round(src.height / scale))
            data = src.read([1, 2, 3], out_shape=(3, out_h, out_w),
                            resampling=rasterio.enums.Resampling.bilinear)
            img = np.moveaxis(data, 0, -1)
            img = img.astype(np.float32)
            lo, hi = float(np.nanpercentile(img, 2)), float(np.nanpercentile(img, 98))
            if hi > lo:
                img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(img, extent=extent, origin="upper", aspect="equal")

        dx, dy = tracker.alignment_shifts.get(1, (0.0, 0.0))
        for _, row in consensus_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            from shapely.affinity import translate
            geom_raw = translate(geom, xoff=-dx, yoff=-dy)
            if geom_raw.geom_type == "Polygon":
                xs, ys = geom_raw.exterior.xy
                ax.plot(xs, ys, color="#ff7f0e", linewidth=0.4, alpha=0.7)
            elif geom_raw.geom_type == "MultiPolygon":
                for part in geom_raw.geoms:
                    xs, ys = part.exterior.xy
                    ax.plot(xs, ys, color="#ff7f0e", linewidth=0.4, alpha=0.7)

        ax.set_title(f"Consensus crowns overlaid on OM01: {stem}\n{len(consensus_gdf)} crowns")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved overlay: {out_path}")
    except Exception as e:
        print(f"Warning: could not generate overlay PNG: {e}")


def save_alignment_viz(
    tracker,
    om_stems: Dict[int, str],
    out_path: Path,
) -> None:
    """Save a bar chart of per-OM alignment shifts (dx, dy)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        om_ids = sorted(tracker.om_ids)
        dxs = [tracker.alignment_shifts.get(oid, (0.0, 0.0))[0] for oid in om_ids]
        dys = [tracker.alignment_shifts.get(oid, (0.0, 0.0))[1] for oid in om_ids]
        labels = [f"OM{oid}\n{om_stems.get(oid, '')}" for oid in om_ids]
        x = np.arange(len(om_ids))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(om_ids) * 1.2), 4))
        bars_dx = ax.bar(x - width / 2, dxs, width, label="dx (X shift)", color="#1f77b4", alpha=0.8)
        bars_dy = ax.bar(x + width / 2, dys, width, label="dy (Y shift)", color="#ff7f0e", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Shift (CRS units)")
        ax.set_title(f"Alignment shifts per OM (method: {getattr(tracker, 'alignment_method', 'unknown')})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved alignment viz: {out_path.name}")
    except Exception as e:
        print(f"  Warning: alignment viz failed: {e}")


def save_graph_stats_viz(
    quality_metrics: dict,
    num_oms: int,
    out_dir: Path,
) -> None:
    """Save match-rate bar chart and chain-length histogram to out_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Match rates by OM pair
        match_rates = quality_metrics.get("match_rate_by_om_pair", {})
        if match_rates:
            pairs_sorted = sorted(match_rates.keys(),
                                  key=lambda k: int(k.split("->")[0]))
            rates = [match_rates[p].get("rate", 0) if isinstance(match_rates[p], dict)
                     else float(match_rates[p])
                     for p in pairs_sorted]
            fig, ax = plt.subplots(figsize=(max(6, len(pairs_sorted) * 0.9), 4))
            colors = ["#2ca02c" if r <= 1.0 else "#d62728" for r in rates]
            ax.bar(pairs_sorted, rates, color=colors, alpha=0.85)
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="1.0 ideal")
            ax.set_xlabel("OM pair")
            ax.set_ylabel("Match rate")
            ax.set_title("Match rate per consecutive OM pair\n(>1.0 = many-to-many, bad OM likely)")
            ax.legend()
            plt.xticks(rotation=30, ha="right", fontsize=8)
            plt.tight_layout()
            mr_path = out_dir / "match_rates_by_pair.png"
            plt.savefig(str(mr_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved match rates chart: {mr_path.name}")

        # 2) Chain length distribution
        chain_dist = quality_metrics.get("chain_length_distribution", {})
        if chain_dist:
            lengths = sorted(int(k) for k in chain_dist.keys())
            # keys may be int or str depending on source
            xlabels = [str(l) for l in lengths]
            counts = [int(chain_dist.get(l) or chain_dist.get(str(l)) or 0) for l in lengths]
            fig, ax = plt.subplots(figsize=(max(6, len(lengths) * 0.8), 4))
            bar_colors = ["#d62728" if l == num_oms else "#1f77b4" for l in lengths]
            ax.bar(xlabels, counts, color=bar_colors, alpha=0.85)
            ax.set_xlabel("Chain length")
            ax.set_ylabel("Number of chains")
            title = "Chain length distribution"
            if num_oms in lengths:
                title += f"\n(red = full-length chains, len={num_oms})"
            else:
                title += f"\n(full-length = {num_oms}, none present)"
            ax.set_title(title)
            plt.tight_layout()
            cl_path = out_dir / "chain_length_distribution.png"
            plt.savefig(str(cl_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved chain length histogram: {cl_path.name}")

    except Exception as e:
        print(f"  Warning: graph stats viz failed: {e}")


def save_raw_overlay_png(
    tracker,
    pairs: List[Tuple[str, str, str]],
    consensus_gdf_raw,
    out_path: Path,
    title_suffix: str = "RAW (before dedup)",
) -> None:
    """Generate overlay of raw (pre-dedup) consensus crowns on OM1."""
    try:
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from shapely.affinity import translate

        _gpkg, ortho_path, stem = pairs[0]
        with rasterio.open(ortho_path) as src:
            scale = max(src.width, src.height) / 2000 if max(src.width, src.height) > 2000 else 1.0
            out_w = int(round(src.width / scale))
            out_h = int(round(src.height / scale))
            data = src.read([1, 2, 3], out_shape=(3, out_h, out_w),
                            resampling=rasterio.enums.Resampling.bilinear)
            img = np.moveaxis(data, 0, -1).astype(np.float32)
            lo, hi = float(np.nanpercentile(img, 2)), float(np.nanpercentile(img, 98))
            if hi > lo:
                img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(img, extent=extent, origin="upper", aspect="equal")

        dx, dy = tracker.alignment_shifts.get(1, (0.0, 0.0))
        for _, row in consensus_gdf_raw.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            geom_raw = translate(geom, xoff=-dx, yoff=-dy)
            if geom_raw.geom_type == "Polygon":
                xs, ys = geom_raw.exterior.xy
                ax.plot(xs, ys, color="#1f77b4", linewidth=0.4, alpha=0.6)
            elif geom_raw.geom_type == "MultiPolygon":
                for part in geom_raw.geoms:
                    xs, ys = part.exterior.xy
                    ax.plot(xs, ys, color="#1f77b4", linewidth=0.4, alpha=0.6)

        ax.set_title(f"Consensus crowns overlaid on OM01 ({title_suffix}): {stem}\n{len(consensus_gdf_raw)} crowns")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved raw overlay: {out_path.name}")
    except Exception as e:
        print(f"  Warning: could not generate raw overlay PNG: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Step 2: Crown Tracking")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-threshold-tag", default="conf_0p45")
    parser.add_argument("--align-threshold-tag", default="conf_0p65")
    parser.add_argument("--align-method", default="pcc_tiled")
    parser.add_argument("--base-max-dist", type=float, default=30.0)
    parser.add_argument("--overlap-gate", type=float, default=0.10)
    parser.add_argument("--min-base-similarity", type=float, default=0.30)
    parser.add_argument("--classify-mode", default="balanced")
    parser.add_argument("--min-partial-len", type=int, default=None,
                        help="Minimum chain length for partial chains. "
                             "Default: 5 (matches notebook baseline).")
    parser.add_argument("--min-partial-ratio", type=float, default=0.9,
                        help="Min one-to-one ratio for partial chains (default: 0.9, matches notebook baseline)")
    parser.add_argument("--disable-gap-fill", action="store_true",
                        help="Disable multithreshold gap-fill augmentation used by the notebook pipeline")
    parser.add_argument("--gapfill-min-threshold-tag", default="conf_0p15",
                        help="Lowest threshold layer considered during gap-fill augmentation")
    parser.add_argument("--gapfill-max-centroid-dist", type=float, default=25.0,
                        help="Maximum centroid distance for multithreshold gap-fill candidates")
    parser.add_argument("--gapfill-min-iou", type=float, default=0.20,
                        help="Minimum IoU for multithreshold gap-fill candidates")
    parser.add_argument("--gapfill-duplicate-iou", type=float, default=0.70,
                        help="IoU above which a gap-fill candidate is treated as duplicate")
    parser.add_argument("--dedup-iou", type=float, default=0.75)
    parser.add_argument("--dedup-containment-buffer", type=float, default=5.0)
    parser.add_argument("--skip-chain-viz", action="store_true",
                        help="Skip chain strip visualizations (faster)")
    parser.add_argument("--skip-consensus-viz", action="store_true",
                        help="Skip consensus strip visualizations (faster)")
    parser.add_argument("--skip-diagnostics", action="store_true",
                        help="Skip all diagnostic PNG/CSV outputs (match rates, chain histogram, overlay)")
    parser.add_argument("--skip-if-done", action="store_true",
                        help="Skip if consensus GPKG already exists")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    project_root = Path(config["project_root"])
    tracking_dir = Path(config["tracking_dir"])
    run_tag = config.get("run_name", "pipeline")

    tracking_dir.mkdir(parents=True, exist_ok=True)

    consensus_gpkg = tracking_dir / "consensus_crowns_complete_all.gpkg"

    if args.skip_if_done and consensus_gpkg.exists():
        print(f"[SKIP] Tracking already done: {consensus_gpkg}")
        return 0

    # --- Inject app dir into sys.path ---
    setup_app_dir(project_root)
    from tree_tracking import TreeTrackingGraph

    # --- Build pairs and om_stems ---
    pairs, om_stems = build_pairs_and_om_stems(config)
    num_oms = len(pairs)

    print(f"Run tag: {run_tag}")
    print(f"OMs: {num_oms}")
    for i, (gpkg, tif, stem) in enumerate(pairs, 1):
        exists_g = Path(gpkg).exists()
        exists_t = Path(tif).exists()
        status = "OK" if (exists_g and exists_t) else f"gpkg={'✓' if exists_g else '✗'} tif={'✓' if exists_t else '✗'}"
        print(f"  OM{i:02d}: {stem} [{status}]")

    missing = [stem for (gpkg, tif, stem) in pairs if not Path(gpkg).exists() or not Path(tif).exists()]
    if missing:
        print(f"\nERROR: Missing files for: {missing}", file=sys.stderr)
        print("Run step 1 first (crown detection).", file=sys.stderr)
        return 1

    crowns_dir = Path(config["crowns_dir"])
    om_dir = Path(config["om_dir"])

    # --- Initialize tracker ---
    tracker = TreeTrackingGraph(
        auto_discover=False,
        multithresh_dir=str(crowns_dir),
        ortho_dir=str(om_dir),
        output_dir=str(tracking_dir),
        simplify_tol=1.0,
        resize_factor=0.1,
        max_crowns_preview=200,
    )
    tracker.file_pairs = [(gpkg, tif) for gpkg, tif, _ in pairs]
    tracker.om_ids = list(range(1, num_oms + 1))
    tracker.base_threshold_tag = None

    # --- Load and align ---
    print(f"\nLoading multithreshold crowns (base={args.base_threshold_tag}, "
          f"align={args.align_method}/{args.align_threshold_tag})...")
    tracker.load_multithreshold_data(
        base_threshold_tag=args.base_threshold_tag,
        load_images=True,
        align=True,
        align_method=args.align_method,
        align_threshold_tag=args.align_threshold_tag,
    )
    print("Alignment shifts:")
    for om_id in tracker.om_ids:
        dx, dy = tracker.alignment_shifts.get(om_id, (0.0, 0.0))
        print(f"  OM{om_id:02d} ({om_stems[om_id]}): dx={dx:.2f}, dy={dy:.2f}")

    # --- Save alignment diagnostics ---
    diag_dir = tracking_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    alignment_csv_path = diag_dir / "alignment_shifts.csv"
    with open(alignment_csv_path, "w") as f:
        f.write("om_id,stem,dx,dy\n")
        for om_id in tracker.om_ids:
            dx, dy = tracker.alignment_shifts.get(om_id, (0.0, 0.0))
            f.write(f"{om_id},{om_stems[om_id]},{dx:.6f},{dy:.6f}\n")
    print(f"  Saved alignment CSV: {alignment_csv_path.name}")
    save_alignment_viz(tracker, om_stems, diag_dir / "alignment_shifts.png")

    # --- Build graph ---
    print(f"\nBuilding tracking graph (base_max_dist={args.base_max_dist}, "
          f"classify_mode={args.classify_mode})...")
    tracker.case_configs = tracker.make_strict_aligned_configs()
    tracker.build_graph_conditional(
        base_max_dist=args.base_max_dist,
        overlap_gate=args.overlap_gate,
        min_base_similarity=args.min_base_similarity,
        classify_mode=args.classify_mode,
    )
    print(f"Graph: {tracker.G.number_of_nodes()} nodes, {tracker.G.number_of_edges()} edges")

    # --- Quality report ---
    print("\nGenerating quality report...")
    quality_text, quality_metrics = tracker.quality_report()
    tracker.save_text(quality_text, "tracking_quality_report.txt")
    tracker.save_json(quality_metrics, "tracking_quality_metrics.json")
    print(f"  Overall match rate: {quality_metrics.get('overall_match_rate', 0):.3f}")
    print(f"  Avg chain length: {quality_metrics.get('average_chain_length', 0):.2f}")

    # --- Save graph diagnostic visualizations ---
    if not args.skip_diagnostics:
        print("\nSaving graph diagnostic visualizations...")
        save_graph_stats_viz(quality_metrics, num_oms, diag_dir)
    else:
        print("  [skip-diagnostics] Skipping graph stat charts")

    # --- Assemble chains ---
    print("\nAssembling high-quality chains...")
    hq_result = tracker.assemble_high_quality_chains()
    cats = hq_result.get("categories", {})
    n_clean = len(hq_result.get("clean_chains", []))
    n_branching = len(hq_result.get("branching_chains", []))
    n_backbones = len(hq_result.get("extracted_backbones", []))
    print(f"  Full chains: {n_clean}")
    print(f"  Branching chains: {n_branching}")
    print(f"  Extracted backbones: {n_backbones}")

    gapfill_summary = {
        "enabled": not args.disable_gap_fill,
        "augmented_nodes": 0,
        "by_threshold": {},
    }
    if not args.disable_gap_fill:
        print("\nRunning multithreshold gap-fill augmentation...")
        mt_cache, thresholds_desc = tracker.build_multithreshold_cache(
            include_base=False,
            min_threshold_tag=args.gapfill_min_threshold_tag,
        )
        gapfill_summary = tracker.augment_partial_chains_with_multithreshold(
            base_chain_categories=cats,
            mt_cache=mt_cache,
            thresholds_desc=thresholds_desc,
            max_centroid_dist=args.gapfill_max_centroid_dist,
            min_iou=args.gapfill_min_iou,
            duplicate_iou=args.gapfill_duplicate_iou,
        )
        gapfill_summary["enabled"] = True
        print(f"  Gap-fill augmented nodes: {gapfill_summary.get('augmented_nodes', 0)}")
        if gapfill_summary.get("by_threshold"):
            print(f"  By threshold: {gapfill_summary['by_threshold']}")

        if gapfill_summary.get("augmented_nodes", 0) > 0:
            print("  Recomputing quality report after gap-fill...")
            quality_text, quality_metrics = tracker.quality_report()
            tracker.save_text(quality_text, "tracking_quality_report.txt")
            tracker.save_json(quality_metrics, "tracking_quality_metrics.json")
            print(f"  Updated overall match rate: {quality_metrics.get('overall_match_rate', 0):.3f}")
            print(f"  Updated avg chain length: {quality_metrics.get('average_chain_length', 0):.2f}")

        print("  Reassembling chains after gap-fill...")
        hq_result = tracker.assemble_high_quality_chains()
        cats = hq_result.get("categories", {})
        n_clean = len(hq_result.get("clean_chains", []))
        n_branching = len(hq_result.get("branching_chains", []))
        n_backbones = len(hq_result.get("extracted_backbones", []))
        print(f"  Full chains: {n_clean}")
        print(f"  Branching chains: {n_branching}")
        print(f"  Extracted backbones: {n_backbones}")

    if args.min_partial_len is None:
        min_partial_len = 5  # matches notebook baseline for both SIT (14 OMs) and LHC (8 OMs)
        print(f"  min_partial_len: default → {min_partial_len}")
    else:
        min_partial_len = args.min_partial_len
        print(f"  min_partial_len: {min_partial_len} (explicit)")

    all_extracted_chains = tracker.select_consensus_source_chains(
        hq_result,
        include_partial=True,
        min_partial_len=min_partial_len,
        min_partial_one_to_one_ratio=args.min_partial_ratio,
    )
    n_partial_added = len(all_extracted_chains) - (n_clean + n_backbones)
    print(f"  Partial chains added (len>={min_partial_len}, ratio>={args.min_partial_ratio}): {n_partial_added}")
    print(f"  Total chains for consensus: {len(all_extracted_chains)}")

    # Save chain breakdown to diagnostics
    chain_breakdown = {
        "num_oms": num_oms,
        "full_chains": n_clean,
        "branching_chains": n_branching,
        "extracted_backbones": n_backbones,
        "partial_chains_added": n_partial_added,
        "total_consensus_chains": len(all_extracted_chains),
        "min_partial_len": min_partial_len,
        "min_partial_ratio": args.min_partial_ratio,
        "gap_fill": gapfill_summary,
    }
    chain_breakdown_path = diag_dir / "chain_breakdown.json"
    chain_breakdown_path.write_text(json.dumps(chain_breakdown, indent=2))
    print(f"  Saved chain breakdown: {chain_breakdown_path.name}")
    print(f"  Total chains for consensus: {len(all_extracted_chains)}")

    # --- Chain visualizations (optional) ---
    if not args.skip_chain_viz:
        print("\nGenerating chain visualizations...")
        chain_viz_dir = tracking_dir / "chain_viz"
        try:
            tracker.visualize_all_chains(all_extracted_chains, str(chain_viz_dir))
            print(f"  Saved to: {chain_viz_dir}")
        except Exception as e:
            print(f"  Warning: chain viz failed: {e}")

    # --- Generate consensus crowns ---
    print("\nGenerating consensus crowns...")
    consensus_gdf_raw = tracker.generate_consensus_crowns(all_extracted_chains)
    print(f"  Raw consensus crowns: {len(consensus_gdf_raw)}")

    # Save raw
    consensus_gpkg_raw = tracking_dir / "consensus_crowns_complete_all_raw.gpkg"
    consensus_gdf_raw.to_file(str(consensus_gpkg_raw), driver="GPKG",
                              layer="consensus_crowns_raw")
    print(f"  Saved raw: {consensus_gpkg_raw.name}")

    # Save raw overlay (before dedup) for visual inspection
    if not args.skip_diagnostics:
        print("\nSaving raw consensus overlay (before dedup)...")
        raw_overlay_png = diag_dir / "consensus_overlay_om1_raw.png"
        save_raw_overlay_png(tracker, pairs, consensus_gdf_raw, raw_overlay_png)
    else:
        print("  [skip-diagnostics] Skipping raw overlay PNG")

    # --- Deduplicate ---
    print(f"\nDeduplicating (iou_threshold={args.dedup_iou}, "
          f"containment_buffer={args.dedup_containment_buffer})...")
    consensus_gdf, dedup_summary = tracker.deduplicate_crowns(
        consensus_gdf_raw,
        iou_threshold=args.dedup_iou,
        drop_contained=True,
        containment_buffer=args.dedup_containment_buffer,
    )
    print(f"  After dedup: {len(consensus_gdf)} crowns "
          f"(dropped {dedup_summary.get('dropped_iou', 0)} IoU + "
          f"{dedup_summary.get('dropped_contained', 0)} contained)")

    # Save cleaned
    consensus_gdf.to_file(str(consensus_gpkg), driver="GPKG",
                          layer="consensus_crowns")
    print(f"  Saved cleaned: {consensus_gpkg.name}")

    # --- Consensus visualizations (optional) ---
    if not args.skip_consensus_viz:
        print("\nGenerating consensus visualizations...")
        consensus_viz_dir = tracking_dir / "consensus_viz"
        try:
            tracker.visualize_all_consensus_chains(all_extracted_chains,
                                                   str(consensus_viz_dir))
            print(f"  Saved to: {consensus_viz_dir}")
        except Exception as e:
            print(f"  Warning: consensus viz failed: {e}")

    # --- Overlay PNG ---
    overlay_png = tracking_dir / "consensus_overlay_om1.png"
    save_overlay_png(tracker, pairs, consensus_gdf, overlay_png)

    # --- Save OM1 GeoJSON for phenology ---
    print("\nSaving OM1 phenology GeoJSON...")
    geojson_path = tracking_dir / "consensus_crowns_om1_phenology.geojson"
    try:
        import geopandas as gpd
        import rasterio

        om1_ortho = pairs[0][1]
        with rasterio.open(om1_ortho) as src:
            raster_crs = src.crs

        geojson_gdf = consensus_gdf.copy()
        if geojson_gdf.crs is None and raster_crs:
            geojson_gdf = geojson_gdf.set_crs(raster_crs, allow_override=True)
        elif raster_crs and geojson_gdf.crs != raster_crs:
            geojson_gdf = geojson_gdf.to_crs(raster_crs)

        geojson_gdf.to_file(str(geojson_path), driver="GeoJSON")
        print(f"  Saved: {geojson_path.name}")
    except Exception as e:
        print(f"  Warning: failed to save phenology GeoJSON: {e}")

    # --- Summary ---
    summary = {
        "run_tag": run_tag,
        "num_oms": num_oms,
        "chains_total": len(all_extracted_chains),
        "consensus_raw": len(consensus_gdf_raw),
        "consensus_cleaned": len(consensus_gdf),
        "min_partial_len_used": min_partial_len,
        "min_partial_ratio_used": args.min_partial_ratio,
        "dedup_summary": dedup_summary,
        "quality_metrics": quality_metrics,
        "align_method": args.align_method,
        "base_threshold_tag": args.base_threshold_tag,
        "align_threshold_tag": args.align_threshold_tag,
        "gap_fill": gapfill_summary,
        "alignment_shifts": {
            str(oid): list(tracker.alignment_shifts.get(oid, (0.0, 0.0)))
            for oid in tracker.om_ids
        },
        "chain_breakdown": chain_breakdown,
        "diagnostics_dir": str(diag_dir),
    }
    summary_path = tracking_dir / "consensus_crowns_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary: {summary_path}")

    # --- Human-readable diagnostics report ---
    diag_report_lines = [
        "=" * 60,
        "TRACKING DIAGNOSTICS REPORT",
        "=" * 60,
        f"Run tag:         {run_tag}",
        f"Num OMs:         {num_oms}",
        f"OMs included:    {[om_stems[i] for i in sorted(om_stems.keys())]}",
        "",
        "ALIGNMENT",
        f"  Method:          {args.align_method}",
        f"  Threshold tag:   {args.align_threshold_tag}",
    ]
    for om_id in sorted(tracker.om_ids):
        dx, dy = tracker.alignment_shifts.get(om_id, (0.0, 0.0))
        diag_report_lines.append(f"  OM{om_id:02d} {om_stems[om_id]:40s}  dx={dx:+.3f}  dy={dy:+.3f}")
    diag_report_lines += [
        "",
        "GRAPH / TRACKING",
        f"  Nodes (trees):     {quality_metrics.get('total_trees_detected', '?')}",
        f"  Edges (matches):   {quality_metrics.get('total_edges', quality_metrics.get('successful_matches', '?'))}",
        f"  Overall match rate: {quality_metrics.get('overall_match_rate', 0):.3f}",
        f"  Avg chain length:  {quality_metrics.get('average_chain_length', 0):.2f}",
        f"  Max chain length:  {quality_metrics.get('max_chain_length', '?')}",
        "",
        "CHAIN ASSEMBLY",
        f"  Full chains:          {n_clean}",
        f"  Branching chains:     {n_branching}",
        f"  Extracted backbones:  {n_backbones}",
        f"  Gap-fill augmented:   {gapfill_summary.get('augmented_nodes', 0)}",
        f"  Partial chains added: {n_partial_added}  (min_len={min_partial_len}, min_ratio={args.min_partial_ratio})",
        f"  Total for consensus:  {len(all_extracted_chains)}",
        "",
        "CONSENSUS CROWNS",
        f"  Raw (before dedup):   {len(consensus_gdf_raw)}",
        f"  Dropped (IoU):        {dedup_summary.get('dropped_iou', 0)}",
        f"  Dropped (contained):  {dedup_summary.get('dropped_contained', 0)}",
        f"  Final (after dedup):  {len(consensus_gdf)}",
        "",
        "SAVED DIAGNOSTICS",
        f"  diagnostics/alignment_shifts.csv",
        f"  diagnostics/alignment_shifts.png",
        f"  diagnostics/match_rates_by_pair.png",
        f"  diagnostics/chain_length_distribution.png",
        f"  diagnostics/chain_breakdown.json",
        f"  diagnostics/consensus_overlay_om1_raw.png",
        "=" * 60,
    ]
    diag_report = "\n".join(diag_report_lines)
    diag_report_path = diag_dir / "tracking_diagnostics_report.txt"
    diag_report_path.write_text(diag_report)
    print(diag_report)
    print(f"\nDiagnostics saved to: {diag_dir}")

    # --- Update config ---
    config["tracking_dir"] = str(tracking_dir)
    config["consensus_gpkg"] = str(consensus_gpkg)
    config["consensus_gpkg_raw"] = str(consensus_gpkg_raw)
    config["consensus_geojson_om1"] = str(geojson_path)
    config["consensus_overlay_om1"] = str(overlay_png)
    config["tracking_quality_report"] = str(tracking_dir / "tracking_quality_report.txt")
    config["tracking_quality_metrics"] = str(tracking_dir / "tracking_quality_metrics.json")
    # Save alignment shifts so steps 3/4 use identical registration
    config["alignment_shifts"] = {
        str(oid): list(tracker.alignment_shifts.get(oid, (0.0, 0.0)))
        for oid in tracker.om_ids
    }
    config["align_method"] = args.align_method
    config["base_threshold_tag"] = args.base_threshold_tag
    config["align_threshold_tag"] = args.align_threshold_tag
    if "02_crown_tracking" not in config["steps_completed"]:
        config["steps_completed"].append("02_crown_tracking")
    save_config(config, config_path)
    print(f"Config updated: {config_path}")

    print(f"\nStep 2 complete. {len(consensus_gdf)} consensus crowns saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
