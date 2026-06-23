#!/usr/bin/env python3
"""
Pipeline Step 3: Phenology Analysis → tree_master_geojson

For each consensus crown across all orthomosaics, extracts image patch features
and computes deciduous/phenophase scores. Produces a single canonical GeoJSON
(tree_master_geojson.geojson) with all observations, tracking metadata, and
phenology embedded per crown. No tree/non-tree classification is performed.

Requires: dpm-tracking conda environment

Usage:
    python 03_phenology_analysis.py --config /path/to/pipeline_config.json \\
        [--dataset-id DATASET_ID]   # default: from run_name in config
        [--veg-min 0.45] [--ds-thresh 0.70] [--on-thresh 0.65] [--off-thresh 0.35] \\
        [--skip-if-done]

Reads:
    <output_dir>/pipeline_config.json
    <tracking_dir>/consensus_crowns_complete_all.gpkg   (from step 2)

Writes:
    <phenology_dir>/tree_master_geojson.geojson   (canonical output)
    <phenology_dir>/phenology_features_raw.csv    (convenience)
    <phenology_dir>/leafshed_tree_scores.csv      (convenience)
    <phenology_dir>/leafshed_phenophase_by_om.csv (convenience)
    <phenology_dir>/leafshed_normalizers.json
    <phenology_dir>/leafshed_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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


def build_pairs_and_om_stems(config: dict) -> Tuple[List[Tuple[str, str, str]], Dict[int, str]]:
    crowns_dir = Path(config["crowns_dir"])
    om_dir = Path(config["om_dir"])
    pairs = []
    om_stems = {}
    for i, (gpkg_raw, tif_raw, stem) in enumerate(config["pairs"], 1):
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


def _safe_float(x) -> Optional[float]:
    """Convert to float, returning None for NaN/inf."""
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _robust_zscore(series: np.ndarray) -> np.ndarray:
    """Robust z-score: (x - median) / (0.5*(Q75-Q25)) per series."""
    vals = np.asarray(series, dtype=float)
    med = float(np.nanmedian(vals))
    q25 = float(np.nanpercentile(vals, 25))
    q75 = float(np.nanpercentile(vals, 75))
    iqr = q75 - q25
    if iqr < 1e-12:
        return np.where(np.isfinite(vals), 0.0, np.nan)
    return (vals - med) / (0.5 * iqr)


def _percentile_rank(series: np.ndarray) -> np.ndarray:
    """Percentile rank of each element among finite values in series (0..1)."""
    vals = np.asarray(series, dtype=float)
    out = np.full_like(vals, np.nan)
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 2:
        return out
    finite_vals = vals[finite_mask]
    n = finite_vals.size
    for i, v in enumerate(vals):
        if np.isfinite(v):
            rank = float(np.sum(finite_vals < v)) + 0.5 * float(np.sum(finite_vals == v))
            out[i] = rank / n
    return out


def _linear_slope(om_ids: List[int], values: np.ndarray) -> Optional[float]:
    """Slope per-OM from linear regression on finite values."""
    vals = np.asarray(values, dtype=float)
    x = np.asarray(om_ids, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(x)
    if mask.sum() < 2:
        return None
    coeffs = np.polyfit(x[mask], vals[mask], 1)
    return _safe_float(coeffs[0])


def _amplitude(values: np.ndarray) -> Optional[float]:
    """Max - min of finite values."""
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        return None
    return _safe_float(float(finite.max()) - float(finite.min()))


def _get_clean_series(sub_df: pd.DataFrame, om_ids: List[int], feat: str) -> np.ndarray:
    """Extract per-OM values for a feature, masking bad observations."""
    sub = sub_df.set_index("om_id")
    bad = sub.get("is_bad_observation", pd.Series(False, index=sub.index)).astype(bool)
    vals = np.array([
        float(sub.loc[oid, feat]) if oid in sub.index else np.nan
        for oid in om_ids
    ], dtype=float)
    for idx, oid in enumerate(om_ids):
        if oid in bad.index and bool(bad.loc[oid]):
            vals[idx] = np.nan
    return vals


def build_temporal_summary(sub_df: pd.DataFrame, om_ids: List[int]) -> dict:
    """Compute temporal summary (amplitudes, slopes, clean obs count) for one crown."""
    bad_mask = sub_df.get("is_bad_observation", pd.Series(False, index=sub_df.index)).astype(bool)
    patch_exists = sub_df.get("patch_exists", pd.Series(True, index=sub_df.index)).astype(bool)
    clean_mask = patch_exists & ~bad_mask
    clean_sub = sub_df[clean_mask]

    n_obs_clean = int(clean_mask.sum())
    om_start = int(clean_sub["om_id"].min()) if len(clean_sub) else None
    om_end = int(clean_sub["om_id"].max()) if len(clean_sub) else None

    summary: dict = {"n_obs_clean": n_obs_clean, "om_start": om_start, "om_end": om_end}

    amp_features = ["gcc_mean", "rcc_mean", "veg_fraction_hsv", "gray_entropy", "laplacian_var"]
    slope_features = ["gcc_mean", "rcc_mean", "veg_fraction_hsv", "gray_entropy"]

    for feat in amp_features:
        if feat not in sub_df.columns:
            continue
        vals = _get_clean_series(sub_df, om_ids, feat)
        summary[f"{feat}_amplitude"] = _amplitude(vals)

    for feat in slope_features:
        if feat not in sub_df.columns:
            continue
        vals = _get_clean_series(sub_df, om_ids, feat)
        summary[f"{feat}_slope_per_om"] = _linear_slope(om_ids, vals)

    return summary


def build_per_crown_normalizers(
    sub_df: pd.DataFrame, om_ids: List[int], feat_cols: List[str]
) -> Dict[str, Dict[int, Optional[float]]]:
    """Per-crown, per-feature robust z-scores and percentile ranks across OMs."""
    result: Dict[str, Dict[int, Optional[float]]] = {}
    for feat in feat_cols:
        if feat not in sub_df.columns:
            continue
        vals = _get_clean_series(sub_df, om_ids, feat)
        rz = _robust_zscore(vals)
        pct = _percentile_rank(vals)
        result[f"{feat}_rz"] = {int(oid): _safe_float(rz[i]) for i, oid in enumerate(om_ids)}
        result[f"{feat}_pct"] = {int(oid): _safe_float(pct[i]) for i, oid in enumerate(om_ids)}
    return result


def build_observations(
    sub_df: pd.DataFrame,
    om_stems: Dict[int, str],
    crown_normalizers: Dict[str, Dict[int, Optional[float]]],
    pheno_for_crown: Dict[int, dict],
    crown_id: str,
    feature_cols: List[str],
) -> List[dict]:
    """Build per-OM observation list for one crown."""
    skip_cols = {"chain_id", "om_id", "date_label", "patch_exists", "patch_h", "patch_w"}
    sub_indexed = sub_df.set_index("om_id")
    observations = []

    for om_id in sorted(sub_indexed.index.unique()):
        row = sub_indexed.loc[om_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        stem = om_stems.get(int(om_id), f"OM{om_id}")
        patch_exists = bool(row.get("patch_exists", False))
        ph = int(row.get("patch_h")) if patch_exists and row.get("patch_h") is not None else None
        pw = int(row.get("patch_w")) if patch_exists and row.get("patch_w") is not None else None
        is_bad = bool(row.get("is_bad_observation", False)) if patch_exists else True
        status = "ok" if (patch_exists and not is_bad) else ("bad" if patch_exists else "missing")

        crop_path = f"crops/{crown_id}/OM{int(om_id):02d}_{stem}.png"

        # raw features
        features_raw: dict = {}
        for col in feature_cols:
            if col in skip_cols:
                continue
            val = row.get(col, np.nan)
            if col == "is_bad_observation":
                features_raw[col] = bool(val)
            else:
                features_raw[col] = _safe_float(val)

        # normalized features
        features_normalized: dict = {}
        for feat in ["gcc_mean", "rcc_mean", "veg_fraction_hsv", "gray_entropy", "laplacian_var"]:
            rz_key, pct_key = f"{feat}_rz", f"{feat}_pct"
            if rz_key in crown_normalizers and int(om_id) in crown_normalizers[rz_key]:
                features_normalized[f"{feat}_rz_date"] = crown_normalizers[rz_key][int(om_id)]
            if pct_key in crown_normalizers and int(om_id) in crown_normalizers[pct_key]:
                features_normalized[f"{feat}_pct_date"] = crown_normalizers[pct_key][int(om_id)]

        # phenology per OM
        pheno_info: dict = {}
        if int(om_id) in pheno_for_crown:
            ph_row = pheno_for_crown[int(om_id)]
            pheno_info = {
                "phenophase": ph_row.get("phenophase"),
                "veg_hat": _safe_float(ph_row.get("veg_fraction_hsv_hat")),
                "veg_norm": _safe_float(ph_row.get("veg_fraction_hsv_norm")),
                "trough_om": int(ph_row["trough_om"]) if ph_row.get("trough_om") is not None else None,
            }

        observations.append({
            "om_id": int(om_id),
            "date_label": f"OM{int(om_id)}",
            "stem": stem,
            "patch": {
                "patch_exists": patch_exists,
                "patch_h": ph,
                "patch_w": pw,
                "status": status,
                "crop_path": crop_path,
            },
            "features_raw": features_raw,
            "features_normalized": features_normalized,
            "phenology": pheno_info,
        })

    return observations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Step 3: Phenology Analysis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-id", default=None,
                        help="Dataset identifier (default: from run_name in config)")
    parser.add_argument("--base-threshold-tag", default="conf_0p45")
    parser.add_argument("--align-threshold-tag", default="conf_0p65")
    parser.add_argument("--align-method", default="pcc_tiled")
    parser.add_argument("--veg-min", type=float, default=0.45,
                        help="Minimum vegetation fraction threshold (default: 0.45)")
    parser.add_argument("--ds-thresh", type=float, default=0.70,
                        help="Deciduous score threshold (default: 0.70)")
    parser.add_argument("--on-thresh", type=float, default=0.65,
                        help="Phenophase 'leaf-on' threshold (default: 0.65)")
    parser.add_argument("--off-thresh", type=float, default=0.35,
                        help="Phenophase 'leaf-off' threshold (default: 0.35)")
    parser.add_argument("--skip-if-done", action="store_true",
                        help="Skip if tree_master_geojson.geojson already exists")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    project_root = Path(config["project_root"])
    phenology_dir = Path(config["phenology_dir"])
    tracking_dir = Path(config["tracking_dir"])

    phenology_dir.mkdir(parents=True, exist_ok=True)

    master_geojson_path = phenology_dir / "tree_master_geojson.geojson"
    if args.skip_if_done and master_geojson_path.exists():
        print(f"[SKIP] Phenology already done: {master_geojson_path}")
        return 0

    # dataset_id: explicit arg → run_name → 'dataset'
    dataset_id = (args.dataset_id or config.get("run_name") or "dataset").strip()

    # --- Load consensus crowns ---
    consensus_gpkg = Path(config.get("consensus_gpkg",
                                     tracking_dir / "consensus_crowns_complete_all.gpkg"))
    if not consensus_gpkg.exists():
        print(f"ERROR: consensus crowns not found: {consensus_gpkg}", file=sys.stderr)
        print("Run step 2 first (crown tracking).", file=sys.stderr)
        return 1

    import geopandas as gpd
    crowns = gpd.read_file(str(consensus_gpkg))
    if crowns.empty:
        print(f"ERROR: no crowns in {consensus_gpkg.name}", file=sys.stderr)
        return 1

    if "chain_id" not in crowns.columns:
        crowns = crowns.reset_index(drop=True)
        crowns["chain_id"] = crowns.index.astype(int)
    crowns["chain_id"] = pd.to_numeric(crowns["chain_id"], errors="coerce")
    crowns = crowns.dropna(subset=["chain_id"]).copy()
    crowns["chain_id"] = crowns["chain_id"].astype(int)
    crowns = crowns.reset_index(drop=True)
    print(f"Loaded {len(crowns)} consensus crowns from {consensus_gpkg.name}")

    crs_str = str(crowns.crs) if crowns.crs is not None else "unknown"

    # --- Set up app imports ---
    setup_app_dir(project_root)
    from tree_tracking import TreeTrackingGraph
    from phenology_leafshed import (
        LeafShedConfig, PatchQCConfig, VegMaskConfig,
        compute_leafshed_scores, compute_patch_features, save_normalizers_json,
    )

    # --- Build pairs ---
    pairs, om_stems = build_pairs_and_om_stems(config)
    crowns_dir = Path(config["crowns_dir"])
    om_dir = Path(config["om_dir"])
    num_oms = len(pairs)

    missing = [stem for (gpkg, tif, stem) in pairs if not Path(tif).exists()]
    if missing:
        print(f"ERROR: Missing TIF files for: {missing}", file=sys.stderr)
        return 1

    # --- Initialize tracker (alignment only) ---
    print(f"\nInitializing tracker (alignment, no images) for {num_oms} OMs...")
    tracker = TreeTrackingGraph(
        auto_discover=False,
        multithresh_dir=str(crowns_dir),
        ortho_dir=str(om_dir),
        output_dir=str(phenology_dir),
        simplify_tol=1.0,
        resize_factor=0.1,
        max_crowns_preview=200,
    )
    tracker.file_pairs = [(gpkg, tif) for gpkg, tif, _ in pairs]
    tracker.om_ids = list(range(1, num_oms + 1))
    tracker.base_threshold_tag = None

    # Load saved alignment shifts from step 2 config (ensures identical registration)
    saved_shifts_raw = config.get("alignment_shifts", {})
    saved_shifts = {int(k): (float(v[0]), float(v[1])) for k, v in saved_shifts_raw.items()} if saved_shifts_raw else {}

    if saved_shifts:
        print(f"  Using saved alignment shifts from step 2 ({len(saved_shifts)} OMs)")
        tracker.load_multithreshold_data(
            base_threshold_tag=args.base_threshold_tag,
            load_images=False,
            align=False,
        )
        # Inject saved shifts and apply to crown geometries
        tracker.alignment_shifts = saved_shifts
        from shapely.affinity import affine_transform as shapely_affine
        for om_id in tracker.om_ids:
            dx, dy = saved_shifts.get(om_id, (0.0, 0.0))
            if om_id == tracker.om_ids[0] or (dx == 0.0 and dy == 0.0):
                continue
            gdf = tracker.crowns_gdfs.get(om_id)
            if gdf is None or gdf.empty:
                continue
            params = (1.0, 0.0, 0.0, 1.0, dx, dy)
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].apply(
                lambda g: shapely_affine(g, params) if g is not None else g
            )
            tracker.crowns_gdfs[om_id] = gdf
            tracker.crown_attrs[om_id] = [
                tracker._compute_crown_attributes(row.geometry)
                for _, row in gdf.iterrows()
            ]
    else:
        print("  WARNING: No saved alignment shifts found in config, recomputing alignment")
        tracker.load_multithreshold_data(
            base_threshold_tag=args.base_threshold_tag,
            load_images=False,
            align=True,
            align_method=args.align_method,
            align_threshold_tag=args.align_threshold_tag,
        )

    print("Alignment shifts (first 3):", {
        k: tracker.alignment_shifts[k]
        for k in sorted(tracker.alignment_shifts)[:3]
    })

    # --- Extract patch features ---
    n_crowns = len(crowns)
    print(f"\nExtracting patch features for {n_crowns} crowns × {num_oms} OMs...")
    veg_cfg = VegMaskConfig()
    qc_cfg = PatchQCConfig()
    records: List[Dict] = []

    for idx, row in crowns.iterrows():
        chain_id = int(row["chain_id"])
        poly = row.geometry
        if poly is None or poly.is_empty:
            continue
        for om_id in tracker.om_ids:
            patch = tracker.extract_patch_for_polygon(int(om_id), poly)
            feats = compute_patch_features(patch, veg_cfg=veg_cfg, qc_cfg=qc_cfg)
            rec: Dict = {
                "chain_id": chain_id,
                "om_id": int(om_id),
                "date_label": f"OM{int(om_id)}",
                "patch_exists": bool(isinstance(patch, np.ndarray) and patch.size > 0),
                "patch_h": int(patch.shape[0]) if isinstance(patch, np.ndarray) else None,
                "patch_w": int(patch.shape[1]) if isinstance(patch, np.ndarray) else None,
            }
            rec.update(feats)
            records.append(rec)

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{n_crowns} crowns...", end="\r")

    print(f"\n  Total feature records: {len(records)}")
    features_df = pd.DataFrame(records).sort_values(["chain_id", "om_id"]).reset_index(drop=True)
    if "is_bad_observation" not in features_df.columns:
        features_df["is_bad_observation"] = False
    features_df["is_bad_observation"] = features_df["is_bad_observation"].astype(bool)

    raw_csv = phenology_dir / "phenology_features_raw.csv"
    features_df.to_csv(raw_csv, index=False)
    print(f"Wrote: {raw_csv}")

    # --- Compute leafshed scores ---
    print("\nComputing leafshed scores and phenophases...")
    cfg = LeafShedConfig(
        veg_min_threshold=args.veg_min,
        ds_threshold=args.ds_thresh,
        phenophase_on=args.on_thresh,
        phenophase_off=args.off_thresh,
    )
    tree_scores_df, phenophase_df, normalizers = compute_leafshed_scores(
        features_df, om_ids=tracker.om_ids, cfg=cfg
    )
    print(f"  Scored: {len(tree_scores_df)} crowns")

    # Save convenience CSVs
    scores_csv = phenology_dir / "leafshed_tree_scores.csv"
    tree_scores_df.to_csv(scores_csv, index=False)
    print(f"Wrote: {scores_csv}")

    phases_csv = phenology_dir / "leafshed_phenophase_by_om.csv"
    phenophase_df.to_csv(phases_csv, index=False)
    print(f"Wrote: {phases_csv}")

    save_normalizers_json(phenology_dir / "leafshed_normalizers.json", normalizers)
    print(f"Wrote: {phenology_dir / 'leafshed_normalizers.json'}")

    (phenology_dir / "leafshed_config.json").write_text(json.dumps(
        {
            "leafshed": asdict(cfg),
            "veg_mask": asdict(veg_cfg),
            "qc": asdict(qc_cfg),
            "align": {
                "method": args.align_method,
                "align_threshold_tag": args.align_threshold_tag,
                "base_threshold_tag": args.base_threshold_tag,
            },
        },
        indent=2, sort_keys=True,
    ))

    # Index scores and phenophase by chain_id for quick lookup
    scores_idx = tree_scores_df.set_index("chain_id").to_dict(orient="index")
    pheno_by_chain: Dict[int, Dict[int, dict]] = {}
    for _, prow in phenophase_df.iterrows():
        cid = int(prow["chain_id"])
        oid = int(prow["om_id"])
        pheno_by_chain.setdefault(cid, {})[oid] = prow.to_dict()

    feature_cols = list(features_df.columns)
    norm_feat_cols = ["gcc_mean", "rcc_mean", "veg_fraction_hsv", "gray_entropy", "laplacian_var"]

    # --- Build tree_master_geojson ---
    print(f"\nBuilding tree_master_geojson for {n_crowns} crowns...")

    orthomosaics_meta = [
        {"om_id": int(om_id), "date_label": f"OM{int(om_id)}", "stem": om_stems[int(om_id)]}
        for om_id in sorted(tracker.om_ids)
    ]

    features_list = []
    for crown_index, crow_row in crowns.iterrows():
        chain_id = int(crow_row["chain_id"])
        chain_length = int(crow_row.get("chain_length", 0)) if "chain_length" in crow_row.index else 0
        quality = str(crow_row.get("quality", "")) if "quality" in crow_row.index else ""
        avg_similarity = _safe_float(crow_row.get("avg_similarity", np.nan)) if "avg_similarity" in crow_row.index else None
        source_chain_idx = int(crow_row.get("source_chain_idx", 0)) if "source_chain_idx" in crow_row.index else 0

        crown_id = f"crown_{int(crown_index):04d}"
        feature_uid = f"{dataset_id}:{crown_id}"

        geom = crow_row.geometry
        geom_json = json.loads(gpd.GeoDataFrame(geometry=[geom]).to_json())["features"][0]["geometry"] if geom else None

        sub_df = features_df[features_df["chain_id"] == chain_id].copy()
        temporal_summary = build_temporal_summary(sub_df, tracker.om_ids)
        crown_normalizers = build_per_crown_normalizers(sub_df, tracker.om_ids, norm_feat_cols)

        sc = scores_idx.get(chain_id, {})
        classification = {
            "deciduous_score": _safe_float(sc.get("deciduous_score", np.nan)),
            "is_deciduous": bool(sc.get("is_deciduous", False)),
            "leaf_off_start_om": None,
            "full_leaf_off_om": None,
            "leaf_on_return_om": None,
        }
        for k in ["leaf_off_start_om", "full_leaf_off_om", "leaf_on_return_om"]:
            v = _safe_float(sc.get(k, np.nan))
            classification[k] = int(v) if v is not None else None

        observations = build_observations(
            sub_df, om_stems, crown_normalizers, pheno_by_chain.get(chain_id, {}),
            crown_id, feature_cols,
        )

        features_list.append({
            "type": "Feature",
            "id": feature_uid,
            "geometry": geom_json,
            "properties": {
                "ids": {
                    "dataset_id": dataset_id,
                    "chain_id": chain_id,
                    "source_chain_idx": source_chain_idx,
                    "crown_index": int(crown_index),
                    "crown_id": crown_id,
                    "crown_label": crown_id,
                    "feature_uid": feature_uid,
                },
                "tracking": {
                    "chain_length": chain_length,
                    "quality": quality,
                    "avg_similarity": avg_similarity,
                    "consensus_method": "medoid",
                    "dedup_status": "kept",
                },
                "classification": classification,
                "temporal_summary": temporal_summary,
                "field_data": None,
                "observations": observations,
                "assets": {
                    "viewer_base_image": None,
                    "viewer_pixel_geojson": None,
                    "crops_dir": f"crops/{crown_id}",
                },
                "alternate_geometries": {"pixel_underlays": {}},
            },
        })

        if (crown_index + 1) % 20 == 0:
            print(f"  Built {crown_index + 1}/{n_crowns} features...", end="\r")

    print(f"\n  Built {len(features_list)} features")

    master_geojson = {
        "type": "FeatureCollection",
        "schema_name": "tree_master_geojson",
        "schema_version": "1.0",
        "dataset": {
            "dataset_id": dataset_id,
            "site_name": dataset_id,
            "crs": crs_str,
            "num_oms": num_oms,
        },
        "orthomosaics": orthomosaics_meta,
        "phenology_config": {
            "leafshed_config": asdict(cfg),
            "normalizers": normalizers,
        },
        "viewer": None,
        "features": features_list,
    }

    print(f"\nSaving tree_master_geojson ({len(features_list)} features)...")
    master_geojson_path.write_text(json.dumps(master_geojson, indent=2, default=str))
    print(f"Wrote: {master_geojson_path}")

    # --- Summary ---
    n_decid = sum(1 for f in features_list if f["properties"]["classification"]["is_deciduous"])
    print(f"\nPhenology summary: {len(features_list)} crowns | "
          f"deciduous: {n_decid} ({n_decid/max(len(features_list),1):.1%})")

    # --- Update config ---
    config["phenology_dir"] = str(phenology_dir)
    config["tree_master_geojson"] = str(master_geojson_path)
    config["phenology_scores_csv"] = str(scores_csv)
    config["phenology_features_csv"] = str(raw_csv)
    if "03_phenology_analysis" not in config["steps_completed"]:
        config["steps_completed"].append("03_phenology_analysis")
    save_config(config, config_path)
    print(f"Config updated: {config_path}")

    print(f"\nStep 3 complete. tree_master_geojson: {master_geojson_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
