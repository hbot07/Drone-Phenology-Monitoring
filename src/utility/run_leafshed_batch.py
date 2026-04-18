from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for _ in range(12):
        if (p / "output").exists() and (p / "src").exists() and (p / "input").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Could not locate repo root (expected output/, src/, input/)")


def infer_dataset_from_run(run_dir: Path) -> Optional[str]:
    name = run_dir.name.lower()
    if "sit" in name:
        return "SIT"
    if "lhc" in name:
        return "LHC"

    for gpkg in run_dir.glob("consensus_crowns_complete_all*.gpkg"):
        stem = gpkg.stem.lower()
        if "_sit" in stem or stem.endswith("sit"):
            return "SIT"
        if "_lhc" in stem or stem.endswith("lhc"):
            return "LHC"
    return None


def discover_tracking_runs(output_root: Path) -> List[Path]:
    runs: List[Path] = []
    for d in sorted(output_root.iterdir()):
        if not d.is_dir():
            continue
        if "tracking" not in d.name.lower():
            continue
        # Any run folder that contains a consensus crowns GPKG is eligible.
        if any(d.glob("consensus_crowns*.gpkg")):
            runs.append(d)
    return runs


def dataset_pairs(root: Path, dataset: str) -> Tuple[List[Tuple[str, str, str]], Dict[int, str], Path, Path]:
    dataset = dataset.upper().strip()
    if dataset == "SIT":
        multithresh_dir = root / "output" / "detectree_om_sit_multithreshold" / "crowns_multithreshold"
        ortho_dir = root / "input" / "input_om_sit"
        om_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        pairs: List[Tuple[str, str, str]] = []
        om_stems: Dict[int, str] = {}
        for idx, om_num in enumerate(om_ids, start=1):
            gpkg = multithresh_dir / f"OM{om_num}_multithreshold.gpkg"
            tif = ortho_dir / f"sit_om{om_num}.tif"
            stem = f"sit_om{om_num}"
            if not gpkg.exists() or not tif.exists():
                raise FileNotFoundError(f"Missing SIT file(s) for {stem}: {gpkg.exists()=}, {tif.exists()=}")
            pairs.append((str(gpkg), str(tif), stem))
            om_stems[idx] = stem
        return pairs, om_stems, multithresh_dir, ortho_dir

    if dataset == "LHC":
        multithresh_dir = root / "output" / "detectree_om_lhc_multithreshold_smaller_tiles" / "crowns_multithreshold"
        ortho_dir = root / "input" / "input_om_lhc"
        stems = [
            "odm_orthophoto25_10_25",
            "odm_orthophoto9_11_25",
            "odm_orthophoto20_11_25",
            "odm_orthophoto26_11_25",
            "odm_orthophoto11_1_26",
            "odm_orthophoto4_02_26",
            "odm_orthophoto20_02_26",
            "odm_orthophoto7_03_26",
        ]
        pairs = []
        om_stems = {}
        for idx, stem in enumerate(stems, start=1):
            gpkg = multithresh_dir / f"{stem}_multithreshold.gpkg"
            tif = ortho_dir / f"{stem}.tif"
            if not gpkg.exists() or not tif.exists():
                raise FileNotFoundError(f"Missing LHC file(s) for {stem}: {gpkg.exists()=}, {tif.exists()=}")
            pairs.append((str(gpkg), str(tif), stem))
            om_stems[idx] = stem
        return pairs, om_stems, multithresh_dir, ortho_dir

    raise ValueError(f"Unknown dataset {dataset!r}; expected SIT or LHC")


def choose_consensus_gpkg(run_dir: Path) -> Path:
    """Pick the best available consensus-crowns artifact within a run directory."""

    patterns = [
        # Prefer cleaned/deduped.
        "consensus_crowns_complete_all*_dedup_*.gpkg",
        # Standard exports (SIT/LHC reruns).
        "consensus_crowns_complete_all*.gpkg",
        # Older runs.
        "consensus_crowns_smallchains*.gpkg",
        "consensus_crowns_all_sources*.gpkg",
        # Last resort.
        "consensus_crowns*.gpkg",
    ]

    for pat in patterns:
        candidates = sorted(run_dir.glob(pat))
        if candidates:
            # Prefer the first in sorted order to keep deterministic.
            return candidates[0]

    raise FileNotFoundError(f"No consensus crowns gpkg found in: {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch compute leaf-shed DS + phenophases for all tracking runs")
    parser.add_argument("--runs", nargs="*", default=None, help="Specific run dirs under output/ to process")
    parser.add_argument("--align-method", default="pcc_tiled", help="Alignment method passed to tracker.load_multithreshold_data")
    parser.add_argument("--align-threshold-tag", default="conf_0p65", help="Threshold layer tag for crowns-based alignment (if used)")
    parser.add_argument("--base-threshold-tag", default="conf_0p45", help="Base threshold tag for loading crowns")
    parser.add_argument("--veg-min", type=float, default=0.45)
    parser.add_argument("--ds-thresh", type=float, default=0.70)
    parser.add_argument("--on", type=float, default=0.65)
    parser.add_argument("--off", type=float, default=0.35)
    # Non-tree filter thresholds (meant to remove a small number of obvious non-tree artifacts).
    parser.add_argument("--gcc-mean-nontree", type=float, default=0.5)
    parser.add_argument("--veg-mean-nontree", type=float, default=0.4)
    parser.add_argument("--gcc-amp-nontree", type=float, default=0.04)
    parser.add_argument("--veg-amp-nontree", type=float, default=0.1)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if output/<run>/phenology/leafshed_tree_scores.csv already exists",
    )
    args = parser.parse_args()

    root = find_repo_root()
    app_dir = root / "src" / "flask_app_tracking"
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    from tree_tracking import TreeTrackingGraph

    from phenology_leafshed import LeafShedConfig, NonTreeThresholds, PatchQCConfig, VegMaskConfig
    from phenology_leafshed import apply_non_tree_thresholds, compute_leafshed_scores, compute_patch_features, save_normalizers_json

    output_root = root / "output"
    run_dirs: List[Path]
    if args.runs:
        run_dirs = [output_root / r for r in args.runs]
    else:
        run_dirs = discover_tracking_runs(output_root)

    if not run_dirs:
        print("No tracking runs discovered under output/.")
        return 2

    cfg = LeafShedConfig(
        veg_min_threshold=float(args.veg_min),
        ds_threshold=float(args.ds_thresh),
        phenophase_on=float(args.on),
        phenophase_off=float(args.off),
    )
    non_tree_thr = NonTreeThresholds(
        gcc_mean_thresh=float(args.gcc_mean_nontree) if args.gcc_mean_nontree is not None else None,
        veg_mean_thresh=float(args.veg_mean_nontree) if args.veg_mean_nontree is not None else None,
        gcc_amp_thresh=float(args.gcc_amp_nontree) if args.gcc_amp_nontree is not None else None,
        veg_amp_thresh=float(args.veg_amp_nontree) if args.veg_amp_nontree is not None else None,
    )

    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"Skip missing run dir: {run_dir}")
            continue

        dataset = infer_dataset_from_run(run_dir)
        if dataset is None:
            print(f"Skip (cannot infer dataset SIT/LHC): {run_dir}")
            continue

        phen_out = run_dir / "phenology"
        scores_csv = phen_out / "leafshed_tree_scores.csv"
        if args.skip_existing and scores_csv.exists():
            print(f"Skip existing: {scores_csv}")
            continue

        try:
            consensus_gpkg = choose_consensus_gpkg(run_dir)
        except Exception as e:
            print(f"Skip (no usable consensus crowns gpkg): {run_dir} ({e})")
            continue

        print("\n=== Run ===")
        print("run_dir:", run_dir)
        print("dataset:", dataset)
        print("consensus_gpkg:", consensus_gpkg.name)

        # Build aligned tracker (matches notebook behaviour: OM ids 1..N with stem mapping)
        try:
            pairs, om_stems, multithresh_dir, ortho_dir = dataset_pairs(root, dataset)
            tracker = TreeTrackingGraph(
                auto_discover=False,
                multithresh_dir=str(multithresh_dir),
                ortho_dir=str(ortho_dir),
                output_dir=str(run_dir),
                simplify_tol=1.0,
                resize_factor=0.1,
                max_crowns_preview=200,
            )
            tracker.file_pairs = [(g, o) for g, o, _ in pairs]
            tracker.om_ids = list(range(1, len(pairs) + 1))
            tracker.base_threshold_tag = None
            tracker.load_multithreshold_data(
                base_threshold_tag=str(args.base_threshold_tag),
                load_images=False,
                align=True,
                align_method=str(args.align_method),
                align_threshold_tag=str(args.align_threshold_tag),
            )
            print(
                "alignment_shifts (first 3):",
                {k: tracker.alignment_shifts[k] for k in sorted(tracker.alignment_shifts)[:3]},
            )
        except Exception as e:
            print(f"Skip (failed to init/align tracker): {run_dir} ({e})")
            continue

        try:
            crowns = gpd.read_file(consensus_gpkg)
        except Exception as e:
            print(f"Skip (failed reading {consensus_gpkg.name}): {e}")
            continue

        if crowns.empty:
            print("No consensus crowns in gpkg; skipping")
            continue
        if "chain_id" not in crowns.columns:
            crowns = crowns.reset_index(drop=True)
            crowns["chain_id"] = crowns.index.astype(int)
        crowns["chain_id"] = pd.to_numeric(crowns["chain_id"], errors="coerce")
        crowns = crowns.dropna(subset=["chain_id"]).copy()
        crowns["chain_id"] = crowns["chain_id"].astype(int)

        phen_out.mkdir(parents=True, exist_ok=True)

        # Extract features per (tree, om)
        records: List[Dict[str, object]] = []
        veg_cfg = VegMaskConfig()
        qc_cfg = PatchQCConfig()
        for _, row in crowns.iterrows():
            chain_id = int(row["chain_id"])
            poly = row.geometry
            if poly is None or poly.is_empty:
                continue
            for om_id in tracker.om_ids:
                patch = tracker.extract_patch_for_polygon(int(om_id), poly)
                feats = compute_patch_features(patch, veg_cfg=veg_cfg, qc_cfg=qc_cfg)
                stem = om_stems.get(int(om_id), f"OM{om_id}")
                rec: Dict[str, object] = {
                    "chain_id": chain_id,
                    "om_id": int(om_id),
                    "date_label": stem,
                    "patch_exists": bool(isinstance(patch, np.ndarray) and patch.size > 0),
                    "patch_h": int(patch.shape[0]) if isinstance(patch, np.ndarray) else np.nan,
                    "patch_w": int(patch.shape[1]) if isinstance(patch, np.ndarray) else np.nan,
                }
                rec.update(feats)
                records.append(rec)

        features_df = pd.DataFrame(records).sort_values(["chain_id", "om_id"]).reset_index(drop=True)
        raw_csv = phen_out / "phenology_features_raw.csv"
        features_df.to_csv(raw_csv, index=False)
        print("wrote:", raw_csv)

        # DS + phenophases
        tree_scores_df, phenophase_df, normalizers = compute_leafshed_scores(features_df, om_ids=tracker.om_ids, cfg=cfg)

        # Non-tree filtering should be based on raw (no-QC-masking) signal amplitudes.
        feats_for_amp = features_df.copy()
        if "patch_exists" in feats_for_amp.columns:
            feats_for_amp = feats_for_amp[feats_for_amp["patch_exists"].astype(bool)].copy()
        feats_for_amp["gcc_mean"] = pd.to_numeric(feats_for_amp.get("gcc_mean"), errors="coerce")
        feats_for_amp["veg_fraction_hsv"] = pd.to_numeric(feats_for_amp.get("veg_fraction_hsv"), errors="coerce")

        grp = feats_for_amp.groupby("chain_id")
        gcc_amp_raw = (grp["gcc_mean"].max() - grp["gcc_mean"].min()).rename("gcc_amp_for_nontree")
        veg_amp_raw = (grp["veg_fraction_hsv"].max() - grp["veg_fraction_hsv"].min()).rename("veg_amp_for_nontree")
        gcc_mean_raw = grp["gcc_mean"].mean().rename("gcc_mean_for_nontree")
        veg_mean_raw = grp["veg_fraction_hsv"].mean().rename("veg_mean_for_nontree")
        n_obs_raw = grp["gcc_mean"].apply(lambda s: int(s.notna().sum())).rename("n_gcc_obs_for_nontree")

        tree_scores_df = tree_scores_df.merge(gcc_amp_raw, on="chain_id", how="left")
        tree_scores_df = tree_scores_df.merge(veg_amp_raw, on="chain_id", how="left")
        tree_scores_df = tree_scores_df.merge(gcc_mean_raw, on="chain_id", how="left")
        tree_scores_df = tree_scores_df.merge(veg_mean_raw, on="chain_id", how="left")
        tree_scores_df = tree_scores_df.merge(n_obs_raw, on="chain_id", how="left")

        # If a crown has too few observations, do not label it non-tree.
        tree_scores_df.loc[pd.to_numeric(tree_scores_df["n_gcc_obs_for_nontree"], errors="coerce") < 3, "gcc_amp_for_nontree"] = np.nan
        tree_scores_df.loc[pd.to_numeric(tree_scores_df["n_gcc_obs_for_nontree"], errors="coerce") < 3, "veg_amp_for_nontree"] = np.nan

        tree_scores_df = apply_non_tree_thresholds(tree_scores_df, thresholds=non_tree_thr)

        # Filter out non-tree crowns (the non-tree classifier exists purely to remove
        # a few stray non-tree objects that made it into consensus crowns).
        keep_ids = set(tree_scores_df.loc[tree_scores_df["is_tree"].astype(bool), "chain_id"].astype(int).tolist())
        drop_ids = set(tree_scores_df.loc[~tree_scores_df["is_tree"].astype(bool), "chain_id"].astype(int).tolist())
        if drop_ids:
            print(f"filter non-tree: drop {len(drop_ids)} crowns, keep {len(keep_ids)}")

            dropped_df = tree_scores_df[tree_scores_df["chain_id"].astype(int).isin(drop_ids)].copy()
            drop_cols = [
                c
                for c in [
                    "chain_id",
                    "is_non_tree",
                    "gcc_mean_for_nontree",
                    "veg_mean_for_nontree",
                    "gcc_amp_for_nontree",
                    "veg_amp_for_nontree",
                    "n_gcc_obs_for_nontree",
                    "A_gcc",
                    "A_veg",
                    "deciduous_score",
                    "is_deciduous",
                ]
                if c in dropped_df.columns
            ]
            dropped_csv = phen_out / "non_tree_dropped_crowns.csv"
            dropped_df[drop_cols].sort_values(["gcc_amp_for_nontree", "veg_amp_for_nontree"], ascending=True).to_csv(
                dropped_csv, index=False
            )
            print("wrote:", dropped_csv)

        tree_scores_df = tree_scores_df[tree_scores_df["chain_id"].astype(int).isin(keep_ids)].copy()
        phenophase_df = phenophase_df[phenophase_df["chain_id"].astype(int).isin(keep_ids)].copy()

        # Save a tree-only features file for convenience.
        feats_tree = features_df[features_df["chain_id"].astype(int).isin(keep_ids)].copy()
        feats_tree_csv = phen_out / "phenology_features_raw_tree_only.csv"
        feats_tree.to_csv(feats_tree_csv, index=False)
        print("wrote:", feats_tree_csv)

        # Export a tree-only consensus crowns artifact alongside phenology outputs.
        crowns_tree = crowns[crowns["chain_id"].astype(int).isin(keep_ids)].copy()
        tree_gpkg = phen_out / f"{consensus_gpkg.stem}_tree_only.gpkg"
        crowns_tree.to_file(tree_gpkg, driver="GPKG", layer="consensus_crowns_tree_only")
        print("wrote:", tree_gpkg)

        if drop_ids:
            crowns_non_tree = crowns[crowns["chain_id"].astype(int).isin(drop_ids)].copy()
            non_tree_gpkg = phen_out / f"{consensus_gpkg.stem}_non_tree_only.gpkg"
            crowns_non_tree.to_file(non_tree_gpkg, driver="GPKG", layer="consensus_crowns_non_tree_only")
            print("wrote:", non_tree_gpkg)

        phases_csv = phen_out / "leafshed_phenophase_by_om.csv"
        tree_scores_df.to_csv(scores_csv, index=False)
        phenophase_df.to_csv(phases_csv, index=False)
        print("wrote:", scores_csv)
        print("wrote:", phases_csv)

        save_normalizers_json(phen_out / "leafshed_normalizers.json", normalizers)
        (phen_out / "leafshed_config.json").write_text(
            __import__("json").dumps(
                {
                    "leafshed": asdict(cfg),
                    "veg_mask": asdict(veg_cfg),
                    "qc": asdict(qc_cfg),
                    "non_tree_thresholds": asdict(non_tree_thr),
                    "align": {
                        "method": str(args.align_method),
                        "align_threshold_tag": str(args.align_threshold_tag),
                        "base_threshold_tag": str(args.base_threshold_tag),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )

        n_trees = int(tree_scores_df.shape[0])
        n_decid = int(tree_scores_df["is_deciduous"].sum())
        print(f"trees scored: {n_trees} | deciduous: {n_decid} ({n_decid/max(n_trees,1):.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
