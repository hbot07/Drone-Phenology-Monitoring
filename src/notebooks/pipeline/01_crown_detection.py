#!/usr/bin/env python3
"""
Pipeline Step 1: Multi-threshold Detectree2 Crown Detection

Runs Detectree2 on each orthomosaic in the pipeline and saves multi-threshold
crown GeoPackages (one GPKG per OM, one layer per confidence threshold).

Requires: base conda environment (has detectree2, detectron2)

Usage:
    python 01_crown_detection.py --config /path/to/pipeline_config.json \\
        [--device cpu] [--threads 6] [--skip-existing] [--no-cleanup]

Reads:
    <output_dir>/pipeline_config.json

Writes:
    <output_dir>/01_detectree/crowns_multithreshold/{stem}_multithreshold.gpkg
    <output_dir>/01_detectree/crowns_multithreshold/{stem}_multithreshold.meta.json
    <output_dir>/01_detectree/run_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def save_config(config: dict, config_path: Path) -> None:
    config_path.write_text(json.dumps(config, indent=2))


def layer_name(conf: float) -> str:
    return f"conf_{str(round(float(conf), 2)).replace('.', 'p')}"


def ensure_proj_data_dir() -> None:
    """Set PROJ data directory if pyproj cannot find it automatically."""
    import pyproj
    try:
        _ = pyproj.datadir.get_data_dir()
        return
    except Exception:
        pass

    candidates = [
        os.environ.get("PROJ_DATA"),
        os.environ.get("PROJ_LIB"),
        os.path.expanduser("~/anaconda3/share/proj"),
        os.path.expanduser("~/miniconda3/share/proj"),
        "/opt/homebrew/share/proj",
        "/usr/local/share/proj",
    ]
    try:
        import subprocess
        result = subprocess.run(["conda", "info", "--base"], capture_output=True, text=True)
        conda_base = result.stdout.strip()
        if conda_base:
            candidates.append(os.path.join(conda_base, "share", "proj"))
    except Exception:
        pass

    for cand in candidates:
        if cand and os.path.exists(cand):
            pyproj.datadir.set_data_dir(cand)
            os.environ["PROJ_DATA"] = cand
            os.environ["PROJ_LIB"] = cand
            print(f"PROJ data directory set to: {cand}")
            return
    raise RuntimeError("Valid PROJ data directory not found; set PROJ_DATA/PROJ_LIB manually.")


def validate_multithreshold_store(
    gpkg_path: Path,
    meta_path: Path,
    stem: str,
    thresholds: List[float],
) -> dict:
    """Return {'ok': True/False, 'reason': ...}."""
    if not gpkg_path.exists():
        return {"ok": False, "reason": "gpkg_missing"}
    if not meta_path.exists():
        return {"ok": False, "reason": "meta_missing"}

    try:
        import fiona
        layers = list(fiona.listlayers(str(gpkg_path)))
    except Exception as e:
        return {"ok": False, "reason": f"fiona_error: {e}"}

    expected = [layer_name(c) for c in thresholds]
    missing = sorted(set(expected) - set(layers))
    if missing:
        return {"ok": False, "reason": "missing_layers", "missing_layers": missing}

    try:
        _ = json.loads(meta_path.read_text())
    except Exception as e:
        return {"ok": False, "reason": f"meta_read_failed: {e}"}

    return {"ok": True, "reason": "ok"}


def clean_bad_geojson_names(geo_folder: Path) -> int:
    bad = list(geo_folder.glob("*_None.geojson"))
    for p in bad:
        try:
            p.unlink()
        except Exception:
            pass
    return len(bad)


def _ensure_png_tile_dir(tiles_path: Path) -> Path:
    """Create a directory containing only PNG tiles (symlinks) for inference."""
    png_dir = tiles_path / "_png_tiles"
    png_dir.mkdir(parents=True, exist_ok=True)
    for png in tiles_path.glob("*.png"):
        link = png_dir / png.name
        if link.exists():
            continue
        try:
            link.symlink_to(png)
        except Exception:
            shutil.copy2(png, link)
    return png_dir


def build_multithreshold_store(
    stem: str,
    base_gdf,
    gpkg_path: Path,
    meta_path: Path,
    thresholds: List[float],
    fixed_iou: float,
    simplify_tol: float,
) -> dict:
    import geopandas as gpd
    from detectree2.models.outputs import clean_crowns

    if gpkg_path.exists():
        gpkg_path.unlink()

    layer_counts: Dict[str, int] = {}
    for conf in thresholds:
        g = base_gdf.copy()
        if simplify_tol and simplify_tol > 0:
            g = g.set_geometry(g.geometry.simplify(simplify_tol))
        g = clean_crowns(g, fixed_iou, float(conf))
        g = g[g.is_valid]
        lyr = layer_name(conf)
        g["orthomosaic"] = stem
        g["confidence"] = float(conf)
        g["threshold_tag"] = lyr
        g.to_file(str(gpkg_path), layer=lyr, driver="GPKG")
        layer_counts[lyr] = int(len(g))
        print(f"    [{lyr}] {len(g)} crowns")

    meta = {
        "orthomosaic": stem,
        "gpkg_path": str(gpkg_path),
        "thresholds": [float(x) for x in thresholds],
        "layer_counts": layer_counts,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def process_one_orthomosaic(
    om_path: Path,
    stem: str,
    crowns_dir: Path,
    work_dir: Path,
    cfg_detectree,
    thresholds: List[float],
    tile_width: int,
    tile_height: int,
    tile_buffer: int,
    fixed_iou: float,
    simplify_tol: float,
    dtype_bool: bool,
    skip_if_valid: bool,
    cleanup_work: bool,
) -> dict:
    """Process a single orthomosaic. Returns a status dict."""
    from detectree2.preprocessing.tiling import tile_data
    from detectree2.models.predict import predict_on_data
    from detectree2.models.outputs import project_to_geojson, stitch_crowns
    from detectron2.engine import DefaultPredictor

    gpkg_path = crowns_dir / f"{stem}_multithreshold.gpkg"
    meta_path = crowns_dir / f"{stem}_multithreshold.meta.json"

    if skip_if_valid:
        v = validate_multithreshold_store(gpkg_path, meta_path, stem, thresholds)
        if v.get("ok"):
            print(f"  [SKIP] {stem} — crowns already valid")
            meta = json.loads(meta_path.read_text())
            return {"stem": stem, "status": "skipped", "meta": meta}

    tiles_path = work_dir / stem
    geo_folder = tiles_path / "predictions_geo"
    tiles_path.mkdir(parents=True, exist_ok=True)

    # --- Tile ---
    if not (geo_folder.exists() and list(geo_folder.glob("*.geojson"))):
        print(f"  [TILE] {stem}")
        tile_data(
            str(om_path),
            str(tiles_path),
            tile_buffer,
            tile_width,
            tile_height,
            dtype_bool=dtype_bool,
        )

        predictor = DefaultPredictor(cfg_detectree)
        png_tiles_dir = _ensure_png_tile_dir(tiles_path)
        print(f"  [PREDICT] {stem} ({png_tiles_dir.name}/)")
        predict_on_data(
            directory=str(png_tiles_dir),
            out_folder="predictions",
            predictor=predictor,
        )

        predictions_folder = png_tiles_dir / "predictions"
        print(f"  [GEO] {stem}")
        project_to_geojson(str(tiles_path), str(predictions_folder), str(geo_folder))

    if not (geo_folder.exists() and list(geo_folder.glob("*.geojson"))):
        return {
            "stem": stem,
            "status": "failed",
            "reason": f"No geojson predictions in {geo_folder}",
        }

    # --- Stitch ---
    print(f"  [STITCH] {stem}")
    removed = clean_bad_geojson_names(geo_folder)
    if removed:
        print(f"    Removed {removed} invalid geojson files")
    base_gdf = stitch_crowns(str(geo_folder), 1)
    base_gdf = base_gdf[base_gdf.is_valid]
    print(f"    Raw crowns: {len(base_gdf)}")

    # --- Build multi-threshold GPKG ---
    print(f"  [EXPORT] {stem} → {gpkg_path.name}")
    meta = build_multithreshold_store(
        stem=stem,
        base_gdf=base_gdf,
        gpkg_path=gpkg_path,
        meta_path=meta_path,
        thresholds=thresholds,
        fixed_iou=fixed_iou,
        simplify_tol=simplify_tol,
    )

    # --- Cleanup work dir ---
    if cleanup_work and tiles_path.exists():
        shutil.rmtree(tiles_path, ignore_errors=True)
        print(f"  [CLEANUP] removed work dir: {tiles_path}")

    return {"stem": stem, "status": "ok", "meta": meta}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Step 1: Crown Detection")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.json")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="CPU thread count for inference (default: 6)",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated confidence thresholds (default: 0.15,0.20,...,0.65)",
    )
    parser.add_argument(
        "--fixed-iou",
        type=float,
        default=0.7,
        help="Fixed IoU threshold for clean_crowns (default: 0.7)",
    )
    parser.add_argument(
        "--simplify-tol",
        type=float,
        default=0.3,
        help="Simplification tolerance (default: 0.3)",
    )
    parser.add_argument(
        "--dtype-bool",
        action="store_true",
        default=True,
        help="Use dtype_bool=True for tile_data (default: True)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip OMs whose GPKG already exists and is valid (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="Keep work directory after detection (default: cleanup)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    om_dir = Path(config["om_dir"])
    output_dir = Path(config["output_dir"])
    model_path = Path(config["model_path"])
    stems = config["om_stems"]
    tile_width = int(config.get("tile_width", 25))
    tile_height = int(config.get("tile_height", 25))
    tile_buffer = int(config.get("tile_buffer", 15))

    detectree_dir = Path(config["detectree_dir"])
    crowns_dir = detectree_dir / "crowns_multithreshold"
    work_dir = detectree_dir / "work_detectree"
    crowns_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.thresholds:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
    else:
        thresholds = [round(0.15 + 0.05 * i, 2) for i in range(11)]  # 0.15..0.65

    # --- Thread and env setup ---
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except Exception:
        pass

    ensure_proj_data_dir()

    # Verify torch/numpy compatibility
    try:
        import numpy as _np
        _ = torch.as_tensor(_np.zeros((2, 2, 3), dtype=_np.float32))
    except Exception as e:
        print(
            f"ERROR: torch/numpy incompatibility: {e}\n"
            "Fix: install numpy<2 in the base conda env (e.g. numpy==1.26.*).",
            file=sys.stderr,
        )
        return 1

    # --- Detectree2 imports ---
    from detectree2.models.train import setup_cfg

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    cfg_detectree = setup_cfg(update_model=str(model_path))
    try:
        cfg_detectree.MODEL.DEVICE = args.device
    except Exception:
        pass

    print(f"Device: {args.device} | Threads: {args.threads}")
    print(f"Model: {model_path.name}")
    print(f"Thresholds: {thresholds}")
    print(f"Tile: {tile_width}x{tile_height} buffer={tile_buffer}")
    print(f"OMs to process: {len(stems)}\n")

    run_rows = []
    failed = []

    for i, stem in enumerate(stems, 1):
        om_path = om_dir / f"{stem}.tif"
        if not om_path.exists():
            print(f"[{i}/{len(stems)}] ERROR: {stem} — TIF not found: {om_path}")
            failed.append(stem)
            continue

        print(f"\n[{i}/{len(stems)}] === {stem} ===")
        result = process_one_orthomosaic(
            om_path=om_path,
            stem=stem,
            crowns_dir=crowns_dir,
            work_dir=work_dir,
            cfg_detectree=cfg_detectree,
            thresholds=thresholds,
            tile_width=tile_width,
            tile_height=tile_height,
            tile_buffer=tile_buffer,
            fixed_iou=args.fixed_iou,
            simplify_tol=args.simplify_tol,
            dtype_bool=args.dtype_bool,
            skip_if_valid=args.skip_existing,
            cleanup_work=not args.no_cleanup,
        )
        run_rows.append(result)
        if result["status"] == "failed":
            failed.append(stem)
            print(f"  FAILED: {result.get('reason', '?')}")
        else:
            print(f"  Done: {result['status']}")

    # --- Save run summary ---
    summary = {
        "run_name": config.get("run_name"),
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "stems_processed": len(run_rows),
        "stems_failed": failed,
        "device": args.device,
        "thresholds": thresholds,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tile_buffer": tile_buffer,
        "fixed_iou": args.fixed_iou,
        "rows": run_rows,
    }
    summary_path = detectree_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nRun summary: {summary_path}")

    # --- Update config ---
    config["crowns_dir"] = str(crowns_dir)
    config["detectree_dir"] = str(detectree_dir)
    config["detectree_thresholds"] = thresholds
    if "01_crown_detection" not in config["steps_completed"]:
        config["steps_completed"].append("01_crown_detection")
    save_config(config, config_path)
    print(f"Config updated: {config_path}")

    if failed:
        print(f"\nWARNING: {len(failed)} OM(s) failed: {failed}", file=sys.stderr)
        return 2
    print(f"\nStep 1 complete. {len(run_rows)} OM(s) processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
