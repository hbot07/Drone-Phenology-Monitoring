from __future__ import annotations

import json
import os
import glob
import warnings
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from rasterio.plot import show

from detectree2.preprocessing.tiling import tile_data
from detectree2.models.train import setup_cfg
from detectree2.models.predict import instances_to_coco_json
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
from detectron2.engine import DefaultPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "input"
OM_DIR = INPUT_DIR / "other_area_oms"
MODEL_PATH = INPUT_DIR / "detectree_models" / "250312_flexi.pth"
TILES_DIR = INPUT_DIR / "tiles_other_area"
OUTPUT_CROWNS_DIR = PROJECT_ROOT / "output" / "other_area_crowns"
OUTPUT_VIZ_DIR = PROJECT_ROOT / "output" / "other_area_overlays"

BUFFER = 20
TILE_WIDTH = 45
TILE_HEIGHT = 45
DTYPE_BOOL = True
IOU_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.45
SIMPLIFY_TOLERANCE = 0.3
DEVICE = "cpu"


def ensure_proj_data_dir() -> None:
    try:
        pyproj.datadir.get_data_dir()
        return
    except Exception:
        pass

    candidates = []
    env_proj = os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")
    if env_proj:
        candidates.append(env_proj)

    for candidate in [
        os.path.expanduser("~/anaconda3/share/proj"),
        os.path.expanduser("~/miniconda3/share/proj"),
        "/opt/homebrew/share/proj",
        "/usr/local/share/proj",
    ]:
        candidates.append(candidate)

    try:
        import subprocess

        result = subprocess.run(["conda", "info", "--base"], capture_output=True, text=True, check=False)
        conda_base = result.stdout.strip()
        if conda_base:
            candidates.append(os.path.join(conda_base, "share", "proj"))
    except Exception:
        pass

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            pyproj.datadir.set_data_dir(candidate)
            os.environ["PROJ_DATA"] = candidate
            os.environ["PROJ_LIB"] = candidate
            print(f"Using PROJ data directory: {candidate}")
            return

    raise RuntimeError("Unable to locate a valid PROJ data directory.")


def tile_orthomosaic(img_path: Path, tiles_path: Path) -> None:
    tiles_path.mkdir(parents=True, exist_ok=True)
    tile_data(str(img_path), str(tiles_path), BUFFER, TILE_WIDTH, TILE_HEIGHT, dtype_bool=DTYPE_BOOL)


def setup_detection_config(trained_model_path: Path):
    cfg = setup_cfg(update_model=str(trained_model_path))
    try:
        cfg.MODEL.DEVICE = DEVICE
    except Exception:
        pass
    return cfg


def predict_tree_crowns(tiles_path: Path, cfg) -> None:
    predictor = DefaultPredictor(cfg)
    pred_dir = tiles_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    png_tiles = sorted(tiles_path.glob("*.png"))
    print(f"Predicting {len(png_tiles)} PNG tiles")

    for i, tile_path in enumerate(png_tiles, start=1):
        image = cv2.imread(str(tile_path))
        if image is None:
            print(f"Failed to read tile: {tile_path}")
            continue

        outputs = predictor(np.ascontiguousarray(image))
        evaluations = instances_to_coco_json(outputs["instances"].to("cpu"), str(tile_path))

        output_file = pred_dir / f"Prediction_{tile_path.stem}.json"
        with open(output_file, "w") as dest:
            json.dump(evaluations, dest)

        if i % 50 == 0:
            print(f"Predicted {i} tiles")


def reproject_predictions(tiles_path: Path) -> Path:
    predictions_folder = tiles_path / "predictions"
    geo_predictions_folder = tiles_path / "predictions_geo"
    project_to_geojson(str(tiles_path), str(predictions_folder), str(geo_predictions_folder))
    return geo_predictions_folder


def detect_on_orthomosaic(img_path: Path, tiles_path: Path, trained_model_path: Path):
    tile_orthomosaic(img_path, tiles_path)
    cfg = setup_detection_config(trained_model_path)
    predict_tree_crowns(tiles_path, cfg)
    return reproject_predictions(tiles_path)


def stitch_and_clean_crowns(geo_predictions_folder: Path):
    crowns = stitch_crowns(str(geo_predictions_folder), 1)
    crowns = crowns[crowns.is_valid]
    if SIMPLIFY_TOLERANCE and SIMPLIFY_TOLERANCE > 0:
        crowns = crowns.set_geometry(crowns.geometry.simplify(SIMPLIFY_TOLERANCE))
    crowns = clean_crowns(crowns, IOU_THRESHOLD, CONFIDENCE_THRESHOLD)
    crowns = crowns[crowns.is_valid]
    return crowns


def clean_bad_geojson_names(geo_folder: Path) -> int:
    bad_files = glob.glob(str(geo_folder / "*_None.geojson"))
    for file_path in bad_files:
        try:
            os.remove(file_path)
        except Exception as exc:
            print(f"Warning: could not remove {file_path}: {exc}")
    return len(bad_files)


def save_overlay(img_path: Path, crowns: gpd.GeoDataFrame, output_path: Path) -> None:
    with rasterio.open(img_path) as src:
        raster_crs = src.crs

    if crowns.crs != raster_crs:
        crowns = crowns.to_crs(raster_crs)

    fig, ax = plt.subplots(figsize=(14, 14))
    with rasterio.open(img_path) as src:
        if src.count >= 3:
            image = src.read([1, 2, 3]).astype(float)
            for band_index in range(3):
                band = image[band_index]
                valid = band[np.isfinite(band) & (band > 0)]
                if valid.size:
                    p2, p98 = np.percentile(valid, (2, 98))
                    if p98 > p2:
                        image[band_index] = np.clip((band - p2) / (p98 - p2), 0, 1)
            show(image, transform=src.transform, ax=ax)
        else:
            show(src, ax=ax, cmap="gray")

    crowns.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.9, alpha=0.9)
    ax.set_title(f"{img_path.stem} crowns at confidence {CONFIDENCE_THRESHOLD:.2f}")
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_proj_data_dir()
    warnings.filterwarnings("ignore", category=UserWarning)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not OM_DIR.exists():
        raise FileNotFoundError(f"Missing input folder: {OM_DIR}")

    OUTPUT_CROWNS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    om_paths = sorted(OM_DIR.glob("*.tif"))
    if not om_paths:
        raise FileNotFoundError(f"No orthomosaics found in {OM_DIR}")

    print(f"Found {len(om_paths)} orthomosaics in {OM_DIR}")
    print(f"Using model: {MODEL_PATH}")
    print(f"Saving crowns to: {OUTPUT_CROWNS_DIR}")
    print(f"Saving overlays to: {OUTPUT_VIZ_DIR}")

    for img_path in om_paths:
        tiles_path = TILES_DIR / img_path.stem
        geo_predictions_folder = tiles_path / "predictions_geo"
        crowns_path = OUTPUT_CROWNS_DIR / f"{img_path.stem}_crowns.gpkg"
        overlay_path = OUTPUT_VIZ_DIR / f"{img_path.stem}_overlay.png"

        print(f"\n=== Processing {img_path.name} ===")
        if crowns_path.exists() and overlay_path.exists():
            print(f"Outputs already exist, skipping: {crowns_path.name}")
            continue

        if not (geo_predictions_folder.exists() and list(geo_predictions_folder.glob("*.geojson"))):
            print("Running detection...")
            geo_predictions_folder = detect_on_orthomosaic(img_path, tiles_path, MODEL_PATH)
        else:
            print("Reusing existing predictions_geo folder")

        removed = clean_bad_geojson_names(geo_predictions_folder)
        if removed:
            print(f"Removed {removed} invalid geojson files")

        crowns = stitch_and_clean_crowns(geo_predictions_folder)
        crowns.to_file(crowns_path, driver="GPKG")
        print(f"Saved crowns: {crowns_path}")

        save_overlay(img_path, crowns, overlay_path)
        print(f"Saved overlay: {overlay_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()