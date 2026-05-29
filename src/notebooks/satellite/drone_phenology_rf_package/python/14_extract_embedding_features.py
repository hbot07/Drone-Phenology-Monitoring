#!/usr/bin/env python3
"""
Extract pretrained visual embedding features for tree crowns from Sentinel-2 patches.

Instead of hand-crafted spectral indices, this script:
  1. Downloads a small RGB+NIR patch (64×64 px at 10m = 640m × 640m) centred on each crown.
  2. Runs it through a pretrained ViT encoder (DINOv2-small or DINOv2-base by default).
  3. Uses the [CLS] token embedding (384-d or 768-d) as features.
  4. Optionally fuses with the standard spectral features from an existing CSV.

Why DINOv2?
  - Self-supervised on 142M images → strong generic visual representations.
  - No fine-tuning needed; the CLS token captures texture, shape, colour structure.
  - Outperforms CLIP for dense visual tasks on natural images.
  - Freely available via torch.hub or HuggingFace.

Why NOT GEE embeddings?
  - Google's Vertex AI CLIP API requires billing + OAuth; not needed here.
  - DINOv2 runs locally on CPU in seconds per patch at this scale (391 crowns × N seasons).

Usage (single year, embeddings only):
  python python/14_extract_embedding_features.py \
    --year 2025 \
    --seasons premonsoon mar apr \
    --patch-px 64 \
    --model dinov2_vits14 \
    --out-csv exports/dino_embeddings_premonsoon_mar_apr_2025.csv

Usage (fuse with spectral CSV):
  python python/14_extract_embedding_features.py \
    --year 2025 \
    --seasons premonsoon mar apr \
    --fuse-csv exports/stac_s2_features_2022_2025_buffer10_items4_label_acacia.csv \
    --out-csv exports/fused_dino_spectral_2025.csv

Available DINOv2 models (from torch.hub facebookresearch/dinov2):
  dinov2_vits14    →  384-dim  (fast, recommended)
  dinov2_vitb14    →  768-dim
  dinov2_vitl14    → 1024-dim
  dinov2_vitg14    → 1536-dim  (very slow on CPU)
"""
from __future__ import annotations

import argparse
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd

_BASE_PATH = Path(__file__).with_name("03_extract_sentinel2_stac_features.py")
_s2 = SourceFileLoader("s2extract", str(_BASE_PATH)).load_module()

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPORT_DIR = ROOT / "exports"

# ── Season windows available for embedding extraction ─────────────────────────
EMBEDDING_SEASONS = {
    # fine-grained Jan-May windows (most phenologically informative)
    "jan":          ("01-01", "02-01"),
    "feb":          ("02-01", "03-01"),
    "mar":          ("03-01", "04-01"),
    "apr":          ("04-01", "05-01"),
    "may":          ("05-01", "06-01"),
    "janfeb":       ("01-01", "03-01"),
    "marapr":       ("03-01", "05-01"),
    # standard seasons
    "winter":       ("12-01", "03-01"),  # straddles year boundary, handled below
    "premonsoon":   ("03-01", "06-01"),
    "monsoon":      ("06-01", "10-01"),
    "postmonsoon":  ("10-01", "12-01"),
}

DINO_INPUT_MEAN = [0.485, 0.456, 0.406]
DINO_INPUT_STD  = [0.229, 0.224, 0.225]


def load_dino_model(model_name: str):
    """Load a DINOv2 model via torch.hub. Falls back to dinov2_vits14."""
    try:
        import torch
        model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
        model.eval()
        return model, torch
    except Exception as e:
        print(f"[WARN] Could not load {model_name} via torch.hub: {e}", file=sys.stderr)
        print("[WARN] Trying dinov2_vits14 …", file=sys.stderr)
        import torch
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
        model.eval()
        return model, torch


def patch_from_arrays(
    arrays: dict[str, np.ndarray],
    crown_mask: np.ndarray,
    patch_px: int,
) -> np.ndarray:
    """
    Crop a square patch centred on the crown mask bounding box.
    Returns float32 array of shape (3, patch_px, patch_px) in RGB order,
    normalised to [0, 1] from the Sentinel-2 reflectance range (0–10000).
    """
    import cv2

    rows, cols = np.where(crown_mask)
    if len(rows) == 0:
        return np.zeros((3, patch_px, patch_px), dtype=np.float32)

    cy = int(rows.mean())
    cx = int(cols.mean())
    h, w = crown_mask.shape
    half = patch_px // 2

    # Pad to ensure we can always crop
    def safe_slice(arr2d, cy, cx, half, h, w):
        padded = np.pad(arr2d, half, constant_values=np.nan)
        return padded[cy: cy + 2 * half, cx: cx + 2 * half]

    red   = safe_slice(arrays.get("B4", np.zeros_like(crown_mask, dtype=float)), cy, cx, half, h, w)
    green = safe_slice(arrays.get("B3", np.zeros_like(crown_mask, dtype=float)), cy, cx, half, h, w)
    blue  = safe_slice(arrays.get("B2", np.zeros_like(crown_mask, dtype=float)), cy, cx, half, h, w)

    # Sentinel-2 L2A surface reflectance is 0–10000; normalise to [0,1]
    rgb = np.stack([red, green, blue], axis=0).astype(np.float32) / 10000.0
    rgb = np.nan_to_num(rgb, nan=0.0).clip(0, 1)

    # Resize to patch_px × patch_px using cv2 if not already correct size
    if rgb.shape[1] != patch_px or rgb.shape[2] != patch_px:
        rgb_hw = np.transpose(rgb, (1, 2, 0))
        rgb_hw = cv2.resize(rgb_hw, (patch_px, patch_px), interpolation=cv2.INTER_LINEAR)
        rgb = np.transpose(rgb_hw, (2, 0, 1))

    # Normalize for DINOv2 (ImageNet stats)
    for c in range(3):
        rgb[c] = (rgb[c] - DINO_INPUT_MEAN[c]) / DINO_INPUT_STD[c]

    return rgb


def extract_embeddings_for_season(
    crowns,
    catalog,
    year: int,
    season_name: str,
    patch_px: int,
    max_cloud: float,
    max_items: int,
    pad_pixels: int,
    dino_model,
    torch,
    embed_dim: int,
) -> np.ndarray:
    """
    Returns array of shape (n_crowns, embed_dim).
    Averages embeddings over up to max_items STAC items for the season.
    """
    import rasterio

    bounds_wgs = list(map(float, crowns.total_bounds))
    fallback_lat = float((bounds_wgs[1] + bounds_wgs[3]) / 2.0)

    start_str, end_str = EMBEDDING_SEASONS[season_name]
    if season_name == "winter":
        # Winter straddles year boundary
        season = _s2.Season("winter", f"{year - 1}-{start_str}", f"{year}-{end_str}")
    else:
        season = _s2.Season(season_name, f"{year}-{start_str}", f"{year}-{end_str}")

    items = _s2.query_items(catalog, bounds_wgs, season, max_cloud, max_items)
    if not items:
        print(f"  No items for {season_name} {year}, filling zeros", flush=True)
        return np.zeros((len(crowns), embed_dim), dtype=np.float32)

    # Pick the EPSG from items
    epsgs = [_s2.sentinel_epsg_from_item(it, fallback_lat) for it in items]
    from collections import Counter
    epsg = Counter(epsgs).most_common(1)[0][0]
    dst_crs = f"EPSG:{epsg}"
    bounds_proj = _s2.transform_bbox_wgs84(bounds_wgs, dst_crs)

    # Build extraction geometries (10m buffer = ~crown centroid area)
    import geopandas as gpd
    geoms_buffer = _s2.extraction_geometries(crowns, dst_crs, "buffer", 10.0)
    geoms_point  = _s2.extraction_geometries(crowns, dst_crs, "point",  0.0)

    all_embeddings = []  # list of (n_crowns, embed_dim) arrays

    for item in items:
        print(f"  Embedding [{season_name}/{year}] {item.id}", flush=True)
        with rasterio.Env(GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_CONNECTTIMEOUT="20",
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            arrays, valid, transform = _s2.read_item_stack(item, bounds_proj, pad_pixels)

        masks = _s2.build_masks(geoms_buffer, transform, valid.shape, True)

        patches = []
        for mask in masks:
            p = patch_from_arrays(arrays, mask, patch_px)
            patches.append(p)

        patches_tensor = torch.tensor(np.stack(patches, axis=0))  # (N, 3, patch_px, patch_px)
        with torch.no_grad():
            emb = dino_model(patches_tensor).cpu().numpy()  # (N, embed_dim)
        all_embeddings.append(emb)

    if not all_embeddings:
        return np.zeros((len(crowns), embed_dim), dtype=np.float32)

    # Average over items
    return np.mean(all_embeddings, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--seasons", nargs="+",
        default=["jan", "feb", "mar", "apr", "may", "premonsoon", "postmonsoon"],
        help=f"Season names to embed. Choices: {list(EMBEDDING_SEASONS.keys())}",
    )
    ap.add_argument("--patch-px", type=int, default=64,
                    help="Patch size in pixels (must be divisible by 14 for ViT patch size).")
    ap.add_argument("--model", default="dinov2_vits14",
                    choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
                    help="DINOv2 model variant.")
    ap.add_argument("--max-cloud", type=float, default=70.0)
    ap.add_argument("--max-items-per-season", type=int, default=3)
    ap.add_argument("--pad-pixels", type=int, default=40)
    ap.add_argument("--fuse-csv", default=None,
                    help="If given, merge embeddings with this existing spectral feature CSV.")
    ap.add_argument("--min-crown-area-m2", type=float, default=None,
                    help="If set, skip crowns smaller than this area (in m²).")
    ap.add_argument("--crowns", default=None)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    # Snap patch size to nearest multiple of 14 (DINOv2 patch stride)
    patch_px = max(14, (args.patch_px // 14) * 14)
    if patch_px != args.patch_px:
        print(f"[INFO] Snapped patch size {args.patch_px} → {patch_px} (multiple of 14)")

    import geopandas as gpd
    import planetary_computer as pc
    from pystac_client import Client

    crowns_path = args.crowns or str(DATA_DIR / "iitd_sv_crowns_master_wgs84.geojson")
    crowns = gpd.read_file(crowns_path)
    crowns = crowns[crowns.geometry.notnull() & ~crowns.geometry.is_empty].reset_index(drop=True)

    if args.min_crown_area_m2 is not None:
        crowns_proj = crowns.to_crs("EPSG:32643")
        keep = crowns_proj.geometry.area >= args.min_crown_area_m2
        n_before = len(crowns)
        crowns = crowns[keep].reset_index(drop=True)
        print(f"[INFO] Crown area filter ≥{args.min_crown_area_m2}m²: kept {len(crowns)}/{n_before}")

    print(f"[INFO] Loading DINOv2 model: {args.model} …", flush=True)
    dino_model, torch = load_dino_model(args.model)

    # Get embedding dim from a dummy forward pass
    dummy = torch.zeros(1, 3, patch_px, patch_px)
    with torch.no_grad():
        embed_dim = dino_model(dummy).shape[1]
    print(f"[INFO] Embedding dim: {embed_dim}", flush=True)

    catalog = Client.open(_s2.STAC_API)

    prop_cols = [c for c in crowns.columns if c != "geometry"]
    result = crowns[prop_cols].copy()

    for season in args.seasons:
        if season not in EMBEDDING_SEASONS:
            print(f"[WARN] Unknown season {season!r}, skipping.")
            continue
        emb = extract_embeddings_for_season(
            crowns, catalog, args.year, season,
            patch_px, args.max_cloud, args.max_items_per_season, args.pad_pixels,
            dino_model, torch, embed_dim,
        )
        for d in range(embed_dim):
            result[f"dino_{season}_d{d:04d}"] = emb[:, d]
        print(f"[INFO] Season {season}: embeddings added ({embed_dim} dims)", flush=True)

    if args.fuse_csv:
        spectral = pd.read_csv(args.fuse_csv)
        result = result.merge(spectral.drop(columns=[c for c in prop_cols if c in spectral.columns
                                                      and c != "crown_uid"], errors="ignore"),
                              on="crown_uid", how="inner")
        print(f"[INFO] Fused with {args.fuse_csv}: {len(result)} rows after merge")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"\n[DONE] Wrote {out}  ({len(result)} rows, {len(result.columns)} columns)")
    print(f"       Embedding columns: {embed_dim * len(args.seasons)}")
    print(f"       Run model sweep:   python python/06_model_sweep.py --csv {out} --label label_acacia --split random")


if __name__ == "__main__":
    main()
