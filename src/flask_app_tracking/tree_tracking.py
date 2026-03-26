import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import rasterio
from rasterio.mask import mask
from shapely.affinity import affine_transform, translate
from shapely.geometry import Polygon, mapping

try:
    import fiona

    FIONA_AVAILABLE = True
except Exception:
    FIONA_AVAILABLE = False

try:
    from skimage.registration import phase_cross_correlation
    from skimage.transform import resize

    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


@dataclass
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


class TreeTrackingGraph:
    def __init__(
        self,
        crown_dir: Optional[str] = None,
        ortho_dir: Optional[str] = None,
        output_dir: str = "../../output",
        simplify_tol: float = 1.0,
        max_crowns_preview: int = 200,
        auto_discover: bool = True,
        iou_threshold: float = 0.15,
        resize_factor: float = 0.1,
        max_crowns: Optional[int] = None,
        multithresh_dir: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir
        self.simplify_tol = simplify_tol
        self.max_crowns_preview = max_crowns_preview
        self.max_crowns = max_crowns if max_crowns is not None else max_crowns_preview
        self.resize_factor = resize_factor
        self.iou_threshold = iou_threshold

        self.crown_dir = crown_dir
        self.ortho_dir = ortho_dir
        self.multithresh_dir = multithresh_dir

        self.file_pairs: List[Tuple[str, Optional[str]]] = []
        self.crown_files: List[str] = []
        self.ortho_files: List[str] = []
        self.om_ids: List[int] = []

        self.crowns_gdfs: Dict[int, gpd.GeoDataFrame] = {}
        self.crown_attrs: Dict[int, List[Dict[str, Any]]] = {}
        self.crown_images: Dict[int, List[Optional[np.ndarray]]] = {}
        self.crown_crs: Dict[int, Optional[Any]] = {}
        self.ortho_crs: Dict[int, Optional[Any]] = {}
        self.alignment_shifts: Dict[int, Tuple[float, float]] = {}
        self.alignment_transforms: Dict[int, Tuple[float, float, float, float, float, float]] = {}
        self.alignment_debug: Dict[int, Dict[str, Any]] = {}

        self.multithreshold_layers: Dict[int, List[str]] = {}
        self.base_threshold_tag: Optional[str] = None

        self.G: nx.DiGraph = nx.DiGraph()
        self.match_case_mode: str = "balanced"
        self.case_configs: Dict[str, MatchCaseConfig] = self._ultra_relaxed_case_configs()
        self.case_order: List[str] = ["one_to_one", "containment", "nearby"]
        self.last_case_counts: Dict[str, int] = {}
        self.last_selected_counts: Dict[str, int] = {}

        if auto_discover:
            if self.multithresh_dir:
                self.discover_multithreshold_files()
            else:
                self.discover_files()

    @staticmethod
    def _extract_numeric_id(name: str) -> Optional[int]:
        m = re.search(r"(\d+)", os.path.basename(name))
        return int(m.group(1)) if m else None

    def _finalize_pairs(self, pairs: List[Tuple[str, Optional[str]]], om_ids: List[int]) -> None:
        pairs_with_id = sorted([(oid, cf, of) for oid, (cf, of) in zip(om_ids, pairs)], key=lambda x: x[0])
        self.file_pairs = [(cf, of) for _, cf, of in pairs_with_id]
        self.om_ids = [oid for oid, _, _ in pairs_with_id]
        self.crown_files = [cf for cf, _ in self.file_pairs]
        self.ortho_files = [of for _, of in self.file_pairs if of]

    def discover_files(self) -> None:
        crown_dir = self.crown_dir or "../../input/input_crowns"
        ortho_dir = self.ortho_dir or "../../input/input_om"
        if not os.path.isdir(crown_dir):
            raise FileNotFoundError(f"Crown directory not found: {crown_dir}")
        self.crown_dir = crown_dir
        self.ortho_dir = ortho_dir

        crown_files = [os.path.join(crown_dir, f) for f in os.listdir(crown_dir) if f.lower().endswith(".gpkg")]
        ortho_files = [os.path.join(ortho_dir, f) for f in os.listdir(ortho_dir) if f.lower().endswith(".tif")] if os.path.isdir(ortho_dir) else []
        if not crown_files:
            raise FileNotFoundError(f"No .gpkg files found in {crown_dir}")

        crowns_by_id = {}
        for cf in crown_files:
            cid = self._extract_numeric_id(cf)
            crowns_by_id[cid if cid is not None else cf] = cf

        orthos_by_id = {}
        for of in ortho_files:
            oid = self._extract_numeric_id(of)
            orthos_by_id[oid if oid is not None else of] = of

        numeric_ids = sorted(
            set(k for k in crowns_by_id.keys() if isinstance(k, int))
            & set(k for k in orthos_by_id.keys() if isinstance(k, int))
        )

        pairs: List[Tuple[str, Optional[str]]] = []
        om_ids: List[int] = []
        if numeric_ids:
            for nid in numeric_ids:
                pairs.append((crowns_by_id[nid], orthos_by_id.get(nid)))
                om_ids.append(nid)
            crown_only = sorted(k for k in crowns_by_id.keys() if isinstance(k, int) and k not in numeric_ids)
            for nid in crown_only:
                pairs.append((crowns_by_id[nid], None))
                om_ids.append(nid)
        else:
            crowns_sorted = sorted(crown_files)
            orthos_sorted = sorted(ortho_files)
            for i, cf in enumerate(crowns_sorted):
                pairs.append((cf, orthos_sorted[i] if i < len(orthos_sorted) else None))
                om_ids.append(i + 1)

        self._finalize_pairs(pairs, om_ids)

    def discover_multithreshold_files(self) -> None:
        if not self.multithresh_dir:
            raise ValueError("multithresh_dir is required")
        mdir = Path(self.multithresh_dir)
        if not mdir.exists():
            raise FileNotFoundError(f"Multi-threshold directory not found: {mdir}")
        ortho_dir = self.ortho_dir or "../../input/input_om"
        self.ortho_dir = ortho_dir

        gpkg_files = sorted(mdir.glob("OM*_multithreshold.gpkg"))
        ortho_files = sorted(Path(ortho_dir).glob("*.tif")) if os.path.isdir(ortho_dir) else []

        pairs: List[Tuple[str, Optional[str]]] = []
        om_ids: List[int] = []
        for gpkg in gpkg_files:
            om_num = int(gpkg.stem.split("_")[0].replace("OM", ""))
            ortho_match = None
            for ortho in ortho_files:
                if f"om{om_num}" in ortho.stem.lower():
                    ortho_match = str(ortho)
                    break
            pairs.append((str(gpkg), ortho_match))
            om_ids.append(om_num)

        self._finalize_pairs(pairs, om_ids)

    @staticmethod
    def _read_ortho_lowres(path: str, max_size: int = 1200):
        with rasterio.open(path) as src:
            scale = max(src.width, src.height) / max_size if max(src.width, src.height) > max_size else 1.0
            out_w = int(round(src.width / scale))
            out_h = int(round(src.height / scale))
            data = src.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)
            rgb = np.moveaxis(data, 0, -1).astype(np.float32)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
            gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
            scale_x = src.width / out_w
            scale_y = src.height / out_h
            transform = src.transform * rasterio.transform.Affine.scale(scale_x, scale_y)
            return gray, transform

    @staticmethod
    def _bounds_from_transform(h: int, w: int, transform: rasterio.transform.Affine):
        corners = [(0, 0), (0, w), (h, 0), (h, w)]
        xs, ys = [], []
        for row, col in corners:
            x, y = rasterio.transform.xy(transform, row, col, offset="ul")
            xs.append(x)
            ys.append(y)
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _safe_window(win: rasterio.windows.Window, height: int, width: int) -> rasterio.windows.Window:
        col_off = int(max(0, min(width, win.col_off)))
        row_off = int(max(0, min(height, win.row_off)))
        max_w = max(0, width - col_off)
        max_h = max(0, height - row_off)
        w = int(max(0, min(max_w, win.width)))
        h = int(max(0, min(max_h, win.height)))
        return rasterio.windows.Window(col_off=col_off, row_off=row_off, width=w, height=h)

    @classmethod
    def _crop_to_overlap(cls, ref_img, mov_img, ref_transform, mov_transform):
        ref_h, ref_w = ref_img.shape[:2]
        mov_h, mov_w = mov_img.shape[:2]
        ref_left, ref_bottom, ref_right, ref_top = cls._bounds_from_transform(ref_h, ref_w, ref_transform)
        mov_left, mov_bottom, mov_right, mov_top = cls._bounds_from_transform(mov_h, mov_w, mov_transform)
        left = max(ref_left, mov_left)
        right = min(ref_right, mov_right)
        bottom = max(ref_bottom, mov_bottom)
        top = min(ref_top, mov_top)
        if left >= right or bottom >= top:
            return None, None
        ref_win = rasterio.windows.from_bounds(left, bottom, right, top, ref_transform).round_offsets().round_lengths()
        mov_win = rasterio.windows.from_bounds(left, bottom, right, top, mov_transform).round_offsets().round_lengths()
        ref_win = cls._safe_window(ref_win, ref_h, ref_w)
        mov_win = cls._safe_window(mov_win, mov_h, mov_w)
        if ref_win.width < 2 or ref_win.height < 2 or mov_win.width < 2 or mov_win.height < 2:
            return None, None
        ref_crop = ref_img[
            int(ref_win.row_off) : int(ref_win.row_off + ref_win.height),
            int(ref_win.col_off) : int(ref_win.col_off + ref_win.width),
        ]
        mov_crop = mov_img[
            int(mov_win.row_off) : int(mov_win.row_off + mov_win.height),
            int(mov_win.col_off) : int(mov_win.col_off + mov_win.width),
        ]
        return ref_crop, mov_crop

    @staticmethod
    def _match_shape(ref_crop, mov_crop):
        if ref_crop.shape == mov_crop.shape:
            return ref_crop, mov_crop
        if not SKIMAGE_AVAILABLE:
            min_h = min(ref_crop.shape[0], mov_crop.shape[0])
            min_w = min(ref_crop.shape[1], mov_crop.shape[1])
            return ref_crop[:min_h, :min_w], mov_crop[:min_h, :min_w]
        mov_resized = resize(mov_crop, ref_crop.shape, preserve_range=True, anti_aliasing=True)
        return ref_crop, mov_resized

    @classmethod
    def _phase_corr_shift(cls, ref_gray, mov_gray, ref_transform, mov_transform):
        if not SKIMAGE_AVAILABLE:
            return None
        ref_crop, mov_crop = cls._crop_to_overlap(ref_gray, mov_gray, ref_transform, mov_transform)
        if ref_crop is None or mov_crop is None:
            return None
        ref_crop, mov_crop = cls._match_shape(ref_crop, mov_crop)
        shift, _, _ = phase_cross_correlation(ref_crop, mov_crop, upsample_factor=10)
        shift_row, shift_col = shift
        dx = shift_col * ref_transform.a
        dy = shift_row * ref_transform.e
        return float(dx), float(dy)

    @classmethod
    def _phase_corr_shift_tiled(
        cls,
        ref_gray,
        mov_gray,
        ref_transform,
        mov_transform,
        tiles: int = 4,
        min_tile_size: int = 160,
        upsample_factor: int = 10,
        min_valid_tiles: int = 4,
    ) -> Optional[Tuple[float, float]]:
        shift, _ = cls._phase_corr_shift_tiled_debug(
            ref_gray,
            mov_gray,
            ref_transform,
            mov_transform,
            tiles=tiles,
            min_tile_size=min_tile_size,
            upsample_factor=upsample_factor,
            min_valid_tiles=min_valid_tiles,
        )
        return shift

    @classmethod
    def _phase_corr_shift_tiled_debug(
        cls,
        ref_gray,
        mov_gray,
        ref_transform,
        mov_transform,
        tiles: int = 4,
        min_tile_size: int = 160,
        upsample_factor: int = 10,
        min_valid_tiles: int = 4,
        min_texture_std: float = 0.02,
        max_error: float = 1.0,
        max_abs_shift_fraction: float = 0.35,
        inlier_sigma: float = 3.0,
        min_inliers: int = 3,
    ) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
        """Robust PCC: estimate shift as the median of many local tile PCC shifts.

        Returns both the (dx, dy) shift in map units and a debug dict describing
        tile acceptance/rejection and inlier filtering.
        """
        debug: Dict[str, Any] = {
            "method": "pcc_tiled",
            "tiles": int(tiles),
            "min_tile_size": int(min_tile_size),
            "upsample_factor": int(upsample_factor),
            "min_valid_tiles": int(min_valid_tiles),
            "min_texture_std": float(min_texture_std),
            "max_error": float(max_error),
            "max_abs_shift_fraction": float(max_abs_shift_fraction),
            "inlier_sigma": float(inlier_sigma),
            "min_inliers": int(min_inliers),
            "fallback": "",
            "rejected": {"low_texture": 0, "high_error": 0, "too_large_shift": 0, "exception": 0},
        }

        if not SKIMAGE_AVAILABLE:
            debug["fallback"] = "skimage_missing"
            return None, debug

        ref_crop, mov_crop = cls._crop_to_overlap(ref_gray, mov_gray, ref_transform, mov_transform)
        if ref_crop is None or mov_crop is None:
            debug["fallback"] = "no_overlap"
            return None, debug
        ref_crop, mov_crop = cls._match_shape(ref_crop, mov_crop)

        h, w = ref_crop.shape[:2]
        debug["overlap_shape"] = (int(h), int(w))
        if h < min_tile_size or w < min_tile_size:
            debug["fallback"] = "full_pcc_small_overlap"
            return cls._phase_corr_shift(ref_gray, mov_gray, ref_transform, mov_transform), debug

        t = max(2, int(tiles))
        tile_h = h // t
        tile_w = w // t
        debug["n_tile_positions"] = int(t * t)
        if tile_h < min_tile_size or tile_w < min_tile_size:
            debug["fallback"] = "full_pcc_tiles_too_small"
            return cls._phase_corr_shift(ref_gray, mov_gray, ref_transform, mov_transform), debug

        # Shift sanity bound: prevent tiles from voting for absurd jumps.
        # Bound is proportional to overlap extent.
        overlap_width_mu = float(w * ref_transform.a)
        overlap_height_mu = float(h * ref_transform.e)
        max_dx = float(abs(overlap_width_mu) * float(max_abs_shift_fraction))
        max_dy = float(abs(overlap_height_mu) * float(max_abs_shift_fraction))

        candidates: List[Tuple[float, float, float]] = []  # (dx, dy, error)
        for i in range(t):
            for j in range(t):
                r0 = i * tile_h
                r1 = (i + 1) * tile_h if i < t - 1 else h
                c0 = j * tile_w
                c1 = (j + 1) * tile_w if j < t - 1 else w
                ref_tile = ref_crop[r0:r1, c0:c1]
                mov_tile = mov_crop[r0:r1, c0:c1]
                if ref_tile.shape[0] < min_tile_size or ref_tile.shape[1] < min_tile_size:
                    continue

                # Simple texture gate: avoid tiles that are mostly uniform (PCC unstable).
                ref_std = float(np.std(ref_tile))
                mov_std = float(np.std(mov_tile))
                if not np.isfinite(ref_std) or not np.isfinite(mov_std) or ref_std < min_texture_std or mov_std < min_texture_std:
                    debug["rejected"]["low_texture"] += 1
                    continue

                try:
                    shift, error, _ = phase_cross_correlation(ref_tile, mov_tile, upsample_factor=upsample_factor)
                except Exception:
                    debug["rejected"]["exception"] += 1
                    continue
                if not np.isfinite(error) or float(error) > float(max_error):
                    debug["rejected"]["high_error"] += 1
                    continue
                shift_row, shift_col = shift
                dx = float(shift_col * ref_transform.a)
                dy = float(shift_row * ref_transform.e)
                if abs(dx) > max_dx or abs(dy) > max_dy:
                    debug["rejected"]["too_large_shift"] += 1
                    continue

                candidates.append((dx, dy, float(error)))

        debug["n_valid_tiles"] = int(len(candidates))
        if len(candidates) < int(min_valid_tiles):
            debug["fallback"] = "full_pcc_insufficient_tiles"
            return cls._phase_corr_shift(ref_gray, mov_gray, ref_transform, mov_transform), debug

        dxs = np.array([c[0] for c in candidates], dtype=float)
        dys = np.array([c[1] for c in candidates], dtype=float)
        errs = np.array([c[2] for c in candidates], dtype=float)
        debug["median_error_all"] = float(np.median(errs)) if errs.size else float("nan")

        # Robust inlier selection around the (dx,dy) median.
        dx_med = float(np.median(dxs))
        dy_med = float(np.median(dys))
        resid = np.sqrt((dxs - dx_med) ** 2 + (dys - dy_med) ** 2)
        resid_med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - resid_med)))
        scale = 1.4826 * mad if mad > 0 else 0.0
        thr = float(inlier_sigma) * scale if scale > 0 else float("inf")
        inliers = resid <= thr

        if int(inliers.sum()) < int(min_inliers):
            # If MAD collapses or we over-filtered, fall back to using all candidates.
            inliers = np.ones_like(resid, dtype=bool)
            debug["fallback"] = "no_inliers_using_all_tiles"
        else:
            debug["fallback"] = ""

        debug["n_inliers"] = int(inliers.sum())
        if errs.size:
            debug["median_error_inliers"] = float(np.median(errs[inliers]))

        dx_out = float(np.median(dxs[inliers]))
        dy_out = float(np.median(dys[inliers]))
        return (dx_out, dy_out), debug

    @classmethod
    def _ecc_shift(
        cls,
        ref_gray,
        mov_gray,
        ref_transform,
        mov_transform,
        motion: str = "euclidean",
        number_of_iterations: int = 250,
        termination_eps: float = 1e-6,
    ) -> Optional[Tuple[float, float]]:
        """Estimate translation-like shift using OpenCV ECC on overlap region.

        ECC can be more stable than PCC when there is small rotation/illumination changes.
        Returns (dx, dy) in map units.
        """
        if not OPENCV_AVAILABLE:
            return None
        ref_crop, mov_crop = cls._crop_to_overlap(ref_gray, mov_gray, ref_transform, mov_transform)
        if ref_crop is None or mov_crop is None:
            return None
        ref_crop, mov_crop = cls._match_shape(ref_crop, mov_crop)

        ref = ref_crop.astype(np.float32)
        mov = mov_crop.astype(np.float32)
        ref = ref - float(np.mean(ref))
        mov = mov - float(np.mean(mov))
        ref_std = float(np.std(ref))
        mov_std = float(np.std(mov))
        if ref_std > 0:
            ref = ref / ref_std
        if mov_std > 0:
            mov = mov / mov_std

        motion = (motion or "euclidean").lower().strip()
        if motion == "translation":
            warp_mode = cv2.MOTION_TRANSLATION
        else:
            warp_mode = cv2.MOTION_EUCLIDEAN
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(number_of_iterations), float(termination_eps))
        try:
            cv2.findTransformECC(ref, mov, warp, warp_mode, criteria, inputMask=None, gaussFiltSize=5)
        except Exception:
            return None

        tx = float(warp[0, 2])
        ty = float(warp[1, 2])
        dx = tx * float(ref_transform.a)
        dy = ty * float(ref_transform.e)
        return float(dx), float(dy)

    def align_to_reference(self, reference_om_id: Optional[int] = None, max_preview: int = 1200) -> Dict[int, Tuple[float, float]]:
        return self.align_to_reference_with_method(reference_om_id=reference_om_id, max_preview=max_preview, method="pcc")

    @staticmethod
    def _as_homog(params: Tuple[float, float, float, float, float, float]) -> np.ndarray:
        a, b, d, e, xoff, yoff = params
        return np.array([[a, b, xoff], [d, e, yoff], [0.0, 0.0, 1.0]], dtype=float)

    @staticmethod
    def _from_homog(M: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        return (float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1]), float(M[0, 2]), float(M[1, 2]))

    @staticmethod
    def _invert_affine_params(params: Tuple[float, float, float, float, float, float]) -> Tuple[float, float, float, float, float, float]:
        a, b, d, e, xoff, yoff = params
        det = a * e - b * d
        if abs(det) < 1e-12:
            return (1.0, 0.0, 0.0, 1.0, -xoff, -yoff)
        ia = e / det
        ib = -b / det
        id_ = -d / det
        ie = a / det
        ixoff = -(ia * xoff + ib * yoff)
        iyoff = -(id_ * xoff + ie * yoff)
        return (float(ia), float(ib), float(id_), float(ie), float(ixoff), float(iyoff))

    @staticmethod
    def _apply_affine_to_gdf(gdf: gpd.GeoDataFrame, params: Tuple[float, float, float, float, float, float]) -> gpd.GeoDataFrame:
        out = gdf.copy()
        out["geometry"] = out["geometry"].apply(lambda g: affine_transform(g, params) if g is not None else g)
        return out

    @staticmethod
    def _kp_to_map_xy(kp_xy: np.ndarray, transform: rasterio.transform.Affine) -> np.ndarray:
        if kp_xy.size == 0:
            return np.zeros((0, 2), dtype=float)
        cols = kp_xy[:, 0]
        rows = kp_xy[:, 1]
        xs = transform.c + cols * transform.a + rows * transform.b
        ys = transform.f + cols * transform.d + rows * transform.e
        return np.column_stack([xs, ys]).astype(float)

    @classmethod
    def _orb_affine_step(
        cls,
        ref_gray: np.ndarray,
        mov_gray: np.ndarray,
        ref_transform: rasterio.transform.Affine,
        mov_transform: rasterio.transform.Affine,
        nfeatures: int = 5000,
        ransac_thresh_map_units: float = 2.5,
        min_inliers: int = 25,
    ) -> Tuple[Optional[Tuple[float, float, float, float, float, float]], Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "method": "orb_affine",
            "nfeatures": int(nfeatures),
            "ransac_thresh_map_units": float(ransac_thresh_map_units),
            "min_inliers": int(min_inliers),
        }
        if not OPENCV_AVAILABLE:
            debug["error"] = "opencv_unavailable"
            return None, debug

        try:
            ref_u8 = np.clip(ref_gray * 255.0, 0, 255).astype(np.uint8)
            mov_u8 = np.clip(mov_gray * 255.0, 0, 255).astype(np.uint8)
            orb = cv2.ORB_create(nfeatures=nfeatures)
            kp1, des1 = orb.detectAndCompute(ref_u8, None)
            kp2, des2 = orb.detectAndCompute(mov_u8, None)
            debug["kp_ref"] = 0 if kp1 is None else int(len(kp1))
            debug["kp_mov"] = 0 if kp2 is None else int(len(kp2))
            if des1 is None or des2 is None or len(des1) < 20 or len(des2) < 20:
                debug["error"] = "insufficient_descriptors"
                return None, debug

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            knn = matcher.knnMatch(des2, des1, k=2)  # query=mov, train=ref
            good = []
            for m, n in knn:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            debug["matches_good"] = int(len(good))
            if len(good) < 40:
                debug["error"] = "insufficient_matches"
                return None, debug

            mov_xy = np.array([kp2[m.queryIdx].pt for m in good], dtype=float)
            ref_xy = np.array([kp1[m.trainIdx].pt for m in good], dtype=float)

            mov_map = cls._kp_to_map_xy(mov_xy, mov_transform)
            ref_map = cls._kp_to_map_xy(ref_xy, ref_transform)

            M, inliers = cv2.estimateAffinePartial2D(
                mov_map,
                ref_map,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thresh_map_units,
                maxIters=5000,
                confidence=0.995,
                refineIters=10,
            )
            if M is None or inliers is None:
                debug["error"] = "affine_estimation_failed"
                return None, debug
            inliers = inliers.reshape(-1).astype(bool)
            n_in = int(inliers.sum())
            debug["inliers"] = n_in
            if n_in < min_inliers:
                debug["error"] = "too_few_inliers"
                return None, debug

            mov_in = mov_map[inliers]
            ref_in = ref_map[inliers]
            pred = (M[:, :2] @ mov_in.T).T + M[:, 2]
            resid = np.linalg.norm(ref_in - pred, axis=1)
            debug["rmse_map_units"] = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")

            params = (float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1]), float(M[0, 2]), float(M[1, 2]))
            return params, debug
        except Exception as e:
            debug["error"] = f"exception: {e}"
            return None, debug

    def _centroid_alignment_subset(self, om_id: int, threshold_tag: Optional[str]) -> np.ndarray:
        gdf = self.crowns_gdfs.get(om_id)
        if gdf is None or gdf.empty:
            return np.zeros((0, 2), dtype=float)

        if threshold_tag and "threshold_tag" in gdf.columns:
            sub = gdf[gdf["threshold_tag"] == threshold_tag]
            if not sub.empty:
                gdf = sub
        geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros((0, 2), dtype=float)
        return np.array([[g.centroid.x, g.centroid.y] for g in geoms], dtype=float)

    @staticmethod
    def _robust_median_shift(prev_xy: np.ndarray, curr_xy: np.ndarray, max_match_dist: float = 30.0, min_pairs: int = 15) -> Optional[Tuple[float, float]]:
        if prev_xy.size == 0 or curr_xy.size == 0:
            return None
        if prev_xy.shape[0] < min_pairs or curr_xy.shape[0] < min_pairs:
            return None

        nn = NearestNeighbors(n_neighbors=1).fit(prev_xy)
        dists, idxs = nn.kneighbors(curr_xy)
        dists = dists.reshape(-1)
        idxs = idxs.reshape(-1)

        good = np.isfinite(dists) & (dists <= max_match_dist)
        if good.sum() < min_pairs:
            return None

        matched_prev = prev_xy[idxs[good]]
        matched_curr = curr_xy[good]
        deltas = matched_prev - matched_curr
        dx = float(np.median(deltas[:, 0]))
        dy = float(np.median(deltas[:, 1]))
        return dx, dy

    def align_to_reference_with_method(
        self,
        reference_om_id: Optional[int] = None,
        max_preview: int = 1200,
        method: str = "pcc",
        align_threshold_tag: Optional[str] = "conf_0p65",
        max_match_dist: float = 30.0,
        min_pairs: int = 15,
    ) -> Dict[int, Tuple[float, float]]:
        method = (method or "pcc").lower().strip()
        if reference_om_id is None and self.om_ids:
            reference_om_id = self.om_ids[0]
        if reference_om_id is None:
            return {}

        if method not in {"pcc", "pcc_tiled", "ecc", "crowns", "orb_affine"}:
            raise ValueError(
                f"Unknown alignment method: {method!r}. Use 'pcc', 'pcc_tiled', 'ecc', 'crowns', or 'orb_affine'."
            )

        self.alignment_debug = {}

        if method in {"pcc", "pcc_tiled", "ecc"}:
            if not SKIMAGE_AVAILABLE:
                return {om_id: (0.0, 0.0) for om_id in self.om_ids}

            ortho_gray = {}
            ortho_transform = {}
            for om_id, (_, ortho_file) in zip(self.om_ids, self.file_pairs):
                if ortho_file and os.path.exists(ortho_file):
                    gray, transform = self._read_ortho_lowres(ortho_file, max_preview)
                    ortho_gray[om_id] = gray
                    ortho_transform[om_id] = transform
                else:
                    ortho_gray[om_id] = None
                    ortho_transform[om_id] = None

            shifts = {reference_om_id: (0.0, 0.0)}
            for idx in range(1, len(self.om_ids)):
                prev_id = self.om_ids[idx - 1]
                curr_id = self.om_ids[idx]
                ref_gray = ortho_gray.get(prev_id)
                mov_gray = ortho_gray.get(curr_id)
                ref_transform = ortho_transform.get(prev_id)
                mov_transform = ortho_transform.get(curr_id)
                if ref_gray is None or mov_gray is None or ref_transform is None or mov_transform is None:
                    shifts[curr_id] = shifts.get(prev_id, (0.0, 0.0))
                    continue
                shift = None
                if method == "ecc":
                    shift = self._ecc_shift(ref_gray, mov_gray, ref_transform, mov_transform, motion="euclidean")
                    if shift is None:
                        shift = self._ecc_shift(ref_gray, mov_gray, ref_transform, mov_transform, motion="translation")
                elif method == "pcc_tiled":
                    shift, dbg = self._phase_corr_shift_tiled_debug(ref_gray, mov_gray, ref_transform, mov_transform)
                    self.alignment_debug[curr_id] = dbg
                if shift is None:
                    shift = self._phase_corr_shift(ref_gray, mov_gray, ref_transform, mov_transform)
                if shift is None:
                    shifts[curr_id] = shifts.get(prev_id, (0.0, 0.0))
                    continue
                dx, dy = shift
                prev_dx, prev_dy = shifts.get(prev_id, (0.0, 0.0))
                shifts[curr_id] = (prev_dx + dx, prev_dy + dy)

            transforms = {om_id: (1.0, 0.0, 0.0, 1.0, float(dx), float(dy)) for om_id, (dx, dy) in shifts.items()}

        elif method == "crowns":
            if not SKLEARN_AVAILABLE:
                return {om_id: (0.0, 0.0) for om_id in self.om_ids}

            shifts = {reference_om_id: (0.0, 0.0)}
            for idx in range(1, len(self.om_ids)):
                prev_id = self.om_ids[idx - 1]
                curr_id = self.om_ids[idx]

                prev_xy = self._centroid_alignment_subset(prev_id, align_threshold_tag)
                curr_xy = self._centroid_alignment_subset(curr_id, align_threshold_tag)
                delta = self._robust_median_shift(prev_xy, curr_xy, max_match_dist=max_match_dist, min_pairs=min_pairs)
                if delta is None:
                    shifts[curr_id] = shifts.get(prev_id, (0.0, 0.0))
                    continue
                dx_step, dy_step = delta
                prev_dx, prev_dy = shifts.get(prev_id, (0.0, 0.0))
                shifts[curr_id] = (prev_dx + dx_step, prev_dy + dy_step)

            transforms = {om_id: (1.0, 0.0, 0.0, 1.0, float(dx), float(dy)) for om_id, (dx, dy) in shifts.items()}

        else:
            if not OPENCV_AVAILABLE:
                return {om_id: (0.0, 0.0) for om_id in self.om_ids}

            ortho_gray: Dict[int, Optional[np.ndarray]] = {}
            ortho_transform: Dict[int, Optional[rasterio.transform.Affine]] = {}
            for om_id, (_, ortho_file) in zip(self.om_ids, self.file_pairs):
                if ortho_file and os.path.exists(ortho_file):
                    gray, transform = self._read_ortho_lowres(ortho_file, max_preview)
                    ortho_gray[om_id] = gray
                    ortho_transform[om_id] = transform
                else:
                    ortho_gray[om_id] = None
                    ortho_transform[om_id] = None

            transforms: Dict[int, Tuple[float, float, float, float, float, float]] = {reference_om_id: (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)}
            for idx in range(1, len(self.om_ids)):
                prev_id = self.om_ids[idx - 1]
                curr_id = self.om_ids[idx]
                ref_gray = ortho_gray.get(prev_id)
                mov_gray = ortho_gray.get(curr_id)
                ref_transform = ortho_transform.get(prev_id)
                mov_transform = ortho_transform.get(curr_id)
                if ref_gray is None or mov_gray is None or ref_transform is None or mov_transform is None:
                    transforms[curr_id] = transforms.get(prev_id, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
                    self.alignment_debug[curr_id] = {"method": "orb_affine", "error": "missing_ortho"}
                    continue

                step_params, dbg = self._orb_affine_step(ref_gray, mov_gray, ref_transform, mov_transform)
                self.alignment_debug[curr_id] = dbg
                if step_params is None:
                    # fallback: carry previous transform
                    transforms[curr_id] = transforms.get(prev_id, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
                    continue

                prev_h = self._as_homog(transforms.get(prev_id, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)))
                step_h = self._as_homog(step_params)
                curr_h = prev_h @ step_h
                transforms[curr_id] = self._from_homog(curr_h)

            shifts = {om_id: (float(params[4]), float(params[5])) for om_id, params in transforms.items()}

        self.alignment_transforms = transforms

        for om_id in self.om_ids:
            if om_id == reference_om_id:
                continue
            params = transforms.get(om_id, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
            aligned_gdf = self._apply_affine_to_gdf(self.crowns_gdfs[om_id], params)
            self.crowns_gdfs[om_id] = aligned_gdf
            self.crown_attrs[om_id] = [self._compute_crown_attributes(row.geometry) for _, row in aligned_gdf.iterrows()]

        self.alignment_shifts = shifts
        return shifts

    @staticmethod
    def _compute_crown_attributes(geometry) -> Dict[str, Any]:
        centroid = geometry.centroid
        area = geometry.area
        perimeter = geometry.length
        bounds = geometry.bounds
        compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
        try:
            min_rect = geometry.minimum_rotated_rectangle
            coords = list(min_rect.exterior.coords)
            side1 = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
            side2 = np.linalg.norm(np.array(coords[1]) - np.array(coords[2]))
            major_axis = max(side1, side2)
            minor_axis = min(side1, side2)
            eccentricity = minor_axis / major_axis if major_axis > 0 else 1.0
        except Exception:
            eccentricity = 1.0
        aspect_ratio = (bounds[3] - bounds[1]) / (bounds[2] - bounds[0]) if bounds[2] != bounds[0] else 1.0
        return {
            "geometry": geometry,
            "centroid": centroid,
            "area": area,
            "perimeter": perimeter,
            "compactness": compactness,
            "eccentricity": eccentricity,
            "aspect_ratio": aspect_ratio,
            "bounds": bounds,
        }

    @staticmethod
    def _compute_iou(g1, g2) -> float:
        try:
            intersection = g1.intersection(g2).area
            union = g1.union(g2).area
        except Exception:
            intersection = 0.0
            union = g1.area + g2.area
        return intersection / union if union > 0 else 0.0

    def compute_iou(self, g1, g2) -> float:
        return self._compute_iou(g1, g2)

    def _weighted_similarity(
        self,
        a1: Dict[str, Any],
        a2: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        max_dist: float = 100.0,
    ) -> Tuple[float, Dict[str, float]]:
        if weights is None:
            weights = {"spatial": 0.4, "area": 0.2, "shape": 0.2, "iou": 0.2}
        centroid_dist = a1["centroid"].distance(a2["centroid"])
        spatial_sim = max(0.0, 1.0 - (centroid_dist / max_dist))
        area_sim = min(a1["area"], a2["area"]) / max(a1["area"], a2["area"]) if max(a1["area"], a2["area"]) > 0 else 0.0
        compactness_sim = 1.0 - abs(a1["compactness"] - a2["compactness"])
        eccentricity_sim = 1.0 - abs(a1["eccentricity"] - a2["eccentricity"])
        shape_sim = (compactness_sim + eccentricity_sim) / 2.0
        iou_sim = self._compute_iou(a1["geometry"], a2["geometry"])
        total = (
            weights.get("spatial", 0.0) * spatial_sim
            + weights.get("area", 0.0) * area_sim
            + weights.get("shape", 0.0) * shape_sim
            + weights.get("iou", 0.0) * iou_sim
        )
        return total, {
            "spatial": float(spatial_sim),
            "area": float(area_sim),
            "shape": float(shape_sim),
            "iou": float(iou_sim),
            "total": float(total),
        }

    def compute_weighted_similarity(self, attrs1, attrs2, weights=None):
        return self._weighted_similarity(attrs1, attrs2, weights=weights)

    def _compute_pair_metrics(self, prev_attrs: Dict[str, Any], curr_attrs: Dict[str, Any], max_dist: float) -> Dict[str, float]:
        prev_geom = prev_attrs["geometry"]
        curr_geom = curr_attrs["geometry"]
        try:
            intersection_area = prev_geom.intersection(curr_geom).area
        except Exception:
            intersection_area = 0.0
        try:
            union_area = prev_geom.union(curr_geom).area
        except Exception:
            union_area = prev_attrs["area"] + curr_attrs["area"] - intersection_area

        prev_area = prev_attrs["area"] if prev_attrs["area"] > 0 else 1e-6
        curr_area = curr_attrs["area"] if curr_attrs["area"] > 0 else 1e-6
        overlap_prev = intersection_area / prev_area
        overlap_curr = intersection_area / curr_area
        iou = intersection_area / union_area if union_area > 0 else 0.0
        centroid_dist = prev_attrs["centroid"].distance(curr_attrs["centroid"])
        base_similarity, parts = self._weighted_similarity(prev_attrs, curr_attrs, max_dist=max_dist)

        prev_radius = np.sqrt(prev_area / np.pi)
        curr_radius = np.sqrt(curr_area / np.pi)
        mean_radius = max((prev_radius + curr_radius) / 2.0, 1e-3)
        area_ratio = curr_area / prev_area if prev_area > 0 else np.inf
        balanced_area_ratio = 0.0 if (not np.isfinite(area_ratio) or area_ratio <= 0) else (area_ratio if area_ratio <= 1 else 1 / area_ratio)

        try:
            prev_contains_curr = prev_geom.buffer(0).contains(curr_geom)
        except Exception:
            prev_contains_curr = False
        try:
            curr_contains_prev = curr_geom.buffer(0).contains(prev_geom)
        except Exception:
            curr_contains_prev = False

        return {
            "intersection_area": float(intersection_area),
            "union_area": float(union_area),
            "overlap_prev": float(overlap_prev),
            "overlap_curr": float(overlap_curr),
            "iou": float(iou),
            "centroid_dist": float(centroid_dist),
            "base_similarity": float(base_similarity),
            "spatial_similarity": float(parts["spatial"]),
            "area_similarity": float(parts["area"]),
            "shape_similarity": float(parts["shape"]),
            "mean_radius": float(mean_radius),
            "area_ratio": float(area_ratio if np.isfinite(area_ratio) else 0.0),
            "balanced_area_ratio": float(balanced_area_ratio),
            "prev_contains_curr": bool(prev_contains_curr),
            "curr_contains_prev": bool(curr_contains_prev),
        }

    def _classify_match_case(
        self,
        prev_node: Tuple[int, int],
        curr_node: Tuple[int, int],
        features: Dict[str, float],
        prev_overlap_counts: Dict[Tuple[int, int], int],
        curr_overlap_counts: Dict[Tuple[int, int], int],
        overlap_gate: float,
        mode: str = "balanced",
    ) -> str:
        if features["prev_contains_curr"] or features["curr_contains_prev"]:
            return "containment"
        overlap_prev = features["overlap_prev"]
        overlap_curr = features["overlap_curr"]
        iou = features["iou"]
        centroid_dist = features["centroid_dist"]
        prev_count = prev_overlap_counts.get(prev_node, 0)
        curr_count = curr_overlap_counts.get(curr_node, 0)

        if mode == "strict":
            min_overlap = max(overlap_gate, 0.30)
            min_iou = max(overlap_gate / 2, 0.10)
            if prev_count == 1 and curr_count == 1 and overlap_prev >= min_overlap and overlap_curr >= min_overlap and iou >= min_iou:
                return "one_to_one"
            if centroid_dist < 30.0 and (overlap_prev >= overlap_gate or overlap_curr >= overlap_gate):
                return "nearby"
        elif mode == "lite":
            min_overlap = max(overlap_gate, 0.10)
            min_iou = max(overlap_gate * 0.5, 0.04)
            if overlap_prev >= min_overlap and overlap_curr >= min_overlap and iou >= min_iou and centroid_dist < 50.0:
                return "one_to_one"
            near_gate = max(overlap_gate * 0.5, 0.01)
            if centroid_dist < 50.0 and (overlap_prev >= near_gate or overlap_curr >= near_gate):
                return "nearby"
        else:
            min_overlap = max(overlap_gate, 0.15)
            min_iou = max(overlap_gate * 0.5, 0.05)
            if overlap_prev >= min_overlap and overlap_curr >= min_overlap and iou >= min_iou and centroid_dist < 40.0:
                return "one_to_one"
            near_gate = max(overlap_gate * 0.5, 0.02)
            if centroid_dist < 35.0 and (overlap_prev >= near_gate or overlap_curr >= near_gate):
                return "nearby"
        return "none"

    def _score_candidate(self, base_similarity: float, similarity_parts: Dict[str, float], features: Dict[str, float], config: MatchCaseConfig) -> float:
        centroid_factor = 1.0 - min(1.0, features["centroid_dist"] / (features["mean_radius"] * 3.0))
        components = {
            "base": base_similarity,
            "spatial": similarity_parts.get("spatial", 0.0),
            "area": similarity_parts.get("area", 0.0),
            "shape": similarity_parts.get("shape", 0.0),
            "iou": features["iou"],
            "overlap_prev": features["overlap_prev"],
            "overlap_curr": features["overlap_curr"],
            "centroid": max(0.0, centroid_factor),
            "area_balance": features.get("balanced_area_ratio", 0.0),
        }
        score = 0.0
        for key, weight in config.scoring_weights.items():
            score += weight * components.get(key, 0.0)
        return score

    def _select_candidates_by_case(
        self,
        candidates: List[Dict[str, Any]],
        configs: Dict[str, MatchCaseConfig],
        case_order: List[str],
        max_dist: float,
        min_base_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        used_prev: Dict[Tuple[int, int], int] = defaultdict(int)
        used_curr: Dict[Tuple[int, int], int] = defaultdict(int)

        for case_name in case_order:
            config = configs.get(case_name)
            if not config:
                continue
            case_candidates = [c for c in candidates if c["case"] == case_name]
            if not case_candidates:
                continue

            enriched: List[Dict[str, Any]] = []
            for cand in case_candidates:
                features = cand["features"]
                if config.max_centroid_dist is not None and features["centroid_dist"] > config.max_centroid_dist:
                    continue
                if features["iou"] < config.min_iou or features["overlap_prev"] < config.min_overlap_prev or features["overlap_curr"] < config.min_overlap_curr:
                    continue
                base_similarity, parts = self._weighted_similarity(
                    cand["prev_attrs"],
                    cand["curr_attrs"],
                    weights=config.base_similarity_weights,
                    max_dist=max_dist,
                )
                if min_base_similarity and base_similarity < min_base_similarity:
                    continue
                score = self._score_candidate(base_similarity, parts, features, config)
                if score < config.similarity_threshold:
                    continue
                cand["base_similarity"] = float(base_similarity)
                cand["similarity_parts"] = {k: float(v) for k, v in parts.items()}
                cand["score"] = float(score)
                enriched.append(cand)

            if not enriched:
                continue
            enriched.sort(key=lambda c: c["score"], reverse=True)
            for cand in enriched:
                prev_node = cand["prev_node"]
                curr_node = cand["curr_node"]
                if not config.allow_multiple and (used_prev.get(prev_node, 0) or used_curr.get(curr_node, 0)):
                    continue
                if config.max_edges_per_prev is not None and used_prev[prev_node] >= config.max_edges_per_prev:
                    continue
                if config.max_edges_per_curr is not None and used_curr[curr_node] >= config.max_edges_per_curr:
                    continue
                selected.append(cand)
                used_prev[prev_node] += 1
                used_curr[curr_node] += 1
        return selected

    def _ultra_relaxed_case_configs(self) -> Dict[str, MatchCaseConfig]:
        return {
            "one_to_one": MatchCaseConfig(
                name="one_to_one",
                base_similarity_weights={"spatial": 0.50, "area": 0.15, "shape": 0.05, "iou": 0.30},
                scoring_weights={"base": 0.60, "iou": 0.10, "overlap_prev": 0.05, "overlap_curr": 0.05, "centroid": 0.20},
                similarity_threshold=0.25,
                min_iou=0.01,
                min_overlap_prev=0.10,
                min_overlap_curr=0.10,
                max_centroid_dist=50.0,
                allow_multiple=True,
                max_edges_per_prev=3,
                max_edges_per_curr=3,
            ),
            "containment": MatchCaseConfig(
                name="containment",
                base_similarity_weights={"spatial": 0.50, "area": 0.20, "shape": 0.10, "iou": 0.20},
                scoring_weights={"base": 0.40, "overlap_prev": 0.30, "overlap_curr": 0.30},
                similarity_threshold=0.25,
                min_overlap_prev=0.25,
                min_overlap_curr=0.25,
                max_centroid_dist=150.0,
                allow_multiple=True,
                max_edges_per_prev=2,
                max_edges_per_curr=2,
            ),
            "nearby": MatchCaseConfig(
                name="nearby",
                base_similarity_weights={"spatial": 0.85, "area": 0.10, "shape": 0.03, "iou": 0.02},
                scoring_weights={"base": 0.70, "centroid": 0.30},
                similarity_threshold=0.20,
                min_overlap_prev=0.05,
                min_overlap_curr=0.05,
                max_centroid_dist=200.0,
                allow_multiple=True,
                max_edges_per_prev=2,
                max_edges_per_curr=2,
            ),
        }

    def make_strict_aligned_configs(self) -> Dict[str, MatchCaseConfig]:
        return {
            "one_to_one": MatchCaseConfig(
                name="one_to_one",
                base_similarity_weights={"iou": 0.45, "spatial": 0.35, "area": 0.15, "shape": 0.05},
                scoring_weights={"base": 0.40, "iou": 0.30, "overlap_prev": 0.10, "overlap_curr": 0.10, "centroid": 0.10},
                similarity_threshold=0.40,
                min_iou=0.30,
                min_overlap_prev=0.20,
                min_overlap_curr=0.20,
                max_centroid_dist=10.0,
                allow_multiple=True,
                max_edges_per_prev=2,
                max_edges_per_curr=2,
            ),
            "containment": MatchCaseConfig(
                name="containment",
                base_similarity_weights={"iou": 0.40, "spatial": 0.35, "area": 0.20, "shape": 0.05},
                scoring_weights={"base": 0.30, "overlap_prev": 0.35, "overlap_curr": 0.35},
                similarity_threshold=0.35,
                min_iou=0.30,
                min_overlap_prev=0.30,
                min_overlap_curr=0.30,
                max_centroid_dist=15.0,
                allow_multiple=True,
                max_edges_per_prev=2,
                max_edges_per_curr=2,
            ),
            "nearby": MatchCaseConfig(
                name="nearby",
                base_similarity_weights={"iou": 0.35, "spatial": 0.45, "area": 0.15, "shape": 0.05},
                scoring_weights={"base": 0.60, "centroid": 0.40},
                similarity_threshold=0.30,
                min_iou=0.10,
                min_overlap_prev=0.10,
                min_overlap_curr=0.10,
                max_centroid_dist=20.0,
                allow_multiple=True,
                max_edges_per_prev=2,
                max_edges_per_curr=2,
            ),
        }

    def _check_crs_consistency(self) -> None:
        if not self.crowns_gdfs:
            return

        base_crs = None
        for om_id in self.om_ids:
            gdf = self.crowns_gdfs.get(om_id)
            if gdf is not None and gdf.crs is not None:
                base_crs = gdf.crs
                break
        if base_crs is None:
            return

        for om_id in self.om_ids:
            gdf = self.crowns_gdfs.get(om_id)
            if gdf is None or gdf.empty:
                continue
            if gdf.crs is None:
                self.crowns_gdfs[om_id] = gdf.set_crs(base_crs, allow_override=True)
                self.crown_crs[om_id] = base_crs
                continue
            if gdf.crs != base_crs:
                self.crowns_gdfs[om_id] = gdf.to_crs(base_crs)
                self.crown_crs[om_id] = base_crs

    def load_data(
        self,
        load_images: bool = False,
        align: bool = False,
        reference_om_id: Optional[int] = None,
        align_method: str = "pcc",
        align_threshold_tag: Optional[str] = "conf_0p65",
    ) -> None:
        self.crowns_gdfs.clear()
        self.crown_attrs.clear()
        self.crown_images.clear()
        self.crown_crs.clear()
        self.ortho_crs.clear()

        for om_id, (crown_file, ortho_file) in zip(self.om_ids, self.file_pairs):
            gdf = gpd.read_file(crown_file)
            gdf = gdf.reset_index(drop=True)
            self.crowns_gdfs[om_id] = gdf
            self.crown_crs[om_id] = gdf.crs

            if load_images and ortho_file and os.path.exists(ortho_file):
                with rasterio.open(ortho_file) as src:
                    self.ortho_crs[om_id] = src.crs
                    patches: List[Optional[np.ndarray]] = []
                    for _, row in gdf.iterrows():
                        try:
                            out_image, _ = mask(src, [mapping(row.geometry)], crop=True)
                            img_patch = np.moveaxis(out_image, 0, -1)
                        except Exception:
                            img_patch = None
                        patches.append(img_patch)
                self.crown_images[om_id] = patches
            else:
                self.crown_images[om_id] = [None] * len(gdf)
                if ortho_file and os.path.exists(ortho_file):
                    with rasterio.open(ortho_file) as src:
                        self.ortho_crs[om_id] = src.crs
                else:
                    self.ortho_crs[om_id] = None

            self.crown_attrs[om_id] = [self._compute_crown_attributes(row.geometry) for _, row in gdf.iterrows()]

        self._check_crs_consistency()
        if align:
            self.align_to_reference_with_method(
                reference_om_id=reference_om_id,
                method=align_method,
                align_threshold_tag=align_threshold_tag,
            )

    @staticmethod
    def _threshold_tag_to_float(tag: str) -> float:
        try:
            return float(tag.replace("conf_", "").replace("p", "."))
        except Exception:
            return float("nan")

    @staticmethod
    def _float_to_threshold_tag(val: float) -> str:
        return f"conf_{val:.2f}".replace(".", "p")

    def _list_threshold_layers(self, gpkg_path: str) -> List[str]:
        meta_path = os.path.splitext(gpkg_path)[0] + ".meta.json"
        meta_layers: List[str] = []
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                for t in meta.get("thresholds", []):
                    if isinstance(t, str) and t.startswith("conf_"):
                        meta_layers.append(t)
                    elif isinstance(t, (int, float)):
                        meta_layers.append(self._float_to_threshold_tag(float(t)))
                meta_layers = list(dict.fromkeys(meta_layers))
            except Exception:
                meta_layers = []
        actual_layers: List[str] = []
        if FIONA_AVAILABLE:
            try:
                actual_layers = list(fiona.listlayers(gpkg_path))
            except Exception:
                actual_layers = []
        if actual_layers:
            layers = [l for l in meta_layers if l in actual_layers] if meta_layers else actual_layers
        else:
            layers = meta_layers
        if not layers:
            raise RuntimeError(f"No layers found in {gpkg_path}")
        return sorted(layers, key=self._threshold_tag_to_float)

    @staticmethod
    def _load_layer_gdf(gpkg_path: str, layer_tag: str) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(gpkg_path, layer=layer_tag).reset_index(drop=True)
        if "threshold_tag" not in gdf.columns:
            gdf["threshold_tag"] = layer_tag
        if "confidence" not in gdf.columns:
            gdf["confidence"] = np.nan
        if "orthomosaic" not in gdf.columns:
            gdf["orthomosaic"] = os.path.basename(gpkg_path)
        if "is_augmented" not in gdf.columns:
            gdf["is_augmented"] = False
        return gdf

    def load_multithreshold_data(
        self,
        base_threshold_tag: Optional[str] = None,
        load_images: bool = False,
        align: bool = True,
        include_base_for_non_reference: bool = True,
        align_method: str = "pcc",
        align_threshold_tag: Optional[str] = "conf_0p65",
    ) -> None:
        self.crowns_gdfs.clear()
        self.crown_attrs.clear()
        self.crown_images.clear()
        self.crown_crs.clear()
        self.ortho_crs.clear()
        self.multithreshold_layers = {}

        for om_id, (crown_file, ortho_file) in zip(self.om_ids, self.file_pairs):
            layers = self._list_threshold_layers(crown_file)
            self.multithreshold_layers[om_id] = layers
            if not layers:
                continue
            base_layer = base_threshold_tag if (base_threshold_tag and base_threshold_tag in layers) else max(layers, key=self._threshold_tag_to_float)
            if self.base_threshold_tag is None:
                self.base_threshold_tag = base_layer

            if om_id == min(self.om_ids):
                gdf = self._load_layer_gdf(crown_file, base_layer)
            else:
                chosen_layers = [l for l in layers if include_base_for_non_reference or l != base_layer]
                pieces = [self._load_layer_gdf(crown_file, lyr) for lyr in chosen_layers]
                gdf = pd.concat(pieces, ignore_index=True) if pieces else self._load_layer_gdf(crown_file, base_layer)
                gdf["geom_wkt"] = gdf.geometry.apply(lambda g: g.buffer(0).wkt)
                gdf = gdf.drop_duplicates(subset="geom_wkt").drop(columns="geom_wkt").reset_index(drop=True)

            self.crowns_gdfs[om_id] = gdf
            self.crown_crs[om_id] = gdf.crs

            if load_images and ortho_file and os.path.exists(ortho_file):
                with rasterio.open(ortho_file) as src:
                    self.ortho_crs[om_id] = src.crs
                    patches: List[Optional[np.ndarray]] = []
                    for _, row in gdf.iterrows():
                        try:
                            out_image, _ = mask(src, [mapping(row.geometry)], crop=True)
                            img_patch = np.moveaxis(out_image, 0, -1)
                        except Exception:
                            img_patch = None
                        patches.append(img_patch)
                self.crown_images[om_id] = patches
            else:
                self.crown_images[om_id] = [None] * len(gdf)
                self.ortho_crs[om_id] = None

            self.crown_attrs[om_id] = [self._compute_crown_attributes(row.geometry) for _, row in gdf.iterrows()]

        if align:
            self.align_to_reference_with_method(
                reference_om_id=min(self.om_ids) if self.om_ids else None,
                method=align_method,
                align_threshold_tag=align_threshold_tag,
            )

    def _align_gdf_to_tracker(self, gdf: gpd.GeoDataFrame, om_id: int) -> gpd.GeoDataFrame:
        params = self.alignment_transforms.get(om_id)
        if params is None:
            dx, dy = self.alignment_shifts.get(om_id, (0.0, 0.0))
            params = (1.0, 0.0, 0.0, 1.0, float(dx), float(dy))
        return self._apply_affine_to_gdf(gdf, params)

    def build_multithreshold_cache(
        self,
        include_base: bool = False,
        min_threshold_tag: Optional[str] = None,
    ) -> Tuple[Dict[int, Dict[str, gpd.GeoDataFrame]], List[str]]:
        cache: Dict[int, Dict[str, gpd.GeoDataFrame]] = {}
        all_thresholds = set()
        base_tag = self.base_threshold_tag
        base_val = self._threshold_tag_to_float(base_tag) if base_tag else None
        min_val = self._threshold_tag_to_float(min_threshold_tag) if min_threshold_tag else None

        for om_id, (crown_file, _) in zip(self.om_ids, self.file_pairs):
            layers = self.multithreshold_layers.get(om_id) or self._list_threshold_layers(crown_file)
            filtered = []
            for layer in layers:
                if not include_base and base_tag and layer == base_tag:
                    continue
                val = self._threshold_tag_to_float(layer)
                if min_val is not None and (np.isnan(val) or val < min_val):
                    continue
                if base_val is not None and not include_base and val > base_val:
                    continue
                filtered.append(layer)
            cache[om_id] = {}
            for layer in filtered:
                gdf = self._load_layer_gdf(crown_file, layer)
                cache[om_id][layer] = self._align_gdf_to_tracker(gdf, om_id)
                all_thresholds.add(layer)

        thresholds_desc = sorted(all_thresholds, key=self._threshold_tag_to_float, reverse=True)
        return cache, thresholds_desc

    def reset_graph(self) -> None:
        self.G = nx.DiGraph()

    def build_graph_conditional(
        self,
        case_configs: Optional[Dict[str, MatchCaseConfig]] = None,
        case_order: Optional[List[str]] = None,
        base_max_dist: float = 200.0,
        overlap_gate: float = 0.05,
        min_base_similarity: float = 0.0,
        max_candidates_per_prev: Optional[int] = None,
        max_candidates_per_curr: Optional[int] = None,
        classify_mode: Optional[str] = None,
    ) -> None:
        if not self.crowns_gdfs:
            self.load_data(load_images=False)

        self.reset_graph()
        configs = {name: replace(cfg) for name, cfg in (case_configs or self.case_configs).items()}
        order = case_order or self.case_order
        mode = classify_mode or self.match_case_mode
        self.last_case_counts = {}
        self.last_selected_counts = {name: 0 for name in configs.keys()}

        for idx in range(len(self.om_ids)):
            om_id = self.om_ids[idx]
            gdf = self.crowns_gdfs[om_id]
            for crown_id, _ in gdf.iterrows():
                attrs = self.crown_attrs[om_id][crown_id]
                self.G.add_node((om_id, crown_id), **attrs)
            if idx == 0:
                continue

            prev_om = self.om_ids[idx - 1]
            prev_nodes = [(prev_om, i) for i in range(len(self.crowns_gdfs[prev_om]))]
            curr_nodes = [(om_id, j) for j in range(len(gdf))]
            candidates: List[Dict[str, Any]] = []
            overlap_counts_prev: Dict[Tuple[int, int], int] = defaultdict(int)
            overlap_counts_curr: Dict[Tuple[int, int], int] = defaultdict(int)

            for prev_node in prev_nodes:
                prev_attrs = self.crown_attrs[prev_om][prev_node[1]]
                for curr_node in curr_nodes:
                    curr_attrs = self.crown_attrs[om_id][curr_node[1]]
                    features = self._compute_pair_metrics(prev_attrs, curr_attrs, max_dist=base_max_dist)
                    if features["centroid_dist"] > base_max_dist:
                        continue
                    cand = {
                        "prev_node": prev_node,
                        "curr_node": curr_node,
                        "prev_attrs": prev_attrs,
                        "curr_attrs": curr_attrs,
                        "features": features,
                    }
                    candidates.append(cand)
                    if features["overlap_prev"] >= overlap_gate:
                        overlap_counts_prev[prev_node] += 1
                    if features["overlap_curr"] >= overlap_gate:
                        overlap_counts_curr[curr_node] += 1

            for cand in candidates:
                cand["case"] = self._classify_match_case(
                    cand["prev_node"],
                    cand["curr_node"],
                    cand["features"],
                    overlap_counts_prev,
                    overlap_counts_curr,
                    overlap_gate,
                    mode=mode,
                )
            candidates = [cand for cand in candidates if cand["case"] != "none"]

            if max_candidates_per_prev is not None:
                grouped_prev: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
                for cand in candidates:
                    grouped_prev[cand["prev_node"]].append(cand)
                trimmed: List[Dict[str, Any]] = []
                for group in grouped_prev.values():
                    group.sort(key=lambda c: (-c["features"]["iou"], c["features"]["centroid_dist"]))
                    trimmed.extend(group[:max_candidates_per_prev])
                candidates = trimmed

            if max_candidates_per_curr is not None:
                grouped_curr: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
                for cand in candidates:
                    grouped_curr[cand["curr_node"]].append(cand)
                trimmed_curr: List[Dict[str, Any]] = []
                for group in grouped_curr.values():
                    group.sort(key=lambda c: (-c["features"]["iou"], c["features"]["centroid_dist"]))
                    trimmed_curr.extend(group[:max_candidates_per_curr])
                candidates = trimmed_curr

            case_counts = defaultdict(int)
            for cand in candidates:
                case_counts[cand["case"]] += 1
            for case_name, count in case_counts.items():
                self.last_case_counts[case_name] = self.last_case_counts.get(case_name, 0) + count

            selected = self._select_candidates_by_case(candidates, configs, order, base_max_dist, min_base_similarity=min_base_similarity)
            for cand in selected:
                case_name = cand["case"]
                features = cand["features"]
                similarity_parts = cand.get("similarity_parts", {})
                self.G.add_edge(
                    cand["prev_node"],
                    cand["curr_node"],
                    similarity=float(cand.get("score", features["base_similarity"])),
                    method="conditional",
                    case=case_name,
                    overlap_prev=float(features["overlap_prev"]),
                    overlap_curr=float(features["overlap_curr"]),
                    iou=float(features["iou"]),
                    centroid_distance=float(features["centroid_dist"]),
                    base_similarity=float(cand.get("base_similarity", features["base_similarity"])),
                    spatial_similarity=float(similarity_parts.get("spatial", features["spatial_similarity"])),
                    area_similarity=float(similarity_parts.get("area", features["area_similarity"])),
                    shape_similarity=float(similarity_parts.get("shape", features["shape_similarity"])),
                )
                self.last_selected_counts[case_name] = self.last_selected_counts.get(case_name, 0) + 1

    def _extract_all_chains(self) -> List[List[Tuple[int, int]]]:
        visited = set()
        chains: List[List[Tuple[int, int]]] = []
        chain_starts = [n for n in self.G.nodes if not list(self.G.predecessors(n))]
        for start_node in chain_starts:
            if start_node in visited:
                continue
            chain = self._greedy_chain(start_node)
            chains.append(chain)
            visited.update(chain)
        remaining = set(self.G.nodes) - visited
        for node in remaining:
            chains.append([node])
        return chains

    def _greedy_chain(self, start_node: Tuple[int, int]) -> List[Tuple[int, int]]:
        chain = [start_node]
        current = start_node
        while True:
            successors = list(self.G.successors(current))
            if not successors:
                break
            best_successor = max(successors, key=lambda n: self.G[current][n].get("similarity", 0.0))
            chain.append(best_successor)
            current = best_successor
        return chain

    def get_matching_chain(self, start_om_id: Any, crown_id: Optional[int] = None) -> List[Tuple[int, int]]:
        node = start_om_id if isinstance(start_om_id, tuple) else (int(start_om_id), int(crown_id))
        if node not in self.G:
            raise ValueError(f"Node {node} not in graph")
        return self._greedy_chain(node)

    def categorize_chains(self) -> Dict[str, List[List[Tuple[int, int]]]]:
        all_chains = self._extract_all_chains()
        max_chain_length = len(self.om_ids)
        full_chains_width_1 = []
        full_chains_branching = []
        partial_chains_long = []
        partial_chains_short = []

        for chain in all_chains:
            chain_length = len(chain)
            if chain_length == max_chain_length:
                has_branching = any(self.G.in_degree(node) > 1 or self.G.out_degree(node) > 1 for node in chain)
                if has_branching:
                    full_chains_branching.append(chain)
                else:
                    full_chains_width_1.append(chain)
            elif chain_length >= 3:
                partial_chains_long.append(chain)
            elif chain_length == 2:
                partial_chains_short.append(chain)

        return {
            "full_width_1": full_chains_width_1,
            "full_branching": full_chains_branching,
            "partial_long": partial_chains_long,
            "partial_short": partial_chains_short,
            "singleton": [c for c in all_chains if len(c) == 1],
        }

    def extract_backbone_mixed(self, chain_nodes: List[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
        edges_by_step = {}
        for i in range(len(chain_nodes) - 1):
            node1 = chain_nodes[i]
            node2 = chain_nodes[i + 1]
            step_edges = []
            if self.G.has_edge(node1, node2):
                edge_data = self.G.edges[node1, node2]
                step_edges.append(
                    {
                        "node1": node1,
                        "node2": node2,
                        "match_type": edge_data.get("case", "unknown"),
                        "similarity": edge_data.get("similarity", 0.0),
                        "iou": edge_data.get("iou", 0.0),
                    }
                )
            for successor in self.G.successors(node1):
                if successor != node2 and successor[0] == node2[0]:
                    edge_data = self.G.edges[node1, successor]
                    step_edges.append(
                        {
                            "node1": node1,
                            "node2": successor,
                            "match_type": edge_data.get("case", "unknown"),
                            "similarity": edge_data.get("similarity", 0.0),
                            "iou": edge_data.get("iou", 0.0),
                        }
                    )
            if not step_edges:
                return None
            edges_by_step[i] = step_edges

        quality_map = {"one_to_one": 3, "containment": 2, "nearby": 1}
        path = []
        for i in range(len(chain_nodes) - 1):
            best = max(edges_by_step[i], key=lambda e: (quality_map.get(e["match_type"], 0), e["similarity"]))
            path.append(best)
        return {
            "edges": path,
            "crowns": [chain_nodes[0]] + [e["node2"] for e in path],
            "avg_similarity": float(np.mean([e["similarity"] for e in path])) if path else 0.0,
            "edge_types": [e["match_type"] for e in path],
            "one_to_one_count": sum(1 for e in path if e["match_type"] == "one_to_one"),
        }

    def assemble_high_quality_chains(self) -> Dict[str, Any]:
        categories = self.categorize_chains()
        clean_chains = categories["full_width_1"]
        branching_chains = categories["full_branching"]
        extracted_backbones = []
        for chain in branching_chains:
            backbone = self.extract_backbone_mixed(chain)
            if backbone:
                extracted_backbones.append(backbone)
        all_extracted_chains = clean_chains + extracted_backbones
        return {
            "categories": categories,
            "clean_chains": clean_chains,
            "branching_chains": branching_chains,
            "extracted_backbones": extracted_backbones,
            "all_extracted_chains": all_extracted_chains,
        }

    def _chain_one_to_one_ratio(self, chain: List[Tuple[int, int]]) -> float:
        if len(chain) < 2:
            return 0.0
        total = len(chain) - 1
        one_to_one = 0
        for i in range(total):
            u, v = chain[i], chain[i + 1]
            if not self.G.has_edge(u, v):
                return 0.0
            if self.G.edges[u, v].get("case") == "one_to_one":
                one_to_one += 1
        return float(one_to_one / total) if total > 0 else 0.0

    def select_consensus_source_chains(
        self,
        hq_result: Dict[str, Any],
        include_partial: bool = True,
        min_partial_len: int = 5,
        min_partial_one_to_one_ratio: float = 0.9,
    ) -> List[List[Tuple[int, int]]]:
        """Build consensus input chains from high-quality chains plus optional filtered partial chains."""
        selected = list(hq_result.get("all_extracted_chains", []))
        if not include_partial:
            return selected

        categories = hq_result.get("categories", {})
        partial_candidates = categories.get("partial_long", []) + categories.get("partial_short", [])
        for chain in partial_candidates:
            if len(chain) < min_partial_len:
                continue
            if self._chain_one_to_one_ratio(chain) < min_partial_one_to_one_ratio:
                continue
            selected.append(chain)
        return selected

    def _append_crown_to_tracker(self, om_id: int, row: Dict[str, Any], threshold_tag: str) -> Tuple[int, int]:
        gdf = self.crowns_gdfs[om_id]
        new_row = dict(row)
        new_row["threshold_tag"] = threshold_tag
        new_row["is_augmented"] = True
        if "confidence" not in new_row:
            new_row["confidence"] = np.nan
        gdf_new = pd.concat([gdf, gpd.GeoDataFrame([new_row], crs=gdf.crs)], ignore_index=True)
        self.crowns_gdfs[om_id] = gdf_new
        new_idx = len(gdf_new) - 1
        attrs = self._compute_crown_attributes(new_row["geometry"])
        self.crown_attrs[om_id].append(attrs)
        self.G.add_node((om_id, new_idx), **attrs)
        return (om_id, new_idx)

    def _add_gap_edge(self, src: Tuple[int, int], dst: Tuple[int, int], max_dist: float) -> None:
        src_attrs = self.crown_attrs[src[0]][src[1]]
        dst_attrs = self.crown_attrs[dst[0]][dst[1]]
        features = self._compute_pair_metrics(src_attrs, dst_attrs, max_dist=max_dist)
        base_similarity, parts = self._weighted_similarity(src_attrs, dst_attrs, max_dist=max_dist)
        self.G.add_edge(
            src,
            dst,
            similarity=float(base_similarity),
            method="gap_fill",
            case="gap_fill",
            overlap_prev=float(features["overlap_prev"]),
            overlap_curr=float(features["overlap_curr"]),
            iou=float(features["iou"]),
            centroid_distance=float(features["centroid_dist"]),
            base_similarity=float(base_similarity),
            spatial_similarity=float(parts["spatial"]),
            area_similarity=float(parts["area"]),
            shape_similarity=float(parts["shape"]),
        )

    def _best_gap_candidate(
        self,
        om_id: int,
        prev_node: Optional[Tuple[int, int]],
        next_node: Optional[Tuple[int, int]],
        mt_cache: Dict[int, Dict[str, gpd.GeoDataFrame]],
        thresholds_desc: List[str],
        max_centroid_dist: float,
        min_iou: float,
        duplicate_iou: float,
    ):
        prev_geom = self.crown_attrs[prev_node[0]][prev_node[1]]["geometry"] if prev_node else None
        next_geom = self.crown_attrs[next_node[0]][next_node[1]]["geometry"] if next_node else None
        existing_geoms = list(self.crowns_gdfs[om_id].geometry)
        best = None
        best_score = -1e9
        best_tag = None

        for tag in thresholds_desc:
            gdf = mt_cache.get(om_id, {}).get(tag)
            if gdf is None or gdf.empty:
                continue
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                if any(self._compute_iou(geom, eg) >= duplicate_iou for eg in existing_geoms):
                    continue
                scores = []
                if prev_geom is not None:
                    dist_prev = geom.centroid.distance(prev_geom.centroid)
                    if dist_prev > max_centroid_dist:
                        continue
                    iou_prev = self._compute_iou(geom, prev_geom)
                    if iou_prev < min_iou:
                        continue
                    scores.append((iou_prev, dist_prev))
                if next_geom is not None:
                    dist_next = geom.centroid.distance(next_geom.centroid)
                    if dist_next > max_centroid_dist:
                        continue
                    iou_next = self._compute_iou(geom, next_geom)
                    if iou_next < min_iou:
                        continue
                    scores.append((iou_next, dist_next))
                if not scores:
                    continue
                score = sum(iou for iou, _ in scores) - (sum(dist for _, dist in scores) / max_centroid_dist)
                if score > best_score:
                    best_score = score
                    best = row
                    best_tag = tag
            if best is not None:
                break
        return best, best_tag

    def augment_partial_chains_with_multithreshold(
        self,
        base_chain_categories: Dict[str, List[List[Tuple[int, int]]]],
        mt_cache: Dict[int, Dict[str, gpd.GeoDataFrame]],
        thresholds_desc: List[str],
        max_centroid_dist: float = 25.0,
        min_iou: float = 0.20,
        duplicate_iou: float = 0.70,
    ) -> Dict[str, Any]:
        augmented = 0
        augmented_by_threshold = defaultdict(int)
        chains_to_extend = base_chain_categories.get("partial_long", []) + base_chain_categories.get("partial_short", [])
        if not self.om_ids:
            return {"augmented_nodes": 0, "by_threshold": {}}

        min_om = min(self.om_ids)
        max_om = max(self.om_ids)
        for chain in chains_to_extend:
            while True:
                last_node = chain[-1]
                last_om = last_node[0]
                if last_om >= max_om or self.G.out_degree(last_node) > 0:
                    break
                target_om = last_om + 1
                if target_om not in self.om_ids:
                    break
                cand, tag = self._best_gap_candidate(
                    target_om,
                    last_node,
                    None,
                    mt_cache,
                    thresholds_desc,
                    max_centroid_dist=max_centroid_dist,
                    min_iou=min_iou,
                    duplicate_iou=duplicate_iou,
                )
                if cand is None:
                    break
                new_node = self._append_crown_to_tracker(target_om, cand.to_dict(), tag)
                self._add_gap_edge(last_node, new_node, max_dist=max_centroid_dist)
                chain.append(new_node)
                augmented += 1
                augmented_by_threshold[tag] += 1

            while True:
                first_node = chain[0]
                first_om = first_node[0]
                if first_om <= min_om or self.G.in_degree(first_node) > 0:
                    break
                target_om = first_om - 1
                if target_om not in self.om_ids:
                    break
                cand, tag = self._best_gap_candidate(
                    target_om,
                    None,
                    first_node,
                    mt_cache,
                    thresholds_desc,
                    max_centroid_dist=max_centroid_dist,
                    min_iou=min_iou,
                    duplicate_iou=duplicate_iou,
                )
                if cand is None:
                    break
                new_node = self._append_crown_to_tracker(target_om, cand.to_dict(), tag)
                self._add_gap_edge(new_node, first_node, max_dist=max_centroid_dist)
                chain.insert(0, new_node)
                augmented += 1
                augmented_by_threshold[tag] += 1

        return {"augmented_nodes": augmented, "by_threshold": dict(augmented_by_threshold)}

    def quality_report(self, save_path: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        G = self.G
        metrics: Dict[str, Any] = {
            "total_trees_detected": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "total_possible_matches": 0,
            "successful_matches": 0,
            "match_rate_by_om_pair": {},
            "chain_length_distribution": {},
            "average_chain_length": 0,
            "median_chain_length": 0,
            "max_chain_length": 0,
        }

        chains = self._extract_all_chains()
        chain_lengths = [len(chain) for chain in chains]
        if chain_lengths:
            metrics["average_chain_length"] = float(np.mean(chain_lengths))
            metrics["median_chain_length"] = float(np.median(chain_lengths))
            metrics["max_chain_length"] = int(max(chain_lengths))
            for length in chain_lengths:
                metrics["chain_length_distribution"][int(length)] = metrics["chain_length_distribution"].get(int(length), 0) + 1

        for i in range(len(self.om_ids) - 1):
            om1, om2 = self.om_ids[i], self.om_ids[i + 1]
            om1_nodes = [n for n in G.nodes if n[0] == om1]
            om2_nodes = [n for n in G.nodes if n[0] == om2]
            matches = sum(1 for u, v in G.edges() if u[0] == om1 and v[0] == om2)
            possible_matches = min(len(om1_nodes), len(om2_nodes))
            rate = matches / possible_matches if possible_matches > 0 else 0.0
            metrics["match_rate_by_om_pair"][f"{om1}->{om2}"] = {"matches": matches, "possible": possible_matches, "rate": float(rate)}
            metrics["total_possible_matches"] += possible_matches
            metrics["successful_matches"] += matches

        metrics["overall_match_rate"] = metrics["successful_matches"] / metrics["total_possible_matches"] if metrics["total_possible_matches"] > 0 else 0.0

        report_lines = [
            "# Tree Tracking Quality Assessment Report",
            f"Total Trees Detected: {metrics['total_trees_detected']}",
            f"Total Tracking Edges: {metrics['total_edges']}",
            f"Overall Match Rate: {metrics['overall_match_rate']:.3f}",
            f"Average Chain Length: {metrics.get('average_chain_length', 0):.2f}",
            f"Maximum Chain Length: {metrics.get('max_chain_length', 0)}",
            "",
            "Match Rates by Orthomosaic Pair:",
        ]
        for pair, data in metrics["match_rate_by_om_pair"].items():
            report_lines.append(f"- {pair}: {data['matches']}/{data['possible']} ({data['rate']:.3f})")
        report_lines.append("")
        report_lines.append("Chain Length Distribution:")
        for length, count in sorted(metrics["chain_length_distribution"].items()):
            report_lines.append(f"- Length {length}: {count} trees")

        if self.last_selected_counts:
            report_lines.append("")
            report_lines.append("Edge selection by case:")
            for case_name, count in sorted(self.last_selected_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                total = self.last_case_counts.get(case_name, 0)
                if total:
                    report_lines.append(f"- {case_name}: {count}/{total} ({count/total:.2f})")
                else:
                    report_lines.append(f"- {case_name}: {count}")

        report = "\n".join(report_lines)
        if save_path:
            with open(save_path, "w") as f:
                f.write(report)
        return report, metrics

    def ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def save_text(self, text: str, filename: str) -> str:
        self.ensure_output_dir()
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(text)
        return path

    def save_json(self, data: Dict[str, Any], filename: str) -> str:
        self.ensure_output_dir()
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def _normalize_chain_data(self, chain_data: Any) -> Tuple[List[Tuple[int, int]], float, str]:
        if isinstance(chain_data, dict) and "edges" in chain_data:
            chain = [chain_data["edges"][0]["node1"]] + [e["node2"] for e in chain_data["edges"]]
            avg_sim = float(chain_data.get("avg_similarity", 0.0))
            oto_count = int(chain_data.get("one_to_one_count", 0))
            quality = "Pure" if oto_count == len(chain) - 1 else "Mixed"
        else:
            chain = chain_data
            avg_sim = 1.0
            quality = "Clean"
        return chain, avg_sim, quality

    def visualize_chain_with_extracted_images(
        self,
        chain: List[Tuple[int, int]],
        title: str = "Chain Example",
        save_path: Optional[str] = None,
        show: bool = True,
        close_fig: bool = True,
        dpi: int = 150,
    ) -> Optional[str]:
        chain_length = len(chain)
        fig = plt.figure(figsize=(5 * chain_length, 10))
        for idx, (om_id, crown_idx) in enumerate(chain):
            gdf = self.crowns_gdfs[om_id]
            crown = gdf.iloc[crown_idx]
            in_deg = self.G.in_degree((om_id, crown_idx)) if (om_id, crown_idx) in self.G else 0
            out_deg = self.G.out_degree((om_id, crown_idx)) if (om_id, crown_idx) in self.G else 0

            ax_poly = plt.subplot(2, chain_length, idx + 1)
            minx, miny, maxx, maxy = crown.geometry.bounds
            margin = max((maxx - minx), (maxy - miny)) * 0.3
            gdf.plot(ax=ax_poly, color="lightgray", edgecolor="gray", alpha=0.3)
            gpd.GeoSeries([crown.geometry]).plot(
                ax=ax_poly,
                facecolor=plt.cm.tab10((om_id - 1) % 10),
                edgecolor="black",
                linewidth=2,
                alpha=0.7,
            )
            centroid = crown.geometry.centroid
            ax_poly.plot(centroid.x, centroid.y, "o", color="yellow", markersize=8, markeredgecolor="black", markeredgewidth=1.5)
            ax_poly.set_xlim(minx - margin, maxx + margin)
            ax_poly.set_ylim(miny - margin, maxy + margin)
            ax_poly.set_aspect("equal")
            ax_poly.set_title(f"OM{om_id} - Crown {crown_idx}\nArea: {crown.geometry.area:.1f}m2\nIn:{in_deg} Out:{out_deg}", fontsize=10)
            ax_poly.grid(True, alpha=0.3)

            ax_img = plt.subplot(2, chain_length, chain_length + idx + 1)
            rgb_image = self.crown_images.get(om_id, [None] * (crown_idx + 1))[crown_idx]
            if rgb_image is not None:
                ax_img.imshow(np.clip(rgb_image[:, :, :3], 0, 255).astype(np.uint8))
                ax_img.set_title(f"Extracted Tree Image\n{rgb_image.shape[0]}x{rgb_image.shape[1]} px", fontsize=9)
            else:
                ax_img.text(0.5, 0.5, "Image not available", ha="center", va="center", fontsize=10, color="red")
                ax_img.set_title("No Image", fontsize=9)
            ax_img.axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        if close_fig:
            plt.close(fig)
        return save_path

    def visualize_all_chains(self, all_extracted_chains: List[Any], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for idx, chain_data in enumerate(all_extracted_chains, 1):
            chain, _, quality = self._normalize_chain_data(chain_data)
            title = f"Chain {idx}/{len(all_extracted_chains)} - {quality}"
            save_path = os.path.join(output_dir, f"chain_{idx:02d}.png")
            self.visualize_chain_with_extracted_images(
                chain=chain,
                title=title,
                save_path=save_path,
                show=False,
                close_fig=True,
                dpi=150,
            )

    def consensus_medoid(self, chain: List[Tuple[int, int]], w_centroid: float = 0.5, w_iou: float = 0.4, w_area: float = 0.1):
        polys = []
        for om_id, crown_idx in chain:
            poly = self.crowns_gdfs[om_id].iloc[crown_idx].geometry
            if poly is not None and not poly.is_empty:
                polys.append(poly.buffer(0))
        if not polys:
            return None
        centroids = [p.centroid for p in polys]
        areas = [p.area for p in polys]
        max_dist = max((c1.distance(c2) for c1 in centroids for c2 in centroids), default=1.0) or 1.0
        best_idx = 0
        best_score = float("inf")
        for i, p in enumerate(polys):
            score = 0.0
            for j, q in enumerate(polys):
                if i == j:
                    continue
                dist = centroids[i].distance(centroids[j]) / max_dist
                iou = self._compute_iou(p, q)
                area_ratio = min(areas[i], areas[j]) / max(areas[i], areas[j]) if max(areas[i], areas[j]) > 0 else 0.0
                score += (w_centroid * dist) + (w_iou * (1.0 - iou)) + (w_area * (1.0 - area_ratio))
            if score < best_score:
                best_score = score
                best_idx = i
        return polys[best_idx]

    def extract_patch_for_polygon(self, om_id: int, polygon_aligned):
        if polygon_aligned is None or polygon_aligned.is_empty:
            return None
        params = self.alignment_transforms.get(om_id)
        if params is None:
            dx, dy = self.alignment_shifts.get(om_id, (0.0, 0.0))
            params = (1.0, 0.0, 0.0, 1.0, float(dx), float(dy))
        inv_params = self._invert_affine_params(params)
        polygon_original = affine_transform(polygon_aligned, inv_params)
        ortho_file = None
        for oid, (_, of) in zip(self.om_ids, self.file_pairs):
            if oid == om_id:
                ortho_file = of
                break
        if not ortho_file or not os.path.exists(ortho_file):
            return None
        try:
            with rasterio.open(ortho_file) as src:
                out_image, _ = mask(src, [mapping(polygon_original)], crop=True)
                return np.moveaxis(out_image, 0, -1)
        except Exception:
            return None

    def generate_consensus_crowns(self, all_extracted_chains: List[Any]) -> gpd.GeoDataFrame:
        consensus_geoms = []
        chain_ids = []
        chain_lengths = []
        chain_qualities = []
        avg_similarities = []
        for idx, chain_data in enumerate(all_extracted_chains, 1):
            chain, avg_sim, quality = self._normalize_chain_data(chain_data)
            consensus_poly = self.consensus_medoid(chain)
            if consensus_poly is not None and not consensus_poly.is_empty:
                consensus_geoms.append(consensus_poly)
                chain_ids.append(idx)
                chain_lengths.append(len(chain))
                chain_qualities.append(quality)
                avg_similarities.append(avg_sim)
        base_crs = self.crowns_gdfs[self.om_ids[0]].crs if self.om_ids else None
        return gpd.GeoDataFrame(
            {
                "chain_id": chain_ids,
                "chain_length": chain_lengths,
                "quality": chain_qualities,
                "avg_similarity": avg_similarities,
                "geometry": consensus_geoms,
            },
            crs=base_crs,
        )

    def visualize_consensus_chain(
        self,
        chain: List[Tuple[int, int]],
        consensus_poly,
        chain_id: int,
        quality: str,
        avg_sim: float,
        save_path: Optional[str] = None,
    ) -> None:
        if consensus_poly is None:
            return
        chain_length = len(chain)
        fig = plt.figure(figsize=(5 * chain_length, 10))
        consensus_centroid = consensus_poly.centroid
        for idx, (om_id, crown_idx) in enumerate(chain):
            gdf = self.crowns_gdfs[om_id]
            crown = gdf.iloc[crown_idx]

            ax_poly = plt.subplot(2, chain_length, idx + 1)
            gdf.plot(ax=ax_poly, color="lightgray", edgecolor="gray", alpha=0.3, linewidth=0.5)
            gpd.GeoSeries([consensus_poly]).plot(ax=ax_poly, facecolor="red", edgecolor="darkred", linewidth=2.5, alpha=0.4)
            gpd.GeoSeries([crown.geometry]).plot(ax=ax_poly, facecolor="none", edgecolor="black", linewidth=2.0, alpha=0.9)
            ax_poly.plot(consensus_centroid.x, consensus_centroid.y, "o", color="yellow", markersize=10, markeredgecolor="black", markeredgewidth=2)
            iou = self._compute_iou(consensus_poly, crown.geometry)
            ax_poly.set_aspect("equal")
            ax_poly.set_title(f"OM{om_id}\nIoU={iou:.3f}", fontsize=11, fontweight="bold")
            ax_poly.grid(True, alpha=0.3)

            ax_img = plt.subplot(2, chain_length, chain_length + idx + 1)
            patch = self.extract_patch_for_polygon(om_id, consensus_poly)
            if patch is not None and patch.size > 0:
                patch_display = np.clip(patch[:, :, :3], 0, 255).astype(np.uint8)
                ax_img.imshow(patch_display)
                ax_img.set_title(f"Consensus Patch\n{patch_display.shape[0]}x{patch_display.shape[1]} px", fontsize=10)
            else:
                ax_img.text(0.5, 0.5, "Image not available", ha="center", va="center", fontsize=10, color="red")
                ax_img.set_title("No Image", fontsize=9)
            ax_img.axis("off")

        fig.suptitle(f"Consensus Crown Chain {chain_id} | Quality: {quality} | Avg Similarity: {avg_sim:.3f}", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_all_consensus_chains(self, all_extracted_chains: List[Any], output_dir_consensus: str) -> None:
        os.makedirs(output_dir_consensus, exist_ok=True)
        for idx, chain_data in enumerate(all_extracted_chains, 1):
            chain, avg_sim, quality = self._normalize_chain_data(chain_data)
            consensus_poly = self.consensus_medoid(chain)
            save_path = os.path.join(output_dir_consensus, f"consensus_chain_{idx:02d}.png")
            self.visualize_consensus_chain(chain, consensus_poly, chain_id=idx, quality=quality, avg_sim=avg_sim, save_path=save_path)

    def plot_matching_chain_plotly(self, chain, highlight_color: str = "orange", normal_color: str = "lightgray"):
        fig = go.Figure()
        for om_id in self.om_ids:
            gdf = self.crowns_gdfs.get(om_id)
            if gdf is None:
                continue
            for _, row in gdf.iterrows():
                if isinstance(row.geometry, Polygon):
                    simple_geom = row.geometry.simplify(self.simplify_tol)
                    x_geo, y_geo = simple_geom.exterior.xy
                    fig.add_trace(
                        go.Scatter(x=list(x_geo), y=list(y_geo), mode="lines", line=dict(color=normal_color, width=1), showlegend=False)
                    )
            chain_node = [node for node in chain if node[0] == om_id]
            if chain_node:
                crown_id = chain_node[0][1]
                if crown_id < len(gdf):
                    row = gdf.iloc[crown_id]
                    if isinstance(row.geometry, Polygon):
                        simple_geom = row.geometry.simplify(self.simplify_tol)
                        x_geo, y_geo = simple_geom.exterior.xy
                        fig.add_trace(
                            go.Scatter(
                                x=list(x_geo),
                                y=list(y_geo),
                                mode="lines",
                                line=dict(color=highlight_color, width=3),
                                name=f"OM{om_id} Crown {crown_id}",
                            )
                        )
        fig.update_layout(title="Matching Chain Highlighted Across OMs", height=600, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    def plot_orthomosaic_with_crowns_plotly(self, om_idx: int = 0):
        if om_idx >= len(self.file_pairs):
            return None
        om_id = self.om_ids[om_idx]
        _, ortho_path = self.file_pairs[om_idx]
        if not ortho_path or not os.path.exists(ortho_path):
            return None
        gdf = self.crowns_gdfs.get(om_id)
        if gdf is None:
            return None

        with rasterio.open(ortho_path) as src:
            ortho_img = src.read([1, 2, 3])
            ortho_img = np.moveaxis(ortho_img, 0, -1)
            if ortho_img.dtype != np.uint8:
                denom = float(np.ptp(ortho_img))
                if denom <= 0:
                    ortho_img = np.zeros_like(ortho_img, dtype=np.uint8)
                else:
                    ortho_img = ((ortho_img - ortho_img.min()) / denom * 255).astype(np.uint8)
            h, w = ortho_img.shape[:2]
            out_h = max(1, int(round(h * self.resize_factor)))
            out_w = max(1, int(round(w * self.resize_factor)))
            bands = src.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)
            ortho_img_small = np.moveaxis(bands, 0, -1)
            transform = src.transform

        fig = go.Figure()
        fig.add_trace(go.Image(z=ortho_img_small))
        for idx, row in gdf.head(self.max_crowns).iterrows():
            if isinstance(row.geometry, Polygon):
                simple_geom = row.geometry.simplify(self.simplify_tol)
                x_geo, y_geo = simple_geom.exterior.xy
                rows, cols = rasterio.transform.rowcol(transform, x_geo, y_geo)
                x_pix = (np.array(cols) * self.resize_factor).tolist()
                y_pix = (np.array(rows) * self.resize_factor).tolist()
                fig.add_trace(
                    go.Scatter(x=x_pix, y=y_pix, mode="lines", name=f"Crown {idx}", customdata=[idx], hoverinfo="name", line=dict(width=2))
                )
        fig.update_layout(title="Orthomosaic with Crowns", dragmode="select", height=800, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    # Backward compatibility: old endpoint name remains callable.
    def process_all_hungarian(self):
        if not self.crowns_gdfs:
            self.load_data(load_images=True, align=True)
        self.build_graph_conditional(
            base_max_dist=30.0,
            overlap_gate=0.10,
            min_base_similarity=0.30,
            classify_mode="balanced",
        )
