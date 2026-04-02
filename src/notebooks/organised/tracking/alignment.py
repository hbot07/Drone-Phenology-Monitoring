from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio

try:
    from skimage.registration import phase_cross_correlation
    from skimage.transform import resize

    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


@dataclass(frozen=True)
class AlignmentResult:
    shifts: Dict[int, Tuple[float, float]]  # cumulative (dx, dy) per om_id in world units
    method: str
    notes: Dict[str, str]


def read_ortho_lowres(path: Path, max_size: int = 1200):
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


def _bounds_from_transform(h: int, w: int, transform: rasterio.transform.Affine):
    corners = [(0, 0), (0, w), (h, 0), (h, w)]
    xs, ys = [], []
    for row, col in corners:
        x, y = rasterio.transform.xy(transform, row, col, offset="ul")
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def _safe_window(win: rasterio.windows.Window, height: int, width: int) -> rasterio.windows.Window:
    col_off = int(max(0, min(width, win.col_off)))
    row_off = int(max(0, min(height, win.row_off)))
    max_w = max(0, width - col_off)
    max_h = max(0, height - row_off)
    w = int(max(0, min(max_w, win.width)))
    h = int(max(0, min(max_h, win.height)))
    return rasterio.windows.Window(col_off=col_off, row_off=row_off, width=w, height=h)


def _crop_to_overlap(ref_img, mov_img, ref_transform, mov_transform):
    ref_h, ref_w = ref_img.shape[:2]
    mov_h, mov_w = mov_img.shape[:2]
    ref_left, ref_bottom, ref_right, ref_top = _bounds_from_transform(ref_h, ref_w, ref_transform)
    mov_left, mov_bottom, mov_right, mov_top = _bounds_from_transform(mov_h, mov_w, mov_transform)

    left = max(ref_left, mov_left)
    right = min(ref_right, mov_right)
    bottom = max(ref_bottom, mov_bottom)
    top = min(ref_top, mov_top)
    if left >= right or bottom >= top:
        return None, None

    ref_win = rasterio.windows.from_bounds(left, bottom, right, top, ref_transform).round_offsets().round_lengths()
    mov_win = rasterio.windows.from_bounds(left, bottom, right, top, mov_transform).round_offsets().round_lengths()
    ref_win = _safe_window(ref_win, ref_h, ref_w)
    mov_win = _safe_window(mov_win, mov_h, mov_w)
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


def _match_shape(ref_crop, mov_crop):
    if ref_crop.shape == mov_crop.shape:
        return ref_crop, mov_crop
    if not SKIMAGE_AVAILABLE:
        min_h = min(ref_crop.shape[0], mov_crop.shape[0])
        min_w = min(ref_crop.shape[1], mov_crop.shape[1])
        return ref_crop[:min_h, :min_w], mov_crop[:min_h, :min_w]
    mov_resized = resize(mov_crop, ref_crop.shape, preserve_range=True, anti_aliasing=True)
    return ref_crop, mov_resized


def phase_corr_shift(ref_gray, mov_gray, ref_transform, mov_transform) -> Optional[Tuple[float, float]]:
    if not SKIMAGE_AVAILABLE:
        return None
    ref_crop, mov_crop = _crop_to_overlap(ref_gray, mov_gray, ref_transform, mov_transform)
    if ref_crop is None or mov_crop is None:
        return None
    ref_crop, mov_crop = _match_shape(ref_crop, mov_crop)
    shift, _, _ = phase_cross_correlation(ref_crop, mov_crop, upsample_factor=10)
    shift_row, shift_col = shift
    dx = shift_col * ref_transform.a
    dy = shift_row * ref_transform.e
    return float(dx), float(dy)


def compute_cumulative_shifts(
    ortho_paths_by_id: Dict[int, Path],
    om_ids: list[int],
    reference_om_id: Optional[int] = None,
    max_preview: int = 1200,
) -> AlignmentResult:
    if not om_ids:
        return AlignmentResult(shifts={}, method="none", notes={"reason": "no_oms"})

    if not SKIMAGE_AVAILABLE:
        return AlignmentResult(
            shifts={om_id: (0.0, 0.0) for om_id in om_ids},
            method="none",
            notes={"reason": "skimage_not_available"},
        )

    if reference_om_id is None:
        reference_om_id = om_ids[0]

    ortho_gray = {}
    ortho_transform = {}
    for om_id in om_ids:
        p = ortho_paths_by_id.get(om_id)
        if p is None or not p.exists():
            ortho_gray[om_id] = None
            ortho_transform[om_id] = None
            continue
        gray, transform = read_ortho_lowres(p, max_preview)
        ortho_gray[om_id] = gray
        ortho_transform[om_id] = transform

    shifts: Dict[int, Tuple[float, float]] = {reference_om_id: (0.0, 0.0)}
    for idx in range(1, len(om_ids)):
        prev_id = om_ids[idx - 1]
        curr_id = om_ids[idx]
        ref_gray = ortho_gray.get(prev_id)
        mov_gray = ortho_gray.get(curr_id)
        ref_transform = ortho_transform.get(prev_id)
        mov_transform = ortho_transform.get(curr_id)

        if ref_gray is None or mov_gray is None or ref_transform is None or mov_transform is None:
            shifts[curr_id] = shifts.get(prev_id, (0.0, 0.0))
            continue

        shift = phase_corr_shift(ref_gray, mov_gray, ref_transform, mov_transform)
        if shift is None:
            shifts[curr_id] = shifts.get(prev_id, (0.0, 0.0))
            continue

        dx, dy = shift
        prev_dx, prev_dy = shifts.get(prev_id, (0.0, 0.0))
        shifts[curr_id] = (prev_dx + dx, prev_dy + dy)

    return AlignmentResult(shifts=shifts, method="phase_cross_correlation", notes={"skimage": "available"})
