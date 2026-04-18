from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.colors as mcolors

    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False


@dataclass(frozen=True)
class PatchQCConfig:
    min_valid_pixel_fraction: float = 0.60
    max_shadow_fraction: float = 0.55
    min_laplacian_var: float = 25.0
    min_valid_px: int = 20


@dataclass(frozen=True)
class VegMaskConfig:
    h_min: float = 0.18
    h_max: float = 0.48
    s_min: float = 0.15
    v_min: float = 0.12


@dataclass(frozen=True)
class LeafShedConfig:
    veg_min_threshold: float = 0.45
    ds_threshold: float = 0.85
    phenophase_on: float = 0.65
    phenophase_off: float = 0.35
    w_veg_amp: float = 0.35
    w_depth: float = 0.30
    w_gcc_amp: float = 0.25
    w_tex: float = 0.10
    a90_quantile: float = 0.90


@dataclass(frozen=True)
class NonTreeThresholds:
    # A crown is considered non-tree if it satisfies the selected rule set.
    # These thresholds are designed to remove a small number of obvious non-tree
    # artifacts that slipped into consensus crowns.
    gcc_mean_thresh: Optional[float] = None
    veg_mean_thresh: Optional[float] = None
    gcc_amp_thresh: Optional[float] = 0.05
    veg_amp_thresh: Optional[float] = None


def shannon_entropy(values_01: np.ndarray, bins: int = 64) -> float:
    if values_01.size == 0:
        return float("nan")
    hist, _ = np.histogram(values_01, bins=bins, range=(0.0, 1.0), density=False)
    total = int(hist.sum())
    if total <= 0:
        return float("nan")
    probs = hist.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return num / (den + eps)


def laplacian_variance(gray_01: np.ndarray) -> float:
    """Compute variance of a simple 4-neighbour Laplacian; no OpenCV dependency."""
    if gray_01.size == 0:
        return float("nan")
    # The QC threshold for Laplacian variance (e.g. 25) is typically defined on
    # 8-bit intensity scale. Our features use gray in 0..1, so rescale.
    g = (gray_01.astype(np.float32) * 255.0)
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        return float("nan")
    # Kernel:
    #  0  1  0
    #  1 -4  1
    #  0  1  0
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[1:-1, 0:-2]
        + g[1:-1, 2:]
        + g[0:-2, 1:-1]
        + g[2:, 1:-1]
    )
    return float(np.var(lap))


def compute_patch_features(
    patch: Optional[np.ndarray],
    *,
    veg_cfg: VegMaskConfig = VegMaskConfig(),
    qc_cfg: PatchQCConfig = PatchQCConfig(),
) -> Dict[str, float]:
    """Compute phenology-relevant features from an RGB crown chip.

    Expects patch as HxWxC (C>=3), values typically 0..255.
    """

    if patch is None or not isinstance(patch, np.ndarray) or patch.size == 0:
        return {}
    if patch.ndim != 3 or patch.shape[2] < 3:
        return {}

    rgb = patch[..., :3].astype(np.float32)

    finite_mask = np.isfinite(rgb).all(axis=2)
    nonzero_mask = (rgb.sum(axis=2) > 0)
    valid_mask = finite_mask & nonzero_mask

    total_px = int(rgb.shape[0] * rgb.shape[1])
    valid_px = int(valid_mask.sum())
    valid_pixel_fraction = valid_px / max(total_px, 1)

    if total_px == 0 or valid_px < qc_cfg.min_valid_px:
        return {
            "valid_pixel_fraction": float(valid_pixel_fraction),
            "is_bad_observation": True,
        }

    rgb01 = np.clip(rgb, 0.0, 255.0) / 255.0
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    denom = r + g + b

    gcc_map = _safe_div(g, denom)
    rcc_map = _safe_div(r, denom)

    if not MPL_AVAILABLE:
        raise RuntimeError(
            "matplotlib is required for HSV conversion (matplotlib.colors.rgb_to_hsv). "
            "Install matplotlib or adjust compute_patch_features to use an alternate HSV conversion."
        )
    hsv = mcolors.rgb_to_hsv(rgb01)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    dark_mask = v < float(veg_cfg.v_min)
    veg_mask = (
        valid_mask
        & (h >= float(veg_cfg.h_min))
        & (h <= float(veg_cfg.h_max))
        & (s >= float(veg_cfg.s_min))
        & (v >= float(veg_cfg.v_min))
    )

    shadow_fraction = float((dark_mask & valid_mask).sum() / max(valid_px, 1))
    veg_fraction_hsv = float(veg_mask.sum() / max(valid_px, 1))

    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    lap_var = laplacian_variance(gray)

    valid_gray = gray[valid_mask]
    valid_gcc = gcc_map[valid_mask]
    valid_rcc = rcc_map[valid_mask]

    gray_entropy = shannon_entropy(np.clip(valid_gray, 0.0, 1.0))

    features: Dict[str, float] = {
        "valid_pixel_fraction": float(valid_pixel_fraction),
        "shadow_fraction": float(shadow_fraction),
        "veg_fraction_hsv": float(veg_fraction_hsv),
        "gcc_mean": float(np.mean(valid_gcc)) if valid_gcc.size else float("nan"),
        "rcc_mean": float(np.mean(valid_rcc)) if valid_rcc.size else float("nan"),
        "gray_entropy": float(gray_entropy),
        "laplacian_var": float(lap_var),
    }

    features["is_bad_observation"] = bool(
        (features["valid_pixel_fraction"] < qc_cfg.min_valid_pixel_fraction)
        or (features["shadow_fraction"] > qc_cfg.max_shadow_fraction)
        or (features["laplacian_var"] < qc_cfg.min_laplacian_var)
    )

    return features


def interp_series(values: np.ndarray, om_ids: List[int]) -> np.ndarray:
    """Linear interpolation over NaNs; edge-fill. om_ids must be increasing."""
    xi = np.asarray(om_ids, dtype=float)
    yi = np.asarray(values, dtype=float)
    valid = np.isfinite(yi)
    if valid.sum() < 2:
        fillval = yi[valid][0] if valid.sum() == 1 else 0.0
        return np.where(valid, yi, float(fillval)).astype(float)

    # Use numpy.interp with edge fill.
    x_valid = xi[valid]
    y_valid = yi[valid]
    y_interp = np.interp(xi, x_valid, y_valid)
    return y_interp.astype(float)


def minmax_norm(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    mn = float(np.nanmin(s))
    mx = float(np.nanmax(s))
    rng = mx - mn
    if not np.isfinite(rng) or rng <= 1e-9:
        return np.zeros_like(s, dtype=float)
    return ((s - mn) / rng).astype(float)


def _q90(values: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.nanquantile(vals, q))


def compute_leafshed_scores(
    features_df: pd.DataFrame,
    *,
    om_ids: List[int],
    cfg: LeafShedConfig = LeafShedConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Compute DS (deciduousness score) and phenophase labels from per-OM features.

    Returns:
    - tree_scores_df: one row per chain_id with components and DS
    - phenophase_df: one row per chain_id × om_id with interpolated series + state
    - normalizers: dict of A90 normalizers used
    """

    required = {"chain_id", "om_id", "veg_fraction_hsv", "gcc_mean", "laplacian_var", "is_bad_observation"}
    missing = sorted(required - set(features_df.columns))
    if missing:
        raise ValueError(f"features_df missing required columns: {missing}")

    df = features_df.copy()
    df["chain_id"] = pd.to_numeric(df["chain_id"], errors="coerce").astype("Int64")
    df["om_id"] = pd.to_numeric(df["om_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["chain_id", "om_id"]).copy()
    df["chain_id"] = df["chain_id"].astype(int)
    df["om_id"] = df["om_id"].astype(int)

    # NaN-out bad observations before interpolation.
    for col in ["veg_fraction_hsv", "gcc_mean", "laplacian_var"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df["is_bad_observation"].astype(bool), col] = np.nan

    tree_ids = sorted(df["chain_id"].unique().tolist())
    om_ids_sorted = list(sorted(om_ids))

    veg_hat_by_tree: Dict[int, np.ndarray] = {}
    gcc_hat_by_tree: Dict[int, np.ndarray] = {}
    tex_hat_by_tree: Dict[int, np.ndarray] = {}

    A_veg = []
    A_gcc = []
    A_tex = []
    min_veg = []

    for tid in tree_ids:
        sub = df[df["chain_id"] == tid].set_index("om_id")
        veg = np.array([sub["veg_fraction_hsv"].get(oid, np.nan) for oid in om_ids_sorted], dtype=float)
        gcc = np.array([sub["gcc_mean"].get(oid, np.nan) for oid in om_ids_sorted], dtype=float)
        tex = np.array([sub["laplacian_var"].get(oid, np.nan) for oid in om_ids_sorted], dtype=float)

        veg_hat = interp_series(veg, om_ids_sorted)
        gcc_hat = interp_series(gcc, om_ids_sorted)
        tex_hat = interp_series(tex, om_ids_sorted)

        veg_hat_by_tree[tid] = veg_hat
        gcc_hat_by_tree[tid] = gcc_hat
        tex_hat_by_tree[tid] = tex_hat

        A_veg.append(float(np.nanmax(veg_hat) - np.nanmin(veg_hat)))
        A_gcc.append(float(np.nanmax(gcc_hat) - np.nanmin(gcc_hat)))
        A_tex.append(float(np.nanmax(tex_hat) - np.nanmin(tex_hat)))
        min_veg.append(float(np.nanmin(veg_hat)))

    A_veg_arr = np.asarray(A_veg, dtype=float)
    A_gcc_arr = np.asarray(A_gcc, dtype=float)
    A_tex_arr = np.asarray(A_tex, dtype=float)
    min_veg_arr = np.asarray(min_veg, dtype=float)

    A90_veg = _q90(A_veg_arr, cfg.a90_quantile)
    A90_gcc = _q90(A_gcc_arr, cfg.a90_quantile)
    A90_tex = _q90(A_tex_arr, cfg.a90_quantile)

    def _amp_score(A: float, A90: float) -> float:
        if not np.isfinite(A) or not np.isfinite(A90) or A90 <= 1e-12:
            return float("nan")
        return float(min(1.0, max(0.0, A / A90)))

    s_veg_amp = np.array([_amp_score(a, A90_veg) for a in A_veg_arr], dtype=float)
    s_gcc_amp = np.array([_amp_score(a, A90_gcc) for a in A_gcc_arr], dtype=float)
    s_tex = np.array([_amp_score(a, A90_tex) for a in A_tex_arr], dtype=float)

    tau_veg = float(cfg.veg_min_threshold)
    s_depth = np.where(
        np.isfinite(min_veg_arr) & (min_veg_arr < tau_veg),
        (tau_veg - min_veg_arr) / max(tau_veg, 1e-12),
        0.0,
    ).astype(float)
    s_depth = np.clip(s_depth, 0.0, 1.0)

    ds = (
        float(cfg.w_veg_amp) * s_veg_amp
        + float(cfg.w_depth) * s_depth
        + float(cfg.w_gcc_amp) * s_gcc_amp
        + float(cfg.w_tex) * s_tex
    ).astype(float)
    is_deciduous = ds >= float(cfg.ds_threshold)

    tree_scores_df = pd.DataFrame(
        {
            "chain_id": tree_ids,
            "A_veg": A_veg_arr,
            "A_gcc": A_gcc_arr,
            "A_tex": A_tex_arr,
            "min_veg": min_veg_arr,
            "s_veg_amp": s_veg_amp,
            "s_depth": s_depth,
            "s_gcc_amp": s_gcc_amp,
            "s_tex": s_tex,
            "deciduous_score": ds,
            "is_deciduous": is_deciduous.astype(bool),
        }
    ).sort_values("deciduous_score", ascending=False)

    # Phenophase per OM.
    rows = []
    for tid in tree_ids:
        veg_hat = veg_hat_by_tree[tid]
        veg_norm = minmax_norm(veg_hat)
        decid = bool(tree_scores_df.loc[tree_scores_df["chain_id"] == tid, "is_deciduous"].iloc[0])
        trough_idx = int(np.nanargmin(veg_hat)) if np.isfinite(veg_hat).any() else 0
        trough_om = int(om_ids_sorted[trough_idx])

        # phenophase label
        for i, oid in enumerate(om_ids_sorted):
            if not decid:
                state = "stable"
            else:
                if veg_norm[i] >= float(cfg.phenophase_on):
                    state = "leaf_on"
                elif veg_norm[i] <= float(cfg.phenophase_off):
                    state = "leaf_off"
                else:
                    state = "transitioning"
            rows.append(
                {
                    "chain_id": int(tid),
                    "om_id": int(oid),
                    "veg_fraction_hsv_hat": float(veg_hat[i]),
                    "veg_fraction_hsv_norm": float(veg_norm[i]),
                    "phenophase": state,
                    "trough_om": trough_om,
                }
            )

    phenophase_df = pd.DataFrame(rows)

    # Event timing (deciduous trees only): add leaf_off_start_om and leaf_on_return_om.
    event_rows = []
    for tid in tree_ids:
        decid = bool(tree_scores_df.loc[tree_scores_df["chain_id"] == tid, "is_deciduous"].iloc[0])
        if not decid:
            event_rows.append({"chain_id": int(tid), "leaf_off_start_om": np.nan, "full_leaf_off_om": np.nan, "leaf_on_return_om": np.nan})
            continue

        sub = phenophase_df[phenophase_df["chain_id"] == tid].sort_values("om_id")
        trough_om = int(sub["trough_om"].iloc[0])

        before = sub[sub["om_id"] < trough_om]
        after = sub[sub["om_id"] >= trough_om]
        # leaf_off_start_om: OM right after last leaf_on before trough.
        leaf_on_before = before[before["phenophase"] == "leaf_on"]
        if len(leaf_on_before) == 0:
            leaf_off_start = np.nan
        else:
            last_leaf_on_om = int(leaf_on_before["om_id"].max())
            cand = sub[sub["om_id"] > last_leaf_on_om]
            leaf_off_start = float(cand["om_id"].min()) if len(cand) else np.nan

        # leaf_on_return_om: first leaf_on at/after trough.
        leaf_on_after = after[after["phenophase"] == "leaf_on"]
        leaf_on_return = float(leaf_on_after["om_id"].min()) if len(leaf_on_after) else np.nan

        event_rows.append(
            {
                "chain_id": int(tid),
                "leaf_off_start_om": leaf_off_start,
                "full_leaf_off_om": float(trough_om),
                "leaf_on_return_om": leaf_on_return,
            }
        )

    events_df = pd.DataFrame(event_rows)
    tree_scores_df = tree_scores_df.merge(events_df, on="chain_id", how="left")

    normalizers = {
        "A90_veg": float(A90_veg),
        "A90_gcc": float(A90_gcc),
        "A90_tex": float(A90_tex),
        "a90_quantile": float(cfg.a90_quantile),
    }
    return tree_scores_df, phenophase_df, normalizers


def apply_non_tree_thresholds(
    tree_scores_df: pd.DataFrame,
    *,
    thresholds: NonTreeThresholds,
) -> pd.DataFrame:
    out = tree_scores_df.copy()
    out["is_non_tree_gcc_only"] = False
    out["is_non_tree_gcc_veg"] = False
    out["is_non_tree"] = False
    out["is_tree"] = True

    gcc_amp_col = "gcc_amp_for_nontree" if "gcc_amp_for_nontree" in out.columns else "A_gcc"
    veg_amp_col = "veg_amp_for_nontree" if "veg_amp_for_nontree" in out.columns else "A_veg"
    gcc_mean_col = "gcc_mean_for_nontree" if "gcc_mean_for_nontree" in out.columns else None
    veg_mean_col = "veg_mean_for_nontree" if "veg_mean_for_nontree" in out.columns else None

    if thresholds.gcc_amp_thresh is not None and gcc_amp_col in out.columns:
        out["is_non_tree_gcc_only"] = pd.to_numeric(out[gcc_amp_col], errors="coerce") <= float(thresholds.gcc_amp_thresh)

    if thresholds.gcc_amp_thresh is not None and thresholds.veg_amp_thresh is not None:
        out["is_non_tree_gcc_veg"] = (
            (pd.to_numeric(out.get(gcc_amp_col), errors="coerce") <= float(thresholds.gcc_amp_thresh))
            & (pd.to_numeric(out.get(veg_amp_col), errors="coerce") <= float(thresholds.veg_amp_thresh))
        )

    # One decision column for filtering.
    # If mean+amp thresholds for both GCC and VEG are provided, require ALL 4 conditions.
    if (
        thresholds.gcc_mean_thresh is not None
        and thresholds.veg_mean_thresh is not None
        and thresholds.gcc_amp_thresh is not None
        and thresholds.veg_amp_thresh is not None
        and gcc_mean_col is not None
        and veg_mean_col is not None
    ):
        out["is_non_tree"] = (
            (pd.to_numeric(out[gcc_mean_col], errors="coerce") <= float(thresholds.gcc_mean_thresh))
            & (pd.to_numeric(out[veg_mean_col], errors="coerce") <= float(thresholds.veg_mean_thresh))
            & (pd.to_numeric(out[gcc_amp_col], errors="coerce") <= float(thresholds.gcc_amp_thresh))
            & (pd.to_numeric(out[veg_amp_col], errors="coerce") <= float(thresholds.veg_amp_thresh))
        ).fillna(False).astype(bool)
    else:
        # Backward-compatible: if veg_amp_thresh is provided, require both low GCC
        # amplitude and low veg amplitude; otherwise use GCC-only.
        if thresholds.gcc_amp_thresh is None:
            out["is_non_tree"] = False
        elif thresholds.veg_amp_thresh is not None:
            out["is_non_tree"] = out["is_non_tree_gcc_veg"].fillna(False).astype(bool)
        else:
            out["is_non_tree"] = out["is_non_tree_gcc_only"].fillna(False).astype(bool)

    out["is_tree"] = (~out["is_non_tree"]).astype(bool)

    return out


def save_normalizers_json(path: Path, normalizers: Dict[str, float]) -> None:
    import json

    path.write_text(json.dumps(normalizers, indent=2, sort_keys=True))
