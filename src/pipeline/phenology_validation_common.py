#!/usr/bin/env python3
"""
Shared helpers for the phenology *validation* tooling
(05_phenology_evidence.py and 06_phenology_sensitivity.py).

Nothing here changes the pipeline. These scripts read the artifacts that
Step 3 already produced (pipeline_config.json + phenology_features_raw.csv,
plus the consensus crowns + OMs when crops are needed) and re-run the exact
same scoring function `compute_leafshed_scores` under different configs.

Design notes
------------
* This module does NOT import `tree_tracking` at import time. Only the
  `build_aligned_tracker()` function needs it, so the sensitivity script
  (which never touches images) can import this module with just pandas/numpy.
* `compute_leafshed_scores` and the config dataclasses come from
  `phenology_leafshed`, which must be importable (same as Step 3: it lives in
  <project_root>/src/flask_app_tracking).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Phenophase colour scheme (used by both the evidence panels and any plots).
# Kept deliberately colour-blind-friendly and consistent with the viewer.
# ---------------------------------------------------------------------------
PHENOPHASE_COLORS: Dict[str, str] = {
    "leaf_on": "#1b7837",        # green
    "leaf_off": "#8c510a",       # brown
    "transitioning": "#f1a340",  # orange
    "stable": "#4575b4",         # blue (evergreen)
    "bad": "#bdbdbd",            # grey
    "missing": "#e0e0e0",        # light grey
}


# ---------------------------------------------------------------------------
# Config / path helpers (mirrors 03_phenology_analysis.py so behaviour matches)
# ---------------------------------------------------------------------------
def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def setup_app_dir(project_root: Path) -> None:
    app_dir = str(project_root / "src" / "flask_app_tracking")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)


def build_pairs_and_om_stems(config: dict) -> Tuple[List[Tuple[str, str, str]], Dict[int, str]]:
    """Identical logic to Step 3, so OM ids line up 1..N with the same stems."""
    crowns_dir = Path(config["crowns_dir"])
    om_dir = Path(config["om_dir"])
    pairs: List[Tuple[str, str, str]] = []
    om_stems: Dict[int, str] = {}
    for i, (gpkg_raw, tif_raw, stem) in enumerate(config["pairs"], 1):
        gpkg_from_config = Path(gpkg_raw)
        tif_from_config = Path(tif_raw)
        gpkg = str(gpkg_from_config) if gpkg_from_config.exists() else str(crowns_dir / f"{stem}_multithreshold.gpkg")
        tif = str(tif_from_config) if tif_from_config.exists() else str(om_dir / f"{stem}.tif")
        pairs.append((gpkg, tif, stem))
        om_stems[i] = stem
    return pairs, om_stems


# ---------------------------------------------------------------------------
# Features / scoring
# ---------------------------------------------------------------------------
def load_features_df(phenology_dir: Path) -> pd.DataFrame:
    """Load the per-(chain, OM) feature table written by Step 3."""
    raw_csv = phenology_dir / "phenology_features_raw.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Could not find {raw_csv}. Run Step 3 (03_phenology_analysis.py) first."
        )
    df = pd.read_csv(raw_csv)
    if "is_bad_observation" not in df.columns:
        df["is_bad_observation"] = False
    df["is_bad_observation"] = df["is_bad_observation"].astype(bool)
    return df


def om_ids_from_features(features_df: pd.DataFrame) -> List[int]:
    """The OM ids actually present in the run, sorted. Robust to OM exclusions."""
    return sorted(int(x) for x in pd.unique(features_df["om_id"].dropna()))


def clean_obs_counts(features_df: pd.DataFrame) -> pd.Series:
    """Per-chain count of usable observations (patch present AND not bad).

    Matches the `clean_mask` definition in Step 3's build_temporal_summary.
    """
    df = features_df.copy()
    patch_exists = df.get("patch_exists", pd.Series(True, index=df.index)).astype(bool)
    bad = df.get("is_bad_observation", pd.Series(False, index=df.index)).astype(bool)
    df["_clean"] = patch_exists & ~bad
    return df.groupby("chain_id")["_clean"].sum().astype(int)


# ---------------------------------------------------------------------------
# Representative chain selection
# ---------------------------------------------------------------------------
def select_representative_chains(
    tree_scores_df: pd.DataFrame,
    *,
    n_total: int = 24,
    min_clean_obs: int = 6,
    clean_counts: Optional[pd.Series] = None,
    ds_threshold: float = 0.70,
) -> Dict[str, List[int]]:
    """Choose a defensible, spread-out set of chains for the evidence figures.

    Returns a dict with three buckets so the paper narrative can group them:
      - "clear_deciduous": highest deciduous_score
      - "borderline":      deciduous_score nearest to ds_threshold
      - "clear_evergreen": lowest deciduous_score

    Only chains with >= min_clean_obs usable observations are eligible, so the
    crops are actually interpretable. Selecting across the *whole* DS range (not
    just obvious cases) is what makes the threshold placement defensible.
    """
    df = tree_scores_df.copy()
    df = df[np.isfinite(pd.to_numeric(df["deciduous_score"], errors="coerce"))]

    if clean_counts is not None:
        elig = clean_counts[clean_counts >= min_clean_obs].index
        df = df[df["chain_id"].isin(elig)]

    df = df.sort_values("deciduous_score", ascending=False).reset_index(drop=True)
    if df.empty:
        return {"clear_deciduous": [], "borderline": [], "clear_evergreen": []}

    # Split budget roughly in thirds.
    k = max(1, n_total // 3)

    clear_decid = df.head(k)["chain_id"].astype(int).tolist()
    clear_ever = df.tail(k)["chain_id"].astype(int).tolist()

    # Borderline: nearest to the threshold, excluding ones already picked.
    picked = set(clear_decid) | set(clear_ever)
    remaining = df[~df["chain_id"].isin(picked)].copy()
    remaining["_dist"] = (pd.to_numeric(remaining["deciduous_score"], errors="coerce") - ds_threshold).abs()
    borderline = (
        remaining.sort_values("_dist").head(n_total - len(picked))["chain_id"].astype(int).tolist()
    )

    return {
        "clear_deciduous": clear_decid,
        "borderline": borderline,
        "clear_evergreen": clear_ever,
    }


def flatten_selection(sel: Dict[str, List[int]]) -> List[int]:
    seen: set = set()
    out: List[int] = []
    for bucket in ("clear_deciduous", "borderline", "clear_evergreen"):
        for cid in sel.get(bucket, []):
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out


# ---------------------------------------------------------------------------
# Tracker construction (only needed by the evidence script; imports tree_tracking)
# ---------------------------------------------------------------------------
def build_aligned_tracker(
    config: dict,
    *,
    base_threshold_tag: str = "conf_0p45",
    align_threshold_tag: str = "conf_0p65",
    align_method: str = "pcc_tiled",
):
    """Rebuild the SAME aligned tracker Step 3 used, so extracted patches are
    spatially identical to what produced the features. Returns (tracker, om_stems).

    This is a faithful copy of the tracker-setup block in Step 3's main().
    """
    project_root = Path(config["project_root"])
    phenology_dir = Path(config["phenology_dir"])
    crowns_dir = Path(config["crowns_dir"])
    om_dir = Path(config["om_dir"])

    setup_app_dir(project_root)
    from tree_tracking import TreeTrackingGraph  # noqa: WPS433 (local import by design)

    pairs, om_stems = build_pairs_and_om_stems(config)
    num_oms = len(pairs)

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

    saved_shifts_raw = config.get("alignment_shifts", {})
    saved_shifts = (
        {int(k): (float(v[0]), float(v[1])) for k, v in saved_shifts_raw.items()}
        if saved_shifts_raw
        else {}
    )

    if saved_shifts:
        tracker.load_multithreshold_data(
            base_threshold_tag=base_threshold_tag, load_images=False, align=False
        )
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
                tracker._compute_crown_attributes(row.geometry) for _, row in gdf.iterrows()
            ]
    else:
        tracker.load_multithreshold_data(
            base_threshold_tag=base_threshold_tag,
            load_images=False,
            align=True,
            align_method=align_method,
            align_threshold_tag=align_threshold_tag,
        )

    return tracker, om_stems


# ===========================================================================
# FLIP-VISUALISATION HELPERS (used by 06's --visualize-flips stage)
# ---------------------------------------------------------------------------
# These require imagery (the aligned tracker) only for crop extraction; all
# plotting works on plain numpy arrays so it is unit-testable without a tracker.
# ===========================================================================
def build_geom_by_chain(config: dict, tracking_dir: Optional[Path] = None) -> Dict[int, Tuple[int, object]]:
    """Read consensus crowns and return chain_id -> (crown_index, geometry)."""
    import geopandas as gpd

    phenology_dir = Path(config["phenology_dir"])
    tdir = Path(tracking_dir) if tracking_dir else Path(config.get("tracking_dir", phenology_dir))
    consensus_gpkg = Path(config.get("consensus_gpkg", tdir / "consensus_crowns_complete_all.gpkg"))
    crowns = gpd.read_file(str(consensus_gpkg))
    if "chain_id" not in crowns.columns:
        crowns = crowns.reset_index(drop=True)
        crowns["chain_id"] = crowns.index.astype(int)
    crowns["chain_id"] = pd.to_numeric(crowns["chain_id"], errors="coerce")
    crowns = crowns.dropna(subset=["chain_id"]).reset_index(drop=True)
    crowns["chain_id"] = crowns["chain_id"].astype(int)
    out: Dict[int, Tuple[int, object]] = {}
    for crown_index, row in crowns.iterrows():
        out[int(row["chain_id"])] = (int(crown_index), row.geometry)
    return out


def extract_crops_for_chains(
    tracker,
    geom_by_chain: Dict[int, Tuple[int, object]],
    chain_ids: List[int],
    om_ids: List[int],
    om_stems: Dict[int, str],
    *,
    save_root: Optional[Path] = None,
) -> Dict[int, Dict[str, dict]]:
    """Extract crops ONCE for each chain (imagery is config-independent).

    Returns {chain_id: {"patches": {om_id: arr|None}, "statuses": {om_id: str}}}.
    If save_root given, also writes PNGs under save_root/chain_XXXX/.
    """
    cache: Dict[int, Dict[str, dict]] = {}
    for cid in chain_ids:
        if cid not in geom_by_chain:
            continue
        _, geom = geom_by_chain[cid]
        patches: Dict[int, Optional[np.ndarray]] = {}
        statuses: Dict[int, str] = {}
        for oid in om_ids:
            patch = None
            if geom is not None and not geom.is_empty:
                try:
                    patch = tracker.extract_patch_for_polygon(int(oid), geom)
                except Exception:
                    patch = None
            ok = isinstance(patch, np.ndarray) and patch.size > 0
            patches[oid] = patch if ok else None
            statuses[oid] = "ok" if ok else "missing"
            if ok and save_root is not None:
                _save_crop_png(patch, Path(save_root) / f"chain_{cid:04d}" /
                               f"OM{oid:02d}_{om_stems.get(oid, f'OM{oid}')}.png")
        cache[cid] = {"patches": patches, "statuses": statuses}
    return cache


def _save_crop_png(patch: Optional[np.ndarray], path: Path) -> bool:
    if patch is None or not isinstance(patch, np.ndarray) or patch.size == 0:
        return False
    arr = patch[..., :3] if patch.ndim == 3 and patch.shape[2] >= 3 else patch
    arr = np.clip(np.nan_to_num(arr, nan=0.0), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(arr).save(str(path))
    except Exception:
        import matplotlib.image as mpimg
        mpimg.imsave(str(path), arr)
    return True


def veg_curves_from_phenophase(phenophase_df: pd.DataFrame) -> Dict[int, Dict[str, Dict[int, float]]]:
    """Extract per-chain veg_hat and veg_norm dicts from a phenophase table.

    These curves are config-INDEPENDENT (interpolation + minmax only), so we
    compute them once from the baseline scoring and reuse for every experiment.
    """
    out: Dict[int, Dict[str, Dict[int, float]]] = {}
    for _, r in phenophase_df.iterrows():
        cid = int(r["chain_id"]); oid = int(r["om_id"])
        d = out.setdefault(cid, {"veg_hat": {}, "veg_norm": {}})
        d["veg_hat"][oid] = float(r.get("veg_fraction_hsv_hat", np.nan))
        d["veg_norm"][oid] = float(r.get("veg_fraction_hsv_norm", np.nan))
    return out


def population_veg_amplitude(features_df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    """Per-chain veg-fraction amplitude (max-min over clean obs) + summary stats.

    This is the INDEPENDENT imagery-based evidence of deciduousness used to argue
    which label is correct (a crown that truly drops its leaves has high amplitude).
    """
    df = features_df.copy()
    patch_exists = df.get("patch_exists", pd.Series(True, index=df.index)).astype(bool)
    bad = df.get("is_bad_observation", pd.Series(False, index=df.index)).astype(bool)
    df = df[patch_exists & ~bad]
    df["veg_fraction_hsv"] = pd.to_numeric(df["veg_fraction_hsv"], errors="coerce")
    amp = df.groupby("chain_id")["veg_fraction_hsv"].agg(lambda s: float(np.nanmax(s) - np.nanmin(s)))
    stats = {
        "median": float(np.nanmedian(amp.values)) if len(amp) else float("nan"),
        "q25": float(np.nanpercentile(amp.values, 25)) if len(amp) else float("nan"),
        "q75": float(np.nanpercentile(amp.values, 75)) if len(amp) else float("nan"),
    }
    return amp, stats


def flip_verdict(call_base: bool, amp: float, amp_median: float) -> str:
    """Heuristic read of whether the IMAGERY supports the BASELINE call.

    NOT automatic ground truth — a transparent proxy: crowns with veg amplitude
    above the population median 'look deciduous'; below it 'look evergreen'.
    """
    looks_decid = np.isfinite(amp) and np.isfinite(amp_median) and amp >= amp_median
    imagery = "DECIDUOUS-looking" if looks_decid else "EVERGREEN-looking"
    # baseline call agrees with imagery?
    supports = (call_base and looks_decid) or ((not call_base) and (not looks_decid))
    who = "baseline appears CORRECT (alt likely wrong)" if supports else "baseline questionable here"
    return f"imagery: {imagery} -> {who}"


def render_flip_panel(
    *,
    chain_id: int,
    experiment_label: str,
    om_ids: List[int],
    stems: Dict[int, str],
    patches: Dict[int, Optional[np.ndarray]],
    statuses: Dict[int, str],
    veg_hat: Dict[int, float],
    veg_norm: Dict[int, float],
    ds_base: float,
    ds_alt: float,
    thr_base: float,
    thr_alt: float,
    call_base: bool,
    call_alt: bool,
    amp: float,
    amp_median: float,
    out_path: Path,
    max_cols: int = 8,
) -> None:
    """One panel per flipped crown: crop strip + veg curve (evidence) + DS gauge
    (why it flipped) + a verdict header. Peak/trough crops are highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    oms = list(om_ids)
    vh = np.array([veg_hat.get(o, np.nan) for o in oms], dtype=float)
    peak_om = oms[int(np.nanargmax(vh))] if np.isfinite(vh).any() else None
    trough_om = oms[int(np.nanargmin(vh))] if np.isfinite(vh).any() else None
    drop_pct = float("nan")
    if peak_om is not None and np.isfinite(veg_hat.get(peak_om, np.nan)) and veg_hat.get(peak_om, 0) > 0:
        drop_pct = 100.0 * (veg_hat[peak_om] - veg_hat[trough_om]) / veg_hat[peak_om]

    ncols = min(max_cols, len(oms))
    nrows_crops = int(np.ceil(len(oms) / ncols))

    fig = plt.figure(figsize=(2.0 * ncols, 2.2 * nrows_crops + 3.4))
    # Outer: crop area (top) and analysis row (bottom), aligned via one nested grid.
    outer = GridSpec(2, 1, figure=fig, height_ratios=[nrows_crops, 1.6], hspace=0.28)
    top = GridSpecFromSubplotSpec(nrows_crops, ncols, subplot_spec=outer[0], hspace=0.55, wspace=0.12)
    bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[3, 1], wspace=0.25)

    for i, oid in enumerate(oms):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(top[r, c])
        patch = patches.get(oid)
        ok = statuses.get(oid, "missing") == "ok" and patch is not None
        if ok:
            arr = np.clip(np.nan_to_num(patch[..., :3], nan=0.0), 0, 255).astype(np.uint8)
            ax.imshow(arr)
        else:
            ax.imshow(np.full((10, 10, 3), 235, dtype=np.uint8))
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#888")
        ax.set_xticks([]); ax.set_yticks([])
        if oid == peak_om:
            col, lw, tag = "#1b7837", 3.5, " PEAK"
        elif oid == trough_om:
            col, lw, tag = "#8c510a", 3.5, " TROUGH"
        else:
            col, lw, tag = "#cccccc", 1.2, ""
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(lw)
        ax.set_title(f"OM{oid}{tag}\nveg={veg_hat.get(oid, float('nan')):.2f}", fontsize=6.5)

    # bottom-left: veg curve (evidence of deciduousness)
    axc = fig.add_subplot(bottom[0, 0])
    x = np.array(oms, dtype=float)
    axc.plot(x, vh, "-o", color="#333", ms=4, lw=1.6)
    if peak_om is not None:
        axc.scatter([peak_om], [veg_hat[peak_om]], color="#1b7837", zorder=5, s=60, label="peak")
        axc.scatter([trough_om], [veg_hat[trough_om]], color="#8c510a", zorder=5, s=60, label="trough")
    axc.set_xlabel("OM id"); axc.set_ylabel("veg fraction (hat)")
    axc.set_xticks(oms)
    axc.set_title(f"peak->trough drop = {drop_pct:.0f}%   |   amplitude = {amp:.3f} "
                  f"(pop. median {amp_median:.3f})", fontsize=8)
    axc.legend(fontsize=7, loc="best")

    # bottom-right: DS gauge (why it flipped)
    axg = fig.add_subplot(bottom[0, 1])
    axg.set_xlim(0, 1); axg.set_ylim(0, 1)
    thr_hi = max(thr_base, thr_alt)
    axg.axvspan(thr_hi, 1.0, color="#1b7837", alpha=0.08)
    axg.axvspan(0.0, min(thr_base, thr_alt), color="#8c510a", alpha=0.08)
    axg.axvline(thr_base, color="#4575b4", ls="--", lw=1.5)
    axg.text(thr_base, 0.92, f"base thr {thr_base:.2f}", rotation=90, fontsize=6.5,
             color="#4575b4", va="top", ha="right")
    if abs(thr_alt - thr_base) > 1e-9:
        axg.axvline(thr_alt, color="#d73027", ls="--", lw=1.5)
        axg.text(thr_alt, 0.92, f"alt thr {thr_alt:.2f}", rotation=90, fontsize=6.5,
                 color="#d73027", va="top", ha="left")
    axg.scatter([ds_base], [0.5], color="#4575b4", s=90, zorder=5, label=f"DS base {ds_base:.2f}")
    if abs(ds_alt - ds_base) > 1e-9:
        axg.scatter([ds_alt], [0.35], color="#d73027", s=90, zorder=5, label=f"DS alt {ds_alt:.2f}")
    axg.set_yticks([]); axg.set_xlabel("deciduous score")
    axg.set_title("why it flipped", fontsize=8)
    axg.legend(fontsize=6.5, loc="lower center")

    b = "DECIDUOUS" if call_base else "EVERGREEN"
    a = "DECIDUOUS" if call_alt else "EVERGREEN"
    verdict = flip_verdict(call_base, amp, amp_median)
    fig.suptitle(
        f"chain {chain_id}   |   {experiment_label}\n"
        f"baseline = {b}   ->   alt = {a}      [{verdict}]",
        fontsize=11, y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_flip_contact_sheet(
    *,
    experiment_label: str,
    rows: List[dict],           # each: chain_id, peak_patch, trough_patch, veg_hat(list), call_base, call_alt, amp, verdict
    om_ids: List[int],
    out_path: Path,
    per_page: int = 12,
) -> List[Path]:
    """Compact scannable sheet: one row per flipped crown = peak | trough | sparkline | text.
    Paginates if more than per_page crowns. Returns list of written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: List[Path] = []
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)] or [[]]
    for pi, page in enumerate(pages):
        n = max(1, len(page))
        fig, axes = plt.subplots(n, 4, figsize=(11, 1.5 * n),
                                 gridspec_kw={"width_ratios": [1, 1, 2.2, 3]})
        if n == 1:
            axes = np.array([axes])
        for ri, item in enumerate(page):
            axp, axt, axs, axx = axes[ri]
            for ax, key, title in ((axp, "peak_patch", "peak"), (axt, "trough_patch", "trough")):
                arr = item.get(key)
                if isinstance(arr, np.ndarray) and arr.size:
                    ax.imshow(np.clip(np.nan_to_num(arr[..., :3], nan=0.0), 0, 255).astype(np.uint8))
                else:
                    ax.imshow(np.full((10, 10, 3), 235, dtype=np.uint8))
                ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=7)
            vh = np.array(item["veg_hat"], dtype=float)
            axs.plot(om_ids, vh, "-o", ms=2, color="#333"); axs.set_xticks([])
            axs.set_yticks([]); axs.set_title("veg curve", fontsize=7)
            axx.axis("off")
            b = "DECID" if item["call_base"] else "EVER"
            a = "DECID" if item["call_alt"] else "EVER"
            axx.text(0.0, 0.5,
                     f"chain {item['chain_id']}   {b} -> {a}\n"
                     f"amp={item['amp']:.3f}   {item['verdict']}",
                     fontsize=8, va="center", family="monospace")
        for ri in range(len(page), n):
            for ax in axes[ri]:
                ax.axis("off")
        fig.suptitle(f"Flip contact sheet — {experiment_label}  (page {pi+1}/{len(pages)})",
                     fontsize=11, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = out_path if len(pages) == 1 else out_path.with_name(out_path.stem + f"_p{pi+1}" + out_path.suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    return written
