#!/usr/bin/env python3
"""Generate thesis-ready LHC tables and figures from existing pipeline outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any


def _set_mpl_cache(run_dir: Path) -> None:
    mpl_dir = run_dir / "_mplconfig"
    cache_dir = run_dir / "_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def parse_lhc_date(stem: str) -> date | None:
    match = re.search(r"odm_orthophoto_?(\d{1,2})_(\d{1,2})_(\d{2})$", stem)
    if not match:
        return None
    day, month, year = map(int, match.groups())
    return date(2000 + year, month, day)


def date_label(stem: str) -> str:
    parsed = parse_lhc_date(stem)
    return parsed.isoformat() if parsed else stem


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def require_imports():
    missing = []
    modules = {}
    for name in ["geopandas", "numpy", "pandas", "rasterio"]:
        try:
            modules[name] = __import__(name)
        except Exception as exc:  # pragma: no cover - human-facing script
            missing.append(f"{name}: {exc}")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        modules["plt"] = plt
    except Exception as exc:  # pragma: no cover
        missing.append(f"matplotlib: {exc}")
    try:
        from PIL import Image, ImageDraw

        modules["Image"] = Image
        modules["ImageDraw"] = ImageDraw
    except Exception as exc:  # pragma: no cover
        missing.append(f"Pillow: {exc}")

    if missing:
        raise SystemExit(
            "Missing Python packages needed for thesis artifact generation.\n"
            "Use the project .venv or activate dpm-tracking/detectree.\n"
            + "\n".join(missing)
        )
    return modules


def ensure_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "tables": run_dir / "tables",
        "figures": run_dir / "figures",
        "notes": run_dir / "notes",
        "copied_figures": run_dir / "copied_pipeline_figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_orthomosaic_inventory(args, pd, rasterio) -> Any:
    config = load_json(args.pipeline_output / "pipeline_config.json", {})
    included = set(config.get("om_stems", []))
    excluded = set(config.get("exclude_stems", []))
    rows = []
    for tif in sorted(args.om_dir.glob("*.tif")) + sorted(args.om_dir.glob("*.TIF")):
        stem = tif.stem
        try:
            with rasterio.open(tif) as src:
                width = src.width
                height = src.height
                crs = str(src.crs)
                res_x, res_y = src.res
                bounds = src.bounds
                area_m2 = abs((bounds.right - bounds.left) * (bounds.top - bounds.bottom))
        except Exception as exc:
            width = height = None
            crs = f"ERROR: {exc}"
            res_x = res_y = area_m2 = None
        rows.append(
            {
                "stem": stem,
                "date": date_label(stem),
                "status_in_lhc_pipeline_fixed": (
                    "included" if stem in included else "excluded" if stem in excluded else "available_not_used"
                ),
                "file_mb": round(tif.stat().st_size / (1024 * 1024), 2),
                "width_px": width,
                "height_px": height,
                "crs": crs,
                "pixel_size_x": res_x,
                "pixel_size_y": res_y,
                "area_m2": area_m2,
                "path": str(tif),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "stem"], kind="stable")
    return df


def detection_counts(args, pd, gpd) -> Any:
    config = load_json(args.pipeline_output / "pipeline_config.json", {})
    included = set(config.get("om_stems", []))
    excluded = set(config.get("exclude_stems", []))
    rows = []
    if not args.crowns_dir.exists():
        return pd.DataFrame(rows)
    for gpkg in sorted(args.crowns_dir.glob("*_multithreshold.gpkg")):
        stem = gpkg.name.replace("_multithreshold.gpkg", "")
        status = "included" if stem in included else "excluded" if stem in excluded else "available_not_used"
        try:
            layers = gpd.list_layers(gpkg)["name"].tolist()
        except Exception as exc:
            rows.append(
                {
                    "stem": stem,
                    "date": date_label(stem),
                    "status_in_lhc_pipeline_fixed": status,
                    "layer": "ERROR",
                    "count": None,
                    "error": str(exc),
                }
            )
            continue
        for layer in layers:
            try:
                count = len(gpd.read_file(gpkg, layer=layer))
                err = ""
            except Exception as exc:
                count = None
                err = str(exc)
            rows.append(
                {
                    "stem": stem,
                    "date": date_label(stem),
                    "status_in_lhc_pipeline_fixed": status,
                    "layer": layer,
                    "count": count,
                    "error": err,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "stem", "layer"], kind="stable")
    return df


def tracking_tables(args, pd) -> dict[str, Any]:
    summary = load_json(args.pipeline_output / "02_tracking" / "consensus_crowns_summary.json", {})
    metrics = load_json(args.pipeline_output / "02_tracking" / "tracking_quality_metrics.json", {})
    config = load_json(args.pipeline_output / "pipeline_config.json", {})

    tracking_summary = pd.DataFrame(
        [
            {
                "run_name": config.get("run_name"),
                "num_oms": summary.get("num_oms"),
                "excluded_stems": ", ".join(config.get("exclude_stems", [])),
                "align_method": summary.get("align_method") or config.get("align_method"),
                "base_threshold_tag": summary.get("base_threshold_tag") or config.get("base_threshold_tag"),
                "align_threshold_tag": summary.get("align_threshold_tag") or config.get("align_threshold_tag"),
                "total_detected_nodes": metrics.get("total_trees_detected"),
                "total_edges": metrics.get("total_edges"),
                "overall_match_rate": metrics.get("overall_match_rate"),
                "chains_total": summary.get("chains_total"),
                "consensus_raw": summary.get("consensus_raw"),
                "consensus_cleaned": summary.get("consensus_cleaned"),
                "dedup_dropped_total": summary.get("dedup_summary", {}).get("dropped_total"),
                "full_chains": summary.get("chain_breakdown", {}).get("full_chains"),
                "branching_chains": summary.get("chain_breakdown", {}).get("branching_chains"),
                "partial_chains_added": summary.get("chain_breakdown", {}).get("partial_chains_added"),
                "min_partial_len": summary.get("min_partial_len_used"),
                "min_partial_ratio": summary.get("min_partial_ratio_used"),
            }
        ]
    )

    match_rows = []
    stems = config.get("om_stems", [])
    for pair, item in metrics.get("match_rate_by_om_pair", {}).items():
        a, b = [int(x) for x in pair.split("->")]
        match_rows.append(
            {
                "pair": pair,
                "from_om": a,
                "to_om": b,
                "from_stem": stems[a - 1] if a - 1 < len(stems) else "",
                "to_stem": stems[b - 1] if b - 1 < len(stems) else "",
                "matches": item.get("matches"),
                "possible": item.get("possible"),
                "rate": item.get("rate"),
            }
        )
    match_rates = pd.DataFrame(match_rows)

    chain_rows = [
        {"chain_length": int(length), "count": int(count)}
        for length, count in metrics.get("chain_length_distribution", {}).items()
    ]
    chain_dist = pd.DataFrame(chain_rows)
    if not chain_dist.empty:
        chain_dist = chain_dist.sort_values("chain_length")

    shifts = pd.DataFrame()
    shift_path = args.pipeline_output / "02_tracking" / "diagnostics" / "alignment_shifts.csv"
    if shift_path.exists():
        shifts = pd.read_csv(shift_path)
        if "stem" in shifts:
            shifts["date"] = shifts["stem"].map(date_label)
            shifts["shift_magnitude"] = (shifts["dx"] ** 2 + shifts["dy"] ** 2) ** 0.5

    return {
        "tracking_summary": tracking_summary,
        "match_rates": match_rates,
        "chain_length_distribution": chain_dist,
        "alignment_shifts": shifts,
        "summary_json": summary,
        "metrics_json": metrics,
        "config_json": config,
    }


def phenology_tables(args, pd) -> dict[str, Any]:
    scores_path = args.pipeline_output / "03_phenology" / "leafshed_tree_scores.csv"
    features_path = args.pipeline_output / "03_phenology" / "phenology_features_raw.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    features = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()

    rows = []
    if not scores.empty:
        deciduous_count = int(scores["is_deciduous"].astype(bool).sum())
        rows.append(
            {
                "n_crowns_scored": int(len(scores)),
                "deciduous_count": deciduous_count,
                "evergreen_or_stable_count": int(len(scores) - deciduous_count),
                "deciduous_fraction": deciduous_count / len(scores),
                "mean_deciduous_score": float(scores["deciduous_score"].mean()),
                "median_deciduous_score": float(scores["deciduous_score"].median()),
                "min_deciduous_score": float(scores["deciduous_score"].min()),
                "max_deciduous_score": float(scores["deciduous_score"].max()),
            }
        )
    summary = pd.DataFrame(rows)

    event_counts = pd.DataFrame()
    if not scores.empty:
        event_frames = []
        for col in ["leaf_off_start_om", "full_leaf_off_om", "leaf_on_return_om"]:
            counts = scores[col].dropna().astype(int).value_counts().sort_index()
            event_frames.append(pd.DataFrame({"event": col, "om_id": counts.index, "count": counts.values}))
        event_counts = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    per_om = pd.DataFrame()
    if not features.empty:
        numeric_cols = [
            "valid_pixel_fraction",
            "shadow_fraction",
            "veg_fraction_hsv",
            "gcc_mean",
            "rcc_mean",
            "gray_entropy",
            "laplacian_var",
        ]
        agg = features.groupby(["om_id", "date_label"], dropna=False).agg(
            observations=("chain_id", "count"),
            bad_observations=("is_bad_observation", "sum"),
            **{f"{col}_mean": (col, "mean") for col in numeric_cols if col in features.columns},
            **{f"{col}_median": (col, "median") for col in numeric_cols if col in features.columns},
        )
        per_om = agg.reset_index()
        per_om["bad_observation_fraction"] = per_om["bad_observations"] / per_om["observations"]

    return {
        "scores": scores,
        "features": features,
        "phenology_summary": summary,
        "phenology_event_counts": event_counts,
        "per_om_feature_summary": per_om,
    }


def save_tables(dirs: dict[str, Path], tables: dict[str, Any]) -> None:
    for name, df in tables.items():
        if hasattr(df, "to_csv"):
            df.to_csv(dirs["tables"] / f"{name}.csv", index=False)


def plot_detection_counts(df, fig_path: Path, plt) -> None:
    if df.empty or "ERROR" in set(df["layer"]):
        return
    pivot = df.pivot_table(index="date", columns="layer", values="count", aggfunc="first")
    keep = [col for col in ["conf_0p15", "conf_0p45", "conf_0p65"] if col in pivot.columns]
    if not keep:
        keep = list(pivot.columns[: min(5, len(pivot.columns))])
    ax = pivot[keep].plot(marker="o", figsize=(10, 4))
    excluded_dates = sorted(df.loc[df["status_in_lhc_pipeline_fixed"] == "excluded", "date"].unique())
    for excluded_date in excluded_dates:
        ax.axvline(excluded_date, color="0.4", linestyle=":", linewidth=1.2)
        ax.text(excluded_date, ax.get_ylim()[1], "excluded", rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("Orthomosaic date")
    ax.set_ylabel("Detected crowns")
    ax.set_title("LHC Detectree2 crown counts by confidence layer")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Layer")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_match_rates(df, fig_path: Path, plt) -> None:
    if df.empty:
        return
    labels = [f"{r.from_om}->{r.to_om}" for r in df.itertuples()]
    colors = ["#b34d4d" if r > 1.0 else "#4c78a8" for r in df["rate"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, df["rate"], color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Consecutive OM pair")
    ax.set_ylabel("Match rate")
    ax.set_title("LHC graph match rate by orthomosaic pair")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)


def plot_chain_distribution(df, fig_path: Path, plt) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["chain_length"].astype(str), df["count"], color="#5f8f5f")
    ax.set_xlabel("Chain length")
    ax.set_ylabel("Number of chains")
    ax.set_title("LHC temporal chain length distribution")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)


def plot_alignment(df, fig_path: Path, plt) -> None:
    if df.empty:
        return
    labels = [f"OM{int(x)}" for x in df["om_id"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(labels, df["dx"], marker="o", label="dx")
    ax.plot(labels, df["dy"], marker="o", label="dy")
    ax.plot(labels, df["shift_magnitude"], marker="o", label="magnitude", linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Orthomosaic")
    ax.set_ylabel("Shift in CRS units")
    ax.set_title("LHC cumulative alignment shifts")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)


def plot_deciduous_scores(scores, fig_path: Path, plt) -> None:
    if scores.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores["deciduous_score"], bins=20, color="#8a6fba", edgecolor="white")
    ax.axvline(0.70, color="black", linestyle="--", label="pipeline threshold 0.70")
    ax.set_xlabel("Deciduousness score")
    ax.set_ylabel("Crowns")
    ax.set_title("LHC deciduousness score distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)


def plot_veg_timeseries(features, fig_path: Path, plt) -> None:
    if features.empty:
        return
    grouped = features.groupby(["om_id", "date_label"])["veg_fraction_hsv"].agg(["median", "mean"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"OM{int(x)}" for x in grouped["om_id"]]
    ax.plot(labels, grouped["median"], marker="o", label="median")
    ax.plot(labels, grouped["mean"], marker="o", label="mean")
    ax.set_xlabel("Orthomosaic")
    ax.set_ylabel("Vegetation fraction (HSV)")
    ax.set_title("LHC crown-level vegetation signal across dates")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)


def make_orthomosaic_contact_sheet(inventory, fig_path: Path, rasterio, np, plt) -> None:
    if inventory.empty:
        return
    rows = inventory.sort_values(["date", "stem"]).to_dict("records")
    n = len(rows)
    cols = 3
    grid_rows = math.ceil(n / cols)
    fig, axes = plt.subplots(grid_rows, cols, figsize=(12, 4 * grid_rows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, row in zip(axes_flat, rows):
        path = Path(row["path"])
        try:
            with rasterio.open(path) as src:
                max_dim = max(src.width, src.height)
                scale = max_dim / 700 if max_dim > 700 else 1.0
                out_h = max(1, int(src.height / scale))
                out_w = max(1, int(src.width / scale))
                indexes = [1, 2, 3] if src.count >= 3 else [1]
                data = src.read(indexes, out_shape=(len(indexes), out_h, out_w))
                img = np.moveaxis(data, 0, -1).astype("float32")
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                lo, hi = np.nanpercentile(img, [2, 98])
                if hi > lo:
                    img = np.clip((img - lo) / (hi - lo), 0, 1)
                ax.imshow(img)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", fontsize=7)
        ax.set_title(f"{row['date']}\n{row['status_in_lhc_pipeline_fixed']}", fontsize=9)
        ax.axis("off")
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle("Available LHC orthomosaics", y=0.995)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)


def make_crown_trajectory(scores, crops_dir: Path, fig_path: Path, Image, ImageDraw) -> dict[str, Any]:
    if scores.empty or not crops_dir.exists():
        return {}
    ranked = scores.sort_values(["is_deciduous", "deciduous_score"], ascending=[False, False])
    selected = None
    crop_files: list[Path] = []
    for row in ranked.itertuples():
        folder = crops_dir / f"crown_{int(row.chain_id):04d}"
        files = sorted(folder.glob("OM*.png"))
        if len(files) >= 2:
            selected = row
            crop_files = files
            break
    if selected is None:
        return {}

    thumbs = []
    for path in crop_files:
        im = Image.open(path).convert("RGB")
        im.thumbnail((180, 180))
        canvas = Image.new("RGB", (190, 215), "white")
        canvas.paste(im, ((190 - im.width) // 2, 5))
        draw = ImageDraw.Draw(canvas)
        label = path.stem.split("_", 1)[0]
        draw.text((8, 190), label, fill="black")
        thumbs.append(canvas)
    out = Image.new("RGB", (190 * len(thumbs), 245), "white")
    for i, thumb in enumerate(thumbs):
        out.paste(thumb, (i * 190, 0))
    draw = ImageDraw.Draw(out)
    draw.text(
        (8, 225),
        f"chain_id={int(selected.chain_id)}, deciduous_score={float(selected.deciduous_score):.3f}",
        fill="black",
    )
    out.save(fig_path)
    return {
        "chain_id": int(selected.chain_id),
        "deciduous_score": float(selected.deciduous_score),
        "is_deciduous": bool(selected.is_deciduous),
        "crop_count": len(crop_files),
        "source_crop_dir": str(crops_dir / f"crown_{int(selected.chain_id):04d}"),
    }


def copy_existing_figures(args, dirs: dict[str, Path]) -> list[str]:
    copied = []
    candidates = [
        args.pipeline_output / "02_tracking" / "diagnostics" / "alignment_shifts.png",
        args.pipeline_output / "02_tracking" / "diagnostics" / "match_rates_by_pair.png",
        args.pipeline_output / "02_tracking" / "diagnostics" / "chain_length_distribution.png",
        args.pipeline_output / "02_tracking" / "consensus_overlay_om1.png",
        args.pipeline_output / "04_viewer" / "phenology_overview.png",
    ]
    for src in candidates:
        if src.exists():
            dst = dirs["copied_figures"] / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def markdown_summary(args, dirs, all_tables, metadata) -> None:
    inv = all_tables["lhc_orthomosaic_inventory"]
    tracking = all_tables["lhc_tracking_summary"]
    phen = all_tables["lhc_phenology_summary"]
    lines = [
        "# LHC Thesis Artifact Run",
        "",
        "This run extracts paper-support artifacts from the currently available LHC data and existing pipeline output.",
        "",
        "## Sources",
        "",
        f"- Orthomosaics: `{args.om_dir}`",
        f"- Pipeline output: `{args.pipeline_output}`",
        f"- Crown detections: `{args.crowns_dir}`",
        "",
        "## Key Numbers",
        "",
    ]
    if not inv.empty:
        lines.append(f"- Available LHC orthomosaics in input folder: **{len(inv)}**")
        status_counts = inv["status_in_lhc_pipeline_fixed"].value_counts().to_dict()
        lines.append(f"- Pipeline usage status: `{status_counts}`")
    if not tracking.empty:
        row = tracking.iloc[0].to_dict()
        lines.extend(
            [
                f"- Pipeline OMs used: **{row.get('num_oms')}**",
                f"- Excluded stems: `{row.get('excluded_stems')}`",
                f"- Alignment method: `{row.get('align_method')}`",
                f"- Base threshold: `{row.get('base_threshold_tag')}`",
                f"- Cleaned consensus crowns: **{row.get('consensus_cleaned')}**",
                f"- Raw consensus chains before deduplication: **{row.get('consensus_raw')}**",
                f"- Overall match rate: **{row.get('overall_match_rate'):.3f}**",
            ]
        )
    if not phen.empty:
        row = phen.iloc[0].to_dict()
        lines.extend(
            [
                f"- Crowns with phenology scores: **{row.get('n_crowns_scored')}**",
                f"- Deciduous crowns at threshold 0.70: **{row.get('deciduous_count')}**",
                f"- Deciduous fraction: **{row.get('deciduous_fraction'):.3f}**",
            ]
        )
    lines.extend(
        [
            "",
            "## Generated Tables",
            "",
            *[f"- `tables/{path.name}`" for path in sorted(dirs["tables"].glob("*.csv"))],
            "",
            "## Generated Figures",
            "",
            *[f"- `figures/{path.name}`" for path in sorted(dirs["figures"].glob("*.png"))],
            "",
            "## Notes For Thesis Use",
            "",
            "- Treat this as an initial LHC subset run: it uses `output/lhc_pipeline_fixed`, which excludes the bad 2025-12-09 orthomosaic.",
            "- The thesis outline says final results should eventually be regenerated on the full LHC/SIT datasets; do not silently present these subset numbers as final full-dataset numbers.",
            "- The generated crown trajectory is chosen automatically by high deciduousness score, so inspect it before using it as an illustrative biological example.",
        ]
    )
    (dirs["notes"] / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(dirs["notes"] / "run_metadata.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--om-dir", type=Path, required=True)
    parser.add_argument("--pipeline-output", type=Path, required=True)
    parser.add_argument("--crowns-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.om_dir = args.om_dir.resolve()
    args.pipeline_output = args.pipeline_output.resolve()
    args.crowns_dir = args.crowns_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    _set_mpl_cache(args.run_dir)
    modules = require_imports()
    pd = modules["pandas"]
    np = modules["numpy"]
    gpd = modules["geopandas"]
    rasterio = modules["rasterio"]
    plt = modules["plt"]
    Image = modules["Image"]
    ImageDraw = modules["ImageDraw"]

    dirs = ensure_dirs(args.run_dir)

    inventory = build_orthomosaic_inventory(args, pd, rasterio)
    detections = detection_counts(args, pd, gpd)
    tracking = tracking_tables(args, pd)
    phenology = phenology_tables(args, pd)

    tables = {
        "lhc_orthomosaic_inventory": inventory,
        "lhc_detection_counts_by_threshold": detections,
        "lhc_tracking_summary": tracking["tracking_summary"],
        "lhc_match_rates": tracking["match_rates"],
        "lhc_chain_length_distribution": tracking["chain_length_distribution"],
        "lhc_alignment_shifts": tracking["alignment_shifts"],
        "lhc_phenology_summary": phenology["phenology_summary"],
        "lhc_phenology_event_counts": phenology["phenology_event_counts"],
        "lhc_per_om_feature_summary": phenology["per_om_feature_summary"],
    }
    save_tables(dirs, tables)
    write_json(dirs["notes"] / "tracking_summary_source.json", tracking["summary_json"])
    write_json(dirs["notes"] / "tracking_metrics_source.json", tracking["metrics_json"])

    plot_detection_counts(detections, dirs["figures"] / "lhc_detection_counts_by_date_threshold.png", plt)
    plot_match_rates(tracking["match_rates"], dirs["figures"] / "lhc_match_rates.png", plt)
    plot_chain_distribution(tracking["chain_length_distribution"], dirs["figures"] / "lhc_chain_length_distribution.png", plt)
    plot_alignment(tracking["alignment_shifts"], dirs["figures"] / "lhc_alignment_shifts.png", plt)
    plot_deciduous_scores(phenology["scores"], dirs["figures"] / "lhc_deciduous_score_hist.png", plt)
    plot_veg_timeseries(phenology["features"], dirs["figures"] / "lhc_veg_fraction_timeseries.png", plt)
    make_orthomosaic_contact_sheet(inventory, dirs["figures"] / "lhc_orthomosaic_contact_sheet.png", rasterio, np, plt)
    trajectory = make_crown_trajectory(
        phenology["scores"],
        args.pipeline_output / "04_viewer" / "crops",
        dirs["figures"] / "lhc_example_deciduous_crown_trajectory.png",
        Image,
        ImageDraw,
    )
    copied_figures = copy_existing_figures(args, dirs)

    metadata = {
        "project_root": str(args.project_root),
        "om_dir": str(args.om_dir),
        "pipeline_output": str(args.pipeline_output),
        "crowns_dir": str(args.crowns_dir),
        "run_dir": str(args.run_dir),
        "selected_trajectory": trajectory,
        "copied_pipeline_figures": copied_figures,
    }
    markdown_summary(args, dirs, tables, metadata)
    print(f"Wrote thesis artifacts to {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
