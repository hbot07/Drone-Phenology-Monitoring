#!/usr/bin/env python3
"""
Create professor-facing data summaries and visualizations for the local
Sentinel-2 crown RF experiment.

Outputs go to:
  output/satellite_rf_weekly_report/
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parents[3]
REPORT_ROOT = REPO_ROOT / "output" / "satellite_rf_weekly_report"
FIG_DIR = REPORT_ROOT / "figures"
TABLE_DIR = REPORT_ROOT / "tables"

DATA_CROWNS = PKG_ROOT / "data" / "iitd_sv_crowns_master_wgs84.geojson"
LABEL_TABLE = PKG_ROOT / "data" / "crown_label_table.csv"
SPECIES_COUNTS = PKG_ROOT / "outputs" / "species_counts_by_area.csv"
CLASSIFIER_SUMMARY = PKG_ROOT / "outputs" / "classifier_label_summary.csv"
ESD_FEATURES = PKG_ROOT / "exports" / "stac_s2_features_2025_buffer20_items4_label_esd.csv"
ACACIA_FEATURES = PKG_ROOT / "exports" / "stac_s2_features_2025_buffer20_items4_label_acacia.csv"
STAC_ITEMS = PKG_ROOT / "outputs" / "stac_meta" / "stac_items_2025_buffer.csv"

METRICS_DIRS = [
    PKG_ROOT / "outputs" / "local_rf_stac",
    PKG_ROOT / "outputs" / "local_rf_stac_acacia_full",
    PKG_ROOT / "outputs" / "local_rf_stac_acacia_full_threshold",
]

LABEL_NAMES = {
    "label_esd": {-1: "ignore", 0: "evergreen", 1: "semi-evergreen", 2: "deciduous"},
    "label_deciduous": {-1: "ignore", 0: "not deciduous", 1: "deciduous"},
    "label_acacia": {-1: "ignore", 0: "non-Acacia", 1: "Acacia"},
    "label_yellow_strict": {-1: "ignore", 0: "not yellow-showy", 1: "yellow-showy"},
    "label_yellow_broad": {-1: "ignore", 0: "not yellow", 1: "yellow"},
    "label_red_showy": {-1: "ignore", 0: "not red-showy", 1: "red-showy"},
    "label_showy_flower": {-1: "ignore", 0: "not showy", 1: "showy"},
}

AREA_COLORS = {
    "A1": "#4e79a7",
    "A2": "#f28e2b",
    "A3": "#e15759",
    "A4": "#76b7b2",
    "A5": "#59a14f",
    "MITTAL": "#edc948",
    "SIT": "#b07aa1",
    "SV_S1": "#ff9da7",
    "SV_S2": "#9c755f",
    "SV_S3": "#bab0ab",
    "SV_S4": "#2f6f73",
}


def ensure_dirs() -> None:
    for folder in [REPORT_ROOT, FIG_DIR, TABLE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def save_bar(series: pd.Series, path: Path, title: str, xlabel: str, ylabel: str, color: str = "#4e79a7", top: int | None = None) -> None:
    data = series.copy()
    if top is not None:
        data = data.head(top)
    fig_h = max(4.2, 0.28 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=180)
    data.sort_values().plot(kind="barh", ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_grouped_label_counts(label_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label_col, names in LABEL_NAMES.items():
        counts = label_table[label_col].value_counts().sort_index()
        for value, count in counts.items():
            value = int(value)
            rows.append(
                {
                    "classifier": label_col,
                    "label_value": value,
                    "label_name": names.get(value, str(value)),
                    "count": int(count),
                    "usable": value != -1,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "classifier_counts_long.csv", index=False)

    usable = out[out["usable"]].copy()
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), dpi=180)
    axes = axes.reshape(-1)
    for ax, (classifier, group) in zip(axes, usable.groupby("classifier", sort=False)):
        group = group.sort_values("label_value")
        ax.bar(group["label_name"], group["count"], color="#4e79a7")
        ax.set_title(classifier)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(usable["classifier"].unique()) :]:
        ax.axis("off")
    fig.suptitle("Usable Label Counts by Classifier", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "classifier_usable_counts.png")
    plt.close(fig)
    return out


def save_species_heatmap(species_counts: pd.DataFrame) -> None:
    top_species = (
        species_counts[species_counts["species_for_count"] != "<unlabelled_or_ambiguous>"]
        .groupby("species_for_count")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(25)
        .index
    )
    pivot = (
        species_counts[species_counts["species_for_count"].isin(top_species)]
        .pivot_table(index="species_for_count", columns="area", values="count", aggfunc="sum", fill_value=0)
        .loc[top_species]
    )
    pivot.to_csv(TABLE_DIR / "top_species_by_area_matrix.csv")

    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Top Species Counts by Area")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = int(pivot.iat[i, j])
            if val:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="crowns")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_species_by_area_heatmap.png")
    plt.close(fig)


def load_metrics() -> pd.DataFrame:
    rows = []
    for folder in METRICS_DIRS:
        if not folder.exists():
            continue
        for path in folder.glob("*metrics.json"):
            data = json.loads(path.read_text())
            rows.append(
                {
                    "run_group": folder.name,
                    "file": path.name,
                    "label": data["label"],
                    "split": data["split"],
                    "holdout": data.get("holdout") or "random",
                    "n_train": data["n_train"],
                    "n_test": data["n_test"],
                    "accuracy": data["accuracy"],
                    "balanced_accuracy": data["balanced_accuracy"],
                    "macro_f1": data["macro_f1"],
                    "threshold": data.get("threshold", np.nan),
                    "positive_recall": data.get("positive_recall", np.nan),
                    "positive_precision": data.get("positive_precision", np.nan),
                    "test_label_counts": data["test_label_counts"],
                    "confusion_matrix": data["confusion_matrix"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "all_rf_metrics_summary.csv", index=False)
    return out


def plot_metrics(metrics: pd.DataFrame) -> None:
    primary = metrics[
        (metrics["run_group"].isin(["local_rf_stac", "local_rf_stac_acacia_full"]))
        & ~metrics["file"].str.contains("threshold", regex=False)
    ].copy()
    primary["experiment"] = primary["label"] + " | " + primary["split"] + " | " + primary["holdout"]
    primary = primary.sort_values(["label", "split", "holdout"])
    fig, ax = plt.subplots(figsize=(12, max(5, 0.34 * len(primary))), dpi=180)
    y = np.arange(len(primary))
    ax.barh(y - 0.18, primary["balanced_accuracy"], height=0.36, label="balanced accuracy", color="#4e79a7")
    ax.barh(y + 0.18, primary["macro_f1"], height=0.36, label="macro-F1", color="#f28e2b")
    ax.set_yticks(y)
    ax.set_yticklabels(primary["experiment"], fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_title("Random Forest Metrics")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rf_metrics_balanced_accuracy_macro_f1.png")
    plt.close(fig)


def plot_confusion_matrices(metrics: pd.DataFrame) -> None:
    selected = metrics[
        (
            (metrics["run_group"] == "local_rf_stac_acacia_full")
            & (metrics["label"] == "label_acacia")
            & (metrics["holdout"].isin(["random", "SV_S1", "SV_S3", "SV_S4"]))
        )
        | (
            (metrics["run_group"] == "local_rf_stac")
            & (metrics["label"].isin(["label_esd", "label_deciduous"]))
            & (metrics["holdout"] == "random")
        )
    ].copy()
    selected = selected.sort_values(["label", "holdout"])
    n = len(selected)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.7 * rows), dpi=180)
    axes = np.array(axes).reshape(-1)
    for ax, (_, row) in zip(axes, selected.iterrows()):
        cm = np.asarray(row["confusion_matrix"], dtype=float)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{row['label']} | {row['holdout']}")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "selected_confusion_matrices.png")
    plt.close(fig)


def plot_feature_importance() -> None:
    files = [
        PKG_ROOT / "outputs" / "local_rf_stac_acacia_full" / "label_acacia_random_feature_importance.csv",
        PKG_ROOT / "outputs" / "local_rf_stac" / "label_deciduous_random_feature_importance.csv",
        PKG_ROOT / "outputs" / "local_rf_stac" / "label_esd_random_feature_importance.csv",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=180)
    for ax, path in zip(axes, files):
        if not path.exists():
            ax.axis("off")
            continue
        df = pd.read_csv(path).head(15).iloc[::-1]
        ax.barh(df["feature"], df["importance"], color="#59a14f")
        ax.set_title(path.name.replace("_feature_importance.csv", ""))
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_feature_importances.png")
    plt.close(fig)


def percentile_stretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    arr = arr.astype("float32")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    vmin, vmax = np.nanpercentile(arr[finite], [lo, hi])
    if vmax <= vmin:
        return np.zeros_like(arr)
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)


def stac_asset_key(band: str) -> str:
    return {"B2": "B02", "B3": "B03", "B4": "B04"}.get(band, band)


def query_item_by_id(item_id: str, bbox: list[float], date_text: str):
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    date = pd.Timestamp(date_text).date()
    items = list(
        catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{date}/{date}",
        ).items()
    )
    for item in items:
        if item.id == item_id:
            return item
    if not items:
        raise RuntimeError(f"No STAC items found for {date}")
    return items[0]


def read_rgb_for_overlay(item, bbox: list[float], dst_crs: str, pad_pixels: int = 40):
    tx = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    minlon, minlat, maxlon, maxlat = bbox
    xs, ys = tx.transform([minlon, minlon, maxlon, maxlon], [minlat, maxlat, minlat, maxlat])
    bounds = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

    bands = []
    out_transform = None
    out_shape = None
    for band in ["B4", "B3", "B2"]:
        url = pc.sign(item.assets[stac_asset_key(band)].href)
        with rasterio.Env(GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_CONNECTTIMEOUT="20", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            with rasterio.open(url) as src:
                win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
                col_off = max(0, int(win.col_off) - pad_pixels)
                row_off = max(0, int(win.row_off) - pad_pixels)
                right = min(src.width, int(math.ceil(win.col_off + win.width)) + pad_pixels)
                bottom = min(src.height, int(math.ceil(win.row_off + win.height)) + pad_pixels)
                win = rasterio.windows.Window(col_off, row_off, max(1, right - col_off), max(1, bottom - row_off))
                if out_shape is None:
                    arr = src.read(1, window=win, masked=True).astype("float32")
                    out_shape = arr.shape
                else:
                    arr = src.read(1, window=win, out_shape=out_shape, resampling=Resampling.bilinear, masked=True).astype("float32")
                out_transform = rasterio.windows.transform(win, src.transform)
                bands.append(np.asarray(arr, dtype="float32"))
    rgb = np.dstack([percentile_stretch(b) for b in bands])
    h, w = rgb.shape[:2]
    left, top = out_transform * (0, 0)
    right, bottom = out_transform * (w, h)
    return rgb, (left, right, bottom, top)


def plot_crowns_over_satellite(crowns: gpd.GeoDataFrame, label_table: pd.DataFrame) -> None:
    if not STAC_ITEMS.exists():
        return
    items = pd.read_csv(STAC_ITEMS)
    row = items[items["season"] == "postmonsoon"].sort_values("cloud_cover").iloc[0]
    bbox = list(map(float, crowns.total_bounds))
    item = query_item_by_id(row["id"], bbox, row["datetime"])
    epsg = int(row["epsg"])
    dst_crs = f"EPSG:{epsg}"
    rgb, extent = read_rgb_for_overlay(item, bbox, dst_crs)

    crowns_plot = crowns.to_crs(dst_crs).copy()
    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    ax.imshow(rgb, extent=extent)
    for area, group in crowns_plot.groupby("area"):
        group.boundary.plot(ax=ax, linewidth=0.25, alpha=0.65, color=AREA_COLORS.get(area, "#333333"), label=area)
    ax.set_title(f"Crowns over Sentinel-2 RGB ({pd.Timestamp(row['datetime']).date()})")
    ax.set_axis_off()
    ax.legend(loc="upper left", fontsize=6, ncols=2, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crowns_over_sentinel2_rgb_by_area.png")
    plt.close(fig)

    acacia_ids = set(label_table.loc[label_table["label_acacia"] == 1, "crown_uid"])
    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    ax.imshow(rgb, extent=extent)
    crowns_plot.boundary.plot(ax=ax, linewidth=0.15, alpha=0.25, color="#222222")
    crowns_plot[crowns_plot["crown_uid"].isin(acacia_ids)].boundary.plot(ax=ax, linewidth=0.9, alpha=0.95, color="#d62728")
    ax.set_title("Acacia-Positive Crowns over Sentinel-2 RGB")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acacia_crowns_over_sentinel2_rgb.png")
    plt.close(fig)


def write_markdown_report(label_table: pd.DataFrame, metrics: pd.DataFrame, classifier_counts: pd.DataFrame) -> None:
    total = len(label_table)
    labelled_species = int(label_table["species_clean"].notna().sum())
    ambiguous = int((label_table["species_status"] == "ambiguous_or_unknown").sum())
    missing = int((label_table["species_status"] == "missing").sum())
    species_counts = label_table["species_clean"].dropna().value_counts()
    top_species_lines = "\n".join([f"- {name}: {count}" for name, count in species_counts.head(15).items()])
    area_lines = "\n".join([f"- {area}: {count}" for area, count in label_table["area"].value_counts().items()])

    usable_lines = []
    for classifier, group in classifier_counts[classifier_counts["usable"]].groupby("classifier", sort=False):
        parts = [f"{r.label_name}={int(r['count'])}" for _, r in group.sort_values("label_value").iterrows()]
        usable_lines.append(f"- `{classifier}`: " + ", ".join(parts))

    metrics = metrics.copy()
    metrics["test_class_count"] = metrics["test_label_counts"].apply(lambda x: len(x) if isinstance(x, dict) else np.nan)
    metric_focus = metrics[
        metrics["run_group"].isin(["local_rf_stac", "local_rf_stac_acacia_full", "local_rf_stac_acacia_full_threshold"])
        & (metrics["test_class_count"] >= 2)
    ].copy()
    metric_focus = metric_focus.sort_values("balanced_accuracy", ascending=False).head(12)
    metric_lines = []
    for _, r in metric_focus.iterrows():
        thresh = "" if pd.isna(r["threshold"]) else f", threshold={r['threshold']:.2f}"
        metric_lines.append(
            f"- {r['run_group']} `{r['label']}` {r['split']} {r['holdout']}: "
            f"accuracy={r['accuracy']:.3f}, balanced_accuracy={r['balanced_accuracy']:.3f}, macro_F1={r['macro_f1']:.3f}{thresh}"
        )

    report = f"""# Satellite RF Weekly Report

## What Was Checked

- Read the prepared IITD + Sanjay Van crown package.
- Reviewed the two satellite notebooks:
  - `sat_data.ipynb`: early Planetary Computer/STAC Sentinel-2 querying, RGB overlays, NDVI time series.
  - `sat_data6May26.ipynb`: cleaner SCL-masked Sentinel-2 extraction, crown sampling, weekly/monthly NDVI/EVI features.
- Implemented a package-local non-GEE Sentinel-2 STAC feature extractor.
- Trained Random Forest baselines locally and saved metrics.

## Crown Data

- Total crowns in master GeoJSON: {total}
- Crowns with clean species labels: {labelled_species}
- Ambiguous/unknown species labels: {ambiguous}
- Missing species labels: {missing}
- Areas: {label_table['area'].nunique()}
- Clean species represented: {species_counts.size}

## Crowns by Area

{area_lines}

## Top Clean Species

{top_species_lines}

## Usable Data by Classifier

{chr(10).join(usable_lines)}

## Sentinel-2 Local Feature Extraction

- Source: Microsoft Planetary Computer Sentinel-2 L2A.
- Year: 2025 complete year.
- Geometry: 20 m centroid buffer.
- Seasons: winter, pre-monsoon, monsoon, post-monsoon.
- Per season: 4 lowest-cloud items for this first local run.
- Features: Sentinel-2 bands, vegetation/color indices, seasonal amplitudes.

## First Result Summary

Single-class holdouts are excluded from this ranked list because balanced accuracy can be misleading there.

{chr(10).join(metric_lines)}

## Interpretation

- Best current local signal is Acacia vs non-Acacia.
- Random split is strong, especially after threshold tuning.
- Leave-area-out validation is uneven, so spatial transfer is not solved yet.
- Deciduous vs rest is moderate in random split but weak on area holdouts.
- ESD multiclass is not yet strong enough as a local-only result.
- Yellow/showy flower models currently collapse to the majority class under random split; they need more positives or better flowering-window features.

## Recommended Next Step

Use this local report as a baseline and run the same experiment in GEE next. GEE will let us cheaply sweep more dates, geometry modes, and cloud masks, then export richer feature tables back into this local evaluation framework.
"""
    (REPORT_ROOT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    label_table = pd.read_csv(LABEL_TABLE)
    crowns = gpd.read_file(DATA_CROWNS)
    species_counts = pd.read_csv(SPECIES_COUNTS)

    label_table.to_csv(TABLE_DIR / "crown_label_table_copy.csv", index=False)
    species_counts.to_csv(TABLE_DIR / "species_counts_by_area.csv", index=False)
    pd.read_csv(CLASSIFIER_SUMMARY).to_csv(TABLE_DIR / "classifier_label_summary.csv", index=False)

    label_table["species_for_count"] = label_table["species_clean"].fillna("<unlabelled_or_ambiguous>")
    save_bar(label_table["area"].value_counts(), FIG_DIR / "crowns_by_area.png", "Crowns by Area", "crowns", "area")
    save_bar(label_table["species_status"].value_counts(), FIG_DIR / "species_label_status.png", "Species Label Status", "crowns", "status", "#f28e2b")
    clean_species_counts = label_table.loc[label_table["species_clean"].notna(), "species_clean"].value_counts()
    save_bar(clean_species_counts, FIG_DIR / "top_30_clean_species.png", "Top 30 Clean Species", "crowns", "species", "#59a14f", top=30)
    classifier_counts = save_grouped_label_counts(label_table)
    save_species_heatmap(species_counts)

    metrics = load_metrics()
    plot_metrics(metrics)
    plot_confusion_matrices(metrics)
    plot_feature_importance()
    plot_crowns_over_satellite(crowns, label_table)
    write_markdown_report(label_table, metrics, classifier_counts)
    print(f"Wrote report assets to {REPORT_ROOT}")


if __name__ == "__main__":
    main()
