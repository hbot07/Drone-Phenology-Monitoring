#!/usr/bin/env python3
"""
Generate presentation-grade figures for the Spatial Classifier work.

Outputs are saved to:
  output/presentation_figures/

Run from the repo root:
  python src/notebooks/satellite/drone_phenology_rf_package/python/make_presentation_figures.py
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[5]   # repo root
PKG  = ROOT / "src/notebooks/satellite/drone_phenology_rf_package"
OUT  = ROOT / "output" / "presentation_figures"
OUT.mkdir(parents=True, exist_ok=True)

MULTIYEAR_RANDOM   = PKG / "outputs" / "multiyear_s2_2022_2025_buffer10_random"
ACACIA_HOLDOUTS    = PKG / "outputs" / "multiyear_s2_2022_2025_buffer10_acacia_holdouts"
DECIDUOUS_HOLDOUTS = PKG / "outputs" / "multiyear_s2_2022_2025_buffer10_deciduous_holdouts"
SHOWY_HOLDOUTS     = PKG / "outputs" / "multiyear_s2_2022_2025_buffer10_temporal_showy_holdouts"
FEAT_DIR           = PKG / "outputs" / "local_rf_stac"

LABEL_NICE = {
    "label_acacia":       "Acacia vs Non-Acacia",
    "label_deciduous":    "Deciduous vs Rest",
    "label_esd":          "ESD Multiclass (E/SE/D)",
    "label_showy_flower": "Showy-Flowering vs Rest",
    "label_yellow_strict":"Yellow-Showy vs Rest",
}
MODEL_COLORS = {
    "logistic_l2":       "#4C72B0",
    "svc_rbf":           "#DD8452",
    "rf_balanced":       "#55A868",
    "rf_deeper":         "#C44E52",
    "extra_trees":       "#8172B3",
    "extra_trees_kbest": "#937860",
    "hist_gradient":     "#DA8BC3",
}
SEASON_ORDER = ["winter", "premonsoon", "monsoon", "postmonsoon"]
SEASON_COLORS = ["#5B9BD5", "#ED7D31", "#70AD47", "#FFC000"]

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi":   150,
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sweep_dir(dirpath: Path) -> pd.DataFrame:
    frames = []
    for f in dirpath.glob("*_model_sweep.csv"):
        frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def best_per_label(df: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.DataFrame:
    idx = df.groupby(["label", "split", "holdout"])[metric].idxmax()
    return df.loc[idx].reset_index(drop=True)


def parse_cm(cm_str: str) -> np.ndarray:
    return np.array(ast.literal_eval(cm_str))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – Pipeline overview (text / schematic)
# ─────────────────────────────────────────────────────────────────────────────

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    steps = [
        ("Drone\nCrown\nDelineation\n(3,212 crowns)",   "#AED6F1"),
        ("Ground-Truth\nSpecies Labels\n(372 labeled,\n7 tasks)",   "#A9DFBF"),
        ("Sentinel-2\nL2A  STAC\n4 seasons/yr\n2022–2025",          "#FAD7A0"),
        ("Sentinel-1 SAR\nVV/VH\n2022–2025",                        "#F9E79F"),
        ("Feature\nEngineering\n(bands + indices\n+ temporal)",      "#D7BDE2"),
        ("Model Sweep\n(6 classifiers\n× 2 thresholds\n× 3 splits)","#F5CBA7"),
        ("Best Model\nSelection &\nSpatial\nValidation",             "#FDEBD0"),
    ]
    n = len(steps)
    box_w, box_h, gap = 1.6, 1.0, 0.3
    total_w = n * box_w + (n - 1) * gap
    x0 = (14 - total_w) / 2
    for i, (label, color) in enumerate(steps):
        x = x0 + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, 1.5), box_w, box_h, boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor="#555", facecolor=color
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, 2.0, label, ha="center", va="center",
                fontsize=9, multialignment="center")
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap, 2.0),
                        xytext=(x + box_w, 2.0),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.set_title("End-to-End Spatial Classifier Pipeline", fontsize=16, fontweight="bold", pad=10)
    fig.tight_layout()
    path = OUT / "01_pipeline_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Data overview: crowns & usable label counts
# ─────────────────────────────────────────────────────────────────────────────

def fig_data_overview():
    crowns_by_area = {
        "SV_S2":717, "SV_S1":656, "SV_S3":628, "A5":201,
        "A3":187, "SV_S4":178, "A4":167, "A1":158, "A2":153, "SIT":131, "MITTAL":36
    }
    label_data = {
        "Acacia vs Rest\n(label_acacia)":          [326, 65],
        "Deciduous vs Rest\n(label_deciduous)":    [227, 115],
        "ESD multiclass\n(label_esd)":             [75, 152, 115],
        "Showy-Flower\n(label_showy_flower)":      [292, 63],
        "Yellow-Showy\n(label_yellow_strict)":     [315, 40],
    }
    class_names = {
        "Acacia vs Rest\n(label_acacia)":          ["Non-Acacia", "Acacia"],
        "Deciduous vs Rest\n(label_deciduous)":    ["Not Deciduous", "Deciduous"],
        "ESD multiclass\n(label_esd)":             ["Evergreen", "Semi-Evergreen", "Deciduous"],
        "Showy-Flower\n(label_showy_flower)":      ["Not Showy", "Showy"],
        "Yellow-Showy\n(label_yellow_strict)":     ["Not Yellow-Showy", "Yellow-Showy"],
    }
    palette = ["#5B9BD5", "#ED7D31", "#70AD47"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Left: crowns by area
    ax = axes[0]
    areas = list(crowns_by_area.keys())
    counts = list(crowns_by_area.values())
    bars = ax.barh(areas[::-1], counts[::-1], color="#5B9BD5", edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xlabel("Number of Crowns")
    ax.set_title("Drone Crowns by Study Area\n(Total: 3,212)", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: label class counts
    ax = axes[1]
    y_positions = list(range(len(label_data)))
    y_labels = list(label_data.keys())
    for yi, (task, counts_list) in enumerate(label_data.items()):
        left = 0
        for ci, count in enumerate(counts_list):
            bar = ax.barh(yi, count, left=left, color=palette[ci % len(palette)],
                          edgecolor="white", height=0.6)
            if count > 10:
                ax.text(left + count / 2, yi, str(count),
                        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
            left += count
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Number of Labeled Crowns")
    ax.set_title("Usable Labels per Classifier Task\n(Class 0 / Class 1 / Class 2)", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    legend_patches = [mpatches.Patch(color=palette[i], label=f"Class {i}") for i in range(3)]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "02_data_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Feature engineering: seasons × bands diagram
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_engineering():
    years   = [2022, 2023, 2024, 2025]
    seasons = ["Winter\n(Dec–Feb)", "Pre-Monsoon\n(Mar–May)",
               "Monsoon\n(Jun–Sep)", "Post-Monsoon\n(Oct–Nov)"]
    bands   = ["B2(blue)", "B3(green)", "B4(red)", "B5", "B6", "B7",
               "B8(NIR)", "B8A", "B11(SWIR1)", "B12(SWIR2)",
               "NDVI", "GNDVI", "NDRE", "NDMI", "NBR", "EVI",
               "blue_ratio", "green_ratio", "red_ratio", "yellow_proxy"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: year × season matrix showing feature count
    ax = axes[0]
    data = np.full((len(years), len(seasons)), len(bands))
    im = ax.imshow(data, cmap="YlGn", vmin=18, vmax=22, aspect="auto")
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, fontsize=10)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=11)
    ax.set_xlabel("Season")
    ax.set_ylabel("Year")
    ax.set_title(f"Sentinel-2 Feature Table\n{len(years)} yrs × {len(seasons)} seasons × {len(bands)} features/season\n"
                 f"+ Temporal trend features across years", fontweight="bold")
    for yi in range(len(years)):
        for si in range(len(seasons)):
            ax.text(si, yi, f"{len(bands)}", ha="center", va="center",
                    fontsize=12, color="black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Features per cell", shrink=0.7)

    # Right: feature types breakdown
    ax = axes[1]
    categories = ["Optical Bands\n(B2–B12)", "Spectral\nIndices", "Seasonal\nAmplitudes",
                  "Temporal\nTrend Features\n(multi-year)"]
    counts = [10, 10, 9, 36]   # approx
    colors = ["#5B9BD5", "#70AD47", "#ED7D31", "#D7BDE2"]
    bars = ax.bar(categories, counts, color=colors, edgecolor="white", width=0.6)
    ax.bar_label(bars, padding=3, fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Features")
    ax.set_title("Feature Types per Crown\n(per season, then aggregated)", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.25)

    fig.suptitle("Feature Engineering from Sentinel-2 (STAC, Planetary Computer)", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "03_feature_engineering.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Model suite comparison: best balanced-accuracy per label (random)
# ─────────────────────────────────────────────────────────────────────────────

def fig_model_comparison():
    df = load_sweep_dir(MULTIYEAR_RANDOM)
    if df.empty:
        print("No multiyear random data found, skipping fig 4")
        return

    labels_to_show = [l for l in LABEL_NICE if l in df["label"].unique()]
    models = list(MODEL_COLORS.keys())

    fig, axes = plt.subplots(1, len(labels_to_show), figsize=(4.5 * len(labels_to_show), 5.5),
                             sharey=True)
    if len(labels_to_show) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels_to_show):
        sub = df[df["label"] == label].copy()
        # best (max balanced_accuracy) per model, aggregating over decision variant
        best = sub.groupby("model")["balanced_accuracy"].max().reindex(models).fillna(0)
        best_f1 = sub.groupby("model")["macro_f1"].max().reindex(models).fillna(0)
        x = np.arange(len(models))
        w = 0.38
        bars1 = ax.bar(x - w/2, best.values, width=w,
                       color=[MODEL_COLORS[m] for m in models], alpha=0.85,
                       edgecolor="white", label="Balanced Accuracy")
        bars2 = ax.bar(x + w/2, best_f1.values, width=w,
                       color=[MODEL_COLORS[m] for m in models], alpha=0.45,
                       edgecolor="white", hatch="///", label="Macro F1")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8, rotation=0)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", lw=1, linestyle="--", alpha=0.6)
        ax.set_title(LABEL_NICE.get(label, label), fontweight="bold", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        # mark best
        best_idx = int(best.values.argmax())
        ax.annotate(f"★ {best.values[best_idx]:.2f}",
                    xy=(best_idx - w/2, best.values[best_idx]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9, color="darkred", fontweight="bold")

    axes[0].set_ylabel("Score")
    solid = mpatches.Patch(color="gray", alpha=0.85, label="Balanced Accuracy")
    hatch = mpatches.Patch(color="gray", alpha=0.45, hatch="///", label="Macro F1")
    fig.legend(handles=[solid, hatch], loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Best Model per Classifier Task  –  Multiyear S2 2022–2025  (Random 70/30 Split)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "04_model_comparison_random.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – Spatial generalisation gap: random vs leave-area-out (acacia)
# ─────────────────────────────────────────────────────────────────────────────

def fig_spatial_gap():
    rand_df = load_sweep_dir(MULTIYEAR_RANDOM)
    hold_df = load_sweep_dir(ACACIA_HOLDOUTS)

    if rand_df.empty or hold_df.empty:
        print("Missing data for fig 5, skipping")
        return

    rand_acacia = rand_df[rand_df["label"] == "label_acacia"].copy()
    rand_best   = rand_acacia.groupby("model")["balanced_accuracy"].max()

    hold_acacia = hold_df[hold_df["label"] == "label_acacia"].copy()
    # best balanced_accuracy per holdout per model
    hold_best = hold_acacia.groupby(["holdout", "model"])["balanced_accuracy"].max().unstack(fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: random vs best holdout per model
    ax = axes[0]
    models = list(rand_best.index)
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, rand_best.values, width=w, color="#5B9BD5", label="Random split", alpha=0.85)
    # best holdout per model
    best_holdout_per_model = hold_best.max(axis=0).reindex(models).fillna(0)
    worst_holdout_per_model = hold_best.min(axis=0).reindex(models).fillna(0)
    ax.bar(x + w/2, best_holdout_per_model.values, width=w, color="#ED7D31",
           label="Best area holdout", alpha=0.85)
    ax.errorbar(x + w/2, best_holdout_per_model.values,
                yerr=[best_holdout_per_model.values - worst_holdout_per_model.values,
                      np.zeros(len(models))],
                fmt="none", color="#333", capsize=4, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", lw=1, linestyle="--", alpha=0.6)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Random vs Leave-Area-Out\n(label_acacia)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    # gap annotations
    for xi, m in zip(x, models):
        gap = rand_best[m] - best_holdout_per_model[m]
        if gap > 0.05:
            ax.annotate(f"↓{gap:.2f}", xy=(xi, rand_best[m] - gap / 2),
                        ha="center", fontsize=8, color="darkred")

    # Right: per-holdout-area barplot (best model for each)
    ax = axes[1]
    holdout_areas = sorted(hold_df[hold_df["label"] == "label_acacia"]["holdout"].dropna().unique())
    area_best_rand  = rand_acacia["balanced_accuracy"].max()
    area_bests = []
    for area in holdout_areas:
        sub = hold_df[(hold_df["label"] == "label_acacia") & (hold_df["holdout"] == area)]
        area_bests.append(sub["balanced_accuracy"].max() if not sub.empty else 0)
    y = np.arange(len(holdout_areas) + 1)
    labels_y = ["Random split"] + holdout_areas
    vals = [area_best_rand] + area_bests
    colors = ["#5B9BD5"] + ["#ED7D31"] * len(holdout_areas)
    bars = ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_y, fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.axvline(0.5, color="gray", lw=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Best Balanced Accuracy (any model)")
    ax.set_title("Best Result per Validation Scheme\n(label_acacia)", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Spatial Generalisation: Random vs Leave-Area-Out  –  Acacia Classifier",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "05_spatial_gap_acacia.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – Best confusion matrices (acacia, deciduous, esd, showy)
# ─────────────────────────────────────────────────────────────────────────────

def fig_confusion_matrices():
    rand_df = load_sweep_dir(MULTIYEAR_RANDOM)
    if rand_df.empty:
        print("No data for fig 6, skipping")
        return

    tasks = [
        ("label_acacia",       "Acacia vs Non-Acacia",   ["Non-Acacia (0)", "Acacia (1)"]),
        ("label_deciduous",    "Deciduous vs Rest",       ["Not Deciduous (0)", "Deciduous (1)"]),
        ("label_esd",          "ESD Multiclass",          ["Evergreen (0)", "Semi-Evergreen (1)", "Deciduous (2)"]),
        ("label_showy_flower", "Showy Flower vs Rest",    ["Not Showy (0)", "Showy (1)"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (label, title, class_names) in zip(axes, tasks):
        sub = rand_df[rand_df["label"] == label]
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{title}\n(no data)")
            continue
        # best row by balanced accuracy
        best_row = sub.loc[sub["balanced_accuracy"].idxmax()]
        cm = parse_cm(best_row["confusion_matrix"])
        # normalize by true (row)
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm /= row_sums

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        n = cm.shape[0]
        for i in range(n):
            for j in range(n):
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i,j]:.0%})",
                        ha="center", va="center", fontsize=10 if n <= 2 else 9,
                        color=color, fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_xticklabels(class_names, fontsize=9, rotation=20, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        bal_acc = best_row["balanced_accuracy"]
        model   = best_row["model"]
        ax.set_title(f"{title}\n{model}  |  Bal.Acc={bal_acc:.2f}", fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Recall (row-norm.)")

    fig.suptitle("Confusion Matrices – Best Model per Task  (Multiyear S2 2022–2025, Random Split)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT / "06_confusion_matrices.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 – Feature importances (from local_rf_stac, best available)
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_importances():
    tasks = [
        ("label_acacia_random_feature_importance.csv",    "Acacia vs Non-Acacia"),
        ("label_deciduous_random_feature_importance.csv", "Deciduous vs Rest"),
        ("label_esd_random_feature_importance.csv",       "ESD Multiclass"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    season_palette = {
        "postmonsoon": "#ED7D31",
        "monsoon":     "#70AD47",
        "premonsoon":  "#5B9BD5",
        "winter":      "#A9CCE3",
    }
    def feat_color(fname: str) -> str:
        for s, c in season_palette.items():
            if s in fname:
                return c
        return "#D7BDE2"  # amplitude / other

    for ax, (fname, title) in zip(axes, tasks):
        fpath = FEAT_DIR / fname
        if not fpath.exists():
            ax.axis("off")
            continue
        fi = pd.read_csv(fpath).nlargest(15, "importance")
        colors = [feat_color(f) for f in fi["feature"]]
        bars = ax.barh(fi["feature"][::-1], fi["importance"][::-1],
                       color=colors[::-1], edgecolor="white")
        ax.set_xlabel("Importance Score")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color="#ED7D31", label="Post-Monsoon"),
        mpatches.Patch(color="#70AD47", label="Monsoon"),
        mpatches.Patch(color="#5B9BD5", label="Pre-Monsoon"),
        mpatches.Patch(color="#A9CCE3", label="Winter"),
        mpatches.Patch(color="#D7BDE2", label="Amplitude / Other"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Top-15 Feature Importances by Classifier Task\n(Random Forest, 2025 Single-Year Baseline)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "07_feature_importances.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 – Multiyear improvement: 2025 single-year vs 2022-2025 multiyear
# ─────────────────────────────────────────────────────────────────────────────

def fig_multiyear_improvement():
    # single-year baseline (model_sweeps_s2_corrected)
    single_dir = PKG / "outputs" / "model_sweeps_s2_corrected"
    single_df  = load_sweep_dir(single_dir)
    multi_df   = load_sweep_dir(MULTIYEAR_RANDOM)

    if single_df.empty or multi_df.empty:
        print("Missing data for fig 8, skipping")
        return

    shared_labels = list(set(single_df["label"].unique()) & set(multi_df["label"].unique()))
    shared_labels = [l for l in ["label_acacia", "label_deciduous", "label_esd"] if l in shared_labels]

    metrics = ["balanced_accuracy", "macro_f1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5))

    for ax, metric in zip(axes, metrics):
        single_best = single_df.groupby("label")[metric].max().reindex(shared_labels)
        multi_best  = multi_df.groupby("label")[metric].max().reindex(shared_labels)
        x = np.arange(len(shared_labels))
        w = 0.35
        ax.bar(x - w/2, single_best.values, width=w, color="#A9CCE3",
               label="Single-year (2025)", edgecolor="white", alpha=0.9)
        ax.bar(x + w/2, multi_best.values,  width=w, color="#2E86C1",
               label="Multi-year (2022–2025)", edgecolor="white", alpha=0.9)
        for xi, (s_val, m_val) in enumerate(zip(single_best.values, multi_best.values)):
            delta = m_val - s_val
            if delta > 0.005:
                ax.annotate(f"+{delta:.2f}", xy=(xi + w/2, m_val),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=9, color="darkgreen", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL_NICE.get(l, l) for l in shared_labels], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", lw=1, linestyle="--", alpha=0.6)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_',' ').title()}\nSingle vs Multi-Year", fontweight="bold")
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Impact of Multi-Year Temporal Features  (S2 2025 only  vs  S2 2022–2025)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "08_multiyear_improvement.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 – Summary results table (slide-ready)
# ─────────────────────────────────────────────────────────────────────────────

def fig_results_table():
    rand_df  = load_sweep_dir(MULTIYEAR_RANDOM)
    acacia_h = load_sweep_dir(ACACIA_HOLDOUTS)
    dec_h    = load_sweep_dir(DECIDUOUS_HOLDOUTS)

    rows = []
    for label, nice in LABEL_NICE.items():
        sub = rand_df[rand_df["label"] == label] if not rand_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        best = sub.loc[sub["balanced_accuracy"].idxmax()]
        rows.append({
            "Task": nice,
            "Split": "Random 70/30",
            "Best Model": best["model"],
            "Bal. Acc.": f"{best['balanced_accuracy']:.3f}",
            "Macro F1": f"{best['macro_f1']:.3f}",
        })

    # acacia holdouts
    for holdout in sorted(acacia_h["holdout"].dropna().unique()) if not acacia_h.empty else []:
        sub = acacia_h[(acacia_h["label"] == "label_acacia") & (acacia_h["holdout"] == holdout)]
        if sub.empty:
            continue
        best = sub.loc[sub["balanced_accuracy"].idxmax()]
        rows.append({
            "Task": "Acacia vs Non-Acacia",
            "Split": f"Leave-Area-Out: {holdout}",
            "Best Model": best["model"],
            "Bal. Acc.": f"{best['balanced_accuracy']:.3f}",
            "Macro F1": f"{best['macro_f1']:.3f}",
        })

    # deciduous holdouts
    for holdout in sorted(dec_h["holdout"].dropna().unique()) if not dec_h.empty else []:
        sub = dec_h[(dec_h["label"] == "label_deciduous") & (dec_h["holdout"] == holdout)]
        if sub.empty:
            continue
        best = sub.loc[sub["balanced_accuracy"].idxmax()]
        rows.append({
            "Task": "Deciduous vs Rest",
            "Split": f"Leave-Area-Out: {holdout}",
            "Best Model": best["model"],
            "Bal. Acc.": f"{best['balanced_accuracy']:.3f}",
            "Macro F1": f"{best['macro_f1']:.3f}",
        })

    if not rows:
        print("No rows for results table, skipping")
        return

    df_table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(16, len(rows) * 0.48 + 1.5))
    ax.axis("off")

    col_widths = [0.32, 0.22, 0.18, 0.10, 0.10]
    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Header style
    for j in range(len(df_table.columns)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row zebra
    for i in range(1, len(rows) + 1):
        color = "#EBF5FB" if i % 2 == 0 else "white"
        for j in range(len(df_table.columns)):
            table[i, j].set_facecolor(color)
        # highlight balanced acc column
        val = float(df_table.iloc[i - 1]["Bal. Acc."])
        if val >= 0.85:
            table[i, 3].set_facecolor("#ABEBC6")
        elif val >= 0.70:
            table[i, 3].set_facecolor("#FAD7A0")
        elif val < 0.55:
            table[i, 3].set_facecolor("#FADBD8")

    ax.set_title("Full Results Summary  –  Multiyear Sentinel-2 2022–2025",
                 fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    path = OUT / "09_results_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 – Seasonal signal: illustrative NDVI profile (schematic from data)
# ─────────────────────────────────────────────────────────────────────────────

def fig_seasonal_signal():
    """
    Schematic seasonal NDVI profiles for three phenology types,
    based on the known ecology and consistent with our feature importance
    findings (postmonsoon is most discriminating).
    """
    seasons_x = np.arange(4)
    season_labels = ["Winter\n(Dec–Feb)", "Pre-Monsoon\n(Mar–May)",
                     "Monsoon\n(Jun–Sep)", "Post-Monsoon\n(Oct–Nov)"]

    profiles = {
        "Evergreen\n(e.g. Neem, Peepal)":        [0.72, 0.65, 0.78, 0.75],
        "Semi-Evergreen\n(e.g. Pilkhan, Jamun)":  [0.58, 0.50, 0.75, 0.70],
        "Deciduous\n(e.g. Amaltas, Siris)":       [0.35, 0.25, 0.72, 0.55],
        "Acacia\n(Prosopis Juliflora)":            [0.55, 0.60, 0.68, 0.60],
    }
    colors = ["#2E86C1", "#1ABC9C", "#E67E22", "#884EA0"]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: NDVI profiles
    ax = axes[0]
    for (species, vals), color, marker in zip(profiles.items(), colors, markers):
        ax.plot(seasons_x, vals, "o-", color=color, marker=marker,
                linewidth=2.5, markersize=9, label=species)
        ax.fill_between(seasons_x, vals, alpha=0.08, color=color)
    ax.set_xticks(seasons_x)
    ax.set_xticklabels(season_labels, fontsize=10)
    ax.set_ylabel("Approximate NDVI", fontsize=12)
    ax.set_ylim(0.1, 0.95)
    ax.set_title("Indicative Seasonal NDVI Profiles\nby Phenology Type", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("Post-monsoon is\nmost discriminating\n(highest feature importance)",
                xy=(3, 0.55), xytext=(2.1, 0.35),
                fontsize=9, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2))

    # Right: EVI amplitude illustration
    ax = axes[1]
    amp_data = {
        "Evergreen":    0.06,
        "Semi-Evg.":   0.25,
        "Deciduous":   0.47,
        "Acacia":      0.13,
        "Showy Flower\n(peak bloom)": 0.38,
    }
    amp_colors = ["#2E86C1", "#1ABC9C", "#E67E22", "#884EA0", "#CB4335"]
    bars = ax.bar(list(amp_data.keys()), list(amp_data.values()),
                  color=amp_colors, edgecolor="white", alpha=0.85)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=11, fontweight="bold")
    ax.set_ylabel("Approx. Seasonal EVI Amplitude\n(max – min across seasons)")
    ax.set_title("EVI Amplitude as Phenology Discriminator\n(top feature for multiple tasks)",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 0.65)

    fig.suptitle("Why Seasonal & Temporal Features Work  –  Phenological Signal in Satellite Data",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "10_seasonal_signal.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output directory: {OUT}\n")
    fig_pipeline()
    fig_data_overview()
    fig_feature_engineering()
    fig_model_comparison()
    fig_spatial_gap()
    fig_confusion_matrices()
    fig_feature_importances()
    fig_multiyear_improvement()
    fig_results_table()
    fig_seasonal_signal()
    print("\nAll presentation figures saved.")
