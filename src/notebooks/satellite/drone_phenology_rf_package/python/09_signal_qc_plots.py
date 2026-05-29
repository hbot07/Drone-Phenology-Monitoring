#!/usr/bin/env python3
"""Create crown-level satellite signal sanity plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parents[3]
OUT_ROOT = REPO_ROOT / "output" / "satellite_rf_signal_qc"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"

FUSED = PKG_ROOT / "exports" / "fused_s1_2022_2025_s2_2024_2025_label_acacia.csv"
SV_ONLY = PKG_ROOT / "exports" / "sv_only_fused_s1_s2_2024_2025_label_acacia.csv"

SEASONS = ["winter", "premonsoon", "monsoon", "postmonsoon"]
SEASON_LABELS = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
YEARS = [2024, 2025]
S2_INDICES = ["NDVI", "GNDVI", "NDRE", "NDMI", "EVI", "yellow_proxy"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def ensure_dirs() -> None:
    for path in [OUT_ROOT, FIG_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def s2_col(year: int, season: str, signal: str) -> str:
    return f"y{year}_{season}_{signal}"


def signal_matrix(row: pd.Series, signal: str) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        for season, label in zip(SEASONS, SEASON_LABELS):
            col = s2_col(year, season, signal)
            rows.append(
                {
                    "year": year,
                    "season": season,
                    "season_label": label,
                    "x": f"{year} {label}",
                    "value": row.get(col, np.nan),
                }
            )
    return pd.DataFrame(rows)


def choose_examples(df: pd.DataFrame) -> pd.DataFrame:
    usable = df[df["label_acacia"].isin([0, 1])].copy()
    examples = []
    rng = np.random.default_rng(7)
    for area, group in usable.groupby("area"):
        for label in [1, 0]:
            sub = group[group["label_acacia"] == label].copy()
            if sub.empty:
                continue
            sub["missing_s2"] = sub[[s2_col(y, s, "NDVI") for y in YEARS for s in SEASONS]].isna().mean(axis=1)
            sub["ndvi_amp_2yr"] = sub[[s2_col(y, s, "NDVI") for y in YEARS for s in SEASONS]].max(axis=1) - sub[
                [s2_col(y, s, "NDVI") for y in YEARS for s in SEASONS]
            ].min(axis=1)
            sub = sub.sort_values(["missing_s2", "ndvi_amp_2yr"], ascending=[True, False])
            examples.append(sub.head(2))
            if len(sub) > 2:
                examples.append(sub.sample(min(1, len(sub) - 2), random_state=int(rng.integers(0, 99999))))
    out = pd.concat(examples, ignore_index=True).drop_duplicates("crown_uid")
    cols = ["crown_uid", "area", "species_clean", "species_raw", "tree_type_raw", "label_acacia", "missing_s2", "ndvi_amp_2yr"]
    out[cols].to_csv(TABLE_DIR / "signal_qc_example_crowns.csv", index=False)
    return out


def plot_individual_traces(examples: pd.DataFrame) -> None:
    plot_cols = ["NDVI", "NDMI", "EVI", "yellow_proxy"]
    for _, row in examples.iterrows():
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=180, sharex=True)
        axes = axes.ravel()
        for ax, signal in zip(axes, plot_cols):
            mat = signal_matrix(row, signal)
            ax.plot(range(len(mat)), mat["value"], marker="o", linewidth=1.8, color="#4477AA")
            ax.set_title(signal)
            ax.grid(alpha=0.25)
            ax.set_xticks(range(len(mat)))
            ax.set_xticklabels(mat["x"], rotation=35, ha="right", fontsize=7)
        label = "Acacia" if int(row["label_acacia"]) == 1 else "non-Acacia"
        species = row["species_clean"] if pd.notna(row["species_clean"]) else row["species_raw"]
        fig.suptitle(f"{row['crown_uid']} | {row['area']} | {species} | {label}", fontsize=11)
        fig.tight_layout()
        safe = str(row["crown_uid"]).replace(":", "_").replace("/", "_")
        fig.savefig(FIG_DIR / f"individual_signal_{safe}.png")
        plt.close(fig)


def plot_class_summary(df: pd.DataFrame, source_name: str, stem: str) -> None:
    usable = df[df["label_acacia"].isin([0, 1])].copy()
    records = []
    for signal in S2_INDICES:
        for year in YEARS:
            for season, label in zip(SEASONS, SEASON_LABELS):
                col = s2_col(year, season, signal)
                if col not in usable:
                    continue
                for class_value, group in usable.groupby("label_acacia"):
                    values = group[col].dropna()
                    if values.empty:
                        continue
                    records.append(
                        {
                            "signal": signal,
                            "year": year,
                            "season": season,
                            "x": f"{year} {label}",
                            "class": "Acacia" if class_value == 1 else "non-Acacia",
                            "mean": values.mean(),
                            "p25": values.quantile(0.25),
                            "p75": values.quantile(0.75),
                            "n": len(values),
                        }
                    )
    long = pd.DataFrame(records)
    long.to_csv(TABLE_DIR / f"{stem}_seasonal_signal_summary.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), dpi=180, sharex=True)
    axes = axes.ravel()
    colors = {"Acacia": "#CC6677", "non-Acacia": "#4477AA"}
    for ax, signal in zip(axes, S2_INDICES):
        sig = long[long["signal"] == signal].copy()
        order = sig["x"].drop_duplicates().tolist()
        x_positions = np.arange(len(order))
        for cls, group in sig.groupby("class"):
            group = group.set_index("x").reindex(order)
            ax.plot(x_positions, group["mean"], marker="o", label=cls, color=colors[cls])
            ax.fill_between(x_positions, group["p25"], group["p75"], color=colors[cls], alpha=0.18)
        ax.set_title(signal)
        ax.grid(alpha=0.25)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=7)
    axes[0].legend()
    fig.suptitle(f"{source_name}: Acacia vs non-Acacia seasonal Sentinel-2 signals", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}_class_seasonal_signals.png")
    plt.close(fig)


def plot_area_summary(df: pd.DataFrame) -> None:
    usable = df[df["label_acacia"].isin([0, 1])].copy()
    rows = []
    for area, group in usable.groupby("area"):
        for label, sub in group.groupby("label_acacia"):
            for signal in ["NDVI", "NDMI", "EVI", "yellow_proxy"]:
                vals = [sub[s2_col(y, s, signal)].mean() for y in YEARS for s in SEASONS]
                rows.append(
                    {
                        "area": area,
                        "class": "Acacia" if label == 1 else "non-Acacia",
                        "signal": signal,
                        "mean_all_seasons": np.nanmean(vals),
                        "amp_all_seasons": np.nanmax(vals) - np.nanmin(vals),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "area_class_signal_summary.csv", index=False)

    for signal in ["NDVI", "NDMI", "EVI", "yellow_proxy"]:
        sub = summary[summary["signal"] == signal]
        pivot = sub.pivot_table(index="area", columns="class", values="mean_all_seasons")
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
        pivot.plot(kind="bar", ax=ax, color=["#CC6677", "#4477AA"])
        ax.set_title(f"Mean {signal} by Area and Class")
        ax.set_ylabel(signal)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"area_class_mean_{signal}.png")
        plt.close(fig)


def plot_s1_summary(df: pd.DataFrame) -> None:
    usable = df[df["label_acacia"].isin([0, 1])].copy()
    s1_signals = ["vv_db", "vh_db", "vh_vv_ratio", "vh_minus_vv_db"]
    rows = []
    for signal in s1_signals:
        for season, label in zip(SEASONS, SEASON_LABELS):
            col = f"s1_{season}_{signal}_median"
            if col not in usable:
                continue
            for class_value, group in usable.groupby("label_acacia"):
                vals = group[col].dropna()
                rows.append(
                    {
                        "signal": signal,
                        "season": label,
                        "class": "Acacia" if class_value == 1 else "non-Acacia",
                        "mean": vals.mean(),
                        "p25": vals.quantile(0.25),
                        "p75": vals.quantile(0.75),
                        "n": len(vals),
                    }
                )
    long = pd.DataFrame(rows)
    long.to_csv(TABLE_DIR / "sv_only_s1_seasonal_summary.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
    axes = axes.ravel()
    colors = {"Acacia": "#CC6677", "non-Acacia": "#4477AA"}
    for ax, signal in zip(axes, s1_signals):
        sig = long[long["signal"] == signal]
        order = SEASON_LABELS
        x = np.arange(len(order))
        for cls, group in sig.groupby("class"):
            group = group.set_index("season").reindex(order)
            ax.plot(x, group["mean"], marker="o", color=colors[cls], label=cls)
            ax.fill_between(x, group["p25"], group["p75"], color=colors[cls], alpha=0.18)
        ax.set_title(signal)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("SV-only Sentinel-1 seasonal radar summaries")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sv_only_s1_seasonal_signals.png")
    plt.close(fig)


def plot_valid_pixels_and_missingness(df: pd.DataFrame) -> None:
    rows = []
    for year in YEARS:
        for season, label in zip(SEASONS, SEASON_LABELS):
            for metric in ["valid_pixels_median", "valid_pixels_max", "NDVI", "NDMI", "EVI"]:
                col = s2_col(year, season, metric)
                if col not in df:
                    continue
                rows.append(
                    {
                        "column": col,
                        "year": year,
                        "season": label,
                        "metric": metric,
                        "missing_fraction": df[col].isna().mean(),
                        "median": df[col].median(),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "s2_missingness_and_valid_pixels.csv", index=False)

    valid = out[out["metric"].isin(["valid_pixels_median", "valid_pixels_max"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    valid["x"] = valid["year"].astype(str) + " " + valid["season"] + " " + valid["metric"].str.replace("valid_pixels_", "")
    ax.bar(valid["x"], valid["median"], color="#44AA99")
    ax.set_title("Median Valid Pixels Inside 20 m Buffers")
    ax.set_ylabel("pixels")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s2_valid_pixels_by_season.png")
    plt.close(fig)

    missing = out[out["metric"].isin(["NDVI", "NDMI", "EVI"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    missing["x"] = missing["year"].astype(str) + " " + missing["season"] + " " + missing["metric"]
    ax.bar(missing["x"], missing["missing_fraction"], color="#DDCC77")
    ax.set_ylim(0, 1)
    ax.set_title("Missing Fraction for Core Sentinel-2 Indices")
    ax.set_ylabel("missing fraction")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s2_index_missingness.png")
    plt.close(fig)


def plot_spectral_signatures(df: pd.DataFrame) -> None:
    usable = df[df["label_acacia"].isin([0, 1])].copy()
    for year in YEARS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180, sharey=True)
        axes = axes.ravel()
        for ax, season, label in zip(axes, SEASONS, SEASON_LABELS):
            x = np.arange(len(S2_BANDS))
            for class_value, cls, color in [(1, "Acacia", "#CC6677"), (0, "non-Acacia", "#4477AA")]:
                sub = usable[usable["label_acacia"] == class_value]
                means = [sub[s2_col(year, season, band)].mean() for band in S2_BANDS]
                ax.plot(x, means, marker="o", label=cls, color=color)
            ax.set_title(label)
            ax.set_xticks(x)
            ax.set_xticklabels(S2_BANDS, rotation=45)
            ax.grid(alpha=0.25)
        axes[0].legend()
        fig.suptitle(f"SV-only mean Sentinel-2 spectral signature, {year}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"sv_only_s2_spectral_signature_{year}.png")
        plt.close(fig)


def write_readme(examples: pd.DataFrame) -> None:
    text = f"""# Satellite Signal QC

These plots are for visual inspection of whether extracted satellite features look plausible at crown level.

Use these first:

- `figures/sv_only_class_seasonal_signals.png`: Acacia vs non-Acacia seasonal Sentinel-2 index curves for the SV-only subset.
- `figures/all_usable_class_seasonal_signals.png`: same curves for all usable Acacia rows.
- `figures/sv_only_s1_seasonal_signals.png`: Sentinel-1 radar seasonal summaries.
- `figures/s2_valid_pixels_by_season.png`: checks whether 20 m buffers are getting enough valid pixels.
- `figures/s2_index_missingness.png`: checks seasonal missingness after cloud masking.
- `figures/individual_signal_*.png`: individual crown traces for {len(examples)} selected crowns.

What to look for:

- NDVI/GNDVI/EVI should generally stay in vegetation-like ranges and should not jump randomly every season for every crown.
- Acacia and non-Acacia curves do not need to be perfectly separated; if they overlap, that explains limited classifier transfer.
- Very low valid pixels or high missingness would mean the model is learning imputation artifacts.
- If one area has a different baseline from another, leave-area-out performance will be harder than random split.
"""
    (OUT_ROOT / "README.md").write_text(text)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(FUSED)
    sv = pd.read_csv(SV_ONLY)
    examples = choose_examples(sv)
    plot_individual_traces(examples)
    plot_class_summary(sv, "SV-only", "sv_only")
    plot_class_summary(df[df["label_acacia"].isin([0, 1])], "All usable Acacia rows", "all_usable")
    plot_area_summary(df)
    plot_s1_summary(sv)
    plot_valid_pixels_and_missingness(df)
    plot_spectral_signatures(sv)
    write_readme(examples)
    print(f"Wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
