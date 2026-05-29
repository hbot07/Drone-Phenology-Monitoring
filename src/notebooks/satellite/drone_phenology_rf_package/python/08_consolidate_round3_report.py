#!/usr/bin/env python3
"""Build the round-3 satellite RF experiment report."""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parents[3]
OUT_ROOT = REPO_ROOT / "output" / "satellite_rf_experiments_round3"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"

LABEL_TABLE = PKG_ROOT / "data" / "crown_label_table.csv"
SPECIES_COUNTS = PKG_ROOT / "outputs" / "species_counts_by_area.csv"
FEATURE_TABLES = {
    "S2 2025": PKG_ROOT / "exports" / "stac_s2_features_2025_buffer20_items4_label_acacia.csv",
    "S2 2024+2025": PKG_ROOT / "exports" / "stac_s2_features_2024_2025_buffer20_items4_label_acacia.csv",
    "S1 2022-2025 + S2 2024+2025": PKG_ROOT / "exports" / "fused_s1_2022_2025_s2_2024_2025_label_acacia.csv",
    "SV-only S2 2024+2025": PKG_ROOT / "exports" / "sv_only_s2_2024_2025_label_acacia.csv",
    "SV-only S1+S2": PKG_ROOT / "exports" / "sv_only_fused_s1_s2_2024_2025_label_acacia.csv",
}

MODEL_DIRS = {
    "S2 2025 corrected": PKG_ROOT / "outputs" / "model_sweeps_s2_corrected",
    "S2 2024+2025": PKG_ROOT / "outputs" / "model_sweeps_s2_2024_2025",
    "S1 2022-2025 + S2 2024+2025": PKG_ROOT / "outputs" / "model_sweeps_fused_s1_s2_2024_2025",
    "SV-only S2 2024+2025": PKG_ROOT / "outputs" / "model_sweeps_sv_only_s2_2024_2025",
    "SV-only S1+S2": PKG_ROOT / "outputs" / "model_sweeps_sv_only_fused_s1_s2_2024_2025",
}

LABEL_NAMES = {
    "label_esd": {-1: "ignore", 0: "evergreen", 1: "semi-evergreen", 2: "deciduous"},
    "label_deciduous": {-1: "ignore", 0: "not deciduous", 1: "deciduous"},
    "label_acacia": {-1: "ignore", 0: "non-Acacia", 1: "Acacia"},
    "label_yellow_strict": {-1: "ignore", 0: "not yellow-showy", 1: "yellow-showy"},
    "label_yellow_broad": {-1: "ignore", 0: "not yellow", 1: "yellow"},
    "label_red_showy": {-1: "ignore", 0: "not red-showy", 1: "red-showy"},
    "label_showy_flower": {-1: "ignore", 0: "not showy", 1: "showy"},
}


def ensure_dirs() -> None:
    for path in [OUT_ROOT, FIG_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def parse_cell(value):
    if pd.isna(value):
        return value
    if isinstance(value, str) and value.startswith(("{", "[")):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    return value


def load_sweeps() -> pd.DataFrame:
    frames = []
    for family, folder in MODEL_DIRS.items():
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*_model_sweep.csv")):
            df = pd.read_csv(path)
            df["feature_family"] = family
            df["sweep_file"] = str(path.relative_to(PKG_ROOT))
            for col in ["train_counts", "test_counts", "confusion_matrix", "labels"]:
                if col in df.columns:
                    df[col] = df[col].map(parse_cell)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(TABLE_DIR / "all_model_sweeps.csv", index=False)
    best = (
        out.sort_values(["balanced_accuracy", "macro_f1", "accuracy"], ascending=False)
        .groupby(["feature_family", "label", "split", "holdout"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(TABLE_DIR / "best_by_feature_label_split.csv", index=False)
    return out


def summarize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, names in LABEL_NAMES.items():
        counts = labels[col].value_counts().sort_index()
        for value, count in counts.items():
            rows.append(
                {
                    "classifier": col,
                    "value": int(value),
                    "meaning": names.get(int(value), str(value)),
                    "count": int(count),
                    "usable": int(value) != -1,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "label_counts.csv", index=False)
    usable = out[out["usable"]].copy()

    fig, axes = plt.subplots(4, 2, figsize=(13, 13), dpi=180)
    axes = axes.ravel()
    for ax, (name, group) in zip(axes, usable.groupby("classifier", sort=False)):
        ax.bar(group["meaning"], group["count"], color="#4477AA")
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    for ax in axes[usable["classifier"].nunique() :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "usable_label_counts.png")
    plt.close(fig)
    return out


def summarize_acacia(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = labels[labels["label_acacia"] != -1].copy()
    area = (
        usable.pivot_table(index="area", columns="label_acacia", values="crown_uid", aggfunc="count", fill_value=0)
        .rename(columns={0: "non_acacia", 1: "acacia"})
        .reset_index()
    )
    for col in ["non_acacia", "acacia"]:
        if col not in area:
            area[col] = 0
    area["total_usable"] = area["non_acacia"] + area["acacia"]
    area["acacia_share"] = area["acacia"] / area["total_usable"]
    area.to_csv(TABLE_DIR / "acacia_counts_by_area.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=180)
    x = np.arange(len(area))
    ax.bar(x, area["non_acacia"], label="non-Acacia", color="#88CCEE")
    ax.bar(x, area["acacia"], bottom=area["non_acacia"], label="Acacia", color="#CC6677")
    ax.set_xticks(x)
    ax.set_xticklabels(area["area"], rotation=45, ha="right")
    ax.set_ylabel("usable crowns")
    ax.set_title("Acacia Labels by Area")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acacia_counts_by_area.png")
    plt.close(fig)

    positive = usable[usable["label_acacia"] == 1].copy()
    positive["species_for_plot"] = positive["species_clean"].fillna("<trusted tree_type Acacia>")
    species_area = (
        positive.groupby(["species_for_plot", "area"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    species_area.to_csv(TABLE_DIR / "acacia_positive_species_by_area.csv", index=False)
    pivot = species_area.pivot_table(index="species_for_plot", columns="area", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.45 * len(pivot) + 1.5)), dpi=180)
    left = np.zeros(len(pivot))
    palette = ["#332288", "#117733", "#44AA99", "#DDCC77", "#CC6677", "#AA4499"]
    for idx, col in enumerate(pivot.columns):
        ax.barh(pivot.index, pivot[col], left=left, label=col, color=palette[idx % len(palette)])
        left += pivot[col].to_numpy()
    ax.set_xlabel("positive Acacia crowns")
    ax.set_title("Acacia Positives: Species and Area")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acacia_positive_species_by_area.png")
    plt.close(fig)
    return area, species_area


def summarize_features() -> pd.DataFrame:
    rows = []
    for name, path in FEATURE_TABLES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rows.append(
            {
                "feature_table": name,
                "path": str(path.relative_to(PKG_ROOT)),
                "rows": len(df),
                "columns": len(df.columns),
                "usable_acacia_rows": int((df["label_acacia"] != -1).sum()) if "label_acacia" in df else np.nan,
                "acacia_positive_rows": int((df["label_acacia"] == 1).sum()) if "label_acacia" in df else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "feature_tables_summary.csv", index=False)
    return out


def plot_model_summaries(sweeps: pd.DataFrame) -> pd.DataFrame:
    best = pd.read_csv(TABLE_DIR / "best_by_feature_label_split.csv")
    acacia = best[best["label"] == "label_acacia"].copy()
    acacia["experiment"] = acacia["feature_family"] + " | " + acacia["split"] + " | " + acacia["holdout"].fillna("")
    acacia = acacia.sort_values("balanced_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.32 * len(acacia) + 1.5)), dpi=180)
    ax.barh(acacia["experiment"], acacia["balanced_accuracy"], color="#4477AA")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("best balanced accuracy")
    ax.set_title("Best Acacia Result per Experiment")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acacia_best_by_experiment.png")
    plt.close(fig)

    sv = acacia[acacia["feature_family"].str.startswith("SV-only")].copy()
    sv["holdout_label"] = sv["holdout"].replace("", "random")
    pivot = sv.pivot_table(index="holdout_label", columns="feature_family", values="balanced_accuracy", aggfunc="max")
    order = ["random", "SV_S1", "SV_S2", "SV_S3", "SV_S4"]
    pivot = pivot.reindex([x for x in order if x in pivot.index])
    pivot.to_csv(TABLE_DIR / "sv_only_holdout_comparison.csv")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    pivot.plot(kind="bar", ax=ax, color=["#4477AA", "#CC6677"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("best balanced accuracy")
    ax.set_title("SV-only Acacia Transfer Tests")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sv_only_holdout_comparison.png")
    plt.close(fig)
    return best


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.to_numpy(dtype=str)]
    return "\n".join([header, sep, *rows])


def write_report(
    labels: pd.DataFrame,
    label_counts: pd.DataFrame,
    acacia_area: pd.DataFrame,
    feature_summary: pd.DataFrame,
    best: pd.DataFrame,
) -> None:
    total = len(labels)
    clean = int((labels["species_status"] == "clean").sum())
    missing = int((labels["species_status"] == "missing").sum())
    ambiguous = int(labels["species_status"].isin(["ambiguous_or_unknown", "acacia_unknown"]).sum())
    acacia_pos = int((labels["label_acacia"] == 1).sum())
    acacia_sv = int(labels[(labels["label_acacia"] == 1) & (labels["area"].str.startswith("SV"))].shape[0])
    non_acacia_sv = int(labels[(labels["label_acacia"] == 0) & (labels["area"].str.startswith("SV"))].shape[0])
    non_acacia = int((labels["label_acacia"] == 0).sum())

    def best_line(family: str, split: str, holdout: str = "") -> str:
        rows = best[
            (best["feature_family"] == family)
            & (best["label"] == "label_acacia")
            & (best["split"] == split)
            & (best["holdout"].fillna("") == holdout)
        ]
        if rows.empty:
            return "not run"
        row = rows.sort_values("balanced_accuracy", ascending=False).iloc[0]
        return f"{row['balanced_accuracy']:.3f} BA, {row['accuracy']:.3f} acc, {row['macro_f1']:.3f} macro-F1 ({row['model']}, {row['decision']})"

    classifier_table = (
        label_counts[label_counts["usable"]]
        .assign(label=lambda d: d["meaning"] + "=" + d["count"].astype(str))
        .groupby("classifier")["label"]
        .apply(lambda s: ", ".join(s))
        .reset_index()
    )

    report = f"""# Satellite RF Experiments Round 3

## Data audit

- Total crown rows in the merged IITD + Sanjay Van table: **{total}**.
- Clean species labels: **{clean}**. Missing species labels: **{missing}**. Ambiguous/unknown species labels: **{ambiguous}**.
- Corrected Acacia labels: **{acacia_pos} Acacia** and **{non_acacia} non-Acacia** usable rows.
- Acacia geography is the central caveat: **{acacia_sv}/{acacia_pos} Acacia positives ({100 * acacia_sv / acacia_pos:.1f}%) are from Sanjay Van**, while only **{non_acacia_sv}/{non_acacia} non-Acacia rows ({100 * non_acacia_sv / non_acacia:.1f}%) are from Sanjay Van**.
- I removed the likely erroneous `species_clean=Neem` Acacia positives by making clean species override `tree_type_raw=Acacia`. I still trust `tree_type_raw=Acacia` when the species is missing/ambiguous, as requested.

## Classifier label inventory

{markdown_table(classifier_table)}

## Feature tables

{markdown_table(feature_summary)}

## Acacia distribution by area

{markdown_table(acacia_area)}

## Main results

These numbers are balanced accuracy first, because the positive/negative counts are uneven.

| Experiment | Best result |
|---|---:|
| All areas, S2 2025, random split | {best_line("S2 2025 corrected", "random")} |
| All areas, S2 2024+2025, random split | {best_line("S2 2024+2025", "random")} |
| All areas, S1+S2, random split | {best_line("S1 2022-2025 + S2 2024+2025", "random")} |
| SV-only, S2 2024+2025, random split | {best_line("SV-only S2 2024+2025", "random")} |
| SV-only, S1+S2, random split | {best_line("SV-only S1+S2", "random")} |
| SV-only S2, hold out SV_S1 | {best_line("SV-only S2 2024+2025", "leave_area_out", "SV_S1")} |
| SV-only S2, hold out SV_S2 | {best_line("SV-only S2 2024+2025", "leave_area_out", "SV_S2")} |
| SV-only S2, hold out SV_S3 | {best_line("SV-only S2 2024+2025", "leave_area_out", "SV_S3")} |
| SV-only S2, hold out SV_S4 | {best_line("SV-only S2 2024+2025", "leave_area_out", "SV_S4")} |
| SV-only S1+S2, hold out SV_S1 | {best_line("SV-only S1+S2", "leave_area_out", "SV_S1")} |
| SV-only S1+S2, hold out SV_S2 | {best_line("SV-only S1+S2", "leave_area_out", "SV_S2")} |
| SV-only S1+S2, hold out SV_S3 | {best_line("SV-only S1+S2", "leave_area_out", "SV_S3")} |
| SV-only S1+S2, hold out SV_S4 | {best_line("SV-only S1+S2", "leave_area_out", "SV_S4")} |

## Interpretation

The best random-split Acacia score is strong enough to show that the satellite features contain useful signal. The more honest SV-only leave-area-out tests are mixed: SV_S3 transfers well, SV_S4 is usable but imperfect, and SV_S1/SV_S2 are weak-to-moderate depending on the feature set. This means the current Acacia model is **not yet a robust general Acacia detector**; it is closer to a promising Sanjay Van / Prosopis-heavy detector that still needs more geographically balanced positives.

Adding Sentinel-1 radar and 2024+2025 optical features did not uniformly improve transfer. That is useful negative evidence: the bottleneck is probably not just cloud gaps or missing bands, but the fact that positives are spatially clustered and mostly one Acacia-like species group.

## Figures

- `figures/acacia_counts_by_area.png`
- `figures/acacia_positive_species_by_area.png`
- `figures/usable_label_counts.png`
- `figures/acacia_best_by_experiment.png`
- `figures/sv_only_holdout_comparison.png`

## Tables

- `tables/label_counts.csv`
- `tables/acacia_counts_by_area.csv`
- `tables/acacia_positive_species_by_area.csv`
- `tables/feature_tables_summary.csv`
- `tables/all_model_sweeps.csv`
- `tables/best_by_feature_label_split.csv`
- `tables/sv_only_holdout_comparison.csv`
"""
    (OUT_ROOT / "REPORT.md").write_text(report)


def main() -> None:
    ensure_dirs()
    labels = pd.read_csv(LABEL_TABLE)
    label_counts = summarize_labels(labels)
    acacia_area, _ = summarize_acacia(labels)
    if SPECIES_COUNTS.exists():
        species_counts = pd.read_csv(SPECIES_COUNTS)
        species_counts.to_csv(TABLE_DIR / "species_counts_by_area.csv", index=False)
    feature_summary = summarize_features()
    sweeps = load_sweeps()
    if sweeps.empty:
        raise SystemExit("No model sweep files found")
    best = plot_model_summaries(sweeps)
    write_report(labels, label_counts, acacia_area, feature_summary, best)
    print(f"Wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
