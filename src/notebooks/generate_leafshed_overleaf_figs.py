#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    # This file lives in <root>/src/notebooks/
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    output_dir: Path
    fig_dir: Path

    features_csv: Path
    phenophase_csv: Path
    classification_csv: Path


def _paths(fig_dir: str | None) -> Paths:
    root = _project_root()
    output_dir = root / "output"
    default_fig_dir = root / "src" / "notebooks" / "leafshed_classifier_fig"
    fig_dir_path = Path(fig_dir).expanduser() if fig_dir else default_fig_dir

    return Paths(
        root=root,
        output_dir=output_dir,
        fig_dir=fig_dir_path,
        features_csv=output_dir / "consensus_phenology_features_raw.csv",
        phenophase_csv=output_dir / "leaf_shed_phenophase_per_om.csv",
        classification_csv=output_dir / "leaf_shed_classification.csv",
    )


def _require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def _savefig(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")


def make_pipeline_schematic(out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 2.8))
    ax.set_axis_off()

    steps = [
        "Consensus\nCrown Chips",
        "Signals\nveg_fraction_hsv\n+ gcc_mean",
        "Interpolation\n+ per-tree\nnormalization",
        "Component\nScores",
        "Deciduousness\nScore (DS)",
        "Phenophase\nlabels + events",
    ]

    x0 = 0.02
    x_gap = 0.015
    box_w = (0.96 - x0) / len(steps) - x_gap
    box_h = 0.58
    y = 0.22

    boxes = []
    for i, label in enumerate(steps):
        x = x0 + i * (box_w + x_gap)
        box = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            facecolor="white",
            edgecolor="black",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x + box_w / 2,
            y + box_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            transform=ax.transAxes,
        )
        boxes.append((x, y, box_w, box_h))

    # arrows
    for i in range(len(boxes) - 1):
        x, y, w, h = boxes[i]
        nx, ny, nw, nh = boxes[i + 1]
        ax.annotate(
            "",
            xy=(nx, ny + nh / 2),
            xytext=(x + w, y + h / 2),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        )

    ax.set_title("Leaf-shed classifier workflow (schematic)", fontsize=12)
    _savefig(out_path)
    plt.close(fig)


def make_phenophase_counts(phenophase_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    required = {"om_id", "phenophase"}
    missing = required - set(phenophase_df.columns)
    if missing:
        raise ValueError(f"phenophase_per_om is missing columns: {sorted(missing)}")

    # Count rows per OM and phenophase.
    counts = (
        phenophase_df.groupby(["om_id", "phenophase"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )

    # Keep a stable, readable order.
    phase_order = ["leaf_on", "transitioning", "leaf_off", "stable"]
    present_phases = [p for p in phase_order if p in set(counts["phenophase"])]
    other_phases = sorted(set(counts["phenophase"]) - set(present_phases))
    phases = present_phases + other_phases

    pivot = (
        counts.pivot(index="om_id", columns="phenophase", values="count")
        .fillna(0)
        .astype(int)
    )

    # Ensure all phases are present as columns for consistent stacked ordering.
    for p in phases:
        if p not in pivot.columns:
            pivot[p] = 0
    pivot = pivot[phases]

    # Sort by OM id (numeric if possible).
    try:
        pivot = pivot.sort_index(key=lambda s: s.astype(float))
    except Exception:
        pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(14, 4.2))
    bottom = np.zeros(len(pivot), dtype=float)

    x = np.arange(len(pivot))
    for phase in phases:
        ax.bar(x, pivot[phase].to_numpy(), bottom=bottom, label=phase)
        bottom += pivot[phase].to_numpy()

    ax.set_title("Phenophase counts per orthomosaic (OM)")
    ax.set_xlabel("OM index")
    ax.set_ylabel("# trees")

    # Ticks: show OM ids but avoid illegibility.
    om_labels = [str(v) for v in pivot.index.tolist()]
    max_ticks = 24
    if len(om_labels) > max_ticks:
        step = int(np.ceil(len(om_labels) / max_ticks))
        tick_idx = np.arange(0, len(om_labels), step)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([om_labels[i] for i in tick_idx], rotation=0)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(om_labels, rotation=0)

    ax.legend(ncol=min(4, len(phases)), frameon=False)
    ax.margins(x=0.01)

    _savefig(out_path)
    plt.close(fig)


def _pick_case_trees(classification_df: pd.DataFrame) -> tuple[str, str]:
    required = {"chain_id", "is_deciduous", "deciduousness_score"}
    missing = required - set(classification_df.columns)
    if missing:
        raise ValueError(f"leaf_shed_classification is missing columns: {sorted(missing)}")

    df = classification_df.copy()
    # Coerce to numeric where appropriate.
    df["is_deciduous"] = pd.to_numeric(df["is_deciduous"], errors="coerce")
    df["deciduousness_score"] = pd.to_numeric(df["deciduousness_score"], errors="coerce")

    decid = df[(df["is_deciduous"] == 1) & df["deciduousness_score"].notna()]
    everg = df[(df["is_deciduous"] == 0) & df["deciduousness_score"].notna()]

    if len(decid) == 0 or len(everg) == 0:
        raise ValueError("Not enough deciduous/evergreen trees to build case studies")

    decid_id = str(decid.sort_values("deciduousness_score", ascending=False).iloc[0]["chain_id"])
    everg_id = str(everg.sort_values("deciduousness_score", ascending=True).iloc[0]["chain_id"])
    return decid_id, everg_id


def _prep_tree_timeseries(features_df: pd.DataFrame, chain_id: str) -> pd.DataFrame:
    required = {"chain_id", "om_id", "veg_fraction_hsv", "gcc_mean"}
    missing = required - set(features_df.columns)
    if missing:
        raise ValueError(f"consensus_phenology_features_raw is missing columns: {sorted(missing)}")

    df = features_df[features_df["chain_id"].astype(str) == str(chain_id)].copy()
    if len(df) == 0:
        raise ValueError(f"No rows in features for chain_id={chain_id}")

    # numeric coercion
    df["om_id"] = pd.to_numeric(df["om_id"], errors="coerce")
    df["veg_fraction_hsv"] = pd.to_numeric(df["veg_fraction_hsv"], errors="coerce")
    df["gcc_mean"] = pd.to_numeric(df["gcc_mean"], errors="coerce")

    df = df.dropna(subset=["om_id"]).sort_values("om_id")

    # Interpolate signals over OM index to make the plot smoother and consistent.
    df = df.set_index("om_id")[["veg_fraction_hsv", "gcc_mean"]]
    df = df.sort_index()

    full_index = pd.Index(sorted(df.index.unique()), name="om_id")
    df = df.reindex(full_index)
    df = df.interpolate(method="linear", limit_direction="both")

    df = df.reset_index()
    return df


def _phenophase_for_tree(phenophase_df: pd.DataFrame, chain_id: str) -> pd.DataFrame:
    required = {"chain_id", "om_id", "phenophase"}
    missing = required - set(phenophase_df.columns)
    if missing:
        raise ValueError(f"phenophase_per_om is missing columns: {sorted(missing)}")

    df = phenophase_df[phenophase_df["chain_id"].astype(str) == str(chain_id)].copy()
    df["om_id"] = pd.to_numeric(df["om_id"], errors="coerce")
    df = df.dropna(subset=["om_id"]).sort_values("om_id")
    return df[["om_id", "phenophase"]]


def make_case_studies(
    features_df: pd.DataFrame,
    phenophase_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    decid_id, everg_id = _pick_case_trees(classification_df)

    decid_ts = _prep_tree_timeseries(features_df, decid_id)
    everg_ts = _prep_tree_timeseries(features_df, everg_id)

    decid_ph = _phenophase_for_tree(phenophase_df, decid_id)
    everg_ph = _phenophase_for_tree(phenophase_df, everg_id)

    # Map phenophase -> a light background color.
    phase_color = {
        "leaf_on": (0.80, 0.92, 0.80),
        "transitioning": (1.00, 0.93, 0.80),
        "leaf_off": (0.93, 0.85, 0.80),
        "stable": (0.92, 0.92, 0.92),
    }

    def plot_one(ax, ts: pd.DataFrame, ph: pd.DataFrame, title: str) -> None:
        # Use positional x for clean background spans.
        om_ids = ts["om_id"].to_numpy()
        x = np.arange(len(om_ids))

        # Background shading by phenophase (if available for those OMs).
        ph_map = {float(r.om_id): str(r.phenophase) for r in ph.itertuples(index=False)}
        for i, om in enumerate(om_ids):
            phase = ph_map.get(float(om))
            if not phase:
                continue
            color = phase_color.get(phase, (0.95, 0.95, 0.95))
            ax.axvspan(i - 0.5, i + 0.5, color=color, zorder=0)

        ax.plot(x, ts["veg_fraction_hsv"].to_numpy(), label="veg_fraction_hsv", lw=2)
        ax.plot(x, ts["gcc_mean"].to_numpy(), label="gcc_mean", lw=2)

        ax.set_title(title)
        ax.set_ylabel("signal value")
        ax.set_xlim(-0.5, len(x) - 0.5)

        max_ticks = 14
        if len(om_ids) > max_ticks:
            step = int(np.ceil(len(om_ids) / max_ticks))
            tick_idx = np.arange(0, len(om_ids), step)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([str(int(om_ids[i])) if float(om_ids[i]).is_integer() else str(om_ids[i]) for i in tick_idx])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in om_ids])

        ax.grid(True, alpha=0.25)

    ds_map = {
        str(r.chain_id): float(r.deciduousness_score)
        for r in classification_df[["chain_id", "deciduousness_score"]].itertuples(index=False)
        if pd.notna(r.deciduousness_score)
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    plot_one(
        axes[0],
        decid_ts,
        decid_ph,
        title=f"Deciduous example (chain_id={decid_id}, DS={ds_map.get(decid_id, float('nan')):.3f})",
    )
    plot_one(
        axes[1],
        everg_ts,
        everg_ph,
        title=f"Evergreen example (chain_id={everg_id}, DS={ds_map.get(everg_id, float('nan')):.3f})",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Case studies: signals with phenophase shading", y=0.98)
    axes[1].set_xlabel("OM index")

    _savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Overleaf-ready leaf-shed figures")
    parser.add_argument(
        "--fig-dir",
        default=None,
        help="Output figure directory (defaults to src/notebooks/leafshed_classifier_fig)",
    )
    args = parser.parse_args()

    p = _paths(args.fig_dir)

    _require_exists(p.features_csv)
    _require_exists(p.phenophase_csv)
    _require_exists(p.classification_csv)

    p.fig_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(p.features_csv)
    phenophase_df = pd.read_csv(p.phenophase_csv)
    classification_df = pd.read_csv(p.classification_csv)

    make_pipeline_schematic(p.fig_dir / "leafshed_pipeline_schematic.png")
    make_phenophase_counts(phenophase_df, p.fig_dir / "leafshed_phenophase_counts.png")
    make_case_studies(
        features_df,
        phenophase_df,
        classification_df,
        p.fig_dir / "leafshed_case_studies.png",
    )

    print("Wrote figures to:", p.fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
