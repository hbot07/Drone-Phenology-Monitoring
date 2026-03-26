from __future__ import annotations

import numpy as np
import pandas as pd

POS = {48, 56, 64, 73}

raw = pd.read_csv("output/consensus_phenology_features_raw.csv")
summary = pd.read_csv("output/consensus_phenology_tree_summary.csv")

raw = raw.copy()
raw["is_bad_observation"] = raw["is_bad_observation"].fillna(0).astype(int)

# Use only usable observations for per-chain variance/means
good = raw[(raw["patch_exists"] == 1) & (raw["is_bad_observation"] == 0)].copy()

per_chain = (
    good.groupby("chain_id")
    .agg(
        veg_mean=("veg_fraction_hsv", "mean"),
        veg_std=("veg_fraction_hsv", "std"),
        veg_min=("veg_fraction_hsv", "min"),
        veg_max=("veg_fraction_hsv", "max"),
        gcc_mean=("gcc_mean", "mean"),
        gcc_std=("gcc_mean", "std"),
        shadow_mean=("shadow_fraction", "mean"),
        valid_mean=("valid_pixel_fraction", "mean"),
        lap_mean=("laplacian_var", "mean"),
        lap_std=("laplacian_var", "std"),
        glcm_contrast_mean=("glcm_contrast", "mean"),
        glcm_energy_mean=("glcm_energy", "mean"),
        glcm_homog_mean=("glcm_homogeneity", "mean"),
        entropy_mean=("gray_entropy", "mean"),
        entropy_std=("gray_entropy", "std"),
        gray_std_smooth_mean=("gray_std_smooth", "mean"),
        gray_std_smooth_std=("gray_std_smooth", "std"),
    )
    .reset_index()
)

df = summary.merge(per_chain, on="chain_id", how="left")

cols = [
    # amplitudes from summary
    "gcc_mean_amplitude",
    "rcc_mean_amplitude",
    "veg_fraction_hsv_amplitude",
    "gray_std_smooth_amplitude",
    "gray_entropy_amplitude",
    # non-GCC per-chain variability and levels
    "veg_mean",
    "veg_std",
    "entropy_mean",
    "entropy_std",
    "gray_std_smooth_mean",
    "gray_std_smooth_std",
    "lap_mean",
    "lap_std",
    "glcm_contrast_mean",
    "glcm_energy_mean",
    "glcm_homog_mean",
    "shadow_mean",
    "valid_mean",
]
cols = [c for c in cols if c in df.columns]

pos = df[df["chain_id"].isin(POS)].copy()
neg = df[~df["chain_id"].isin(POS)].copy()

print("POS table:")
print(pos[["chain_id"] + cols].sort_values("chain_id").to_string(index=False))

print("\nPOS ranges:")
for c in cols:
    s = pos[c].astype(float)
    print(f"{c:28s} min={float(np.nanmin(s)):.6g}  max={float(np.nanmax(s)):.6g}")

print("\nHow many negatives are <= POS max (lower is more separating):")
for c in cols:
    mx = float(np.nanmax(pos[c].astype(float)))
    frac = float((neg[c].astype(float) <= mx).mean())
    print(f"{c:28s} POS_max={mx:.6g}  neg_frac<=POSmax={frac:.3f}")
