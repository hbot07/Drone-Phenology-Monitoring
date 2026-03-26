"""Derive a hard-threshold rule to flag known non-tree chains.

This script searches for a small set of threshold clauses (1-3 ANDed clauses)
that flag exactly POS chain IDs using exported phenology metrics.

Outputs:
- output/non_tree_hard_rule.json
- output/non_tree_hard_rule_report.txt

Run with the same env as the notebook kernel (detectree).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


POS = {48, 56, 64, 73}

RAW_CSV = Path("output/consensus_phenology_features_raw.csv")
SUMMARY_CSV = Path("output/consensus_phenology_tree_summary.csv")
OUT_JSON = Path("output/non_tree_hard_rule.json")
OUT_TXT = Path("output/non_tree_hard_rule_report.txt")


def _iqr(series: pd.Series) -> float:
    arr = series.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


@dataclass(frozen=True)
class Clause:
    feature: str
    op: str  # '<=' or '>='
    threshold: float

    def apply(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature].to_numpy(dtype=float)
        ok = np.isfinite(x)
        if self.op == "<=" or self.op == "≤":
            return ok & (x <= self.threshold)
        if self.op == ">=" or self.op == "≥":
            return ok & (x >= self.threshold)
        raise ValueError(f"Unsupported op: {self.op}")

    def to_str(self) -> str:
        return f"({self.feature} {self.op} {self.threshold:.6g})"


def candidate_clauses(df: pd.DataFrame, feature: str) -> list[Clause]:
    x = df[feature].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    vals = np.unique(x)
    if vals.size < 2:
        return []
    vals.sort()
    # midpoints between consecutive unique values ensure stable boundaries
    thresholds = (vals[:-1] + vals[1:]) / 2.0
    thresholds = np.concatenate(([vals[0] - 1e-9], thresholds, [vals[-1] + 1e-9]))
    out: list[Clause] = []
    for t in thresholds:
        out.append(Clause(feature, "<=", float(t)))
        out.append(Clause(feature, ">=", float(t)))
    return out


def build_feature_table(raw: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["is_bad_observation"] = raw["is_bad_observation"].fillna(0).astype(int)

    raw_good = raw[(raw["patch_exists"] == 1) & (raw["is_bad_observation"] == 0)].copy()

    signals = [
        "veg_fraction_hsv",
        "gcc_mean",
        "rcc_mean",
        "gray_std_smooth",
        "gray_entropy",
        "laplacian_var",
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
    ]
    aux = ["valid_pixel_fraction", "shadow_fraction"]

    agg: dict[str, tuple[str, object]] = {}
    for col in signals:
        agg[col + "_mean"] = (col, "mean")
        agg[col + "_std"] = (col, "std")
        agg[col + "_iqr"] = (col, _iqr)
        agg[col + "_min"] = (col, "min")
        agg[col + "_max"] = (col, "max")
    for col in aux:
        agg[col + "_mean"] = (col, "mean")
        agg[col + "_min"] = (col, "min")

    per_chain = raw_good.groupby("chain_id").agg(**agg).reset_index()

    counts = raw.groupby("chain_id").agg(
        n_obs=("om_id", "count"),
        n_good=("is_bad_observation", lambda s: int((s == 0).sum())),
        n_bad=("is_bad_observation", lambda s: int((s == 1).sum())),
        patch_exists_frac=("patch_exists", "mean"),
    ).reset_index()
    counts["bad_frac"] = counts["n_bad"] / counts["n_obs"].clip(lower=1)

    features = per_chain.merge(counts, on="chain_id", how="left").merge(summary, on="chain_id", how="left")

    # Derived ratios capturing low-variance + low-amplitude patterns
    features["veg_amp_over_std"] = features["veg_fraction_hsv_amplitude"] / features["veg_fraction_hsv_std"].replace(0, np.nan)
    features["gcc_amp_over_std"] = features["gcc_mean_amplitude"] / features["gcc_mean_std"].replace(0, np.nan)
    features["veg_rel_amp"] = features["veg_fraction_hsv_amplitude"] / features["veg_fraction_hsv_mean"].replace(0, np.nan)
    features["gcc_rel_amp"] = features["gcc_mean_amplitude"] / features["gcc_mean_mean"].replace(0, np.nan)

    # Keep only chains with core stats
    features = features.dropna(subset=["veg_fraction_hsv_mean", "gcc_mean_mean"])
    features["is_pos"] = features["chain_id"].isin(POS)

    return features


def search_exact_rule(features: pd.DataFrame, curated: list[str]) -> tuple[list[Clause] | None, np.ndarray | None, list[tuple[int, Clause]]]:
    chain_ids = features["chain_id"].to_numpy()

    scored: list[tuple[int, Clause, np.ndarray]] = []  # (false_pos_count, clause, mask)

    for f in curated:
        if f not in features.columns:
            continue
        for cl in candidate_clauses(features, f):
            mask = cl.apply(features)
            picked = set(chain_ids[mask])
            if not (picked >= POS):
                continue
            fp = len(picked - POS)
            scored.append((fp, cl, mask))

    scored.sort(key=lambda t: (t[0], t[1].feature, t[1].op, t[1].threshold))
    best_singles = [(fp, cl) for fp, cl, _ in scored[:50]]

    def picked_ids(mask: np.ndarray) -> set[int]:
        return set(chain_ids[mask])

    # Prefer minimal number of clauses (1, then 2, then 3)
    top_pool_1 = scored[:200]
    for fp, cl, mask in top_pool_1:
        if picked_ids(mask) == POS:
            return [cl], mask, best_singles

    top_pool_2 = scored[:120]
    for (fp1, c1, m1), (fp2, c2, m2) in itertools.combinations(top_pool_2, 2):
        mask = m1 & m2
        if picked_ids(mask) == POS:
            return [c1, c2], mask, best_singles

    top_pool_3 = scored[:60]
    for (fp1, c1, m1), (fp2, c2, m2), (fp3, c3, m3) in itertools.combinations(top_pool_3, 3):
        mask = m1 & m2 & m3
        if picked_ids(mask) == POS:
            return [c1, c2, c3], mask, best_singles

    return None, None, best_singles


def main() -> None:
    if not RAW_CSV.exists() or not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing inputs: {RAW_CSV} or {SUMMARY_CSV}")

    raw = pd.read_csv(RAW_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    features = build_feature_table(raw, summary)

    curated = [
        # low-variance candidates
        "veg_fraction_hsv_std",
        "gcc_mean_std",
        "veg_fraction_hsv_iqr",
        "gcc_mean_iqr",
        # low-amplitude candidates
        "veg_fraction_hsv_amplitude",
        "gcc_mean_amplitude",
        "gray_std_smooth_amplitude",
        "gray_entropy_amplitude",
        # absolute levels (optional)
        "veg_fraction_hsv_mean",
        "gcc_mean_mean",
        # quality/confounds
        "shadow_fraction_mean",
        "valid_pixel_fraction_mean",
        "bad_frac",
        # ratios
        "veg_rel_amp",
        "gcc_rel_amp",
        "veg_amp_over_std",
        "gcc_amp_over_std",
    ]

    rule, mask, best_singles = search_exact_rule(features, curated)

    report_lines: list[str] = []
    report_lines.append(f"Chains in table: {len(features)}")
    report_lines.append(f"POS present: {int(features['is_pos'].sum())} (expect 4)")
    report_lines.append("")
    report_lines.append("Best single-clause candidates (fp <= 15):")
    for fp, cl in best_singles:
        if fp > 15:
            break
        report_lines.append(f"  fp={fp:2d}  {cl.to_str()}")

    out: dict = {
        "pos_chain_ids": [int(x) for x in sorted(POS)],
        "found": rule is not None,
        "curated_features": curated,
    }

    if rule is None or mask is None:
        report_lines.append("")
        report_lines.append("No exact rule found with up to 3 clauses (in this search space).")
    else:
        chain_ids = features["chain_id"].to_numpy()
        flagged = [int(x) for x in sorted(set(chain_ids[mask]))]
        fp = [int(x) for x in sorted(set(flagged) - POS)]
        fn = [int(x) for x in sorted(POS - set(flagged))]

        out.update(
            {
                "n_clauses": len(rule),
                "clauses": [
                    {"feature": c.feature, "op": c.op, "threshold": float(c.threshold)}
                    for c in rule
                ],
                "rule_str": " AND ".join([c.to_str() for c in rule]),
                "flags": flagged,
                "false_positives": fp,
                "false_negatives": fn,
            }
        )

        report_lines.append("")
        report_lines.append(f"FOUND exact rule with {len(rule)} clauses:")
        report_lines.append("  " + out["rule_str"])
        report_lines.append(f"Flags: {flagged}")

        used_features = sorted({c.feature for c in rule})
        snap = features.loc[features["chain_id"].isin(POS), ["chain_id"] + used_features].sort_values("chain_id")
        report_lines.append("")
        report_lines.append("POS snapshot (features used in rule):")
        report_lines.append(snap.to_string(index=False))

    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True))
    OUT_TXT.write_text("\n".join(report_lines) + "\n")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
