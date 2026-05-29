#!/usr/bin/env python3
"""Add explicit temporal/phenology features to a multi-year Sentinel-2 table."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEASONS = ["winter", "premonsoon", "monsoon", "postmonsoon"]
SIGNALS = ["NDVI", "GNDVI", "NDRE", "NDMI", "NBR", "EVI", "green_ratio", "red_ratio", "yellow_proxy"]


def col(year: int, season: str, signal: str) -> str:
    return f"y{year}_{season}_{signal}"


def available_years(df: pd.DataFrame) -> list[int]:
    years = set()
    for c in df.columns:
        m = re.match(r"y(\d{4})_", c)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    years = available_years(out)
    if len(years) < 2:
        return out

    for signal in SIGNALS:
        all_cols = [col(y, s, signal) for y in years for s in SEASONS if col(y, s, signal) in out]
        if all_cols:
            out[f"temporal_{signal}_all_mean"] = out[all_cols].mean(axis=1, skipna=True)
            out[f"temporal_{signal}_all_std"] = out[all_cols].std(axis=1, skipna=True)
            out[f"temporal_{signal}_all_amp"] = out[all_cols].max(axis=1, skipna=True) - out[all_cols].min(axis=1, skipna=True)
            out[f"temporal_{signal}_late_minus_early"] = (
                out[[c for c in all_cols if c.startswith(f"y{years[-1]}_")]].mean(axis=1, skipna=True)
                - out[[c for c in all_cols if c.startswith(f"y{years[0]}_")]].mean(axis=1, skipna=True)
            )

        for season in SEASONS:
            season_cols = [col(y, season, signal) for y in years if col(y, season, signal) in out]
            if len(season_cols) >= 2:
                out[f"temporal_{signal}_{season}_mean"] = out[season_cols].mean(axis=1, skipna=True)
                out[f"temporal_{signal}_{season}_std"] = out[season_cols].std(axis=1, skipna=True)
                out[f"temporal_{signal}_{season}_late_minus_early"] = out[season_cols[-1]] - out[season_cols[0]]
                x = np.asarray(years[: len(season_cols)], dtype=float)
                x = x - x.mean()
                denom = float((x * x).sum())
                vals = out[season_cols].to_numpy(dtype=float)
                centered = vals - np.nanmean(vals, axis=1, keepdims=True)
                slope = np.nansum(centered * x, axis=1) / denom if denom else np.zeros(len(out))
                slope[np.isnan(vals).sum(axis=1) > len(season_cols) - 2] = np.nan
                out[f"temporal_{signal}_{season}_slope"] = slope

        for year in years:
            needed = {s: col(year, s, signal) for s in SEASONS}
            if all(c in out for c in needed.values()):
                out[f"temporal_y{year}_{signal}_post_minus_pre"] = out[needed["postmonsoon"]] - out[needed["premonsoon"]]
                out[f"temporal_y{year}_{signal}_monsoon_minus_pre"] = out[needed["monsoon"]] - out[needed["premonsoon"]]
                out[f"temporal_y{year}_{signal}_winter_minus_pre"] = out[needed["winter"]] - out[needed["premonsoon"]]
                out[f"temporal_y{year}_{signal}_post_minus_monsoon"] = out[needed["postmonsoon"]] - out[needed["monsoon"]]

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out = add_features(df)
    path = Path(args.out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {path} with {len(out)} rows and {len(out.columns)} columns")
    print(f"Added {len(out.columns) - len(df.columns)} temporal columns")


if __name__ == "__main__":
    main()
