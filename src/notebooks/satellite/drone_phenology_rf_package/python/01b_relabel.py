#!/usr/bin/env python3
"""
Fast label re-application from a configs/ experiment subfolder.

Use this whenever you edit a configs/<subfolder>/ JSON (add/remove species,
change class assignments) WITHOUT needing to re-download satellite data.

What it does:
  1. Reads the existing data/iitd_sv_crowns_master_wgs84.geojson.
  2. Re-applies all labels defined in --config-dir.
  3. Writes the updated GeoJSON in place (or to --out-geojson).
  4. Updates label columns in every feature CSV under exports/ by
     joining on crown_uid (spectral features are untouched).

Typical workflow:
  1. Edit configs/baseline/<task>.json  (or create configs/my_experiment/)
  2. python python/01b_relabel.py --config-dir configs/baseline
  3. python python/12_run_model_suite.py ...  (re-run experiments)

Usage:
  python python/01b_relabel.py [--config-dir configs/baseline] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPORTS_DIR = ROOT / "exports"
DEFAULT_CONFIG_DIR = ROOT / "configs" / "baseline"

# Import helpers from 01_prepare_crowns.py
_prep = SourceFileLoader(
    "prep_crowns",
    str(Path(__file__).with_name("01_prepare_crowns.py")),
).load_module()


def relabel_geojson(geojson_path: Path, configs: dict, dry_run: bool) -> dict:
    with geojson_path.open("r", encoding="utf-8") as f:
        gj = json.load(f)

    changed = 0
    for feat in gj["features"]:
        props = feat["properties"]
        species_clean = props.get("species_clean")
        extra = {"tree_type_raw": props.get("tree_type_raw")}

        for task, cfg in configs.items():
            new_val = _prep.apply_label_from_config(species_clean, cfg, extra)
            old_val = props.get(task)
            if old_val != new_val:
                changed += 1
            props[task] = new_val

    print(f"[geojson] {len(gj['features'])} crowns, {changed} label values changed")
    return gj


def update_feature_csvs(geojson: dict, label_cols: list, exports_dir: Path, dry_run: bool) -> None:
    uid_to_labels: dict[str, dict] = {}
    for feat in geojson["features"]:
        p = feat["properties"]
        uid = p.get("crown_uid")
        if uid:
            uid_to_labels[uid] = {col: p.get(col, -1) for col in label_cols}

    csvs = sorted(exports_dir.glob("*.csv"))
    if not csvs:
        print(f"[csvs] No CSVs found in {exports_dir}")
        return

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        if "crown_uid" not in df.columns:
            continue
        present = [c for c in label_cols if c in df.columns]
        if not present:
            continue

        old_vals = df[present].copy()
        for col in present:
            df[col] = df["crown_uid"].map(
                lambda uid, c=col: uid_to_labels.get(uid, {}).get(c, -1)
            )

        n_changed = (df[present] != old_vals).values.sum()
        if n_changed == 0:
            print(f"[csv] {csv_path.name}: no changes")
            continue

        print(f"[csv] {csv_path.name}: {n_changed} label cells updated")
        if not dry_run:
            df.to_csv(csv_path, index=False)
        else:
            print("  (dry-run: not written)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="Path to experiment config subfolder (default: configs/baseline).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change without writing anything.")
    ap.add_argument("--out-geojson", default=None,
                    help="Write updated GeoJSON here instead of in-place.")
    ap.add_argument("--no-csv", action="store_true",
                    help="Skip updating feature CSVs in exports/.")
    args = ap.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = ROOT / config_dir

    configs = _prep.load_label_configs(config_dir)
    print(f"Config dir : {config_dir}")
    print(f"Label tasks: {', '.join(configs)}")

    geojson_path = DATA_DIR / "iitd_sv_crowns_master_wgs84.geojson"
    updated_gj = relabel_geojson(geojson_path, configs, args.dry_run)

    out_path = Path(args.out_geojson) if args.out_geojson else geojson_path
    if not args.dry_run:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(updated_gj, f, ensure_ascii=False, separators=(",", ":"))
        print(f"[geojson] Written: {out_path}")
    else:
        print(f"[geojson] dry-run: would write {out_path}")

    if not args.no_csv:
        update_feature_csvs(updated_gj, list(configs.keys()), EXPORTS_DIR, args.dry_run)

    # Label distribution summary
    rows = [f["properties"] for f in updated_gj["features"]]
    table = pd.DataFrame(rows)
    print("\nLabel distribution after relabeling:")
    for col in configs.keys():
        if col in table.columns:
            vc = table[col].value_counts(dropna=False).sort_index()
            parts = "  ".join(f"{int(k)}:{v}" for k, v in vc.items())
            print(f"  {col:<25} {parts}")


if __name__ == "__main__":
    main()
