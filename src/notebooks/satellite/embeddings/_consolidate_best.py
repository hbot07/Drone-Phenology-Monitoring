#!/usr/bin/env python3
"""Consolidate best result per (feature source, label) from model-sweep CSVs.

Reports the best random-split row and the best leave-area-out row (across all
held-out areas) for every label, grouped by feature source.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

# dir name -> human-readable feature source. Only these dirs are scanned.
SOURCES = {
    "exp01_gee_centroid": "GEE centroid (64d)",
    "exp01_s2temporal": "S2 temporal 2022-25 (724d)",
    "exp01_all3": "GEE+DINO+S2temporal (1172d)",
}


def best_of(df: pd.DataFrame) -> pd.Series | None:
    df = df[df["decision"] != "error"].copy()
    if df.empty or "balanced_accuracy" not in df.columns:
        return None
    return df.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).iloc[0]


def main() -> None:
    rows = []
    for dname, source in SOURCES.items():
        d = OUT / dname
        if not d.exists():
            continue
        files = list(d.glob("*_model_sweep.csv"))
        # group by label
        per_label: dict[str, dict[str, list]] = {}
        for f in files:
            df = pd.read_csv(f)
            if df.empty or "label" not in df.columns:
                continue
            label = str(df["label"].iloc[0])
            split = str(df["split"].iloc[0])
            per_label.setdefault(label, {"random": [], "leave_area_out": []})
            per_label[label].setdefault(split, [])
            per_label[label][split].append(df)
        for label, splits in sorted(per_label.items()):
            rec = {"feature_source": source, "dir": dname, "label": label}
            # random
            rnd = pd.concat(splits.get("random", []), ignore_index=True) if splits.get("random") else pd.DataFrame()
            b = best_of(rnd) if not rnd.empty else None
            rec["rand_bacc"] = round(float(b["balanced_accuracy"]), 3) if b is not None else None
            rec["rand_f1"] = round(float(b["macro_f1"]), 3) if b is not None else None
            rec["rand_model"] = f'{b["model"]}/{b["decision"]}' if b is not None else ""
            rec["rand_n_test"] = int(b["n_test"]) if b is not None else None
            rec["rand_labels"] = b.get("labels", "") if b is not None else ""
            rec["rand_cm"] = b.get("confusion_matrix", "") if b is not None else ""
            # leave-area-out (best across holdouts)
            lao = pd.concat(splits.get("leave_area_out", []), ignore_index=True) if splits.get("leave_area_out") else pd.DataFrame()
            b2 = best_of(lao) if not lao.empty else None
            rec["lao_bacc"] = round(float(b2["balanced_accuracy"]), 3) if b2 is not None else None
            rec["lao_f1"] = round(float(b2["macro_f1"]), 3) if b2 is not None else None
            rec["lao_holdout"] = b2["holdout"] if b2 is not None else ""
            rec["lao_model"] = f'{b2["model"]}/{b2["decision"]}' if b2 is not None else ""
            rec["lao_n_test"] = int(b2["n_test"]) if b2 is not None else None
            rec["lao_labels"] = b2.get("labels", "") if b2 is not None else ""
            rec["lao_cm"] = b2.get("confusion_matrix", "") if b2 is not None else ""
            rows.append(rec)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_colwidth", 60)
    summary_cols = ["feature_source", "label", "rand_bacc", "rand_f1", "rand_model", "lao_bacc", "lao_f1", "lao_holdout", "lao_model"]
    print("=== SUMMARY (balanced acc / macro-F1) ===")
    print(res[summary_cols].to_string(index=False))
    print("\n=== CONFUSION MATRICES (rows=true, cols=pred; class order in *_labels) ===")
    for _, r in res.iterrows():
        print(f"\n{r['feature_source']:30s} {r['label']}")
        print(f"  random  n_test={r['rand_n_test']} labels={r['rand_labels']}  cm={r['rand_cm']}")
        if r["lao_bacc"] is not None:
            print(f"  LAO[{r['lao_holdout']}] n_test={r['lao_n_test']} labels={r['lao_labels']}  cm={r['lao_cm']}")
    res.to_csv(OUT / "_consolidated_best.csv", index=False)
    print(f"\nWrote {OUT / '_consolidated_best.csv'}")


if __name__ == "__main__":
    main()
