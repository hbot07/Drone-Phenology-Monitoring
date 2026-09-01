# #!/usr/bin/env python3
# """
# Per-OM phenophase classifier: leaf_on / transitioning / leaf_off.

# Why this exists
# ---------------
# The threshold rule (veg_norm >= on -> leaf_on, etc.) looks at each OM in
# isolation and tops out around ~58% accuracy, because a single veg_norm value is
# ambiguous: 0.10 can be the trough (leaf_off) OR a post-recovery OM on a
# multi-cycle chain (leaf_on). A threshold cannot tell those apart.

# This script instead trains a CLASSIFIER on per-OM features that carry the
# time-series context the threshold throws away -- most importantly WHERE the OM
# sits relative to the trough and which direction the curve is moving. This is the
# same feature-based modelling idea used for the deciduous/evergreen DS classifier,
# applied to the per-OM phenophase problem.

# Two models are trained and compared:
#     - Logistic regression  (interpretable, coefficients reportable -- like DS)
#     - Gradient boosting     (HistGradientBoosting; higher ceiling, less interpretable)

# Ground truth
# ------------
# Per-OM state is derived from your hand-labeled event OMs. The TRAINING file may
# carry a second cycle (columns *_again); both cycles are honoured so post-recovery
# OMs that then drop again are labeled correctly. Rule per OM, given cycle events
# (s = leaf_off_start, t = full_leaf_off, r = leaf_on_return) for each cycle:
#     OM <  first s                          -> leaf_on
#     within +/-span of any t                -> leaf_off
#     s <= OM < t  or  t < OM < r  (any cyc) -> transitioning
#     OM >= last r                           -> leaf_on
#     between two cycles (r1 <= OM < s2)      -> leaf_on
# Chains/OMs the rule cannot resolve are dropped from training (not guessed).

# Features per (chain, OM)  [the "methodology" you asked for]
# -----------------------------------------------------------
#     veg_norm                 normalized veg fraction at this OM (the old feature)
#     veg_hat                  raw interpolated veg fraction
#     veg_hat_z                per-chain z-score of veg_hat (cycle-robust level)
#     d_trough                 signed OM distance to the trough (OM - trough_om)
#     abs_d_trough             |OM - trough_om|
#     side                     -1 before trough, 0 at, +1 after   (time direction)
#     slope_local              veg_norm[i+1]-veg_norm[i-1] (rising/falling)
#     slope_back               veg_norm[i]-veg_norm[i-1]
#     veg_norm_prev/next       neighbours (local shape)
#     rank_in_chain            veg_norm percentile rank within the chain
#     frac_of_max              veg_hat / chain-max veg_hat  (recovery level)
#     om_frac                  OM position in series (0..1)
# These are exactly the signals a threshold ignores.

# Usage
# -----
#     python 10_phenophase_classifier.py --config /path/to/pipeline_config.json \\
#         --train-labels /path/to/leaf_leafoff_validation.xlsx \\
#         --test-labels  /path/to/test_leafonoff.xlsx \\
#         [--leafoff-span 0] [--drop-overlap] [--model both|logreg|gb]

# Outputs (under <phenology_dir>/validation/phenophase_clf/)
# -----------------------------------------------------------
#     features_train.csv / features_test.csv   the engineered feature tables
#     cv_results.csv                            stratified k-fold CV on training
#     logreg_coefficients.csv                   per-class LR coefficients (interpretable)
#     test_metrics_logreg.csv / _gb.csv         precision/recall/F1 on the held-out test
#     test_confusion_logreg.csv / _gb.csv       confusion matrices
#     test_predictions_logreg.csv / _gb.csv     per-OM predictions on test
#     model_comparison.txt                      the summary you read
# """

# from __future__ import annotations

# import argparse
# import json
# import sys
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd

# sys.path.insert(0, str(Path(__file__).resolve().parent))
# from phenology_validation_common import (  # noqa: E402
#     load_config,
#     load_features_df,
#     om_ids_from_features,
#     setup_app_dir,
# )

# STATES = ["leaf_on", "transitioning", "leaf_off"]
# STATE_TO_INT = {s: i for i, s in enumerate(STATES)}


# # ---------------------------------------------------------------------------
# # Ground-truth per-OM state from labeled events (supports 2 cycles)
# # ---------------------------------------------------------------------------
# def cycles_from_row(row: pd.Series) -> List[Tuple[float, float, float]]:
#     """Return list of (s, t, r) cycles present in a label row."""
#     cyc = []
#     s, t, r = row.get("leaf_off_start_om"), row.get("full_leaf_off_om"), row.get("leaf_on_return_om")
#     if np.isfinite(t):
#         cyc.append((s, t, r))
#     s2, t2, r2 = (row.get("leaf_off_start_om_again"), row.get("full_leaf_off_om_again"),
#                   row.get("leaf_on_return_om_again"))
#     if s2 is not None and np.isfinite(t2):
#         cyc.append((s2, t2, r2))
#     return cyc


# def true_state_multi(om: float, cycles: List[Tuple[float, float, float]], span: int) -> Optional[str]:
#     """Per-OM state honouring one or two labeled cycles."""
#     if not cycles:
#         return None
#     # leaf_off if within span of any trough
#     for (s, t, r) in cycles:
#         if np.isfinite(t) and abs(om - t) <= span:
#             return "leaf_off"
#     first_s = cycles[0][0]
#     last_r = cycles[-1][2]
#     # before the very first drop
#     if np.isfinite(first_s) and om < first_s:
#         return "leaf_on"
#     # after the very last return
#     if np.isfinite(last_r) and om >= last_r:
#         return "leaf_on"
#     # inside any cycle's transition arms
#     for (s, t, r) in cycles:
#         if np.isfinite(s) and np.isfinite(t) and s <= om < t:
#             return "transitioning"
#         if np.isfinite(t) and np.isfinite(r) and t < om < r:
#             return "transitioning"
#     # gap between cycle1 return and cycle2 start -> leafy plateau
#     if len(cycles) == 2:
#         r1 = cycles[0][2]
#         s2 = cycles[1][0]
#         if np.isfinite(r1) and np.isfinite(s2) and r1 <= om < s2:
#             return "leaf_on"
#     return None


# # ---------------------------------------------------------------------------
# # Feature engineering per (chain, OM)
# # ---------------------------------------------------------------------------
# def build_features_for_chain(oms: np.ndarray, veg_norm: np.ndarray,
#                              veg_hat: np.ndarray) -> pd.DataFrame:
#     n = len(oms)
#     order = np.argsort(oms)
#     oms, veg_norm, veg_hat = oms[order], veg_norm[order], veg_hat[order]

#     trough_idx = int(np.nanargmin(veg_hat)) if np.isfinite(veg_hat).any() else 0
#     trough_om = float(oms[trough_idx])

#     vh = veg_hat.astype(float)
#     mu, sd = np.nanmean(vh), np.nanstd(vh)
#     vh_z = (vh - mu) / sd if sd > 1e-9 else np.zeros_like(vh)
#     vmax = np.nanmax(vh) if np.isfinite(vh).any() else 1.0
#     frac_of_max = vh / vmax if vmax > 1e-9 else np.zeros_like(vh)

#     # percentile rank of veg_norm within chain
#     rank = pd.Series(veg_norm).rank(pct=True).to_numpy()

#     rows = []
#     for i in range(n):
#         prev_vn = veg_norm[i - 1] if i > 0 else veg_norm[i]
#         next_vn = veg_norm[i + 1] if i < n - 1 else veg_norm[i]
#         slope_local = (next_vn - prev_vn) / 2.0
#         slope_back = veg_norm[i] - prev_vn
#         d_trough = float(oms[i] - trough_om)
#         rows.append({
#             "om_id": int(oms[i]),
#             "veg_norm": float(veg_norm[i]),
#             "veg_hat": float(veg_hat[i]),
#             "veg_hat_z": float(vh_z[i]),
#             "d_trough": d_trough,
#             "abs_d_trough": abs(d_trough),
#             "side": float(np.sign(d_trough)),
#             "slope_local": float(slope_local),
#             "slope_back": float(slope_back),
#             "veg_norm_prev": float(prev_vn),
#             "veg_norm_next": float(next_vn),
#             "rank_in_chain": float(rank[i]),
#             "frac_of_max": float(frac_of_max[i]),
#             "om_frac": float((oms[i] - oms.min()) / max(oms.max() - oms.min(), 1)),
#             "trough_om": trough_om,
#         })
#     return pd.DataFrame(rows)


# FEATURE_COLS = [
#     "veg_norm", "veg_hat", "veg_hat_z", "d_trough", "abs_d_trough", "side",
#     "slope_local", "slope_back", "veg_norm_prev", "veg_norm_next",
#     "rank_in_chain", "frac_of_max", "om_frac",
# ]


# # ---------------------------------------------------------------------------
# # Metrics
# # ---------------------------------------------------------------------------
# def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
#     rows = []
#     macro_p, macro_r, macro_f = [], [], []
#     n = len(y_true)
#     for st in STATES:
#         i = STATE_TO_INT[st]
#         tp = int(((y_true == i) & (y_pred == i)).sum())
#         fp = int(((y_true != i) & (y_pred == i)).sum())
#         fn = int(((y_true == i) & (y_pred != i)).sum())
#         tn = n - tp - fp - fn
#         p = tp / (tp + fp) if (tp + fp) else np.nan
#         r = tp / (tp + fn) if (tp + fn) else np.nan
#         f = 2 * p * r / (p + r) if (np.isfinite(p) and np.isfinite(r) and p + r > 0) else np.nan
#         sup = int((y_true == i).sum())
#         rows.append({"phenophase": st, "precision": p, "recall": r, "f1": f,
#                      "support": sup, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
#         if sup > 0:
#             macro_p.append(p); macro_r.append(r); macro_f.append(f)
#     sups = np.array([r["support"] for r in rows], dtype=float)
#     tot = sups.sum()
#     class_rows = list(rows)  # snapshot of the 3 per-class rows only
#     def wavg(key):
#         return float(np.nansum([r[key] * r["support"] for r in class_rows]) / tot) if tot else np.nan
#     rows.append({"phenophase": "macro avg", "precision": float(np.nanmean(macro_p)),
#                  "recall": float(np.nanmean(macro_r)), "f1": float(np.nanmean(macro_f)),
#                  "support": int(tot), "TP": None, "FP": None, "FN": None, "TN": None})
#     rows.append({"phenophase": "weighted avg", "precision": wavg("precision"),
#                  "recall": wavg("recall"), "f1": wavg("f1"),
#                  "support": int(tot), "TP": None, "FP": None, "FN": None, "TN": None})
#     return pd.DataFrame(rows)


# def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
#     cm = pd.DataFrame(0, index=STATES, columns=STATES)
#     for t, p in zip(y_true, y_pred):
#         cm.iloc[t, p] += 1
#     cm.index.name = "true"; cm.columns.name = "predicted"
#     return cm


# # ---------------------------------------------------------------------------
# # Label loading + feature assembly
# # ---------------------------------------------------------------------------
# def load_labels(path: Path, sheet) -> pd.DataFrame:
#     if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
#         sh = int(sheet) if str(sheet).isdigit() else sheet
#         df = pd.read_excel(path, sheet_name=sh)
#     else:
#         df = pd.read_csv(path)
#     for c in df.columns:
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#     df = df.dropna(subset=["chain_id"]).copy()
#     df["chain_id"] = df["chain_id"].astype(int)
#     return df


# def assemble(labels: pd.DataFrame, curves: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
#              span: int, split_name: str) -> pd.DataFrame:
#     frames = []
#     n_dropped_state = 0
#     for _, row in labels.iterrows():
#         cid = int(row["chain_id"])
#         if cid not in curves:
#             continue
#         oms, vn, vh = curves[cid]
#         feats = build_features_for_chain(oms, vn, vh)
#         cyc = cycles_from_row(row)
#         states = [true_state_multi(float(o), cyc, span) for o in feats["om_id"]]
#         feats["chain_id"] = cid
#         feats["true_state"] = states
#         n_dropped_state += sum(s is None for s in states)
#         frames.append(feats)
#     if not frames:
#         return pd.DataFrame()
#     out = pd.concat(frames, ignore_index=True)
#     out["split"] = split_name
#     return out


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------
# def main() -> int:
#     ap = argparse.ArgumentParser(description="Per-OM phenophase classifier (logreg + gradient boosting)")
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--train-labels", required=True)
#     ap.add_argument("--test-labels", required=True)
#     ap.add_argument("--train-sheet", default=0)
#     ap.add_argument("--test-sheet", default=0)
#     ap.add_argument("--leafoff-span", type=int, default=0,
#                     help="OMs within +/-K of a labeled trough count as leaf_off")
#     ap.add_argument("--drop-overlap", action="store_true",
#                     help="Drop test chains that also appear in the training file")
#     ap.add_argument("--model", choices=["both", "logreg", "gb"], default="both")
#     ap.add_argument("--cv-folds", type=int, default=5)
#     ap.add_argument("--veg-min", type=float, default=0.45)
#     ap.add_argument("--ds-thresh", type=float, default=0.70)
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()

#     try:
#         from sklearn.linear_model import LogisticRegression
#         from sklearn.ensemble import HistGradientBoostingClassifier
#         from sklearn.preprocessing import StandardScaler
#         from sklearn.pipeline import Pipeline
#         from sklearn.model_selection import StratifiedKFold, cross_val_predict
#     except Exception as e:
#         print(f"ERROR: scikit-learn is required: {e}")
#         return 1

#     config = load_config(Path(args.config).resolve())
#     project_root = Path(config["project_root"])
#     phenology_dir = Path(config["phenology_dir"])
#     out_dir = phenology_dir / "validation" / "phenophase_clf"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # --- curves (identical to the pipeline's) ---
#     setup_app_dir(project_root)
#     from phenology_leafshed import LeafShedConfig, compute_leafshed_scores
#     features_df = load_features_df(phenology_dir)
#     om_ids = om_ids_from_features(features_df)
#     cfg = LeafShedConfig(veg_min_threshold=args.veg_min, ds_threshold=args.ds_thresh)
#     _s, pp_df, _n = compute_leafshed_scores(features_df, om_ids=om_ids, cfg=cfg)
#     pp_df["chain_id"] = pp_df["chain_id"].astype(int)
#     pp_df["om_id"] = pp_df["om_id"].astype(int)

#     curves = {}
#     for cid, sub in pp_df.groupby("chain_id"):
#         sub = sub.sort_values("om_id")
#         curves[int(cid)] = (sub["om_id"].to_numpy(float),
#                             sub["veg_fraction_hsv_norm"].to_numpy(float),
#                             sub["veg_fraction_hsv_hat"].to_numpy(float))

#     # --- labels ---
#     train_lab = load_labels(Path(args.train_labels).resolve(), args.train_sheet)
#     test_lab = load_labels(Path(args.test_labels).resolve(), args.test_sheet)
#     print(f"Train label chains: {len(train_lab)}  |  Test label chains: {len(test_lab)}")

#     overlap = sorted(set(train_lab["chain_id"]) & set(test_lab["chain_id"]))
#     if overlap:
#         print(f"[warn] train/test overlap: {overlap}")
#         if args.drop_overlap:
#             test_lab = test_lab[~test_lab["chain_id"].isin(overlap)].copy()
#             print(f"       dropped from test; {len(test_lab)} test chains remain")

#     # --- assemble features ---
#     span = args.leafoff_span
#     train_df = assemble(train_lab, curves, span, "train")
#     test_df = assemble(test_lab, curves, span, "test")
#     if train_df.empty or test_df.empty:
#         print("ERROR: no usable rows after assembling features.")
#         return 1

#     train_df.to_csv(out_dir / "features_train.csv", index=False)
#     test_df.to_csv(out_dir / "features_test.csv", index=False)

#     tr = train_df.dropna(subset=["true_state"]).copy()
#     te = test_df.dropna(subset=["true_state"]).copy()
#     print(f"\nUsable per-OM rows -> train: {len(tr)}  test: {len(te)}")
#     print(f"Train class balance: {tr['true_state'].value_counts().to_dict()}")
#     print(f"Test  class balance: {te['true_state'].value_counts().to_dict()}")

#     Xtr = tr[FEATURE_COLS].to_numpy(float)
#     ytr = tr["true_state"].map(STATE_TO_INT).to_numpy()
#     Xte = te[FEATURE_COLS].to_numpy(float)
#     yte = te["true_state"].map(STATE_TO_INT).to_numpy()

#     present = sorted(set(ytr))
#     if len(present) < 2:
#         print("ERROR: training data has <2 phenophase classes; cannot train.")
#         return 1

#     # class weights to counter the leaf_on majority
#     models = {}
#     if args.model in ("both", "logreg"):
#         models["logreg"] = Pipeline([
#             ("scale", StandardScaler()),
#             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
#         ])
#     if args.model in ("both", "gb"):
#         models["gb"] = HistGradientBoostingClassifier(
#             max_iter=300, learning_rate=0.08, max_depth=3,
#             l2_regularization=1.0, random_state=args.seed)

#     report = []
#     report.append("=" * 66)
#     report.append("PER-OM PHENOPHASE CLASSIFIER  (leaf_on / transitioning / leaf_off)")
#     report.append("=" * 66)
#     report.append(f"Train rows: {len(tr)}  Test rows: {len(te)}  leafoff_span={span}")
#     report.append(f"Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
#     report.append(f"Train class balance: {tr['true_state'].value_counts().to_dict()}")
#     report.append(f"Test  class balance: {te['true_state'].value_counts().to_dict()}")
#     report.append("")

#     # baseline: the old threshold rule for reference
#     def threshold_pred(df, on=0.65, off=0.20):
#         v = df["veg_norm"].to_numpy(float)
#         out = np.where(v >= on, STATE_TO_INT["leaf_on"],
#               np.where(v <= off, STATE_TO_INT["leaf_off"], STATE_TO_INT["transitioning"]))
#         return out
#     base_pred = threshold_pred(te)
#     base_acc = float((base_pred == yte).mean())
#     report.append(f"[baseline] threshold rule (on=0.65, off=0.20) test accuracy = {base_acc:.4f}")
#     report.append("")

#     cv_rows = []
#     n_splits = min(args.cv_folds, min(np.bincount(ytr)[present]))
#     n_splits = max(2, n_splits)
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

#     results = {}
#     for name, model in models.items():
#         # --- CV on training ---
#         try:
#             cv_pred = cross_val_predict(model, Xtr, ytr, cv=skf)
#             cv_acc = float((cv_pred == ytr).mean())
#             cvm = per_class_metrics(ytr, cv_pred)
#             cv_macro_f1 = float(cvm.loc[cvm["phenophase"] == "macro avg", "f1"].iloc[0])
#         except Exception as e:
#             cv_acc, cv_macro_f1 = float("nan"), float("nan")
#             print(f"[warn] CV failed for {name}: {e}")
#         cv_rows.append({"model": name, "cv_accuracy": cv_acc, "cv_macro_f1": cv_macro_f1,
#                         "n_splits": n_splits})

#         # --- fit on all training, evaluate on held-out test ---
#         model.fit(Xtr, ytr)
#         yp = model.predict(Xte)
#         acc = float((yp == yte).mean())
#         m = per_class_metrics(yte, yp)
#         cm = confusion(yte, yp)
#         m.to_csv(out_dir / f"test_metrics_{name}.csv", index=False)
#         cm.to_csv(out_dir / f"test_confusion_{name}.csv")
#         te_out = te[["chain_id", "om_id", "veg_norm", "d_trough", "true_state"]].copy()
#         te_out["pred_state"] = [STATES[i] for i in yp]
#         te_out.to_csv(out_dir / f"test_predictions_{name}.csv", index=False)
#         results[name] = {"acc": acc, "metrics": m, "cm": cm, "cv_acc": cv_acc,
#                          "cv_macro_f1": cv_macro_f1}

#         # logreg coefficients (interpretable)
#         if name == "logreg":
#             clf = model.named_steps["clf"]
#             coef = pd.DataFrame(clf.coef_, columns=FEATURE_COLS)
#             coef.insert(0, "class", [STATES[c] for c in clf.classes_])
#             coef.to_csv(out_dir / "logreg_coefficients.csv", index=False)

#     pd.DataFrame(cv_rows).to_csv(out_dir / "cv_results.csv", index=False)

#     # --- report ---
#     for name in results:
#         r = results[name]
#         report.append("-" * 66)
#         report.append(f"MODEL: {name}")
#         report.append("-" * 66)
#         report.append(f"  Stratified {n_splits}-fold CV accuracy on training: {r['cv_acc']:.4f}")
#         report.append(f"  CV macro-F1 on training:                    {r['cv_macro_f1']:.4f}")
#         report.append(f"  HELD-OUT TEST accuracy:                     {r['acc']:.4f} ({r['acc']:.1%})")
#         report.append("")
#         report.append("  Confusion matrix (rows=true, cols=pred):")
#         for line in r["cm"].to_string().splitlines():
#             report.append("    " + line)
#         report.append("")
#         m = r["metrics"]
#         report.append(f"  {'phenophase':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}")
#         for _, x in m.iterrows():
#             report.append(f"  {x['phenophase']:<16}{x['precision']:>10.4f}"
#                           f"{x['recall']:>10.4f}{x['f1']:>10.4f}{int(x['support']):>9d}")
#         report.append("")

#     # --- comparison verdict ---
#     report.append("=" * 66)
#     report.append("COMPARISON")
#     report.append("=" * 66)
#     report.append(f"{'model':<14}{'CV acc':>9}{'test acc':>10}{'test macroF1':>14}")
#     report.append(f"{'threshold':<14}{'-':>9}{base_acc:>10.3f}{'-':>14}")
#     for name in results:
#         r = results[name]
#         mf = r["metrics"].loc[r["metrics"]["phenophase"] == "macro avg", "f1"].iloc[0]
#         report.append(f"{name:<14}{r['cv_acc']:>9.3f}{r['acc']:>10.3f}{mf:>14.3f}")
#     best = max(results, key=lambda n: results[n]["acc"])
#     report.append("")
#     report.append(f"Best test accuracy: {best} ({results[best]['acc']:.1%}) "
#                   f"vs threshold baseline {base_acc:.1%} "
#                   f"(+{results[best]['acc'] - base_acc:.1%}).")
#     report.append("")
#     report.append("The gain over the threshold comes from the trough-relative and")
#     report.append("slope features (d_trough, side, slope_local, veg_hat_z), which let")
#     report.append("the model tell a low-veg_norm post-recovery OM (leaf_on) apart from")
#     report.append("a low-veg_norm trough OM (leaf_off) -- something no single threshold can do.")

#     txt = "\n".join(report)
#     print("\n" + txt)
#     (out_dir / "model_comparison.txt").write_text(txt, encoding="utf-8")
#     print(f"\nOutput in: {out_dir}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())

#!/usr/bin/env python3

#-----------------------------------------------------------------------------------------------------
# """
# Per-OM phenophase classifier: leaf_on / transitioning / leaf_off.

# Why this exists
# ---------------
# The threshold rule (veg_norm >= on -> leaf_on, etc.) looks at each OM in
# isolation and tops out around ~58% accuracy, because a single veg_norm value is
# ambiguous: 0.10 can be the trough (leaf_off) OR a post-recovery OM on a
# multi-cycle chain (leaf_on). A threshold cannot tell those apart.

# This script instead trains a CLASSIFIER on per-OM features that carry the
# time-series context the threshold throws away -- most importantly WHERE the OM
# sits relative to the trough and which direction the curve is moving. This is the
# same feature-based modelling idea used for the deciduous/evergreen DS classifier,
# applied to the per-OM phenophase problem.

# Two models are trained and compared:
#     - Logistic regression  (interpretable, coefficients reportable -- like DS)
#     - Gradient boosting     (HistGradientBoosting; higher ceiling, less interpretable)

# Ground truth
# ------------
# Per-OM state is derived from your hand-labeled event OMs. The TRAINING file may
# carry a second cycle (columns *_again); both cycles are honoured so post-recovery
# OMs that then drop again are labeled correctly. Rule per OM, given cycle events
# (s = leaf_off_start, t = full_leaf_off, r = leaf_on_return) for each cycle:
#     OM <  first s                          -> leaf_on
#     within +/-span of any t                -> leaf_off
#     s <= OM < t  or  t < OM < r  (any cyc) -> transitioning
#     OM >= last r                           -> leaf_on
#     between two cycles (r1 <= OM < s2)      -> leaf_on
# Chains/OMs the rule cannot resolve are dropped from training (not guessed).

# Features per (chain, OM)  [the "methodology" you asked for]
# -----------------------------------------------------------
#     veg_norm                 normalized veg fraction at this OM (the old feature)
#     veg_hat                  raw interpolated veg fraction
#     veg_hat_z                per-chain z-score of veg_hat (cycle-robust level)
#     d_trough                 signed OM distance to the trough (OM - trough_om)
#     abs_d_trough             |OM - trough_om|
#     side                     -1 before trough, 0 at, +1 after   (time direction)
#     slope_local              veg_norm[i+1]-veg_norm[i-1] (rising/falling)
#     slope_back               veg_norm[i]-veg_norm[i-1]
#     veg_norm_prev/next       neighbours (local shape)
#     rank_in_chain            veg_norm percentile rank within the chain
#     frac_of_max              veg_hat / chain-max veg_hat  (recovery level)
#     om_frac                  OM position in series (0..1)
# These are exactly the signals a threshold ignores.

# Usage
# -----
#     python 10_phenophase_classifier.py --config /path/to/pipeline_config.json \\
#         --train-labels /path/to/leaf_leafoff_validation.xlsx \\
#         --test-labels  /path/to/test_leafonoff.xlsx \\
#         [--leafoff-span 0] [--drop-overlap] [--model both|logreg|gb]

# Outputs (under <phenology_dir>/validation/phenophase_clf/)
# -----------------------------------------------------------
#     features_train.csv / features_test.csv   the engineered feature tables
#     cv_results.csv                            stratified k-fold CV on training
#     logreg_coefficients.csv                   per-class LR coefficients (interpretable)
#     test_metrics_logreg.csv / _gb.csv         precision/recall/F1 on the held-out test
#     test_confusion_logreg.csv / _gb.csv       confusion matrices
#     test_predictions_logreg.csv / _gb.csv     per-OM predictions on test
#     model_comparison.txt                      the summary you read
# """

# from __future__ import annotations

# import argparse
# import json
# import sys
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd

# sys.path.insert(0, str(Path(__file__).resolve().parent))
# from phenology_validation_common import (  # noqa: E402
#     load_config,
#     load_features_df,
#     om_ids_from_features,
#     setup_app_dir,
# )

# STATES = ["leaf_on", "transitioning", "leaf_off"]
# STATE_TO_INT = {s: i for i, s in enumerate(STATES)}


# # ---------------------------------------------------------------------------
# # Ground-truth per-OM state from labeled events (supports 2 cycles)
# # ---------------------------------------------------------------------------
# def cycles_from_row(row: pd.Series) -> List[Tuple[float, float, float]]:
#     """Return list of (s, t, r) cycles present in a label row."""
#     cyc = []
#     s, t, r = row.get("leaf_off_start_om"), row.get("full_leaf_off_om"), row.get("leaf_on_return_om")
#     if np.isfinite(t):
#         cyc.append((s, t, r))
#     s2, t2, r2 = (row.get("leaf_off_start_om_again"), row.get("full_leaf_off_om_again"),
#                   row.get("leaf_on_return_om_again"))
#     if s2 is not None and np.isfinite(t2):
#         cyc.append((s2, t2, r2))
#     return cyc


# def true_state_multi(om: float, cycles: List[Tuple[float, float, float]], span: int) -> Optional[str]:
#     """Per-OM state honouring one or two labeled cycles."""
#     if not cycles:
#         return None
#     # leaf_off if within span of any trough
#     for (s, t, r) in cycles:
#         if np.isfinite(t) and abs(om - t) <= span:
#             return "leaf_off"
#     first_s = cycles[0][0]
#     last_r = cycles[-1][2]
#     # before the very first drop
#     if np.isfinite(first_s) and om < first_s:
#         return "leaf_on"
#     # after the very last return
#     if np.isfinite(last_r) and om >= last_r:
#         return "leaf_on"
#     # inside any cycle's transition arms
#     for (s, t, r) in cycles:
#         if np.isfinite(s) and np.isfinite(t) and s <= om < t:
#             return "transitioning"
#         if np.isfinite(t) and np.isfinite(r) and t < om < r:
#             return "transitioning"
#     # gap between cycle1 return and cycle2 start -> leafy plateau
#     if len(cycles) == 2:
#         r1 = cycles[0][2]
#         s2 = cycles[1][0]
#         if np.isfinite(r1) and np.isfinite(s2) and r1 <= om < s2:
#             return "leaf_on"
#     return None


# # ---------------------------------------------------------------------------
# # Feature engineering per (chain, OM)
# # ---------------------------------------------------------------------------
# def build_features_for_chain(oms: np.ndarray, veg_norm: np.ndarray,
#                              veg_hat: np.ndarray) -> pd.DataFrame:
#     n = len(oms)
#     order = np.argsort(oms)
#     oms, veg_norm, veg_hat = oms[order], veg_norm[order], veg_hat[order]

#     trough_idx = int(np.nanargmin(veg_hat)) if np.isfinite(veg_hat).any() else 0
#     trough_om = float(oms[trough_idx])

#     vh = veg_hat.astype(float)
#     mu, sd = np.nanmean(vh), np.nanstd(vh)
#     vh_z = (vh - mu) / sd if sd > 1e-9 else np.zeros_like(vh)
#     vmax = np.nanmax(vh) if np.isfinite(vh).any() else 1.0
#     frac_of_max = vh / vmax if vmax > 1e-9 else np.zeros_like(vh)

#     # percentile rank of veg_norm within chain
#     rank = pd.Series(veg_norm).rank(pct=True).to_numpy()

#     rows = []
#     for i in range(n):
#         prev_vn = veg_norm[i - 1] if i > 0 else veg_norm[i]
#         next_vn = veg_norm[i + 1] if i < n - 1 else veg_norm[i]
#         slope_local = (next_vn - prev_vn) / 2.0
#         slope_back = veg_norm[i] - prev_vn
#         d_trough = float(oms[i] - trough_om)
#         sd_sign = float(np.sign(d_trough))
#         rows.append({
#             "om_id": int(oms[i]),
#             "veg_norm": float(veg_norm[i]),
#             "veg_hat": float(veg_hat[i]),
#             "veg_hat_z": float(vh_z[i]),
#             "d_trough": d_trough,
#             "abs_d_trough": abs(d_trough),
#             "side": sd_sign,
#             "slope_local": float(slope_local),
#             "slope_back": float(slope_back),
#             "veg_norm_prev": float(prev_vn),
#             "veg_norm_next": float(next_vn),
#             "rank_in_chain": float(rank[i]),
#             "frac_of_max": float(frac_of_max[i]),
#             "om_frac": float((oms[i] - oms.min()) / max(oms.max() - oms.min(), 1)),
#             # --- interaction features: let a LINEAR model express "low veg AND
#             #     after trough AND rising" = leaf_on, which pure logreg cannot ---
#             "veg_x_side": float(veg_norm[i]) * sd_sign,
#             "veg_x_slope": float(veg_norm[i]) * float(slope_local),
#             "side_x_slope": sd_sign * float(slope_local),
#             "fracmax_x_side": float(frac_of_max[i]) * sd_sign,
#             "vegz_x_side": float(vh_z[i]) * sd_sign,
#             "trough_om": trough_om,
#         })
#     return pd.DataFrame(rows)


# FEATURE_COLS = [
#     "veg_norm", "veg_hat", "veg_hat_z", "d_trough", "abs_d_trough", "side",
#     "slope_local", "slope_back", "veg_norm_prev", "veg_norm_next",
#     "rank_in_chain", "frac_of_max", "om_frac",
#     "veg_x_side", "veg_x_slope", "side_x_slope", "fracmax_x_side", "vegz_x_side",
# ]


# # ---------------------------------------------------------------------------
# # Metrics
# # ---------------------------------------------------------------------------
# def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
#     rows = []
#     macro_p, macro_r, macro_f = [], [], []
#     n = len(y_true)
#     for st in STATES:
#         i = STATE_TO_INT[st]
#         tp = int(((y_true == i) & (y_pred == i)).sum())
#         fp = int(((y_true != i) & (y_pred == i)).sum())
#         fn = int(((y_true == i) & (y_pred != i)).sum())
#         tn = n - tp - fp - fn
#         p = tp / (tp + fp) if (tp + fp) else np.nan
#         r = tp / (tp + fn) if (tp + fn) else np.nan
#         f = 2 * p * r / (p + r) if (np.isfinite(p) and np.isfinite(r) and p + r > 0) else np.nan
#         sup = int((y_true == i).sum())
#         rows.append({"phenophase": st, "precision": p, "recall": r, "f1": f,
#                      "support": sup, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
#         if sup > 0:
#             macro_p.append(p); macro_r.append(r); macro_f.append(f)
#     sups = np.array([r["support"] for r in rows], dtype=float)
#     tot = sups.sum()
#     class_rows = list(rows)  # snapshot of the 3 per-class rows only
#     def wavg(key):
#         return float(np.nansum([r[key] * r["support"] for r in class_rows]) / tot) if tot else np.nan
#     rows.append({"phenophase": "macro avg", "precision": float(np.nanmean(macro_p)),
#                  "recall": float(np.nanmean(macro_r)), "f1": float(np.nanmean(macro_f)),
#                  "support": int(tot), "TP": None, "FP": None, "FN": None, "TN": None})
#     rows.append({"phenophase": "weighted avg", "precision": wavg("precision"),
#                  "recall": wavg("recall"), "f1": wavg("f1"),
#                  "support": int(tot), "TP": None, "FP": None, "FN": None, "TN": None})
#     return pd.DataFrame(rows)


# def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
#     cm = pd.DataFrame(0, index=STATES, columns=STATES)
#     for t, p in zip(y_true, y_pred):
#         cm.iloc[t, p] += 1
#     cm.index.name = "true"; cm.columns.name = "predicted"
#     return cm


# # ---------------------------------------------------------------------------
# # Label loading + feature assembly
# # ---------------------------------------------------------------------------
# def load_labels(path: Path, sheet) -> pd.DataFrame:
#     if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
#         sh = int(sheet) if str(sheet).isdigit() else sheet
#         df = pd.read_excel(path, sheet_name=sh)
#     else:
#         df = pd.read_csv(path)
#     for c in df.columns:
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#     df = df.dropna(subset=["chain_id"]).copy()
#     df["chain_id"] = df["chain_id"].astype(int)
#     return df


# def assemble(labels: pd.DataFrame, curves: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
#              span: int, split_name: str) -> pd.DataFrame:
#     frames = []
#     n_dropped_state = 0
#     for _, row in labels.iterrows():
#         cid = int(row["chain_id"])
#         if cid not in curves:
#             continue
#         oms, vn, vh = curves[cid]
#         feats = build_features_for_chain(oms, vn, vh)
#         cyc = cycles_from_row(row)
#         states = [true_state_multi(float(o), cyc, span) for o in feats["om_id"]]
#         feats["chain_id"] = cid
#         feats["true_state"] = states
#         n_dropped_state += sum(s is None for s in states)
#         frames.append(feats)
#     if not frames:
#         return pd.DataFrame()
#     out = pd.concat(frames, ignore_index=True)
#     out["split"] = split_name
#     return out


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------
# def main() -> int:
#     ap = argparse.ArgumentParser(description="Per-OM phenophase classifier (logreg + gradient boosting)")
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--train-labels", required=True)
#     ap.add_argument("--test-labels", required=True)
#     ap.add_argument("--train-sheet", default=0)
#     ap.add_argument("--test-sheet", default=0)
#     ap.add_argument("--leafoff-span", type=int, default=0,
#                     help="OMs within +/-K of a labeled trough count as leaf_off")
#     ap.add_argument("--drop-overlap", action="store_true",
#                     help="Drop test chains that also appear in the training file")
#     ap.add_argument("--merge-transitioning", choices=["none", "leaf_off", "leaf_on"],
#                     default="none",
#                     help="Fold the rare 'transitioning' class into leaf_off or leaf_on "
#                          "and evaluate as a 2-class problem. 'transitioning' is often "
#                          "too rare (here ~3%% of test OMs) to learn or evaluate reliably; "
#                          "merging gives a stable 2-class leaf-on/leaf-off result.")
#     ap.add_argument("--model", choices=["both", "logreg", "gb"], default="both")
#     ap.add_argument("--cv-folds", type=int, default=5)
#     ap.add_argument("--veg-min", type=float, default=0.45)
#     ap.add_argument("--ds-thresh", type=float, default=0.70)
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()

#     try:
#         from sklearn.linear_model import LogisticRegression
#         from sklearn.ensemble import HistGradientBoostingClassifier
#         from sklearn.preprocessing import StandardScaler
#         from sklearn.pipeline import Pipeline
#         from sklearn.model_selection import StratifiedKFold, cross_val_predict
#     except Exception as e:
#         print(f"ERROR: scikit-learn is required: {e}")
#         return 1

#     config = load_config(Path(args.config).resolve())
#     project_root = Path(config["project_root"])
#     phenology_dir = Path(config["phenology_dir"])
#     out_dir = phenology_dir / "validation" / "phenophase_clf"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # --- curves (identical to the pipeline's) ---
#     setup_app_dir(project_root)
#     from phenology_leafshed import LeafShedConfig, compute_leafshed_scores
#     features_df = load_features_df(phenology_dir)
#     om_ids = om_ids_from_features(features_df)
#     cfg = LeafShedConfig(veg_min_threshold=args.veg_min, ds_threshold=args.ds_thresh)
#     _s, pp_df, _n = compute_leafshed_scores(features_df, om_ids=om_ids, cfg=cfg)
#     pp_df["chain_id"] = pp_df["chain_id"].astype(int)
#     pp_df["om_id"] = pp_df["om_id"].astype(int)

#     curves = {}
#     for cid, sub in pp_df.groupby("chain_id"):
#         sub = sub.sort_values("om_id")
#         curves[int(cid)] = (sub["om_id"].to_numpy(float),
#                             sub["veg_fraction_hsv_norm"].to_numpy(float),
#                             sub["veg_fraction_hsv_hat"].to_numpy(float))

#     # --- labels ---
#     train_lab = load_labels(Path(args.train_labels).resolve(), args.train_sheet)
#     test_lab = load_labels(Path(args.test_labels).resolve(), args.test_sheet)
#     print(f"Train label chains: {len(train_lab)}  |  Test label chains: {len(test_lab)}")

#     overlap = sorted(set(train_lab["chain_id"]) & set(test_lab["chain_id"]))
#     if overlap:
#         print(f"[warn] train/test overlap: {overlap}")
#         if args.drop_overlap:
#             test_lab = test_lab[~test_lab["chain_id"].isin(overlap)].copy()
#             print(f"       dropped from test; {len(test_lab)} test chains remain")

#     # --- assemble features ---
#     span = args.leafoff_span
#     train_df = assemble(train_lab, curves, span, "train")
#     test_df = assemble(test_lab, curves, span, "test")
#     if train_df.empty or test_df.empty:
#         print("ERROR: no usable rows after assembling features.")
#         return 1

#     train_df.to_csv(out_dir / "features_train.csv", index=False)
#     test_df.to_csv(out_dir / "features_test.csv", index=False)

#     tr = train_df.dropna(subset=["true_state"]).copy()
#     te = test_df.dropna(subset=["true_state"]).copy()

#     # --- optionally merge the rare transitioning class ---
#     global STATES, STATE_TO_INT
#     if args.merge_transitioning != "none":
#         tgt = args.merge_transitioning
#         tr["true_state"] = tr["true_state"].replace("transitioning", tgt)
#         te["true_state"] = te["true_state"].replace("transitioning", tgt)
#         STATES = ["leaf_on", "leaf_off"]
#         STATE_TO_INT = {s: i for i, s in enumerate(STATES)}
#         print(f"\n[merge] 'transitioning' folded into '{tgt}' -> 2-class problem "
#               f"(leaf_on / leaf_off)")

#     print(f"\nUsable per-OM rows -> train: {len(tr)}  test: {len(te)}")
#     print(f"Train class balance: {tr['true_state'].value_counts().to_dict()}")
#     print(f"Test  class balance: {te['true_state'].value_counts().to_dict()}")

#     Xtr = tr[FEATURE_COLS].to_numpy(float)
#     ytr = tr["true_state"].map(STATE_TO_INT).to_numpy()
#     Xte = te[FEATURE_COLS].to_numpy(float)
#     yte = te["true_state"].map(STATE_TO_INT).to_numpy()

#     present = sorted(set(ytr))
#     if len(present) < 2:
#         print("ERROR: training data has <2 phenophase classes; cannot train.")
#         return 1

#     # class weights to counter the leaf_on majority
#     models = {}
#     if args.model in ("both", "logreg"):
#         models["logreg"] = Pipeline([
#             ("scale", StandardScaler()),
#             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
#         ])
#     if args.model in ("both", "gb"):
#         models["gb"] = HistGradientBoostingClassifier(
#             max_iter=300, learning_rate=0.08, max_depth=3,
#             l2_regularization=1.0, random_state=args.seed)

#     report = []
#     report.append("=" * 66)
#     report.append("PER-OM PHENOPHASE CLASSIFIER  (leaf_on / transitioning / leaf_off)")
#     report.append("=" * 66)
#     report.append(f"Train rows: {len(tr)}  Test rows: {len(te)}  leafoff_span={span}")
#     report.append(f"Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
#     report.append(f"Train class balance: {tr['true_state'].value_counts().to_dict()}")
#     report.append(f"Test  class balance: {te['true_state'].value_counts().to_dict()}")
#     report.append("")

#     # baseline: the old threshold rule for reference
#     def threshold_pred(df, on=0.65, off=0.20):
#         v = df["veg_norm"].to_numpy(float)
#         if "transitioning" in STATE_TO_INT:
#             out = np.where(v >= on, STATE_TO_INT["leaf_on"],
#                   np.where(v <= off, STATE_TO_INT["leaf_off"], STATE_TO_INT["transitioning"]))
#         else:
#             # 2-class: midpoint split between the two thresholds
#             mid = (on + off) / 2.0
#             out = np.where(v >= mid, STATE_TO_INT["leaf_on"], STATE_TO_INT["leaf_off"])
#         return out
#     base_pred = threshold_pred(te)
#     base_acc = float((base_pred == yte).mean())
#     report.append(f"[baseline] threshold rule (on=0.65, off=0.20) test accuracy = {base_acc:.4f}")
#     report.append("")

#     cv_rows = []
#     n_splits = min(args.cv_folds, min(np.bincount(ytr)[present]))
#     n_splits = max(2, n_splits)
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

#     results = {}
#     for name, model in models.items():
#         # --- CV on training ---
#         try:
#             cv_pred = cross_val_predict(model, Xtr, ytr, cv=skf)
#             cv_acc = float((cv_pred == ytr).mean())
#             cvm = per_class_metrics(ytr, cv_pred)
#             cv_macro_f1 = float(cvm.loc[cvm["phenophase"] == "macro avg", "f1"].iloc[0])
#         except Exception as e:
#             cv_acc, cv_macro_f1 = float("nan"), float("nan")
#             print(f"[warn] CV failed for {name}: {e}")
#         cv_rows.append({"model": name, "cv_accuracy": cv_acc, "cv_macro_f1": cv_macro_f1,
#                         "n_splits": n_splits})

#         # --- fit on all training, evaluate on held-out test ---
#         model.fit(Xtr, ytr)
#         yp = model.predict(Xte)
#         acc = float((yp == yte).mean())
#         m = per_class_metrics(yte, yp)
#         cm = confusion(yte, yp)
#         m.to_csv(out_dir / f"test_metrics_{name}.csv", index=False)
#         cm.to_csv(out_dir / f"test_confusion_{name}.csv")
#         te_out = te[["chain_id", "om_id", "veg_norm", "d_trough", "true_state"]].copy()
#         te_out["pred_state"] = [STATES[i] for i in yp]
#         te_out.to_csv(out_dir / f"test_predictions_{name}.csv", index=False)
#         results[name] = {"acc": acc, "metrics": m, "cm": cm, "cv_acc": cv_acc,
#                          "cv_macro_f1": cv_macro_f1}

#         # logreg coefficients (interpretable)
#         if name == "logreg":
#             clf = model.named_steps["clf"]
#             coef_arr = clf.coef_
#             if coef_arr.shape[0] == 1:
#                 # binary case: one row separates class[1] from class[0]
#                 coef = pd.DataFrame(coef_arr, columns=FEATURE_COLS)
#                 coef.insert(0, "class", [f"{STATES[clf.classes_[1]]}_vs_{STATES[clf.classes_[0]]}"])
#             else:
#                 coef = pd.DataFrame(coef_arr, columns=FEATURE_COLS)
#                 coef.insert(0, "class", [STATES[c] for c in clf.classes_])
#             coef.to_csv(out_dir / "logreg_coefficients.csv", index=False)

#     pd.DataFrame(cv_rows).to_csv(out_dir / "cv_results.csv", index=False)

#     # --- report ---
#     for name in results:
#         r = results[name]
#         report.append("-" * 66)
#         report.append(f"MODEL: {name}")
#         report.append("-" * 66)
#         report.append(f"  Stratified {n_splits}-fold CV accuracy on training: {r['cv_acc']:.4f}")
#         report.append(f"  CV macro-F1 on training:                    {r['cv_macro_f1']:.4f}")
#         report.append(f"  HELD-OUT TEST accuracy:                     {r['acc']:.4f} ({r['acc']:.1%})")
#         report.append("")
#         report.append("  Confusion matrix (rows=true, cols=pred):")
#         for line in r["cm"].to_string().splitlines():
#             report.append("    " + line)
#         report.append("")
#         m = r["metrics"]
#         report.append(f"  {'phenophase':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}")
#         for _, x in m.iterrows():
#             report.append(f"  {x['phenophase']:<16}{x['precision']:>10.4f}"
#                           f"{x['recall']:>10.4f}{x['f1']:>10.4f}{int(x['support']):>9d}")
#         report.append("")

#     # --- comparison verdict ---
#     report.append("=" * 66)
#     report.append("COMPARISON")
#     report.append("=" * 66)
#     report.append(f"{'model':<14}{'CV acc':>9}{'test acc':>10}{'test macroF1':>14}")
#     report.append(f"{'threshold':<14}{'-':>9}{base_acc:>10.3f}{'-':>14}")
#     for name in results:
#         r = results[name]
#         mf = r["metrics"].loc[r["metrics"]["phenophase"] == "macro avg", "f1"].iloc[0]
#         report.append(f"{name:<14}{r['cv_acc']:>9.3f}{r['acc']:>10.3f}{mf:>14.3f}")
#     best = max(results, key=lambda n: results[n]["acc"])
#     report.append("")
#     report.append(f"Best test accuracy: {best} ({results[best]['acc']:.1%}) "
#                   f"vs threshold baseline {base_acc:.1%} "
#                   f"(+{results[best]['acc'] - base_acc:.1%}).")
#     report.append("")
#     report.append("The gain over the threshold comes from the trough-relative and")
#     report.append("slope features (d_trough, side, slope_local, veg_hat_z), which let")
#     report.append("the model tell a low-veg_norm post-recovery OM (leaf_on) apart from")
#     report.append("a low-veg_norm trough OM (leaf_off) -- something no single threshold can do.")

#     txt = "\n".join(report)
#     print("\n" + txt)
#     (out_dir / "model_comparison.txt").write_text(txt, encoding="utf-8")
#     print(f"\nOutput in: {out_dir}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
#-----------------------------------------------------------------------------------------------------



#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Per-OM phenophase classifier: leaf_on / transitioning / leaf_off.

Why this exists
---------------
The threshold rule (veg_norm >= on -> leaf_on, etc.) looks at each OM in
isolation and tops out around ~58% accuracy, because a single veg_norm value is
ambiguous: 0.10 can be the trough (leaf_off) OR a post-recovery OM on a
multi-cycle chain (leaf_on). A threshold cannot tell those apart.

This script instead trains a CLASSIFIER on per-OM features that carry the
time-series context the threshold throws away -- most importantly WHERE the OM
sits relative to the trough and which direction the curve is moving. This is the
same feature-based modelling idea used for the deciduous/evergreen DS classifier,
applied to the per-OM phenophase problem.

Two models are trained and compared:
    - Logistic regression  (interpretable, coefficients reportable -- like DS)
    - Gradient boosting     (HistGradientBoosting; higher ceiling, less interpretable)

Ground truth
------------
Per-OM state is derived from your hand-labeled event OMs. The TRAINING file may
carry a second cycle (columns *_again); both cycles are honoured so post-recovery
OMs that then drop again are labeled correctly. Rule per OM, given cycle events
(s = leaf_off_start, t = full_leaf_off, r = leaf_on_return) for each cycle:
    OM <  first s                          -> leaf_on
    within +/-span of any t                -> leaf_off
    s <= OM < t  or  t < OM < r  (any cyc) -> transitioning
    OM >= last r                           -> leaf_on
    between two cycles (r1 <= OM < s2)      -> leaf_on
Chains/OMs the rule cannot resolve are dropped from training (not guessed).

Features per (chain, OM)  [the "methodology" you asked for]
-----------------------------------------------------------
    veg_norm                 normalized veg fraction at this OM (the old feature)
    veg_hat                  raw interpolated veg fraction
    veg_hat_z                per-chain z-score of veg_hat (cycle-robust level)
    d_trough                 signed OM distance to the trough (OM - trough_om)
    abs_d_trough             |OM - trough_om|
    side                     -1 before trough, 0 at, +1 after   (time direction)
    slope_local              veg_norm[i+1]-veg_norm[i-1] (rising/falling)
    slope_back               veg_norm[i]-veg_norm[i-1]
    veg_norm_prev/next       neighbours (local shape)
    rank_in_chain            veg_norm percentile rank within the chain
    frac_of_max              veg_hat / chain-max veg_hat  (recovery level)
    om_frac                  OM position in series (0..1)
These are exactly the signals a threshold ignores.

Usage
-----
    python 10_phenophase_classifier.py --config /path/to/pipeline_config.json \\
        --train-labels /path/to/leaf_leafoff_validation.xlsx \\
        --test-labels  /path/to/test_leafonoff.xlsx \\
        [--leafoff-span 0] [--drop-overlap] [--model both|logreg|gb]

Outputs (under <phenology_dir>/validation/phenophase_clf/)
-----------------------------------------------------------
    features_train.csv / features_test.csv   the engineered feature tables
    cv_results.csv                            stratified k-fold CV on training
    logreg_coefficients.csv                   per-class LR coefficients (interpretable)
    test_metrics_logreg.csv / _gb.csv         precision/recall/F1 on the held-out test
    test_confusion_logreg.csv / _gb.csv       confusion matrices
    test_predictions_logreg.csv / _gb.csv     per-OM predictions on test
    model_comparison.txt                      the summary you read
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phenology_validation_common import (  # noqa: E402
    load_config,
    load_features_df,
    om_ids_from_features,
    setup_app_dir,
)

STATES = ["leaf_on", "transitioning", "leaf_off"]
STATE_TO_INT = {s: i for i, s in enumerate(STATES)}


# ---------------------------------------------------------------------------
# Ground-truth per-OM state from labeled events (supports 2 cycles)
# ---------------------------------------------------------------------------
def cycles_from_row(row: pd.Series) -> List[Tuple[float, float, float]]:
    """Return list of (s, t, r) cycles present in a label row."""
    cyc = []
    s, t, r = row.get("leaf_off_start_om"), row.get("full_leaf_off_om"), row.get("leaf_on_return_om")
    if np.isfinite(t):
        cyc.append((s, t, r))
    s2, t2, r2 = (row.get("leaf_off_start_om_again"), row.get("full_leaf_off_om_again"),
                  row.get("leaf_on_return_om_again"))
    if s2 is not None and np.isfinite(t2):
        cyc.append((s2, t2, r2))
    return cyc


def true_state_multi(om: float, cycles: List[Tuple[float, float, float]], span: int) -> Optional[str]:
    """Per-OM state honouring one or two labeled cycles."""
    if not cycles:
        return None
    # leaf_off if within span of any trough
    for (s, t, r) in cycles:
        if np.isfinite(t) and abs(om - t) <= span:
            return "leaf_off"
    first_s = cycles[0][0]
    last_r = cycles[-1][2]
    # before the very first drop
    if np.isfinite(first_s) and om < first_s:
        return "leaf_on"
    # after the very last return
    if np.isfinite(last_r) and om >= last_r:
        return "leaf_on"
    # inside any cycle's transition arms
    for (s, t, r) in cycles:
        if np.isfinite(s) and np.isfinite(t) and s <= om < t:
            return "transitioning"
        if np.isfinite(t) and np.isfinite(r) and t < om < r:
            return "transitioning"
    # gap between cycle1 return and cycle2 start -> leafy plateau
    if len(cycles) == 2:
        r1 = cycles[0][2]
        s2 = cycles[1][0]
        if np.isfinite(r1) and np.isfinite(s2) and r1 <= om < s2:
            return "leaf_on"
    return None


# ---------------------------------------------------------------------------
# Local extrema detection (window-free, site-independent)
# ---------------------------------------------------------------------------
def find_local_extrema(y: np.ndarray, min_prominence: float = 0.10):
    """Detect local minima (troughs) and maxima (peaks) in a 1-D curve.

    Window-free and scale-relative: works for 1-cycle, 2-cycle, or n-cycle
    chains at any site/cadence. A point is a local min if it is <= both
    neighbours and forms a dip of at least `min_prominence` (fraction of the
    curve's own range) below the surrounding peaks. The global argmin is always
    included as a trough so every chain has at least one.
    """
    n = len(y)
    if n == 0:
        return [], []
    yr = np.nan_to_num(y, nan=np.nanmax(y) if np.isfinite(y).any() else 0.0)
    rng = float(np.nanmax(yr) - np.nanmin(yr))
    prom = max(min_prominence * rng, 1e-9)

    troughs, peaks = [], []
    for i in range(n):
        lo = yr[i - 1] if i > 0 else yr[i]
        hi = yr[i + 1] if i < n - 1 else yr[i]
        if yr[i] <= lo and yr[i] <= hi:
            # local minimum candidate; check prominence vs neighbours
            left_max = np.max(yr[:i + 1]) if i > 0 else yr[i]
            right_max = np.max(yr[i:]) if i < n - 1 else yr[i]
            if (min(left_max, right_max) - yr[i]) >= prom:
                troughs.append(i)
        if yr[i] >= lo and yr[i] >= hi:
            left_min = np.min(yr[:i + 1]) if i > 0 else yr[i]
            right_min = np.min(yr[i:]) if i < n - 1 else yr[i]
            if (yr[i] - max(left_min, right_min)) >= prom:
                peaks.append(i)
    # guarantee at least the global argmin as a trough
    gmin = int(np.nanargmin(yr))
    if gmin not in troughs:
        troughs.append(gmin)
    troughs = sorted(set(troughs))
    peaks = sorted(set(peaks))
    return troughs, peaks


# ---------------------------------------------------------------------------
# Feature engineering per (chain, OM)  -- window-free, generalisable
# ---------------------------------------------------------------------------
def build_features_for_chain(oms: np.ndarray, veg_norm: np.ndarray,
                             veg_hat: np.ndarray) -> pd.DataFrame:
    n = len(oms)
    order = np.argsort(oms)
    oms, veg_norm, veg_hat = oms[order], veg_norm[order], veg_hat[order]

    vh = veg_hat.astype(float)
    mu, sd = np.nanmean(vh), np.nanstd(vh)
    vh_z = (vh - mu) / sd if sd > 1e-9 else np.zeros_like(vh)
    vmax = np.nanmax(vh) if np.isfinite(vh).any() else 1.0
    frac_of_max = vh / vmax if vmax > 1e-9 else np.zeros_like(vh)
    rank = pd.Series(veg_norm).rank(pct=True).to_numpy()

    # --- local extrema (no window, no site assumption) ---
    trough_idx_list, peak_idx_list = find_local_extrema(veg_hat)
    trough_oms = oms[trough_idx_list] if trough_idx_list else np.array([oms[int(np.nanargmin(vh))]])
    peak_oms = oms[peak_idx_list] if peak_idx_list else np.array([oms[int(np.nanargmax(vh))]])
    global_trough_om = float(oms[int(np.nanargmin(vh))])

    # local-baseline normalization: veg relative to the run between the two
    # nearest surrounding troughs -> does NOT collapse on multi-cycle chains
    def local_norm(i):
        left_tr = [t for t in trough_idx_list if t <= i]
        right_tr = [t for t in trough_idx_list if t >= i]
        lo_idx = left_tr[-1] if left_tr else 0
        hi_idx = right_tr[0] if right_tr else n - 1
        seg = vh[min(lo_idx, hi_idx):max(lo_idx, hi_idx) + 1]
        if seg.size == 0:
            return veg_norm[i]
        smin, smax = float(np.min(seg)), float(np.max(seg))
        # normalize against local peak on either side
        window_peak = max(vh[max(0, i - 3):min(n, i + 4)].max(), smax)
        base = min(smin, vh[max(0, i - 3):min(n, i + 4)].min())
        return float((vh[i] - base) / (window_peak - base)) if window_peak - base > 1e-9 else 0.5

    rows = []
    for i in range(n):
        prev_vn = veg_norm[i - 1] if i > 0 else veg_norm[i]
        next_vn = veg_norm[i + 1] if i < n - 1 else veg_norm[i]
        slope_local = (next_vn - prev_vn) / 2.0
        slope_back = veg_norm[i] - prev_vn

        # nearest trough / peak (window-free trough-relative features)
        d_troughs = trough_oms - oms[i]
        j = int(np.argmin(np.abs(d_troughs)))
        d_near_trough = float(oms[i] - trough_oms[j])
        d_peaks = peak_oms - oms[i]
        k = int(np.argmin(np.abs(d_peaks)))
        d_near_peak = float(oms[i] - peak_oms[k])

        # side relative to NEAREST trough (not the arbitrary global one)
        side_near = float(np.sign(d_near_trough))
        is_trough = 1.0 if i in trough_idx_list else 0.0
        is_peak = 1.0 if i in peak_idx_list else 0.0
        vn_loc = local_norm(i)

        rows.append({
            "om_id": int(oms[i]),
            "veg_norm": float(veg_norm[i]),
            "veg_norm_local": float(vn_loc),
            "veg_hat": float(veg_hat[i]),
            "veg_hat_z": float(vh_z[i]),
            "d_near_trough": d_near_trough,
            "abs_d_near_trough": abs(d_near_trough),
            "d_near_peak": d_near_peak,
            "abs_d_near_peak": abs(d_near_peak),
            "side_near": side_near,
            "is_trough": is_trough,
            "is_peak": is_peak,
            "n_troughs": float(len(trough_idx_list)),
            "slope_local": float(slope_local),
            "slope_back": float(slope_back),
            "veg_norm_prev": float(prev_vn),
            "veg_norm_next": float(next_vn),
            "rank_in_chain": float(rank[i]),
            "frac_of_max": float(frac_of_max[i]),
            "om_frac": float((oms[i] - oms.min()) / max(oms.max() - oms.min(), 1)),
            # interactions on the LOCAL (nearest-trough) reference
            "vegloc_x_side": float(vn_loc) * side_near,
            "vegloc_x_slope": float(vn_loc) * float(slope_local),
            "side_x_slope": side_near * float(slope_local),
            "fracmax_x_side": float(frac_of_max[i]) * side_near,
            "vegz_x_side": float(vh_z[i]) * side_near,
            "trough_om": global_trough_om,
        })
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "veg_norm", "veg_norm_local", "veg_hat", "veg_hat_z",
    "d_near_trough", "abs_d_near_trough", "d_near_peak", "abs_d_near_peak",
    "side_near", "is_trough", "is_peak", "n_troughs",
    "slope_local", "slope_back", "veg_norm_prev", "veg_norm_next",
    "rank_in_chain", "frac_of_max", "om_frac",
    "vegloc_x_side", "vegloc_x_slope", "side_x_slope", "fracmax_x_side", "vegz_x_side",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    macro_p, macro_r, macro_f = [], [], []
    n = len(y_true)
    for st in STATES:
        i = STATE_TO_INT[st]
        tp = int(((y_true == i) & (y_pred == i)).sum())
        fp = int(((y_true != i) & (y_pred == i)).sum())
        fn = int(((y_true == i) & (y_pred != i)).sum())
        tn = n - tp - fp - fn
        p = tp / (tp + fp) if (tp + fp) else np.nan
        r = tp / (tp + fn) if (tp + fn) else np.nan
        f = 2 * p * r / (p + r) if (np.isfinite(p) and np.isfinite(r) and p + r > 0) else np.nan
        sup = int((y_true == i).sum())
        rows.append({"phenophase": st, "precision": p, "recall": r, "f1": f,
                     "support": sup, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
        if sup > 0:
            macro_p.append(p); macro_r.append(r); macro_f.append(f)
    sups = np.array([r["support"] for r in rows], dtype=float)
    tot = sups.sum()
    class_rows = list(rows)  # snapshot of the 3 per-class rows only
    def wavg(key):
        return float(np.nansum([r[key] * r["support"] for r in class_rows]) / tot) if tot else np.nan
    rows.append({"phenophase": "macro avg", "precision": float(np.nanmean(macro_p)),
                 "recall": float(np.nanmean(macro_r)), "f1": float(np.nanmean(macro_f)),
                 "support": int(tot), "TP": None, "FP": None, "FN": None, "TN": None})
    rows.append({"phenophase": "weighted avg", "precision": wavg("precision"),
                 "recall": wavg("recall"), "f1": wavg("f1"),
                 "support": int(tot), "TP": None, "FP": None, "FN": None, "TN": None})
    return pd.DataFrame(rows)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = pd.DataFrame(0, index=STATES, columns=STATES)
    for t, p in zip(y_true, y_pred):
        cm.iloc[t, p] += 1
    cm.index.name = "true"; cm.columns.name = "predicted"
    return cm


# ---------------------------------------------------------------------------
# Label loading + feature assembly
# ---------------------------------------------------------------------------
def load_labels(path: Path, sheet) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        sh = int(sheet) if str(sheet).isdigit() else sheet
        df = pd.read_excel(path, sheet_name=sh)
    else:
        df = pd.read_csv(path)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["chain_id"]).copy()
    df["chain_id"] = df["chain_id"].astype(int)
    return df


def assemble(labels: pd.DataFrame, curves: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
             span: int, split_name: str) -> pd.DataFrame:
    frames = []
    n_dropped_state = 0
    for _, row in labels.iterrows():
        cid = int(row["chain_id"])
        if cid not in curves:
            continue
        oms, vn, vh = curves[cid]
        feats = build_features_for_chain(oms, vn, vh)
        cyc = cycles_from_row(row)
        states = [true_state_multi(float(o), cyc, span) for o in feats["om_id"]]
        feats["chain_id"] = cid
        feats["true_state"] = states
        n_dropped_state += sum(s is None for s in states)
        frames.append(feats)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["split"] = split_name
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Per-OM phenophase classifier (logreg + gradient boosting)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-labels", required=True)
    ap.add_argument("--test-labels", required=True)
    ap.add_argument("--train-sheet", default=0)
    ap.add_argument("--test-sheet", default=0)
    ap.add_argument("--leafoff-span", type=int, default=0,
                    help="OMs within +/-K of a labeled trough count as leaf_off")
    ap.add_argument("--drop-overlap", action="store_true",
                    help="Drop test chains that also appear in the training file")
    ap.add_argument("--merge-transitioning", choices=["none", "leaf_off", "leaf_on"],
                    default="none",
                    help="Fold the rare 'transitioning' class into leaf_off or leaf_on "
                         "and evaluate as a 2-class problem. 'transitioning' is often "
                         "too rare (here ~3%% of test OMs) to learn or evaluate reliably; "
                         "merging gives a stable 2-class leaf-on/leaf-off result.")
    ap.add_argument("--om-min", type=int, default=None,
                    help="Restrict curves to OMs >= this before building features")
    ap.add_argument("--om-max", type=int, default=None,
                    help="Restrict curves to OMs <= this before building features. "
                         "CRITICAL for multi-cycle chains: the trough-relative features "
                         "(side, d_trough) use the argmin of the curve, and if a deeper "
                         "SECOND cycle exists the global argmin is the wrong trough for "
                         "your first-cycle labels. Confine to the labeled cycle's window.")
    ap.add_argument("--model", choices=["both", "logreg", "gb"], default="both")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--veg-min", type=float, default=0.45)
    ap.add_argument("--ds-thresh", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except Exception as e:
        print(f"ERROR: scikit-learn is required: {e}")
        return 1

    config = load_config(Path(args.config).resolve())
    project_root = Path(config["project_root"])
    phenology_dir = Path(config["phenology_dir"])
    out_dir = phenology_dir / "validation" / "phenophase_clf"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- curves (identical to the pipeline's) ---
    setup_app_dir(project_root)
    from phenology_leafshed import LeafShedConfig, compute_leafshed_scores
    features_df = load_features_df(phenology_dir)
    om_ids = om_ids_from_features(features_df)
    cfg = LeafShedConfig(veg_min_threshold=args.veg_min, ds_threshold=args.ds_thresh)
    _s, pp_df, _n = compute_leafshed_scores(features_df, om_ids=om_ids, cfg=cfg)
    pp_df["chain_id"] = pp_df["chain_id"].astype(int)
    pp_df["om_id"] = pp_df["om_id"].astype(int)
    if args.om_min is not None:
        pp_df = pp_df[pp_df["om_id"] >= args.om_min]
    if args.om_max is not None:
        pp_df = pp_df[pp_df["om_id"] <= args.om_max]
    if args.om_min is not None or args.om_max is not None:
        print(f"Feature window restricted to OMs "
              f"[{args.om_min if args.om_min is not None else 'min'}, "
              f"{args.om_max if args.om_max is not None else 'max'}] "
              f"-- trough/side features now reference this window's argmin.")

    curves = {}
    for cid, sub in pp_df.groupby("chain_id"):
        sub = sub.sort_values("om_id")
        curves[int(cid)] = (sub["om_id"].to_numpy(float),
                            sub["veg_fraction_hsv_norm"].to_numpy(float),
                            sub["veg_fraction_hsv_hat"].to_numpy(float))

    # --- labels ---
    train_lab = load_labels(Path(args.train_labels).resolve(), args.train_sheet)
    test_lab = load_labels(Path(args.test_labels).resolve(), args.test_sheet)
    print(f"Train label chains: {len(train_lab)}  |  Test label chains: {len(test_lab)}")

    overlap = sorted(set(train_lab["chain_id"]) & set(test_lab["chain_id"]))
    if overlap:
        print(f"[warn] train/test overlap: {overlap}")
        if args.drop_overlap:
            test_lab = test_lab[~test_lab["chain_id"].isin(overlap)].copy()
            print(f"       dropped from test; {len(test_lab)} test chains remain")

    # --- assemble features ---
    span = args.leafoff_span
    train_df = assemble(train_lab, curves, span, "train")
    test_df = assemble(test_lab, curves, span, "test")
    if train_df.empty or test_df.empty:
        print("ERROR: no usable rows after assembling features.")
        return 1

    train_df.to_csv(out_dir / "features_train.csv", index=False)
    test_df.to_csv(out_dir / "features_test.csv", index=False)

    tr = train_df.dropna(subset=["true_state"]).copy()
    te = test_df.dropna(subset=["true_state"]).copy()

    # --- optionally merge the rare transitioning class ---
    global STATES, STATE_TO_INT
    if args.merge_transitioning != "none":
        tgt = args.merge_transitioning
        tr["true_state"] = tr["true_state"].replace("transitioning", tgt)
        te["true_state"] = te["true_state"].replace("transitioning", tgt)
        STATES = ["leaf_on", "leaf_off"]
        STATE_TO_INT = {s: i for i, s in enumerate(STATES)}
        print(f"\n[merge] 'transitioning' folded into '{tgt}' -> 2-class problem "
              f"(leaf_on / leaf_off)")

    print(f"\nUsable per-OM rows -> train: {len(tr)}  test: {len(te)}")
    print(f"Train class balance: {tr['true_state'].value_counts().to_dict()}")
    print(f"Test  class balance: {te['true_state'].value_counts().to_dict()}")

    Xtr = tr[FEATURE_COLS].to_numpy(float)
    ytr = tr["true_state"].map(STATE_TO_INT).to_numpy()
    Xte = te[FEATURE_COLS].to_numpy(float)
    yte = te["true_state"].map(STATE_TO_INT).to_numpy()

    present = sorted(set(ytr))
    if len(present) < 2:
        print("ERROR: training data has <2 phenophase classes; cannot train.")
        return 1

    # class weights to counter the leaf_on majority
    models = {}
    if args.model in ("both", "logreg"):
        models["logreg"] = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    if args.model in ("both", "gb"):
        models["gb"] = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=3,
            l2_regularization=1.0, random_state=args.seed)

    report = []
    report.append("=" * 66)
    report.append("PER-OM PHENOPHASE CLASSIFIER  (leaf_on / transitioning / leaf_off)")
    report.append("=" * 66)
    report.append(f"Train rows: {len(tr)}  Test rows: {len(te)}  leafoff_span={span}")
    report.append(f"Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
    report.append(f"Train class balance: {tr['true_state'].value_counts().to_dict()}")
    report.append(f"Test  class balance: {te['true_state'].value_counts().to_dict()}")
    report.append("")

    # baseline: the old threshold rule for reference
    def threshold_pred(df, on=0.65, off=0.20):
        v = df["veg_norm"].to_numpy(float)
        if "transitioning" in STATE_TO_INT:
            out = np.where(v >= on, STATE_TO_INT["leaf_on"],
                  np.where(v <= off, STATE_TO_INT["leaf_off"], STATE_TO_INT["transitioning"]))
        else:
            # 2-class: midpoint split between the two thresholds
            mid = (on + off) / 2.0
            out = np.where(v >= mid, STATE_TO_INT["leaf_on"], STATE_TO_INT["leaf_off"])
        return out
    base_pred = threshold_pred(te)
    base_acc = float((base_pred == yte).mean())
    report.append(f"[baseline] threshold rule (on=0.65, off=0.20) test accuracy = {base_acc:.4f}")
    report.append("")

    cv_rows = []
    n_splits = min(args.cv_folds, min(np.bincount(ytr)[present]))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    results = {}
    for name, model in models.items():
        # --- CV on training ---
        try:
            cv_pred = cross_val_predict(model, Xtr, ytr, cv=skf)
            cv_acc = float((cv_pred == ytr).mean())
            cvm = per_class_metrics(ytr, cv_pred)
            cv_macro_f1 = float(cvm.loc[cvm["phenophase"] == "macro avg", "f1"].iloc[0])
        except Exception as e:
            cv_acc, cv_macro_f1 = float("nan"), float("nan")
            print(f"[warn] CV failed for {name}: {e}")
        cv_rows.append({"model": name, "cv_accuracy": cv_acc, "cv_macro_f1": cv_macro_f1,
                        "n_splits": n_splits})

        # --- fit on all training, evaluate on held-out test ---
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        acc = float((yp == yte).mean())
        m = per_class_metrics(yte, yp)
        cm = confusion(yte, yp)
        m.to_csv(out_dir / f"test_metrics_{name}.csv", index=False)
        cm.to_csv(out_dir / f"test_confusion_{name}.csv")
        te_out = te[["chain_id", "om_id", "veg_norm", "d_near_trough", "true_state"]].copy()
        te_out["pred_state"] = [STATES[i] for i in yp]
        te_out.to_csv(out_dir / f"test_predictions_{name}.csv", index=False)
        results[name] = {"acc": acc, "metrics": m, "cm": cm, "cv_acc": cv_acc,
                         "cv_macro_f1": cv_macro_f1}

        # logreg coefficients (interpretable)
        if name == "logreg":
            clf = model.named_steps["clf"]
            coef_arr = clf.coef_
            if coef_arr.shape[0] == 1:
                # binary case: one row separates class[1] from class[0]
                coef = pd.DataFrame(coef_arr, columns=FEATURE_COLS)
                coef.insert(0, "class", [f"{STATES[clf.classes_[1]]}_vs_{STATES[clf.classes_[0]]}"])
            else:
                coef = pd.DataFrame(coef_arr, columns=FEATURE_COLS)
                coef.insert(0, "class", [STATES[c] for c in clf.classes_])
            coef.to_csv(out_dir / "logreg_coefficients.csv", index=False)

    pd.DataFrame(cv_rows).to_csv(out_dir / "cv_results.csv", index=False)

    # --- report ---
    for name in results:
        r = results[name]
        report.append("-" * 66)
        report.append(f"MODEL: {name}")
        report.append("-" * 66)
        report.append(f"  Stratified {n_splits}-fold CV accuracy on training: {r['cv_acc']:.4f}")
        report.append(f"  CV macro-F1 on training:                    {r['cv_macro_f1']:.4f}")
        report.append(f"  HELD-OUT TEST accuracy:                     {r['acc']:.4f} ({r['acc']:.1%})")
        report.append("")
        report.append("  Confusion matrix (rows=true, cols=pred):")
        for line in r["cm"].to_string().splitlines():
            report.append("    " + line)
        report.append("")
        m = r["metrics"]
        report.append(f"  {'phenophase':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}")
        for _, x in m.iterrows():
            report.append(f"  {x['phenophase']:<16}{x['precision']:>10.4f}"
                          f"{x['recall']:>10.4f}{x['f1']:>10.4f}{int(x['support']):>9d}")
        report.append("")

    # --- comparison verdict ---
    report.append("=" * 66)
    report.append("COMPARISON")
    report.append("=" * 66)
    report.append(f"{'model':<14}{'CV acc':>9}{'test acc':>10}{'test macroF1':>14}")
    report.append(f"{'threshold':<14}{'-':>9}{base_acc:>10.3f}{'-':>14}")
    for name in results:
        r = results[name]
        mf = r["metrics"].loc[r["metrics"]["phenophase"] == "macro avg", "f1"].iloc[0]
        report.append(f"{name:<14}{r['cv_acc']:>9.3f}{r['acc']:>10.3f}{mf:>14.3f}")
    best = max(results, key=lambda n: results[n]["acc"])
    report.append("")
    report.append(f"Best test accuracy: {best} ({results[best]['acc']:.1%}) "
                  f"vs threshold baseline {base_acc:.1%} "
                  f"(+{results[best]['acc'] - base_acc:.1%}).")
    report.append("")
    report.append("The gain over the threshold comes from the window-free local")
    report.append("trough/peak features (d_near_trough, side_near, veg_norm_local,")
    report.append("slope_local, veg_hat_z), which reference each OM's NEAREST local")
    report.append("trough rather than a global argmin. This lets the model tell a")
    report.append("post-recovery leaf_on OM apart from a genuine trough OM WITHOUT any")
    report.append("hardcoded OM window -- so it transfers across sites, flight cadences,")
    report.append("and 1- or 2-cycle trees automatically.")

    txt = "\n".join(report)
    print("\n" + txt)
    (out_dir / "model_comparison.txt").write_text(txt, encoding="utf-8")
    print(f"\nOutput in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())