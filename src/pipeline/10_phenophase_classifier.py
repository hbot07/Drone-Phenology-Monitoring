# # #!/usr/bin/env python3
# # """
# # Per-OM phenophase classifier: leaf_on / transitioning / leaf_off.

# # Why this exists
# # ---------------
# # The threshold rule (veg_norm >= on -> leaf_on, etc.) looks at each OM in
# # isolation and tops out around ~58% accuracy, because a single veg_norm value is
# # ambiguous: 0.10 can be the trough (leaf_off) OR a post-recovery OM on a
# # multi-cycle chain (leaf_on). A threshold cannot tell those apart.

# # This script instead trains a CLASSIFIER on per-OM features that carry the
# # time-series context the threshold throws away -- most importantly WHERE the OM
# # sits relative to the trough and which direction the curve is moving. This is the
# # same feature-based modelling idea used for the deciduous/evergreen DS classifier,
# # applied to the per-OM phenophase problem.

# # Two models are trained and compared:
# #     - Logistic regression  (interpretable, coefficients reportable -- like DS)
# #     - Gradient boosting     (HistGradientBoosting; higher ceiling, less interpretable)

# # Ground truth
# # ------------
# # Per-OM state is derived from your hand-labeled event OMs. The TRAINING file may
# # carry a second cycle (columns *_again); both cycles are honoured so post-recovery
# # OMs that then drop again are labeled correctly. Rule per OM, given cycle events
# # (s = leaf_off_start, t = full_leaf_off, r = leaf_on_return) for each cycle:
# #     OM <  first s                          -> leaf_on
# #     within +/-span of any t                -> leaf_off
# #     s <= OM < t  or  t < OM < r  (any cyc) -> transitioning
# #     OM >= last r                           -> leaf_on
# #     between two cycles (r1 <= OM < s2)      -> leaf_on
# # Chains/OMs the rule cannot resolve are dropped from training (not guessed).

# # Features per (chain, OM)  [the "methodology" you asked for]
# # -----------------------------------------------------------
# #     veg_norm                 normalized veg fraction at this OM (the old feature)
# #     veg_hat                  raw interpolated veg fraction
# #     veg_hat_z                per-chain z-score of veg_hat (cycle-robust level)
# #     d_trough                 signed OM distance to the trough (OM - trough_om)
# #     abs_d_trough             |OM - trough_om|
# #     side                     -1 before trough, 0 at, +1 after   (time direction)
# #     slope_local              veg_norm[i+1]-veg_norm[i-1] (rising/falling)
# #     slope_back               veg_norm[i]-veg_norm[i-1]
# #     veg_norm_prev/next       neighbours (local shape)
# #     rank_in_chain            veg_norm percentile rank within the chain
# #     frac_of_max              veg_hat / chain-max veg_hat  (recovery level)
# #     om_frac                  OM position in series (0..1)
# # These are exactly the signals a threshold ignores.

# # Usage
# # -----
# #     python 10_phenophase_classifier.py --config /path/to/pipeline_config.json \\
# #         --train-labels /path/to/leaf_leafoff_validation.xlsx \\
# #         --test-labels  /path/to/test_leafonoff.xlsx \\
# #         [--leafoff-span 0] [--drop-overlap] [--model both|logreg|gb]

# # Outputs (under <phenology_dir>/validation/phenophase_clf/)
# # -----------------------------------------------------------
# #     features_train.csv / features_test.csv   the engineered feature tables
# #     cv_results.csv                            stratified k-fold CV on training
# #     logreg_coefficients.csv                   per-class LR coefficients (interpretable)
# #     test_metrics_logreg.csv / _gb.csv         precision/recall/F1 on the held-out test
# #     test_confusion_logreg.csv / _gb.csv       confusion matrices
# #     test_predictions_logreg.csv / _gb.csv     per-OM predictions on test
# #     model_comparison.txt                      the summary you read
# # """

# # from __future__ import annotations

# # import argparse
# # import json
# # import sys
# # from pathlib import Path
# # from typing import Dict, List, Optional, Tuple

# # import numpy as np
# # import pandas as pd

# # sys.path.insert(0, str(Path(__file__).resolve().parent))
# # from phenology_validation_common import (  # noqa: E402
# #     load_config,
# #     load_features_df,
# #     om_ids_from_features,
# #     setup_app_dir,
# # )

# # STATES = ["leaf_on", "transitioning", "leaf_off"]
# # STATE_TO_INT = {s: i for i, s in enumerate(STATES)}


# # # ---------------------------------------------------------------------------
# # # Ground-truth per-OM state from labeled events (supports 2 cycles)
# # # ---------------------------------------------------------------------------
# # def cycles_from_row(row: pd.Series) -> List[Tuple[float, float, float]]:
# #     """Return list of (s, t, r) cycles present in a label row."""
# #     cyc = []
# #     s, t, r = row.get("leaf_off_start_om"), row.get("full_leaf_off_om"), row.get("leaf_on_return_om")
# #     if np.isfinite(t):
# #         cyc.append((s, t, r))
# #     s2, t2, r2 = (row.get("leaf_off_start_om_again"), row.get("full_leaf_off_om_again"),
# #                   row.get("leaf_on_return_om_again"))
# #     if s2 is not None and np.isfinite(t2):
# #         cyc.append((s2, t2, r2))
# #     return cyc


# # def true_state_multi(om: float, cycles: List[Tuple[float, float, float]], span: int) -> Optional[str]:
# #     """Per-OM state honouring one or two labeled cycles."""
# #     if not cycles:
# #         return None
# #     # leaf_off if within span of any trough
# #     for (s, t, r) in cycles:
# #         if np.isfinite(t) and abs(om - t) <= span:
# #             return "leaf_off"
# #     first_s = cycles[0][0]
# #     last_r = cycles[-1][2]
# #     # before the very first drop
# #     if np.isfinite(first_s) and om < first_s:
# #         return "leaf_on"
# #     # after the very last return
# #     if np.isfinite(last_r) and om >= last_r:
# #         return "leaf_on"
# #     # inside any cycle's transition arms
# #     for (s, t, r) in cycles:
# #         if np.isfinite(s) and np.isfinite(t) and s <= om < t:
# #             return "transitioning"
# #         if np.isfinite(t) and np.isfinite(r) and t < om < r:
# #             return "transitioning"
# #     # gap between cycle1 return and cycle2 start -> leafy plateau
# #     if len(cycles) == 2:
# #         r1 = cycles[0][2]
# #         s2 = cycles[1][0]
# #         if np.isfinite(r1) and np.isfinite(s2) and r1 <= om < s2:
# #             return "leaf_on"
# #     return None


# # # ---------------------------------------------------------------------------
# # # Feature engineering per (chain, OM)
# # # ---------------------------------------------------------------------------
# # def build_features_for_chain(oms: np.ndarray, veg_norm: np.ndarray,
# #                              veg_hat: np.ndarray) -> pd.DataFrame:
# #     n = len(oms)
# #     order = np.argsort(oms)
# #     oms, veg_norm, veg_hat = oms[order], veg_norm[order], veg_hat[order]

# #     trough_idx = int(np.nanargmin(veg_hat)) if np.isfinite(veg_hat).any() else 0
# #     trough_om = float(oms[trough_idx])

# #     vh = veg_hat.astype(float)
# #     mu, sd = np.nanmean(vh), np.nanstd(vh)
# #     vh_z = (vh - mu) / sd if sd > 1e-9 else np.zeros_like(vh)
# #     vmax = np.nanmax(vh) if np.isfinite(vh).any() else 1.0
# #     frac_of_max = vh / vmax if vmax > 1e-9 else np.zeros_like(vh)

# #     # percentile rank of veg_norm within chain
# #     rank = pd.Series(veg_norm).rank(pct=True).to_numpy()

# #     rows = []
# #     for i in range(n):
# #         prev_vn = veg_norm[i - 1] if i > 0 else veg_norm[i]
# #         next_vn = veg_norm[i + 1] if i < n - 1 else veg_norm[i]
# #         slope_local = (next_vn - prev_vn) / 2.0
# #         slope_back = veg_norm[i] - prev_vn
# #         d_trough = float(oms[i] - trough_om)
# #         rows.append({
# #             "om_id": int(oms[i]),
# #             "veg_norm": float(veg_norm[i]),
# #             "veg_hat": float(veg_hat[i]),
# #             "veg_hat_z": float(vh_z[i]),
# #             "d_trough": d_trough,
# #             "abs_d_trough": abs(d_trough),
# #             "side": float(np.sign(d_trough)),
# #             "slope_local": float(slope_local),
# #             "slope_back": float(slope_back),
# #             "veg_norm_prev": float(prev_vn),
# #             "veg_norm_next": float(next_vn),
# #             "rank_in_chain": float(rank[i]),
# #             "frac_of_max": float(frac_of_max[i]),
# #             "om_frac": float((oms[i] - oms.min()) / max(oms.max() - oms.min(), 1)),
# #             "trough_om": trough_om,
# #         })
# #     return pd.DataFrame(rows)


# # FEATURE_COLS = [
# #     "veg_norm", "veg_hat", "veg_hat_z", "d_trough", "abs_d_trough", "side",
# #     "slope_local", "slope_back", "veg_norm_prev", "veg_norm_next",
# #     "rank_in_chain", "frac_of_max", "om_frac",
# # ]


# # # ---------------------------------------------------------------------------
# # # Metrics
# # # ---------------------------------------------------------------------------
# # def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
# #     rows = []
# #     macro_p, macro_r, macro_f = [], [], []
# #     n = len(y_true)
# #     for st in STATES:
# #         i = STATE_TO_INT[st]
# #         tp = int(((y_true == i) & (y_pred == i)).sum())
# #         fp = int(((y_true != i) & (y_pred == i)).sum())
# #         fn = int(((y_true == i) & (y_pred != i)).sum())
# #         tn = n - tp - fp - fn
# #         p = tp / (tp + fp) if (tp + fp) else np.nan
# #         r = tp / (tp + fn) if (tp + fn) else np.nan
# #         f = 2 * p * r / (p + r) if (np.isfinite(p) and np.isfinite(r) and p + r > 0) else np.nan
# #         sup = int((y_true == i).sum())
# #         rows.append({"phenophase": st, "precision": p, "recall": r, "f1": f,
# #                      "support": sup, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
# #         if sup > 0:
# #             macro_p.append(p); macro_r.append(r); macro_f.append(f)
# #     sups = np.array([r["support"] for r in rows], dtype=float)
# #     tot = sups.sum()
# #     class_rows = list(rows)  # snapshot of the 3 per-class rows only
# #     def wavg(key):
# #         return float(np.nansum([r[key] * r["support"] for r in class_rows]) / tot) if tot else np.nan
# #     rows.append({"phenophase": "macro avg", "precision": float(np.nanmean(macro_p)),
# #                  "recall": float(np.nanmean(macro_r)), "f1": float(np.nanmean(macro_f)),
# #                  "support": int(tot), "TP": None, "FP": None, "FN": None, "TN": None})
# #     rows.append({"phenophase": "weighted avg", "precision": wavg("precision"),
# #                  "recall": wavg("recall"), "f1": wavg("f1"),
# #                  "support": int(tot), "TP": None, "FP": None, "FN": None, "TN": None})
# #     return pd.DataFrame(rows)


# # def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
# #     cm = pd.DataFrame(0, index=STATES, columns=STATES)
# #     for t, p in zip(y_true, y_pred):
# #         cm.iloc[t, p] += 1
# #     cm.index.name = "true"; cm.columns.name = "predicted"
# #     return cm


# # # ---------------------------------------------------------------------------
# # # Label loading + feature assembly
# # # ---------------------------------------------------------------------------
# # def load_labels(path: Path, sheet) -> pd.DataFrame:
# #     if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
# #         sh = int(sheet) if str(sheet).isdigit() else sheet
# #         df = pd.read_excel(path, sheet_name=sh)
# #     else:
# #         df = pd.read_csv(path)
# #     for c in df.columns:
# #         df[c] = pd.to_numeric(df[c], errors="coerce")
# #     df = df.dropna(subset=["chain_id"]).copy()
# #     df["chain_id"] = df["chain_id"].astype(int)
# #     return df


# # def assemble(labels: pd.DataFrame, curves: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
# #              span: int, split_name: str) -> pd.DataFrame:
# #     frames = []
# #     n_dropped_state = 0
# #     for _, row in labels.iterrows():
# #         cid = int(row["chain_id"])
# #         if cid not in curves:
# #             continue
# #         oms, vn, vh = curves[cid]
# #         feats = build_features_for_chain(oms, vn, vh)
# #         cyc = cycles_from_row(row)
# #         states = [true_state_multi(float(o), cyc, span) for o in feats["om_id"]]
# #         feats["chain_id"] = cid
# #         feats["true_state"] = states
# #         n_dropped_state += sum(s is None for s in states)
# #         frames.append(feats)
# #     if not frames:
# #         return pd.DataFrame()
# #     out = pd.concat(frames, ignore_index=True)
# #     out["split"] = split_name
# #     return out


# # # ---------------------------------------------------------------------------
# # # Main
# # # ---------------------------------------------------------------------------
# # def main() -> int:
# #     ap = argparse.ArgumentParser(description="Per-OM phenophase classifier (logreg + gradient boosting)")
# #     ap.add_argument("--config", required=True)
# #     ap.add_argument("--train-labels", required=True)
# #     ap.add_argument("--test-labels", required=True)
# #     ap.add_argument("--train-sheet", default=0)
# #     ap.add_argument("--test-sheet", default=0)
# #     ap.add_argument("--leafoff-span", type=int, default=0,
# #                     help="OMs within +/-K of a labeled trough count as leaf_off")
# #     ap.add_argument("--drop-overlap", action="store_true",
# #                     help="Drop test chains that also appear in the training file")
# #     ap.add_argument("--model", choices=["both", "logreg", "gb"], default="both")
# #     ap.add_argument("--cv-folds", type=int, default=5)
# #     ap.add_argument("--veg-min", type=float, default=0.45)
# #     ap.add_argument("--ds-thresh", type=float, default=0.70)
# #     ap.add_argument("--seed", type=int, default=42)
# #     args = ap.parse_args()

# #     try:
# #         from sklearn.linear_model import LogisticRegression
# #         from sklearn.ensemble import HistGradientBoostingClassifier
# #         from sklearn.preprocessing import StandardScaler
# #         from sklearn.pipeline import Pipeline
# #         from sklearn.model_selection import StratifiedKFold, cross_val_predict
# #     except Exception as e:
# #         print(f"ERROR: scikit-learn is required: {e}")
# #         return 1

# #     config = load_config(Path(args.config).resolve())
# #     project_root = Path(config["project_root"])
# #     phenology_dir = Path(config["phenology_dir"])
# #     out_dir = phenology_dir / "validation" / "phenophase_clf"
# #     out_dir.mkdir(parents=True, exist_ok=True)

# #     # --- curves (identical to the pipeline's) ---
# #     setup_app_dir(project_root)
# #     from phenology_leafshed import LeafShedConfig, compute_leafshed_scores
# #     features_df = load_features_df(phenology_dir)
# #     om_ids = om_ids_from_features(features_df)
# #     cfg = LeafShedConfig(veg_min_threshold=args.veg_min, ds_threshold=args.ds_thresh)
# #     _s, pp_df, _n = compute_leafshed_scores(features_df, om_ids=om_ids, cfg=cfg)
# #     pp_df["chain_id"] = pp_df["chain_id"].astype(int)
# #     pp_df["om_id"] = pp_df["om_id"].astype(int)

# #     curves = {}
# #     for cid, sub in pp_df.groupby("chain_id"):
# #         sub = sub.sort_values("om_id")
# #         curves[int(cid)] = (sub["om_id"].to_numpy(float),
# #                             sub["veg_fraction_hsv_norm"].to_numpy(float),
# #                             sub["veg_fraction_hsv_hat"].to_numpy(float))

# #     # --- labels ---
# #     train_lab = load_labels(Path(args.train_labels).resolve(), args.train_sheet)
# #     test_lab = load_labels(Path(args.test_labels).resolve(), args.test_sheet)
# #     print(f"Train label chains: {len(train_lab)}  |  Test label chains: {len(test_lab)}")

# #     overlap = sorted(set(train_lab["chain_id"]) & set(test_lab["chain_id"]))
# #     if overlap:
# #         print(f"[warn] train/test overlap: {overlap}")
# #         if args.drop_overlap:
# #             test_lab = test_lab[~test_lab["chain_id"].isin(overlap)].copy()
# #             print(f"       dropped from test; {len(test_lab)} test chains remain")

# #     # --- assemble features ---
# #     span = args.leafoff_span
# #     train_df = assemble(train_lab, curves, span, "train")
# #     test_df = assemble(test_lab, curves, span, "test")
# #     if train_df.empty or test_df.empty:
# #         print("ERROR: no usable rows after assembling features.")
# #         return 1

# #     train_df.to_csv(out_dir / "features_train.csv", index=False)
# #     test_df.to_csv(out_dir / "features_test.csv", index=False)

# #     tr = train_df.dropna(subset=["true_state"]).copy()
# #     te = test_df.dropna(subset=["true_state"]).copy()
# #     print(f"\nUsable per-OM rows -> train: {len(tr)}  test: {len(te)}")
# #     print(f"Train class balance: {tr['true_state'].value_counts().to_dict()}")
# #     print(f"Test  class balance: {te['true_state'].value_counts().to_dict()}")

# #     Xtr = tr[FEATURE_COLS].to_numpy(float)
# #     ytr = tr["true_state"].map(STATE_TO_INT).to_numpy()
# #     Xte = te[FEATURE_COLS].to_numpy(float)
# #     yte = te["true_state"].map(STATE_TO_INT).to_numpy()

# #     present = sorted(set(ytr))
# #     if len(present) < 2:
# #         print("ERROR: training data has <2 phenophase classes; cannot train.")
# #         return 1

# #     # class weights to counter the leaf_on majority
# #     models = {}
# #     if args.model in ("both", "logreg"):
# #         models["logreg"] = Pipeline([
# #             ("scale", StandardScaler()),
# #             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
# #         ])
# #     if args.model in ("both", "gb"):
# #         models["gb"] = HistGradientBoostingClassifier(
# #             max_iter=300, learning_rate=0.08, max_depth=3,
# #             l2_regularization=1.0, random_state=args.seed)

# #     report = []
# #     report.append("=" * 66)
# #     report.append("PER-OM PHENOPHASE CLASSIFIER  (leaf_on / transitioning / leaf_off)")
# #     report.append("=" * 66)
# #     report.append(f"Train rows: {len(tr)}  Test rows: {len(te)}  leafoff_span={span}")
# #     report.append(f"Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
# #     report.append(f"Train class balance: {tr['true_state'].value_counts().to_dict()}")
# #     report.append(f"Test  class balance: {te['true_state'].value_counts().to_dict()}")
# #     report.append("")

# #     # baseline: the old threshold rule for reference
# #     def threshold_pred(df, on=0.65, off=0.20):
# #         v = df["veg_norm"].to_numpy(float)
# #         out = np.where(v >= on, STATE_TO_INT["leaf_on"],
# #               np.where(v <= off, STATE_TO_INT["leaf_off"], STATE_TO_INT["transitioning"]))
# #         return out
# #     base_pred = threshold_pred(te)
# #     base_acc = float((base_pred == yte).mean())
# #     report.append(f"[baseline] threshold rule (on=0.65, off=0.20) test accuracy = {base_acc:.4f}")
# #     report.append("")

# #     cv_rows = []
# #     n_splits = min(args.cv_folds, min(np.bincount(ytr)[present]))
# #     n_splits = max(2, n_splits)
# #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

# #     results = {}
# #     for name, model in models.items():
# #         # --- CV on training ---
# #         try:
# #             cv_pred = cross_val_predict(model, Xtr, ytr, cv=skf)
# #             cv_acc = float((cv_pred == ytr).mean())
# #             cvm = per_class_metrics(ytr, cv_pred)
# #             cv_macro_f1 = float(cvm.loc[cvm["phenophase"] == "macro avg", "f1"].iloc[0])
# #         except Exception as e:
# #             cv_acc, cv_macro_f1 = float("nan"), float("nan")
# #             print(f"[warn] CV failed for {name}: {e}")
# #         cv_rows.append({"model": name, "cv_accuracy": cv_acc, "cv_macro_f1": cv_macro_f1,
# #                         "n_splits": n_splits})

# #         # --- fit on all training, evaluate on held-out test ---
# #         model.fit(Xtr, ytr)
# #         yp = model.predict(Xte)
# #         acc = float((yp == yte).mean())
# #         m = per_class_metrics(yte, yp)
# #         cm = confusion(yte, yp)
# #         m.to_csv(out_dir / f"test_metrics_{name}.csv", index=False)
# #         cm.to_csv(out_dir / f"test_confusion_{name}.csv")
# #         te_out = te[["chain_id", "om_id", "veg_norm", "d_trough", "true_state"]].copy()
# #         te_out["pred_state"] = [STATES[i] for i in yp]
# #         te_out.to_csv(out_dir / f"test_predictions_{name}.csv", index=False)
# #         results[name] = {"acc": acc, "metrics": m, "cm": cm, "cv_acc": cv_acc,
# #                          "cv_macro_f1": cv_macro_f1}

# #         # logreg coefficients (interpretable)
# #         if name == "logreg":
# #             clf = model.named_steps["clf"]
# #             coef = pd.DataFrame(clf.coef_, columns=FEATURE_COLS)
# #             coef.insert(0, "class", [STATES[c] for c in clf.classes_])
# #             coef.to_csv(out_dir / "logreg_coefficients.csv", index=False)

# #     pd.DataFrame(cv_rows).to_csv(out_dir / "cv_results.csv", index=False)

# #     # --- report ---
# #     for name in results:
# #         r = results[name]
# #         report.append("-" * 66)
# #         report.append(f"MODEL: {name}")
# #         report.append("-" * 66)
# #         report.append(f"  Stratified {n_splits}-fold CV accuracy on training: {r['cv_acc']:.4f}")
# #         report.append(f"  CV macro-F1 on training:                    {r['cv_macro_f1']:.4f}")
# #         report.append(f"  HELD-OUT TEST accuracy:                     {r['acc']:.4f} ({r['acc']:.1%})")
# #         report.append("")
# #         report.append("  Confusion matrix (rows=true, cols=pred):")
# #         for line in r["cm"].to_string().splitlines():
# #             report.append("    " + line)
# #         report.append("")
# #         m = r["metrics"]
# #         report.append(f"  {'phenophase':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}")
# #         for _, x in m.iterrows():
# #             report.append(f"  {x['phenophase']:<16}{x['precision']:>10.4f}"
# #                           f"{x['recall']:>10.4f}{x['f1']:>10.4f}{int(x['support']):>9d}")
# #         report.append("")

# #     # --- comparison verdict ---
# #     report.append("=" * 66)
# #     report.append("COMPARISON")
# #     report.append("=" * 66)
# #     report.append(f"{'model':<14}{'CV acc':>9}{'test acc':>10}{'test macroF1':>14}")
# #     report.append(f"{'threshold':<14}{'-':>9}{base_acc:>10.3f}{'-':>14}")
# #     for name in results:
# #         r = results[name]
# #         mf = r["metrics"].loc[r["metrics"]["phenophase"] == "macro avg", "f1"].iloc[0]
# #         report.append(f"{name:<14}{r['cv_acc']:>9.3f}{r['acc']:>10.3f}{mf:>14.3f}")
# #     best = max(results, key=lambda n: results[n]["acc"])
# #     report.append("")
# #     report.append(f"Best test accuracy: {best} ({results[best]['acc']:.1%}) "
# #                   f"vs threshold baseline {base_acc:.1%} "
# #                   f"(+{results[best]['acc'] - base_acc:.1%}).")
# #     report.append("")
# #     report.append("The gain over the threshold comes from the trough-relative and")
# #     report.append("slope features (d_trough, side, slope_local, veg_hat_z), which let")
# #     report.append("the model tell a low-veg_norm post-recovery OM (leaf_on) apart from")
# #     report.append("a low-veg_norm trough OM (leaf_off) -- something no single threshold can do.")

# #     txt = "\n".join(report)
# #     print("\n" + txt)
# #     (out_dir / "model_comparison.txt").write_text(txt, encoding="utf-8")
# #     print(f"\nOutput in: {out_dir}")
# #     return 0


# # if __name__ == "__main__":
# #     raise SystemExit(main())

# #!/usr/bin/env python3

# #-----------------------------------------------------------------------------------------------------
# # """
# # Per-OM phenophase classifier: leaf_on / transitioning / leaf_off.

# # Why this exists
# # ---------------
# # The threshold rule (veg_norm >= on -> leaf_on, etc.) looks at each OM in
# # isolation and tops out around ~58% accuracy, because a single veg_norm value is
# # ambiguous: 0.10 can be the trough (leaf_off) OR a post-recovery OM on a
# # multi-cycle chain (leaf_on). A threshold cannot tell those apart.

# # This script instead trains a CLASSIFIER on per-OM features that carry the
# # time-series context the threshold throws away -- most importantly WHERE the OM
# # sits relative to the trough and which direction the curve is moving. This is the
# # same feature-based modelling idea used for the deciduous/evergreen DS classifier,
# # applied to the per-OM phenophase problem.

# # Two models are trained and compared:
# #     - Logistic regression  (interpretable, coefficients reportable -- like DS)
# #     - Gradient boosting     (HistGradientBoosting; higher ceiling, less interpretable)

# # Ground truth
# # ------------
# # Per-OM state is derived from your hand-labeled event OMs. The TRAINING file may
# # carry a second cycle (columns *_again); both cycles are honoured so post-recovery
# # OMs that then drop again are labeled correctly. Rule per OM, given cycle events
# # (s = leaf_off_start, t = full_leaf_off, r = leaf_on_return) for each cycle:
# #     OM <  first s                          -> leaf_on
# #     within +/-span of any t                -> leaf_off
# #     s <= OM < t  or  t < OM < r  (any cyc) -> transitioning
# #     OM >= last r                           -> leaf_on
# #     between two cycles (r1 <= OM < s2)      -> leaf_on
# # Chains/OMs the rule cannot resolve are dropped from training (not guessed).

# # Features per (chain, OM)  [the "methodology" you asked for]
# # -----------------------------------------------------------
# #     veg_norm                 normalized veg fraction at this OM (the old feature)
# #     veg_hat                  raw interpolated veg fraction
# #     veg_hat_z                per-chain z-score of veg_hat (cycle-robust level)
# #     d_trough                 signed OM distance to the trough (OM - trough_om)
# #     abs_d_trough             |OM - trough_om|
# #     side                     -1 before trough, 0 at, +1 after   (time direction)
# #     slope_local              veg_norm[i+1]-veg_norm[i-1] (rising/falling)
# #     slope_back               veg_norm[i]-veg_norm[i-1]
# #     veg_norm_prev/next       neighbours (local shape)
# #     rank_in_chain            veg_norm percentile rank within the chain
# #     frac_of_max              veg_hat / chain-max veg_hat  (recovery level)
# #     om_frac                  OM position in series (0..1)
# # These are exactly the signals a threshold ignores.

# # Usage
# # -----
# #     python 10_phenophase_classifier.py --config /path/to/pipeline_config.json \\
# #         --train-labels /path/to/leaf_leafoff_validation.xlsx \\
# #         --test-labels  /path/to/test_leafonoff.xlsx \\
# #         [--leafoff-span 0] [--drop-overlap] [--model both|logreg|gb]

# # Outputs (under <phenology_dir>/validation/phenophase_clf/)
# # -----------------------------------------------------------
# #     features_train.csv / features_test.csv   the engineered feature tables
# #     cv_results.csv                            stratified k-fold CV on training
# #     logreg_coefficients.csv                   per-class LR coefficients (interpretable)
# #     test_metrics_logreg.csv / _gb.csv         precision/recall/F1 on the held-out test
# #     test_confusion_logreg.csv / _gb.csv       confusion matrices
# #     test_predictions_logreg.csv / _gb.csv     per-OM predictions on test
# #     model_comparison.txt                      the summary you read
# # """

# # from __future__ import annotations

# # import argparse
# # import json
# # import sys
# # from pathlib import Path
# # from typing import Dict, List, Optional, Tuple

# # import numpy as np
# # import pandas as pd

# # sys.path.insert(0, str(Path(__file__).resolve().parent))
# # from phenology_validation_common import (  # noqa: E402
# #     load_config,
# #     load_features_df,
# #     om_ids_from_features,
# #     setup_app_dir,
# # )

# # STATES = ["leaf_on", "transitioning", "leaf_off"]
# # STATE_TO_INT = {s: i for i, s in enumerate(STATES)}


# # # ---------------------------------------------------------------------------
# # # Ground-truth per-OM state from labeled events (supports 2 cycles)
# # # ---------------------------------------------------------------------------
# # def cycles_from_row(row: pd.Series) -> List[Tuple[float, float, float]]:
# #     """Return list of (s, t, r) cycles present in a label row."""
# #     cyc = []
# #     s, t, r = row.get("leaf_off_start_om"), row.get("full_leaf_off_om"), row.get("leaf_on_return_om")
# #     if np.isfinite(t):
# #         cyc.append((s, t, r))
# #     s2, t2, r2 = (row.get("leaf_off_start_om_again"), row.get("full_leaf_off_om_again"),
# #                   row.get("leaf_on_return_om_again"))
# #     if s2 is not None and np.isfinite(t2):
# #         cyc.append((s2, t2, r2))
# #     return cyc


# # def true_state_multi(om: float, cycles: List[Tuple[float, float, float]], span: int) -> Optional[str]:
# #     """Per-OM state honouring one or two labeled cycles."""
# #     if not cycles:
# #         return None
# #     # leaf_off if within span of any trough
# #     for (s, t, r) in cycles:
# #         if np.isfinite(t) and abs(om - t) <= span:
# #             return "leaf_off"
# #     first_s = cycles[0][0]
# #     last_r = cycles[-1][2]
# #     # before the very first drop
# #     if np.isfinite(first_s) and om < first_s:
# #         return "leaf_on"
# #     # after the very last return
# #     if np.isfinite(last_r) and om >= last_r:
# #         return "leaf_on"
# #     # inside any cycle's transition arms
# #     for (s, t, r) in cycles:
# #         if np.isfinite(s) and np.isfinite(t) and s <= om < t:
# #             return "transitioning"
# #         if np.isfinite(t) and np.isfinite(r) and t < om < r:
# #             return "transitioning"
# #     # gap between cycle1 return and cycle2 start -> leafy plateau
# #     if len(cycles) == 2:
# #         r1 = cycles[0][2]
# #         s2 = cycles[1][0]
# #         if np.isfinite(r1) and np.isfinite(s2) and r1 <= om < s2:
# #             return "leaf_on"
# #     return None


# # # ---------------------------------------------------------------------------
# # # Feature engineering per (chain, OM)
# # # ---------------------------------------------------------------------------
# # def build_features_for_chain(oms: np.ndarray, veg_norm: np.ndarray,
# #                              veg_hat: np.ndarray) -> pd.DataFrame:
# #     n = len(oms)
# #     order = np.argsort(oms)
# #     oms, veg_norm, veg_hat = oms[order], veg_norm[order], veg_hat[order]

# #     trough_idx = int(np.nanargmin(veg_hat)) if np.isfinite(veg_hat).any() else 0
# #     trough_om = float(oms[trough_idx])

# #     vh = veg_hat.astype(float)
# #     mu, sd = np.nanmean(vh), np.nanstd(vh)
# #     vh_z = (vh - mu) / sd if sd > 1e-9 else np.zeros_like(vh)
# #     vmax = np.nanmax(vh) if np.isfinite(vh).any() else 1.0
# #     frac_of_max = vh / vmax if vmax > 1e-9 else np.zeros_like(vh)

# #     # percentile rank of veg_norm within chain
# #     rank = pd.Series(veg_norm).rank(pct=True).to_numpy()

# #     rows = []
# #     for i in range(n):
# #         prev_vn = veg_norm[i - 1] if i > 0 else veg_norm[i]
# #         next_vn = veg_norm[i + 1] if i < n - 1 else veg_norm[i]
# #         slope_local = (next_vn - prev_vn) / 2.0
# #         slope_back = veg_norm[i] - prev_vn
# #         d_trough = float(oms[i] - trough_om)
# #         sd_sign = float(np.sign(d_trough))
# #         rows.append({
# #             "om_id": int(oms[i]),
# #             "veg_norm": float(veg_norm[i]),
# #             "veg_hat": float(veg_hat[i]),
# #             "veg_hat_z": float(vh_z[i]),
# #             "d_trough": d_trough,
# #             "abs_d_trough": abs(d_trough),
# #             "side": sd_sign,
# #             "slope_local": float(slope_local),
# #             "slope_back": float(slope_back),
# #             "veg_norm_prev": float(prev_vn),
# #             "veg_norm_next": float(next_vn),
# #             "rank_in_chain": float(rank[i]),
# #             "frac_of_max": float(frac_of_max[i]),
# #             "om_frac": float((oms[i] - oms.min()) / max(oms.max() - oms.min(), 1)),
# #             # --- interaction features: let a LINEAR model express "low veg AND
# #             #     after trough AND rising" = leaf_on, which pure logreg cannot ---
# #             "veg_x_side": float(veg_norm[i]) * sd_sign,
# #             "veg_x_slope": float(veg_norm[i]) * float(slope_local),
# #             "side_x_slope": sd_sign * float(slope_local),
# #             "fracmax_x_side": float(frac_of_max[i]) * sd_sign,
# #             "vegz_x_side": float(vh_z[i]) * sd_sign,
# #             "trough_om": trough_om,
# #         })
# #     return pd.DataFrame(rows)


# # FEATURE_COLS = [
# #     "veg_norm", "veg_hat", "veg_hat_z", "d_trough", "abs_d_trough", "side",
# #     "slope_local", "slope_back", "veg_norm_prev", "veg_norm_next",
# #     "rank_in_chain", "frac_of_max", "om_frac",
# #     "veg_x_side", "veg_x_slope", "side_x_slope", "fracmax_x_side", "vegz_x_side",
# # ]


# # # ---------------------------------------------------------------------------
# # # Metrics
# # # ---------------------------------------------------------------------------
# # def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
# #     rows = []
# #     macro_p, macro_r, macro_f = [], [], []
# #     n = len(y_true)
# #     for st in STATES:
# #         i = STATE_TO_INT[st]
# #         tp = int(((y_true == i) & (y_pred == i)).sum())
# #         fp = int(((y_true != i) & (y_pred == i)).sum())
# #         fn = int(((y_true == i) & (y_pred != i)).sum())
# #         tn = n - tp - fp - fn
# #         p = tp / (tp + fp) if (tp + fp) else np.nan
# #         r = tp / (tp + fn) if (tp + fn) else np.nan
# #         f = 2 * p * r / (p + r) if (np.isfinite(p) and np.isfinite(r) and p + r > 0) else np.nan
# #         sup = int((y_true == i).sum())
# #         rows.append({"phenophase": st, "precision": p, "recall": r, "f1": f,
# #                      "support": sup, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
# #         if sup > 0:
# #             macro_p.append(p); macro_r.append(r); macro_f.append(f)
# #     sups = np.array([r["support"] for r in rows], dtype=float)
# #     tot = sups.sum()
# #     class_rows = list(rows)  # snapshot of the 3 per-class rows only
# #     def wavg(key):
# #         return float(np.nansum([r[key] * r["support"] for r in class_rows]) / tot) if tot else np.nan
# #     rows.append({"phenophase": "macro avg", "precision": float(np.nanmean(macro_p)),
# #                  "recall": float(np.nanmean(macro_r)), "f1": float(np.nanmean(macro_f)),
# #                  "support": int(tot), "TP": None, "FP": None, "FN": None, "TN": None})
# #     rows.append({"phenophase": "weighted avg", "precision": wavg("precision"),
# #                  "recall": wavg("recall"), "f1": wavg("f1"),
# #                  "support": int(tot), "TP": None, "FP": None, "FN": None, "TN": None})
# #     return pd.DataFrame(rows)


# # def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
# #     cm = pd.DataFrame(0, index=STATES, columns=STATES)
# #     for t, p in zip(y_true, y_pred):
# #         cm.iloc[t, p] += 1
# #     cm.index.name = "true"; cm.columns.name = "predicted"
# #     return cm


# # # ---------------------------------------------------------------------------
# # # Label loading + feature assembly
# # # ---------------------------------------------------------------------------
# # def load_labels(path: Path, sheet) -> pd.DataFrame:
# #     if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
# #         sh = int(sheet) if str(sheet).isdigit() else sheet
# #         df = pd.read_excel(path, sheet_name=sh)
# #     else:
# #         df = pd.read_csv(path)
# #     for c in df.columns:
# #         df[c] = pd.to_numeric(df[c], errors="coerce")
# #     df = df.dropna(subset=["chain_id"]).copy()
# #     df["chain_id"] = df["chain_id"].astype(int)
# #     return df


# # def assemble(labels: pd.DataFrame, curves: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
# #              span: int, split_name: str) -> pd.DataFrame:
# #     frames = []
# #     n_dropped_state = 0
# #     for _, row in labels.iterrows():
# #         cid = int(row["chain_id"])
# #         if cid not in curves:
# #             continue
# #         oms, vn, vh = curves[cid]
# #         feats = build_features_for_chain(oms, vn, vh)
# #         cyc = cycles_from_row(row)
# #         states = [true_state_multi(float(o), cyc, span) for o in feats["om_id"]]
# #         feats["chain_id"] = cid
# #         feats["true_state"] = states
# #         n_dropped_state += sum(s is None for s in states)
# #         frames.append(feats)
# #     if not frames:
# #         return pd.DataFrame()
# #     out = pd.concat(frames, ignore_index=True)
# #     out["split"] = split_name
# #     return out


# # # ---------------------------------------------------------------------------
# # # Main
# # # ---------------------------------------------------------------------------
# # def main() -> int:
# #     ap = argparse.ArgumentParser(description="Per-OM phenophase classifier (logreg + gradient boosting)")
# #     ap.add_argument("--config", required=True)
# #     ap.add_argument("--train-labels", required=True)
# #     ap.add_argument("--test-labels", required=True)
# #     ap.add_argument("--train-sheet", default=0)
# #     ap.add_argument("--test-sheet", default=0)
# #     ap.add_argument("--leafoff-span", type=int, default=0,
# #                     help="OMs within +/-K of a labeled trough count as leaf_off")
# #     ap.add_argument("--drop-overlap", action="store_true",
# #                     help="Drop test chains that also appear in the training file")
# #     ap.add_argument("--merge-transitioning", choices=["none", "leaf_off", "leaf_on"],
# #                     default="none",
# #                     help="Fold the rare 'transitioning' class into leaf_off or leaf_on "
# #                          "and evaluate as a 2-class problem. 'transitioning' is often "
# #                          "too rare (here ~3%% of test OMs) to learn or evaluate reliably; "
# #                          "merging gives a stable 2-class leaf-on/leaf-off result.")
# #     ap.add_argument("--model", choices=["both", "logreg", "gb"], default="both")
# #     ap.add_argument("--cv-folds", type=int, default=5)
# #     ap.add_argument("--veg-min", type=float, default=0.45)
# #     ap.add_argument("--ds-thresh", type=float, default=0.70)
# #     ap.add_argument("--seed", type=int, default=42)
# #     args = ap.parse_args()

# #     try:
# #         from sklearn.linear_model import LogisticRegression
# #         from sklearn.ensemble import HistGradientBoostingClassifier
# #         from sklearn.preprocessing import StandardScaler
# #         from sklearn.pipeline import Pipeline
# #         from sklearn.model_selection import StratifiedKFold, cross_val_predict
# #     except Exception as e:
# #         print(f"ERROR: scikit-learn is required: {e}")
# #         return 1

# #     config = load_config(Path(args.config).resolve())
# #     project_root = Path(config["project_root"])
# #     phenology_dir = Path(config["phenology_dir"])
# #     out_dir = phenology_dir / "validation" / "phenophase_clf"
# #     out_dir.mkdir(parents=True, exist_ok=True)

# #     # --- curves (identical to the pipeline's) ---
# #     setup_app_dir(project_root)
# #     from phenology_leafshed import LeafShedConfig, compute_leafshed_scores
# #     features_df = load_features_df(phenology_dir)
# #     om_ids = om_ids_from_features(features_df)
# #     cfg = LeafShedConfig(veg_min_threshold=args.veg_min, ds_threshold=args.ds_thresh)
# #     _s, pp_df, _n = compute_leafshed_scores(features_df, om_ids=om_ids, cfg=cfg)
# #     pp_df["chain_id"] = pp_df["chain_id"].astype(int)
# #     pp_df["om_id"] = pp_df["om_id"].astype(int)

# #     curves = {}
# #     for cid, sub in pp_df.groupby("chain_id"):
# #         sub = sub.sort_values("om_id")
# #         curves[int(cid)] = (sub["om_id"].to_numpy(float),
# #                             sub["veg_fraction_hsv_norm"].to_numpy(float),
# #                             sub["veg_fraction_hsv_hat"].to_numpy(float))

# #     # --- labels ---
# #     train_lab = load_labels(Path(args.train_labels).resolve(), args.train_sheet)
# #     test_lab = load_labels(Path(args.test_labels).resolve(), args.test_sheet)
# #     print(f"Train label chains: {len(train_lab)}  |  Test label chains: {len(test_lab)}")

# #     overlap = sorted(set(train_lab["chain_id"]) & set(test_lab["chain_id"]))
# #     if overlap:
# #         print(f"[warn] train/test overlap: {overlap}")
# #         if args.drop_overlap:
# #             test_lab = test_lab[~test_lab["chain_id"].isin(overlap)].copy()
# #             print(f"       dropped from test; {len(test_lab)} test chains remain")

# #     # --- assemble features ---
# #     span = args.leafoff_span
# #     train_df = assemble(train_lab, curves, span, "train")
# #     test_df = assemble(test_lab, curves, span, "test")
# #     if train_df.empty or test_df.empty:
# #         print("ERROR: no usable rows after assembling features.")
# #         return 1

# #     train_df.to_csv(out_dir / "features_train.csv", index=False)
# #     test_df.to_csv(out_dir / "features_test.csv", index=False)

# #     tr = train_df.dropna(subset=["true_state"]).copy()
# #     te = test_df.dropna(subset=["true_state"]).copy()

# #     # --- optionally merge the rare transitioning class ---
# #     global STATES, STATE_TO_INT
# #     if args.merge_transitioning != "none":
# #         tgt = args.merge_transitioning
# #         tr["true_state"] = tr["true_state"].replace("transitioning", tgt)
# #         te["true_state"] = te["true_state"].replace("transitioning", tgt)
# #         STATES = ["leaf_on", "leaf_off"]
# #         STATE_TO_INT = {s: i for i, s in enumerate(STATES)}
# #         print(f"\n[merge] 'transitioning' folded into '{tgt}' -> 2-class problem "
# #               f"(leaf_on / leaf_off)")

# #     print(f"\nUsable per-OM rows -> train: {len(tr)}  test: {len(te)}")
# #     print(f"Train class balance: {tr['true_state'].value_counts().to_dict()}")
# #     print(f"Test  class balance: {te['true_state'].value_counts().to_dict()}")

# #     Xtr = tr[FEATURE_COLS].to_numpy(float)
# #     ytr = tr["true_state"].map(STATE_TO_INT).to_numpy()
# #     Xte = te[FEATURE_COLS].to_numpy(float)
# #     yte = te["true_state"].map(STATE_TO_INT).to_numpy()

# #     present = sorted(set(ytr))
# #     if len(present) < 2:
# #         print("ERROR: training data has <2 phenophase classes; cannot train.")
# #         return 1

# #     # class weights to counter the leaf_on majority
# #     models = {}
# #     if args.model in ("both", "logreg"):
# #         models["logreg"] = Pipeline([
# #             ("scale", StandardScaler()),
# #             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
# #         ])
# #     if args.model in ("both", "gb"):
# #         models["gb"] = HistGradientBoostingClassifier(
# #             max_iter=300, learning_rate=0.08, max_depth=3,
# #             l2_regularization=1.0, random_state=args.seed)

# #     report = []
# #     report.append("=" * 66)
# #     report.append("PER-OM PHENOPHASE CLASSIFIER  (leaf_on / transitioning / leaf_off)")
# #     report.append("=" * 66)
# #     report.append(f"Train rows: {len(tr)}  Test rows: {len(te)}  leafoff_span={span}")
# #     report.append(f"Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
# #     report.append(f"Train class balance: {tr['true_state'].value_counts().to_dict()}")
# #     report.append(f"Test  class balance: {te['true_state'].value_counts().to_dict()}")
# #     report.append("")

# #     # baseline: the old threshold rule for reference
# #     def threshold_pred(df, on=0.65, off=0.20):
# #         v = df["veg_norm"].to_numpy(float)
# #         if "transitioning" in STATE_TO_INT:
# #             out = np.where(v >= on, STATE_TO_INT["leaf_on"],
# #                   np.where(v <= off, STATE_TO_INT["leaf_off"], STATE_TO_INT["transitioning"]))
# #         else:
# #             # 2-class: midpoint split between the two thresholds
# #             mid = (on + off) / 2.0
# #             out = np.where(v >= mid, STATE_TO_INT["leaf_on"], STATE_TO_INT["leaf_off"])
# #         return out
# #     base_pred = threshold_pred(te)
# #     base_acc = float((base_pred == yte).mean())
# #     report.append(f"[baseline] threshold rule (on=0.65, off=0.20) test accuracy = {base_acc:.4f}")
# #     report.append("")

# #     cv_rows = []
# #     n_splits = min(args.cv_folds, min(np.bincount(ytr)[present]))
# #     n_splits = max(2, n_splits)
# #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

# #     results = {}
# #     for name, model in models.items():
# #         # --- CV on training ---
# #         try:
# #             cv_pred = cross_val_predict(model, Xtr, ytr, cv=skf)
# #             cv_acc = float((cv_pred == ytr).mean())
# #             cvm = per_class_metrics(ytr, cv_pred)
# #             cv_macro_f1 = float(cvm.loc[cvm["phenophase"] == "macro avg", "f1"].iloc[0])
# #         except Exception as e:
# #             cv_acc, cv_macro_f1 = float("nan"), float("nan")
# #             print(f"[warn] CV failed for {name}: {e}")
# #         cv_rows.append({"model": name, "cv_accuracy": cv_acc, "cv_macro_f1": cv_macro_f1,
# #                         "n_splits": n_splits})

# #         # --- fit on all training, evaluate on held-out test ---
# #         model.fit(Xtr, ytr)
# #         yp = model.predict(Xte)
# #         acc = float((yp == yte).mean())
# #         m = per_class_metrics(yte, yp)
# #         cm = confusion(yte, yp)
# #         m.to_csv(out_dir / f"test_metrics_{name}.csv", index=False)
# #         cm.to_csv(out_dir / f"test_confusion_{name}.csv")
# #         te_out = te[["chain_id", "om_id", "veg_norm", "d_trough", "true_state"]].copy()
# #         te_out["pred_state"] = [STATES[i] for i in yp]
# #         te_out.to_csv(out_dir / f"test_predictions_{name}.csv", index=False)
# #         results[name] = {"acc": acc, "metrics": m, "cm": cm, "cv_acc": cv_acc,
# #                          "cv_macro_f1": cv_macro_f1}

# #         # logreg coefficients (interpretable)
# #         if name == "logreg":
# #             clf = model.named_steps["clf"]
# #             coef_arr = clf.coef_
# #             if coef_arr.shape[0] == 1:
# #                 # binary case: one row separates class[1] from class[0]
# #                 coef = pd.DataFrame(coef_arr, columns=FEATURE_COLS)
# #                 coef.insert(0, "class", [f"{STATES[clf.classes_[1]]}_vs_{STATES[clf.classes_[0]]}"])
# #             else:
# #                 coef = pd.DataFrame(coef_arr, columns=FEATURE_COLS)
# #                 coef.insert(0, "class", [STATES[c] for c in clf.classes_])
# #             coef.to_csv(out_dir / "logreg_coefficients.csv", index=False)

# #     pd.DataFrame(cv_rows).to_csv(out_dir / "cv_results.csv", index=False)

# #     # --- report ---
# #     for name in results:
# #         r = results[name]
# #         report.append("-" * 66)
# #         report.append(f"MODEL: {name}")
# #         report.append("-" * 66)
# #         report.append(f"  Stratified {n_splits}-fold CV accuracy on training: {r['cv_acc']:.4f}")
# #         report.append(f"  CV macro-F1 on training:                    {r['cv_macro_f1']:.4f}")
# #         report.append(f"  HELD-OUT TEST accuracy:                     {r['acc']:.4f} ({r['acc']:.1%})")
# #         report.append("")
# #         report.append("  Confusion matrix (rows=true, cols=pred):")
# #         for line in r["cm"].to_string().splitlines():
# #             report.append("    " + line)
# #         report.append("")
# #         m = r["metrics"]
# #         report.append(f"  {'phenophase':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}")
# #         for _, x in m.iterrows():
# #             report.append(f"  {x['phenophase']:<16}{x['precision']:>10.4f}"
# #                           f"{x['recall']:>10.4f}{x['f1']:>10.4f}{int(x['support']):>9d}")
# #         report.append("")

# #     # --- comparison verdict ---
# #     report.append("=" * 66)
# #     report.append("COMPARISON")
# #     report.append("=" * 66)
# #     report.append(f"{'model':<14}{'CV acc':>9}{'test acc':>10}{'test macroF1':>14}")
# #     report.append(f"{'threshold':<14}{'-':>9}{base_acc:>10.3f}{'-':>14}")
# #     for name in results:
# #         r = results[name]
# #         mf = r["metrics"].loc[r["metrics"]["phenophase"] == "macro avg", "f1"].iloc[0]
# #         report.append(f"{name:<14}{r['cv_acc']:>9.3f}{r['acc']:>10.3f}{mf:>14.3f}")
# #     best = max(results, key=lambda n: results[n]["acc"])
# #     report.append("")
# #     report.append(f"Best test accuracy: {best} ({results[best]['acc']:.1%}) "
# #                   f"vs threshold baseline {base_acc:.1%} "
# #                   f"(+{results[best]['acc'] - base_acc:.1%}).")
# #     report.append("")
# #     report.append("The gain over the threshold comes from the trough-relative and")
# #     report.append("slope features (d_trough, side, slope_local, veg_hat_z), which let")
# #     report.append("the model tell a low-veg_norm post-recovery OM (leaf_on) apart from")
# #     report.append("a low-veg_norm trough OM (leaf_off) -- something no single threshold can do.")

# #     txt = "\n".join(report)
# #     print("\n" + txt)
# #     (out_dir / "model_comparison.txt").write_text(txt, encoding="utf-8")
# #     print(f"\nOutput in: {out_dir}")
# #     return 0


# # if __name__ == "__main__":
# #     raise SystemExit(main())
# #-----------------------------------------------------------------------------------------------------



# #!/usr/bin/env python3
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
# # Local extrema detection (window-free, site-independent)
# # ---------------------------------------------------------------------------
# def find_local_extrema(y: np.ndarray, min_prominence: float = 0.10):
#     """Detect local minima (troughs) and maxima (peaks) in a 1-D curve.

#     Window-free and scale-relative: works for 1-cycle, 2-cycle, or n-cycle
#     chains at any site/cadence. A point is a local min if it is <= both
#     neighbours and forms a dip of at least `min_prominence` (fraction of the
#     curve's own range) below the surrounding peaks. The global argmin is always
#     included as a trough so every chain has at least one.
#     """
#     n = len(y)
#     if n == 0:
#         return [], []
#     yr = np.nan_to_num(y, nan=np.nanmax(y) if np.isfinite(y).any() else 0.0)
#     rng = float(np.nanmax(yr) - np.nanmin(yr))
#     prom = max(min_prominence * rng, 1e-9)

#     troughs, peaks = [], []
#     for i in range(n):
#         lo = yr[i - 1] if i > 0 else yr[i]
#         hi = yr[i + 1] if i < n - 1 else yr[i]
#         if yr[i] <= lo and yr[i] <= hi:
#             # local minimum candidate; check prominence vs neighbours
#             left_max = np.max(yr[:i + 1]) if i > 0 else yr[i]
#             right_max = np.max(yr[i:]) if i < n - 1 else yr[i]
#             if (min(left_max, right_max) - yr[i]) >= prom:
#                 troughs.append(i)
#         if yr[i] >= lo and yr[i] >= hi:
#             left_min = np.min(yr[:i + 1]) if i > 0 else yr[i]
#             right_min = np.min(yr[i:]) if i < n - 1 else yr[i]
#             if (yr[i] - max(left_min, right_min)) >= prom:
#                 peaks.append(i)
#     # guarantee at least the global argmin as a trough
#     gmin = int(np.nanargmin(yr))
#     if gmin not in troughs:
#         troughs.append(gmin)
#     troughs = sorted(set(troughs))
#     peaks = sorted(set(peaks))
#     return troughs, peaks


# # ---------------------------------------------------------------------------
# # Feature engineering per (chain, OM)  -- window-free, generalisable
# # ---------------------------------------------------------------------------
# def build_features_for_chain(oms: np.ndarray, veg_norm: np.ndarray,
#                              veg_hat: np.ndarray) -> pd.DataFrame:
#     n = len(oms)
#     order = np.argsort(oms)
#     oms, veg_norm, veg_hat = oms[order], veg_norm[order], veg_hat[order]

#     vh = veg_hat.astype(float)
#     mu, sd = np.nanmean(vh), np.nanstd(vh)
#     vh_z = (vh - mu) / sd if sd > 1e-9 else np.zeros_like(vh)
#     vmax = np.nanmax(vh) if np.isfinite(vh).any() else 1.0
#     frac_of_max = vh / vmax if vmax > 1e-9 else np.zeros_like(vh)
#     rank = pd.Series(veg_norm).rank(pct=True).to_numpy()

#     # --- local extrema (no window, no site assumption) ---
#     trough_idx_list, peak_idx_list = find_local_extrema(veg_hat)
#     trough_oms = oms[trough_idx_list] if trough_idx_list else np.array([oms[int(np.nanargmin(vh))]])
#     peak_oms = oms[peak_idx_list] if peak_idx_list else np.array([oms[int(np.nanargmax(vh))]])
#     global_trough_om = float(oms[int(np.nanargmin(vh))])

#     # local-baseline normalization: veg relative to the run between the two
#     # nearest surrounding troughs -> does NOT collapse on multi-cycle chains
#     def local_norm(i):
#         left_tr = [t for t in trough_idx_list if t <= i]
#         right_tr = [t for t in trough_idx_list if t >= i]
#         lo_idx = left_tr[-1] if left_tr else 0
#         hi_idx = right_tr[0] if right_tr else n - 1
#         seg = vh[min(lo_idx, hi_idx):max(lo_idx, hi_idx) + 1]
#         if seg.size == 0:
#             return veg_norm[i]
#         smin, smax = float(np.min(seg)), float(np.max(seg))
#         # normalize against local peak on either side
#         window_peak = max(vh[max(0, i - 3):min(n, i + 4)].max(), smax)
#         base = min(smin, vh[max(0, i - 3):min(n, i + 4)].min())
#         return float((vh[i] - base) / (window_peak - base)) if window_peak - base > 1e-9 else 0.5

#     rows = []
#     for i in range(n):
#         prev_vn = veg_norm[i - 1] if i > 0 else veg_norm[i]
#         next_vn = veg_norm[i + 1] if i < n - 1 else veg_norm[i]
#         slope_local = (next_vn - prev_vn) / 2.0
#         slope_back = veg_norm[i] - prev_vn

#         # nearest trough / peak (window-free trough-relative features)
#         d_troughs = trough_oms - oms[i]
#         j = int(np.argmin(np.abs(d_troughs)))
#         d_near_trough = float(oms[i] - trough_oms[j])
#         d_peaks = peak_oms - oms[i]
#         k = int(np.argmin(np.abs(d_peaks)))
#         d_near_peak = float(oms[i] - peak_oms[k])

#         # side relative to NEAREST trough (not the arbitrary global one)
#         side_near = float(np.sign(d_near_trough))
#         is_trough = 1.0 if i in trough_idx_list else 0.0
#         is_peak = 1.0 if i in peak_idx_list else 0.0
#         vn_loc = local_norm(i)

#         rows.append({
#             "om_id": int(oms[i]),
#             "veg_norm": float(veg_norm[i]),
#             "veg_norm_local": float(vn_loc),
#             "veg_hat": float(veg_hat[i]),
#             "veg_hat_z": float(vh_z[i]),
#             "d_near_trough": d_near_trough,
#             "abs_d_near_trough": abs(d_near_trough),
#             "d_near_peak": d_near_peak,
#             "abs_d_near_peak": abs(d_near_peak),
#             "side_near": side_near,
#             "is_trough": is_trough,
#             "is_peak": is_peak,
#             "n_troughs": float(len(trough_idx_list)),
#             "slope_local": float(slope_local),
#             "slope_back": float(slope_back),
#             "veg_norm_prev": float(prev_vn),
#             "veg_norm_next": float(next_vn),
#             "rank_in_chain": float(rank[i]),
#             "frac_of_max": float(frac_of_max[i]),
#             "om_frac": float((oms[i] - oms.min()) / max(oms.max() - oms.min(), 1)),
#             # interactions on the LOCAL (nearest-trough) reference
#             "vegloc_x_side": float(vn_loc) * side_near,
#             "vegloc_x_slope": float(vn_loc) * float(slope_local),
#             "side_x_slope": side_near * float(slope_local),
#             "fracmax_x_side": float(frac_of_max[i]) * side_near,
#             "vegz_x_side": float(vh_z[i]) * side_near,
#             "trough_om": global_trough_om,
#         })
#     return pd.DataFrame(rows)


# FEATURE_COLS = [
#     "veg_norm", "veg_norm_local", "veg_hat", "veg_hat_z",
#     "d_near_trough", "abs_d_near_trough", "d_near_peak", "abs_d_near_peak",
#     "side_near", "is_trough", "is_peak", "n_troughs",
#     "slope_local", "slope_back", "veg_norm_prev", "veg_norm_next",
#     "rank_in_chain", "frac_of_max", "om_frac",
#     "vegloc_x_side", "vegloc_x_slope", "side_x_slope", "fracmax_x_side", "vegz_x_side",
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
#     ap.add_argument("--om-min", type=int, default=None,
#                     help="Restrict curves to OMs >= this before building features")
#     ap.add_argument("--om-max", type=int, default=None,
#                     help="Restrict curves to OMs <= this before building features. "
#                          "CRITICAL for multi-cycle chains: the trough-relative features "
#                          "(side, d_trough) use the argmin of the curve, and if a deeper "
#                          "SECOND cycle exists the global argmin is the wrong trough for "
#                          "your first-cycle labels. Confine to the labeled cycle's window.")
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
#     if args.om_min is not None:
#         pp_df = pp_df[pp_df["om_id"] >= args.om_min]
#     if args.om_max is not None:
#         pp_df = pp_df[pp_df["om_id"] <= args.om_max]
#     if args.om_min is not None or args.om_max is not None:
#         print(f"Feature window restricted to OMs "
#               f"[{args.om_min if args.om_min is not None else 'min'}, "
#               f"{args.om_max if args.om_max is not None else 'max'}] "
#               f"-- trough/side features now reference this window's argmin.")

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
#         te_out = te[["chain_id", "om_id", "veg_norm", "d_near_trough", "true_state"]].copy()
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
#     report.append("The gain over the threshold comes from the window-free local")
#     report.append("trough/peak features (d_near_trough, side_near, veg_norm_local,")
#     report.append("slope_local, veg_hat_z), which reference each OM's NEAREST local")
#     report.append("trough rather than a global argmin. This lets the model tell a")
#     report.append("post-recovery leaf_on OM apart from a genuine trough OM WITHOUT any")
#     report.append("hardcoded OM window -- so it transfers across sites, flight cadences,")
#     report.append("and 1- or 2-cycle trees automatically.")

#     txt = "\n".join(report)
#     print("\n" + txt)
#     (out_dir / "model_comparison.txt").write_text(txt, encoding="utf-8")
#     print(f"\nOutput in: {out_dir}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())

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
    build_aligned_tracker,
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
# Conceptual / teaching figures (static, no run data needed)
# ---------------------------------------------------------------------------
STATE_FIG_COLORS = {"leaf_on": "#2e7d32", "transitioning": "#f9a825", "leaf_off": "#8d6e63"}


def make_concept_figures(out_dir: Path) -> None:
    """feature_explainer.png, norm_explainer.png, feature_list_slide.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D

    C_ON, C_OFF, C_TRAN = "#2e7d32", "#8d6e63", "#f9a825"
    C_LINE, C_TR, C_PK = "#263238", "#c62828", "#1565c0"

    # ---------- FIG 1: how features read a curve ----------
    oms = np.arange(1, 16)
    veg = np.array([0.92, 0.90, 0.55, 0.20, 0.15, 0.45, 0.80, 0.88, 0.86,
                    0.70, 0.40, 0.12, 0.22, 0.55, 0.85])
    trough_idx, peak_idx = [4, 11], [0, 7, 14]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13.5, 10.2),
                                  height_ratios=[2.45, 1.0],
                                  gridspec_kw={"hspace": 0.32})
    bands = [(0.5, 2.5, C_ON), (2.5, 4.5, C_TRAN), (4.5, 5.5, C_OFF),
             (5.5, 6.5, C_TRAN), (6.5, 9.5, C_ON), (9.5, 11.5, C_TRAN),
             (11.5, 12.5, C_OFF), (12.5, 13.5, C_TRAN), (13.5, 15.5, C_ON)]
    for x0, x1, c in bands:
        ax.axvspan(x0, x1, color=c, alpha=0.08, zorder=0)
    ax.plot(oms, veg, "-", color=C_LINE, lw=2.2, zorder=3)
    sc = {1: C_ON, 2: C_ON, 3: C_TRAN, 4: C_TRAN, 5: C_OFF, 6: C_TRAN, 7: C_ON,
          8: C_ON, 9: C_ON, 10: C_TRAN, 11: C_TRAN, 12: C_OFF, 13: C_TRAN,
          14: C_ON, 15: C_ON}
    for o, v in zip(oms, veg):
        ax.plot(o, v, "o", ms=11, color=sc[o], mec="white", mew=1.4, zorder=5)
    for ti in trough_idx:
        ax.plot(oms[ti], veg[ti], "v", ms=17, color=C_TR, zorder=6, mec="white", mew=1.2)
        ax.annotate("TROUGH\n(local min)", (oms[ti], veg[ti]),
                    xytext=(oms[ti], veg[ti] - 0.13), ha="center", va="top",
                    fontsize=8.5, fontweight="bold", color=C_TR)
    for pi in peak_idx:
        ax.plot(oms[pi], veg[pi], "^", ms=14, color=C_PK, zorder=6, mec="white", mew=1.2)
    ax.annotate("PEAK (local max)", (oms[peak_idx[1]], veg[peak_idx[1]]),
                xytext=(oms[peak_idx[1]], veg[peak_idx[1]] + 0.07), ha="center",
                fontsize=8.5, fontweight="bold", color=C_PK)
    for ti in trough_idx:
        ax.axvline(oms[ti], color=C_TR, ls=":", lw=1.1, alpha=0.5, zorder=1)
    ax.annotate("", xy=(2.7, 0.97), xytext=(4.7, 0.97),
                arrowprops=dict(arrowstyle="->", color=C_TRAN, lw=2))
    ax.text(3.5, 1.005, "side_near = \u22121\n(before trough \u2192 DROPPING)",
            ha="center", fontsize=8.2, color=C_TRAN, fontweight="bold")
    ax.annotate("", xy=(5.3, 0.62), xytext=(6.6, 0.62),
                arrowprops=dict(arrowstyle="->", color=C_ON, lw=2))
    ax.text(6.15, 0.66, "side_near = +1\n(after trough \u2192 RECOVERING)",
            ha="center", fontsize=8.2, color=C_ON, fontweight="bold")
    ax.annotate("Low veg + BEFORE trough\n\u2192 leaf_off / dropping",
                (oms[3], veg[3]), xytext=(1.3, 0.33), fontsize=8, color=C_OFF,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_OFF, lw=1.4))
    ax.annotate("Low veg but AFTER trough\n+ rising \u2192 leaf_on returning",
                (oms[12], veg[12]), xytext=(12.6, 0.30), fontsize=8, color=C_ON,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_ON, lw=1.4))
    ax.text(8.0, 0.30, "n_troughs = 2  \u2192  model knows this crown has TWO leaf-off cycles",
            ha="center", fontsize=8.5, style="italic", color=C_LINE,
            bbox=dict(boxstyle="round,pad=0.35", fc="#eceff1", ec="#b0bec5"))
    ax.set_ylim(-0.02, 1.10); ax.set_xlim(0.3, 15.7); ax.set_xticks(oms)
    ax.set_xlabel("OM (flight date) \u2192", fontsize=10.5)
    ax.set_ylabel("vegetation fraction", fontsize=10.5)
    ax.set_title("How the features read one crown's vegetation curve",
                 fontsize=13.5, fontweight="bold", pad=14)
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=C_ON, ms=11, label='leaf_on'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=C_TRAN, ms=11, label='transitioning'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=C_OFF, ms=11, label='leaf_off'),
           Line2D([0], [0], marker='v', color='w', markerfacecolor=C_TR, ms=13, label='trough'),
           Line2D([0], [0], marker='^', color='w', markerfacecolor=C_PK, ms=12, label='peak')]
    ax.legend(handles=leg, loc="upper center", ncol=5, fontsize=9, frameon=True,
              bbox_to_anchor=(0.5, -0.11))
    ax2.axis("off")
    ax2.set_title("What the model 'sees' at three example OMs  (same low-ish veg, different verdicts)",
                  fontsize=11, fontweight="bold", loc="left", y=1.02)

    def card(x, om, vegv, side, slope, dtr, verdict, vcolor, why):
        box = FancyBboxPatch((x, 0.10), 0.29, 0.78, boxstyle="round,pad=0.012",
                             mutation_aspect=0.5, fc="white", ec=vcolor, lw=2.2,
                             transform=ax2.transAxes)
        ax2.add_patch(box)
        ax2.text(x + 0.145, 0.80, f"OM {om}", transform=ax2.transAxes, ha="center",
                 fontsize=11, fontweight="bold", color=vcolor)
        for i, ln in enumerate([f"veg_norm_local : {vegv}", f"side_near      : {side}",
                                f"slope_local    : {slope}", f"d_near_trough  : {dtr}"]):
            ax2.text(x + 0.02, 0.66 - i * 0.115, ln, transform=ax2.transAxes,
                     fontsize=8.6, family="monospace", color=C_LINE)
        ax2.text(x + 0.145, 0.185, f"\u2192  {verdict}", transform=ax2.transAxes,
                 ha="center", fontsize=9.6, fontweight="bold", color=vcolor)
        ax2.text(x + 0.145, 0.03, why, transform=ax2.transAxes, ha="center",
                 fontsize=7.4, style="italic", color="#546e7a")

    card(0.02, 4, "low", "\u22121 (before)", "falling", "\u22121", "leaf_off / dropping",
         C_OFF, "before the trough + falling")
    card(0.355, 6, "low", "+1 (after)", "rising", "+1", "leaf_on returning",
         C_ON, "past the trough + rising")
    card(0.69, 9, "high", "+1 (after)", "flat", "+2", "leaf_on (stable)",
         C_ON, "well past trough + flat + green")
    fig.savefig(str(out_dir / "feature_explainer.png"), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    # ---------- FIG 2: global vs local normalization ----------
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    veg_raw = np.array([0.92, 0.90, 0.55, 0.30, 0.28, 0.50, 0.82, 0.88, 0.86,
                        0.60, 0.35, 0.08, 0.20, 0.55, 0.85])
    g = (veg_raw - veg_raw.min()) / (veg_raw.max() - veg_raw.min())
    loc = veg_raw.copy()
    for seg in (slice(0, 8), slice(8, 15)):
        s = veg_raw[seg]
        loc[seg] = (s - s.min()) / (s.max() - s.min())
    panels = [(bx1, g, "GLOBAL min-max normalization  (the OLD feature)",
               "OM7\u20139 are FULLY LEAFY, but the deeper 2nd trough\ncrushes them \u2014 "
               "OM10\u201311 collapse toward 0, looking\nbare when they are not."),
              (bx2, loc, "LOCAL normalization  (veg_norm_local)",
               "Each cycle scaled to its OWN range, so leafy OMs\nread ~1.0 and bare OMs "
               "~0.0 in BOTH cycles \u2014\nthe deeper 2nd trough no longer distorts cycle 1.")]
    for bx, series, title, note in panels:
        bx.plot(oms, series, "-o", color=C_LINE, ms=7, zorder=3)
        bx.axhline(0.65, color=C_ON, ls="--", lw=1, alpha=0.7)
        bx.axhline(0.20, color=C_OFF, ls="--", lw=1, alpha=0.7)
        bx.axhspan(0.65, 1.05, color=C_ON, alpha=0.06)
        bx.axhspan(-0.05, 0.20, color=C_OFF, alpha=0.06)
        bx.set_ylim(-0.05, 1.08); bx.set_xlim(0.4, 15.6); bx.set_xticks(oms)
        bx.set_xlabel("OM \u2192", fontsize=10); bx.set_ylabel("normalized value", fontsize=10)
        bx.set_title(title, fontsize=11, fontweight="bold")
        bx.text(0.5, -0.34, note, transform=bx.transAxes, ha="center", va="top",
                fontsize=8.6, color="#37474f",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#cfd8dc"))
    for o in [10, 11]:
        bx1.plot(o, g[o - 1], "o", ms=13, mec=C_TR, mfc="none", mew=2.4, zorder=5)
        bx2.plot(o, loc[o - 1], "o", ms=13, mec=C_ON, mfc="none", mew=2.4, zorder=5)
    fig2.suptitle("The multi-cycle problem local normalization solves",
                  fontsize=13, fontweight="bold", y=1.0)
    fig2.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig2.savefig(str(out_dir / "norm_explainer.png"), dpi=150, bbox_inches="tight",
                 facecolor="white")
    plt.close(fig2)

    # ---------- FIG 3: feature list slide ----------
    make_feature_list_slide(out_dir)


def make_feature_list_slide(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    INK, SUB = "#1a2b3c", "#5a6b7b"
    fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(4, 94, "Per-OM Phenophase Classifier", fontsize=25, fontweight="bold",
            color=INK, va="top")
    ax.text(4, 88.5, "24 features \u2014 each computed within a crown's own time series, "
            "referenced to its nearest automatically-detected local extremum",
            fontsize=11.5, color=SUB, va="top", style="italic")
    groups = [
        ("Vegetation level", "#2e7d32",
         [("veg_norm", "veg fraction, min-max normalized over the full series"),
          ("veg_hat", "raw interpolated veg fraction"),
          ("veg_hat_z", "veg as a z-score within the crown"),
          ("frac_of_max", "veg as a fraction of the crown's greenest OM")]),
        ("Local normalization", "#00838f",
         [("veg_norm_local", "veg normalized to the local between-trough segment")]),
        ("Position vs extrema", "#c62828",
         [("d_near_trough", "signed OM distance to nearest local trough"),
          ("abs_d_near_trough", "unsigned distance to nearest trough"),
          ("d_near_peak", "signed OM distance to nearest local peak"),
          ("abs_d_near_peak", "unsigned distance to nearest peak"),
          ("side_near", "side of nearest trough (\u22121 before, 0 at, +1 after)"),
          ("is_trough", "is this OM itself a local minimum (0/1)"),
          ("is_peak", "is this OM itself a local maximum (0/1)")]),
        ("Cycles", "#6a1b9a",
         [("n_troughs", "number of leaf-off cycles detected in the crown")]),
        ("Direction of change", "#e65100",
         [("slope_local", "curve slope at this OM (avg of change either side)"),
          ("slope_back", "change from the previous OM"),
          ("veg_norm_prev", "veg at the previous OM"),
          ("veg_norm_next", "veg at the next OM")]),
        ("Position in series", "#455a64",
         [("rank_in_chain", "greenness percentile within the crown"),
          ("om_frac", "position in the series (0 = start, 1 = end)")]),
        ("Interaction terms", "#ad1457",
         [("vegloc_x_side", "veg_norm_local \u00d7 side_near"),
          ("vegloc_x_slope", "veg_norm_local \u00d7 slope_local"),
          ("side_x_slope", "side_near \u00d7 slope_local"),
          ("fracmax_x_side", "frac_of_max \u00d7 side_near"),
          ("vegz_x_side", "veg_hat_z \u00d7 side_near")]),
    ]
    col_top, col_w, line_h, header_h = 82, 45, 3.05, 3.2

    def draw_col(gs, x):
        y = col_top
        for gname, color, feats in gs:
            chip = FancyBboxPatch((x, y - header_h + 0.6), col_w, header_h - 0.6,
                                  boxstyle="round,pad=0.15", mutation_aspect=0.5,
                                  fc=color, ec="none")
            ax.add_patch(chip)
            ax.text(x + 0.8, y - header_h / 2 + 0.5, gname.upper(), fontsize=10.5,
                    fontweight="bold", color="white", va="center")
            ax.text(x + col_w - 0.8, y - header_h / 2 + 0.5, f"{len(feats)}", fontsize=10.5,
                    fontweight="bold", color="white", va="center", ha="right")
            y -= header_h + 0.5
            for fname, desc in feats:
                ax.text(x + 1.2, y, fname, fontsize=9.3, fontweight="bold", color=INK,
                        va="top", family="monospace")
                ax.text(x + 15.5, y, desc, fontsize=8.3, color=SUB, va="top")
                y -= line_h
            y -= 1.6
    draw_col(groups[:3], 4)
    draw_col(groups[3:], 52)
    foot = FancyBboxPatch((4, 2.2), 92, 5.4, boxstyle="round,pad=0.2",
                          mutation_aspect=0.5, fc="#eceff1", ec="#b0bec5", lw=1)
    ax.add_patch(foot)
    ax.text(50, 4.9, "All features are self-referenced within each crown \u2192 site-, "
            "flight-cadence-, and species-independent (no hardcoded OM window).",
            fontsize=10, color=INK, ha="center", va="center", fontweight="bold")
    fig.savefig(str(out_dir / "feature_list_slide.png"), dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-chain explainer figures (crops + curve + feature table + decision)
# ---------------------------------------------------------------------------
def make_chain_explainers(out_dir, test_df, fitted_model, model_name, want_chains,
                          config, tracking_dir, base_tag, align_tag, align_method,
                          feature_cols, states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    exp_dir = out_dir / "chain_explainers"
    exp_dir.mkdir(parents=True, exist_ok=True)

    import geopandas as gpd
    consensus_gpkg = Path(config.get(
        "consensus_gpkg", tracking_dir / "consensus_crowns_complete_all.gpkg"))
    crowns = gpd.read_file(str(consensus_gpkg))
    if "chain_id" not in crowns.columns:
        crowns = crowns.reset_index(drop=True)
        crowns["chain_id"] = crowns.index.astype(int)
    crowns["chain_id"] = pd.to_numeric(crowns["chain_id"], errors="coerce")
    crowns = crowns.dropna(subset=["chain_id"]).copy()
    crowns["chain_id"] = crowns["chain_id"].astype(int)
    geom_by_chain = {int(r["chain_id"]): r.geometry for _, r in crowns.iterrows()}

    print("Building aligned tracker for explainer crops...")
    tracker, om_stems = build_aligned_tracker(
        config, base_threshold_tag=base_tag, align_threshold_tag=align_tag,
        align_method=align_method)

    for cid in want_chains:
        sub = test_df[test_df["chain_id"] == cid].copy()
        if sub.empty:
            print(f"  [skip] chain {cid} not in test features")
            continue
        sub = sub.sort_values("om_id").reset_index(drop=True)
        oms = sub["om_id"].to_numpy(int)
        X = sub[feature_cols].to_numpy(float)
        yp = fitted_model.predict(X)
        sub["pred_state"] = [states[i] for i in yp]
        sub.to_csv(exp_dir / f"chain_{cid:04d}_features.csv", index=False)

        geom = geom_by_chain.get(cid)
        patches = {}
        for oid in oms:
            p = None
            if geom is not None and not geom.is_empty:
                try:
                    p = tracker.extract_patch_for_polygon(int(oid), geom)
                except Exception:
                    p = None
            patches[int(oid)] = p

        n = len(oms); ncols = min(19, n)
        fig = plt.figure(figsize=(max(14, 0.95 * n), 12.0))
        gs = GridSpec(3, ncols, figure=fig, height_ratios=[1.0, 2.2, 1.9],
                      hspace=0.40, wspace=0.06)
        for i, oid in enumerate(oms):
            if i >= ncols:
                break
            axi = fig.add_subplot(gs[0, i])
            p = patches.get(int(oid))
            if isinstance(p, np.ndarray) and p.size > 0:
                arr = np.clip(np.nan_to_num(p[..., :3], nan=0), 0, 255).astype(np.uint8)
                axi.imshow(arr)
            else:
                axi.imshow(np.full((10, 10, 3), 235, np.uint8))
                axi.text(0.5, 0.5, "n/a", ha="center", va="center",
                         transform=axi.transAxes, fontsize=7, color="#888")
            axi.set_xticks([]); axi.set_yticks([])
            col = STATE_FIG_COLORS.get(sub["pred_state"].iloc[i], "#bbb")
            for sp in axi.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(3)
            axi.set_title(f"OM{oid}", fontsize=7.5)

        axc = fig.add_subplot(gs[1, :])
        vn = sub["veg_norm"].to_numpy(float)
        vloc = sub["veg_norm_local"].to_numpy(float)
        axc.plot(oms, vn, "-", color="#90a4ae", lw=1.4, label="veg_norm (global)")
        axc.plot(oms, vloc, "-o", color="#263238", lw=2.0, ms=3, label="veg_norm_local")
        for i, oid in enumerate(oms):
            axc.plot(oid, vloc[i], "o", ms=13,
                     color=STATE_FIG_COLORS.get(sub["pred_state"].iloc[i], "#bbb"),
                     mec="white", mew=1.3, zorder=5)
        for i, oid in enumerate(oms):
            if sub["is_trough"].iloc[i] >= 0.5:
                axc.plot(oid, vloc[i], "v", ms=15, color="#c62828", zorder=6, mec="white", mew=1)
            if sub["is_peak"].iloc[i] >= 0.5:
                axc.plot(oid, vloc[i], "^", ms=12, color="#1565c0", zorder=6, mec="white", mew=1)
        for i, oid in enumerate(oms):
            ts = sub["true_state"].iloc[i]
            if isinstance(ts, str):
                axc.plot(oid, -0.08, "s", ms=10, color=STATE_FIG_COLORS.get(ts, "#ddd"),
                         clip_on=False)
        axc.text(oms[0] - 0.6, -0.08, "hand:", ha="right", va="center", fontsize=8)
        axc.set_ylim(-0.14, 1.12); axc.set_xticks(oms)
        axc.set_ylabel("veg (local-normalized)"); axc.set_xlabel("OM")
        axc.legend(loc="upper left", fontsize=8, framealpha=0.9)
        axc.set_title(f"chain {cid}: curve, local extrema, and model decision per OM",
                      fontsize=11, fontweight="bold")

        axt = fig.add_subplot(gs[2, :]); axt.axis("off")
        show = ["veg_norm", "veg_norm_local", "d_near_trough", "side_near",
                "slope_local", "frac_of_max", "is_trough", "n_troughs"]
        header = ["OM"] + show + ["PRED", "hand"]
        cell = []
        for i, oid in enumerate(oms):
            row = [str(int(oid))]
            for f in show:
                v = sub[f].iloc[i]
                row.append(f"{v:.2f}" if isinstance(v, (int, float, np.floating)) else str(v))
            row.append(sub["pred_state"].iloc[i][:5])
            th = sub["true_state"].iloc[i]
            row.append(th[:5] if isinstance(th, str) else "-")
            cell.append(row)
        tab = axt.table(cellText=cell, colLabels=header, loc="center", cellLoc="center")
        tab.auto_set_font_size(False); tab.set_fontsize(7.2); tab.scale(1, 1.25)
        pc, hc = len(header) - 2, len(header) - 1
        for i in range(len(oms)):
            pv = sub["pred_state"].iloc[i]
            tab[(i + 1, pc)].set_facecolor(STATE_FIG_COLORS.get(pv, "#fff"))
            tab[(i + 1, pc)].set_alpha(0.35)
            hv = sub["true_state"].iloc[i]
            if isinstance(hv, str):
                tab[(i + 1, hc)].set_facecolor(STATE_FIG_COLORS.get(hv, "#fff"))
                tab[(i + 1, hc)].set_alpha(0.35)
                if hv != pv:
                    for c in (pc, hc):
                        tab[(i + 1, c)].set_edgecolor("red"); tab[(i + 1, c)].set_linewidth(2)
        acc = float(np.mean([sub["pred_state"].iloc[i] == sub["true_state"].iloc[i]
                             for i in range(len(oms))
                             if isinstance(sub["true_state"].iloc[i], str)]))
        fig.suptitle(f"Phenophase explainer \u2014 chain {cid}  |  model={model_name}  "
                     f"|  per-OM accuracy on this chain = {acc:.0%}",
                     fontsize=13, fontweight="bold", y=0.995)
        fig.savefig(str(exp_dir / f"chain_{cid:04d}_explainer.png"), dpi=145,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved chain_{cid:04d}_explainer.png (acc {acc:.0%})")


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
    ap.add_argument("--ds-thresh", type=float, default=-0.145,
                    help="DS threshold for deciduous classification (default -0.145 "
                         "from your fitted logistic regression)")
    ap.add_argument("--w-veg-amp", type=float, default=-0.4772)
    ap.add_argument("--w-depth", type=float, default=0.7921)
    ap.add_argument("--w-gcc-amp", type=float, default=-0.6147)
    ap.add_argument("--w-tex", type=float, default=0.3949)
    ap.add_argument("--seed", type=int, default=42)
    # --- extra output flags ---
    ap.add_argument("--explain", action="store_true",
                    help="Also render per-chain explainer figures (crops + curve + "
                         "feature table + decision) for test chains. Off by default "
                         "because extracting crops needs the tracker and is slow.")
    ap.add_argument("--explain-chains", default=None,
                    help="Comma-separated test chain_ids to explain (default: first 6). "
                         "Only used with --explain.")
    ap.add_argument("--explain-model", choices=["gb", "logreg"], default="gb",
                    help="Which trained model's decisions to show in explainer figures.")
    ap.add_argument("--no-concept-figs", action="store_true",
                    help="Skip the static conceptual/teaching figures "
                         "(feature_explainer, norm_explainer, feature_list_slide).")
    ap.add_argument("--save-model", action="store_true",
                    help="Persist the trained model(s) + feature list + class map to "
                         "<out_dir>/models/ as .joblib, for reuse by the GeoJSON patcher "
                         "(12_apply_phenophase_to_geojson.py) without retraining.")
    ap.add_argument("--base-threshold-tag", default="conf_0p45")
    ap.add_argument("--align-threshold-tag", default="conf_0p65")
    ap.add_argument("--align-method", default="pcc_tiled")
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
    tracking_dir = Path(config.get("tracking_dir", phenology_dir))
    out_dir = phenology_dir / "validation" / "phenophase_clf"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- curves (identical to the pipeline's) ---
    setup_app_dir(project_root)
    from phenology_leafshed import LeafShedConfig, compute_leafshed_scores
    features_df = load_features_df(phenology_dir)
    om_ids = om_ids_from_features(features_df)
    cfg = LeafShedConfig(veg_min_threshold=args.veg_min, ds_threshold=args.ds_thresh,
                         w_veg_amp=args.w_veg_amp, w_depth=args.w_depth,
                         w_gcc_amp=args.w_gcc_amp, w_tex=args.w_tex)
    scores_df, pp_df, _n = compute_leafshed_scores(features_df, om_ids=om_ids, cfg=cfg)

    # --- filter to DECIDUOUS chains only ---
    # The per-OM phenophase classifier (leaf_on/transitioning/leaf_off) is only
    # meaningful on deciduous trees. Evergreen chains have flat veg curves with
    # no real trough/cycle, so the classifier concepts don't apply physically.
    # Chains not satisfying DS >= ds_threshold are excluded from training/eval;
    # in the GeoJSON patcher they will be assigned leaf_on to every OM.
    scores_df["chain_id"] = scores_df["chain_id"].astype(int)
    deciduous_ids = set(scores_df.loc[scores_df["is_deciduous"], "chain_id"].tolist())
    n_total = int(scores_df["chain_id"].nunique())
    n_decid = len(deciduous_ids)
    print(f"\nDS filter: {n_decid}/{n_total} chains classified as deciduous "
          f"(DS >= {args.ds_thresh}) → only these enter the phenophase classifier.")

    pp_df["chain_id"] = pp_df["chain_id"].astype(int)
    pp_df["om_id"] = pp_df["om_id"].astype(int)
    # restrict pp_df to deciduous chains before building curves
    pp_df = pp_df[pp_df["chain_id"].isin(deciduous_ids)].copy()

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
                         "cv_macro_f1": cv_macro_f1, "fitted": model}

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

    # --- persist trained model(s) for reuse by the GeoJSON patcher ---
    if args.save_model:
        try:
            import joblib
        except Exception as e:
            print(f"[warn] --save-model needs joblib ({e}); skipping model save.")
        else:
            models_dir = out_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            for name in results:
                bundle = {
                    "model": results[name]["fitted"],
                    "feature_cols": list(FEATURE_COLS),
                    "states": list(STATES),
                    "state_to_int": dict(STATE_TO_INT),
                    "leafoff_span": args.leafoff_span,
                    "veg_min": args.veg_min,
                    "ds_thresh": args.ds_thresh,
                    "trained_on_rows": int(len(tr)),
                }
                dest = models_dir / f"phenophase_{name}.joblib"
                joblib.dump(bundle, dest)
                print(f"  saved model: {dest}")

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

    # --- conceptual / teaching figures (static) ---
    if not args.no_concept_figs:
        try:
            make_concept_figures(out_dir)
            print("Wrote concept figures: feature_explainer.png, norm_explainer.png, "
                  "feature_list_slide.png")
        except Exception as e:
            print(f"[warn] concept figures skipped: {e}")

    # --- per-chain explainer figures (needs the tracker; gated behind --explain) ---
    if args.explain:
        if args.explain_chains:
            want = [int(x) for x in args.explain_chains.split(",") if x.strip()]
        else:
            want = test_lab["chain_id"].head(6).tolist()
        em = args.explain_model if args.explain_model in results else best
        fitted = results[em]["fitted"]
        print(f"\nRendering per-chain explainers for {want} using model '{em}'...")
        try:
            make_chain_explainers(
                out_dir, test_df, fitted, em, want, config, tracking_dir,
                args.base_threshold_tag, args.align_threshold_tag, args.align_method,
                FEATURE_COLS, STATES)
        except Exception as e:
            print(f"[warn] chain explainers skipped: {e}")

    print(f"\nOutput in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
