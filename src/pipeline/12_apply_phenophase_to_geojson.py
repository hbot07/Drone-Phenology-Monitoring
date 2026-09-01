# """
# Apply the trained phenophase classifier to tree_master_geojson.geojson.

# What it does
# ------------
# 1. Loads a model saved by 10_phenophase_classifier.py --save-model
#    (or trains one on the spot from the label files if none is given).
# 2. Rebuilds the SAME window-free local-trough features for EVERY crown in the
#    dataset and predicts per-OM phenophase with the trained model.
# 3. Applies temporal smoothing (the isolated-label correction your reviewer asked
#    for): a run of a class shorter than --min-run OMs, sandwiched between a
#    different class, is flipped to its neighbours. Optionally confidence-aware:
#    a label is only flipped if the model was less than --flip-below confident.
# 4. Rewrites each observation's phenology.phenophase in the GeoJSON with the
#    smoothed prediction, and recomputes classification.leaf_off_start_om /
#    full_leaf_off_om / leaf_on_return_om to stay consistent. Everything else in
#    the file is preserved byte-for-byte in structure.

# The original file is never overwritten unless --in-place is given; by default
# the result is written to tree_master_geojson_phenoclf.geojson beside it.

# Usage
# -----
#     # using a saved model (recommended: train once, reuse)
#     python 12_apply_phenophase_to_geojson.py --config /path/pipeline_config.json \\
#         --model /path/validation/phenophase_clf/models/phenophase_gb.joblib \\
#         [--min-run 2] [--flip-below 0.90] [--leafoff-span 0]

#     # or train on the fly from the label files (no saved model)
#     python 12_apply_phenophase_to_geojson.py --config /path/pipeline_config.json \\
#         --train-labels /path/leaf_leafoff_validation.xlsx \\
#         --test-labels  /path/test_leafonoff.xlsx --model-type gb

# Outputs (under <phenology_dir>/)
# --------------------------------
#     tree_master_geojson_phenoclf.geojson   patched copy (default)
#     phenophase_predictions_all.csv         chain_id, om_id, raw pred, smoothed,
#                                            prob per class, whether it was flipped
#     phenophase_patch_report.txt            counts: predicted, flipped, per class
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

# # reuse feature engineering + labeling from script 10
# def _load_clf():
#     import importlib.util
#     here = Path(__file__).resolve().parent
#     for cand in ("10_phenophase_classifier.py", "phenophase_classifier.py"):
#         p = here / cand
#         if p.exists():
#             spec = importlib.util.spec_from_file_location("clf10", p)
#             m = importlib.util.module_from_spec(spec)
#             spec.loader.exec_module(m)
#             return m
#     raise ImportError("Need 10_phenophase_classifier.py next to this script for feature code.")

# from phenology_validation_common import (  # noqa: E402
#     load_config,
#     load_features_df,
#     om_ids_from_features,
#     setup_app_dir,
# )


# # ---------------------------------------------------------------------------
# # Temporal smoothing
# # ---------------------------------------------------------------------------
# def smooth_sequence(states: List[str], probs: Optional[np.ndarray],
#                     state_list: List[str], min_run: int,
#                     flip_below: float) -> Tuple[List[str], List[bool]]:
#     """Flip short isolated runs to their surrounding class.

#     A maximal run of identical labels with length < min_run that is bounded on
#     BOTH sides by the same other label is reassigned to that bounding label.
#     If probs is given, a label is only flipped when its own confidence was
#     < flip_below (so a very confident short event can survive).
#     Returns (smoothed_states, flipped_flags).
#     """
#     n = len(states)
#     out = list(states)
#     flipped = [False] * n
#     if n == 0:
#         return out, flipped

#     # identify maximal runs
#     runs = []  # (start, end_exclusive, label)
#     s = 0
#     for i in range(1, n + 1):
#         if i == n or out[i] != out[s]:
#             runs.append((s, i, out[s]))
#             s = i

#     for ri, (a, b, lab) in enumerate(runs):
#         run_len = b - a
#         if run_len >= min_run:
#             continue
#         left_lab = runs[ri - 1][2] if ri > 0 else None
#         right_lab = runs[ri + 1][2] if ri < len(runs) - 1 else None
#         # only flip if sandwiched by the SAME neighbouring label
#         if left_lab is not None and left_lab == right_lab and left_lab != lab:
#             target = left_lab
#             for j in range(a, b):
#                 if probs is not None and flip_below < 1.0:
#                     conf = float(np.max(probs[j]))
#                     if conf >= flip_below:
#                         continue  # too confident to flip
#                 out[j] = target
#                 flipped[j] = True
#         # edge runs (start/end of series) shorter than min_run: absorb into the
#         # single neighbour if present
#         elif left_lab is None and right_lab is not None and right_lab != lab:
#             for j in range(a, b):
#                 if probs is not None and flip_below < 1.0 and float(np.max(probs[j])) >= flip_below:
#                     continue
#                 out[j] = right_lab; flipped[j] = True
#         elif right_lab is None and left_lab is not None and left_lab != lab:
#             for j in range(a, b):
#                 if probs is not None and flip_below < 1.0 and float(np.max(probs[j])) >= flip_below:
#                     continue
#                 out[j] = left_lab; flipped[j] = True
#     return out, flipped


# def events_from_states(oms: List[int], states: List[str]) -> Dict[str, Optional[int]]:
#     """Derive leaf_off_start / full_leaf_off / leaf_on_return OMs from a smoothed
#     per-OM state sequence (first cycle)."""
#     off_idx = [i for i, s in enumerate(states) if s == "leaf_off"]
#     res = {"leaf_off_start_om": None, "full_leaf_off_om": None, "leaf_on_return_om": None}
#     if not off_idx:
#         return res
#     first_off = off_idx[0]
#     # leaf_off_start = first transitioning/off after the last leaf_on before first_off
#     start = first_off
#     for i in range(first_off - 1, -1, -1):
#         if states[i] == "leaf_on":
#             start = i + 1
#             break
#         start = i
#     res["leaf_off_start_om"] = int(oms[start])
#     res["full_leaf_off_om"] = int(oms[off_idx[len(off_idx) // 2]])  # middle of the off run
#     # leaf_on_return = first leaf_on at/after the last off in this first cluster
#     last_off = first_off
#     for i in range(first_off, len(states)):
#         if states[i] == "leaf_off":
#             last_off = i
#         elif states[i] == "leaf_on" and i > last_off:
#             res["leaf_on_return_om"] = int(oms[i])
#             break
#     return res


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------
# def main() -> int:
#     ap = argparse.ArgumentParser(description="Apply trained phenophase model to tree_master_geojson")
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--model", default=None, help="path to a .joblib saved by script 10")
#     ap.add_argument("--geojson", default=None,
#                     help="path to tree_master_geojson.geojson (default: <phenology_dir>/tree_master_geojson.geojson)")
#     ap.add_argument("--in-place", action="store_true",
#                     help="overwrite the input GeoJSON instead of writing a _phenoclf copy")
#     # smoothing controls
#     ap.add_argument("--min-run", type=int, default=2,
#                     help="runs of a class shorter than this (OMs) get flipped to neighbours (default 2)")
#     ap.add_argument("--flip-below", type=float, default=1.0,
#                     help="only flip a label if model confidence < this (default 1.0 = always flip)")
#     ap.add_argument("--no-smooth", action="store_true", help="skip temporal smoothing")
#     ap.add_argument("--no-events", action="store_true",
#                     help="do not recompute classification event OMs from smoothed labels")
#     # on-the-fly training fallback
#     ap.add_argument("--eval-test-labels", default=None,
#                     help="Score raw vs smoothed predictions against this hand-labeled "
#                          "test file (xlsx/csv). Reports per-OM accuracy/precision/recall/F1 "
#                          "BEFORE and AFTER smoothing, so you can see if smoothing helped.")
#     ap.add_argument("--eval-sheet", default=0)
#     ap.add_argument("--eval-drop-overlap-with", default=None,
#                     help="Optional training-label file; test chains also present there "
#                          "are dropped from the evaluation for a clean held-out score.")
#     ap.add_argument("--train-labels", default=None)
#     ap.add_argument("--test-labels", default=None)
#     ap.add_argument("--train-sheet", default=0)
#     ap.add_argument("--test-sheet", default=0)
#     ap.add_argument("--model-type", choices=["gb", "logreg"], default="gb")
#     ap.add_argument("--leafoff-span", type=int, default=0)
#     ap.add_argument("--veg-min", type=float, default=0.45)
#     ap.add_argument("--ds-thresh", type=float, default=-0.145,
#                     help="DS threshold (default -0.145 from fitted logistic regression)")
#     ap.add_argument("--w-veg-amp", type=float, default=-0.4772)
#     ap.add_argument("--w-depth", type=float, default=0.7921)
#     ap.add_argument("--w-gcc-amp", type=float, default=-0.6147)
#     ap.add_argument("--w-tex", type=float, default=0.3949)
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()

#     clf = _load_clf()
#     config = load_config(Path(args.config).resolve())
#     project_root = Path(config["project_root"])
#     phenology_dir = Path(config["phenology_dir"])

#     gj_path = Path(args.geojson) if args.geojson else phenology_dir / "tree_master_geojson.geojson"
#     if not gj_path.exists():
#         print(f"ERROR: GeoJSON not found: {gj_path}")
#         return 1

#     # --- compute DS scores and identify deciduous chains ---
#     setup_app_dir(project_root)
#     from phenology_leafshed import LeafShedConfig, compute_leafshed_scores
#     features_df = load_features_df(phenology_dir)
#     om_ids = om_ids_from_features(features_df)
#     cfg = LeafShedConfig(veg_min_threshold=args.veg_min, ds_threshold=args.ds_thresh,
#                          w_veg_amp=args.w_veg_amp, w_depth=args.w_depth,
#                          w_gcc_amp=args.w_gcc_amp, w_tex=args.w_tex)
#     scores_df, pp_df, _n = compute_leafshed_scores(features_df, om_ids=om_ids, cfg=cfg)

#     scores_df["chain_id"] = scores_df["chain_id"].astype(int)
#     deciduous_ids = set(scores_df.loc[scores_df["is_deciduous"], "chain_id"].tolist())
#     evergreen_ids = set(scores_df["chain_id"].tolist()) - deciduous_ids
#     n_total = len(deciduous_ids) + len(evergreen_ids)
#     print(f"DS filter: {len(deciduous_ids)}/{n_total} deciduous "
#           f"(DS >= {args.ds_thresh}), {len(evergreen_ids)} evergreen "
#           f"→ classifier runs only on deciduous; evergreen OMs assigned leaf_on.")

#     # --- curves for DECIDUOUS chains only ---
#     pp_df["chain_id"] = pp_df["chain_id"].astype(int)
#     pp_df["om_id"] = pp_df["om_id"].astype(int)
#     pp_df_decid = pp_df[pp_df["chain_id"].isin(deciduous_ids)].copy()
#     curves = {}
#     for cid, sub in pp_df_decid.groupby("chain_id"):
#         sub = sub.sort_values("om_id")
#         curves[int(cid)] = (sub["om_id"].to_numpy(float),
#                             sub["veg_fraction_hsv_norm"].to_numpy(float),
#                             sub["veg_fraction_hsv_hat"].to_numpy(float))

#     # also build a simple om_list for evergreen chains (for GeoJSON patching)
#     evergreen_oms: Dict[int, List[int]] = {}
#     for cid, sub in pp_df[pp_df["chain_id"].isin(evergreen_ids)].groupby("chain_id"):
#         evergreen_oms[int(cid)] = sorted(sub["om_id"].astype(int).tolist())

#     # --- get the model ---
#     if args.model:
#         import joblib
#         bundle = joblib.load(args.model)
#         model = bundle["model"]
#         FEATURE_COLS = bundle["feature_cols"]
#         STATES = bundle["states"]
#         print(f"Loaded model: {args.model}  (trained on {bundle.get('trained_on_rows','?')} rows)")
#     else:
#         if not (args.train_labels and args.test_labels):
#             print("ERROR: give --model, or --train-labels and --test-labels to train on the fly.")
#             return 1
#         from sklearn.ensemble import HistGradientBoostingClassifier
#         from sklearn.linear_model import LogisticRegression
#         from sklearn.preprocessing import StandardScaler
#         from sklearn.pipeline import Pipeline
#         FEATURE_COLS = clf.FEATURE_COLS
#         STATES = clf.STATES
#         STI = clf.STATE_TO_INT
#         train_lab = clf.load_labels(Path(args.train_labels).resolve(), args.train_sheet)
#         train_df = clf.assemble(train_lab, curves, args.leafoff_span, "train")
#         tr = train_df.dropna(subset=["true_state"]).copy()
#         Xtr = tr[FEATURE_COLS].to_numpy(float)
#         ytr = tr["true_state"].map(STI).to_numpy()
#         if args.model_type == "gb":
#             model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
#                                                    max_depth=3, l2_regularization=1.0,
#                                                    random_state=args.seed)
#         else:
#             model = Pipeline([("scale", StandardScaler()),
#                               ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
#         model.fit(Xtr, ytr)
#         print(f"Trained {args.model_type} on {len(tr)} rows (on the fly).")

#     has_proba = hasattr(model, "predict_proba")

#     # --- predict + smooth for every chain ---
#     pred_rows = []
#     smoothed_by_chain: Dict[int, Dict[int, str]] = {}
#     events_by_chain: Dict[int, Dict[str, Optional[int]]] = {}
#     n_flipped_total = 0

#     for cid, (oms, vn, vh) in curves.items():
#         feats = clf.build_features_for_chain(oms.copy(), vn.copy(), vh.copy())
#         X = feats[FEATURE_COLS].to_numpy(float)
#         yi = model.predict(X)
#         raw_states = [STATES[i] for i in yi]
#         probs = model.predict_proba(X) if has_proba else None

#         if args.no_smooth:
#             sm_states, flipped = list(raw_states), [False] * len(raw_states)
#         else:
#             sm_states, flipped = smooth_sequence(raw_states, probs, STATES,
#                                                  args.min_run, args.flip_below)
#         n_flipped_total += sum(flipped)

#         om_list = [int(o) for o in feats["om_id"].tolist()]
#         smoothed_by_chain[int(cid)] = {om: st for om, st in zip(om_list, sm_states)}
#         if not args.no_events:
#             events_by_chain[int(cid)] = events_from_states(om_list, sm_states)

#         for k, om in enumerate(om_list):
#             row = {"chain_id": int(cid), "om_id": om,
#                    "pred_raw": raw_states[k], "pred_smoothed": sm_states[k],
#                    "flipped": bool(flipped[k])}
#             if probs is not None:
#                 for ci, st in enumerate(STATES):
#                     row[f"p_{st}"] = float(probs[k][ci])
#             pred_rows.append(row)

#     pred_df = pd.DataFrame(pred_rows)
#     pred_df.to_csv(phenology_dir / "phenophase_predictions_all.csv", index=False)

#     # --- evaluate raw vs smoothed on a hand-labeled test set ---
#     if args.eval_test_labels:
#         eval_path = Path(args.eval_test_labels)
#         if not eval_path.exists():
#             print(f"[warn] eval labels not found: {eval_path}; skipping evaluation.")
#         else:
#             test_lab = clf.load_labels(eval_path, args.eval_sheet)
#             # optional overlap drop
#             if args.eval_drop_overlap_with:
#                 trp = Path(args.eval_drop_overlap_with)
#                 if trp.exists():
#                     tr_ids = set(clf.load_labels(trp, 0)["chain_id"].astype(int))
#                     ov = sorted(set(test_lab["chain_id"].astype(int)) & tr_ids)
#                     if ov:
#                         test_lab = test_lab[~test_lab["chain_id"].isin(ov)].copy()
#                         print(f"[eval] dropped {len(ov)} train/test overlap chains: {ov}")
#             # true per-OM states via the same rule script 10 uses
#             test_feat = clf.assemble(test_lab, curves, args.leafoff_span, "test")
#             truth = test_feat.dropna(subset=["true_state"])[["chain_id", "om_id", "true_state"]].copy()
#             truth["chain_id"] = truth["chain_id"].astype(int)
#             truth["om_id"] = truth["om_id"].astype(int)

#             merged = truth.merge(pred_df[["chain_id", "om_id", "pred_raw", "pred_smoothed"]],
#                                  on=["chain_id", "om_id"], how="inner")
#             n = len(merged)
#             if n == 0:
#                 print("[eval] no overlapping (chain, om) rows between test labels and predictions.")
#             else:
#                 STI = clf.STATE_TO_INT
#                 yt = merged["true_state"].map(STI).to_numpy()
#                 yr = merged["pred_raw"].map(STI).to_numpy()
#                 ys = merged["pred_smoothed"].map(STI).to_numpy()
#                 acc_raw = float((yr == yt).mean())
#                 acc_sm = float((ys == yt).mean())
#                 m_raw = clf.per_class_metrics(yt, yr)
#                 m_sm = clf.per_class_metrics(yt, ys)
#                 cm_raw = clf.confusion(yt, yr)
#                 cm_sm = clf.confusion(yt, ys)

#                 merged.to_csv(phenology_dir / "phenophase_test_eval_perom.csv", index=False)
#                 m_raw.to_csv(phenology_dir / "phenophase_test_metrics_raw.csv", index=False)
#                 m_sm.to_csv(phenology_dir / "phenophase_test_metrics_smoothed.csv", index=False)

#                 # how many test OMs did smoothing change, and net effect
#                 changed = int((merged["pred_raw"] != merged["pred_smoothed"]).sum())
#                 fixed = int(((yr != yt) & (ys == yt)).sum())     # was wrong, now right
#                 broke = int(((yr == yt) & (ys != yt)).sum())     # was right, now wrong

#                 el = []
#                 el.append("=" * 60)
#                 el.append("TEST-SET EVALUATION: raw vs smoothed")
#                 el.append("=" * 60)
#                 el.append(f"Scored (chain x OM) rows: {n}")
#                 el.append(f"Smoothing changed {changed} test labels: "
#                           f"{fixed} fixed (wrong->right), {broke} broke (right->wrong), "
#                           f"net {fixed - broke:+d}")
#                 el.append("")
#                 el.append(f"Overall accuracy BEFORE smoothing: {acc_raw:.4f} ({acc_raw:.1%})")
#                 el.append(f"Overall accuracy AFTER  smoothing: {acc_sm:.4f} ({acc_sm:.1%})")
#                 el.append(f"Change: {acc_sm - acc_raw:+.4f} ({acc_sm - acc_raw:+.1%})")
#                 el.append("")
#                 el.append("Confusion BEFORE (rows=true, cols=pred):")
#                 el += ["  " + ln for ln in cm_raw.to_string().splitlines()]
#                 el.append("")
#                 el.append("Confusion AFTER (rows=true, cols=pred):")
#                 el += ["  " + ln for ln in cm_sm.to_string().splitlines()]
#                 el.append("")
#                 el.append("Per-class F1  (before -> after):")
#                 for st in clf.STATES:
#                     f_b = m_raw.loc[m_raw["phenophase"] == st, "f1"]
#                     f_a = m_sm.loc[m_sm["phenophase"] == st, "f1"]
#                     if len(f_b) and len(f_a):
#                         el.append(f"  {st:<16}{float(f_b.iloc[0]):.3f}  ->  {float(f_a.iloc[0]):.3f}")
#                 mac_b = float(m_raw.loc[m_raw['phenophase']=='macro avg','f1'].iloc[0])
#                 mac_a = float(m_sm.loc[m_sm['phenophase']=='macro avg','f1'].iloc[0])
#                 el.append(f"  {'macro avg':<16}{mac_b:.3f}  ->  {mac_a:.3f}")
#                 eval_report = "\n".join(el)
#                 print("\n" + eval_report)
#                 (phenology_dir / "phenophase_test_eval_report.txt").write_text(
#                     eval_report, encoding="utf-8")

#     # --- patch the GeoJSON ---
#     gj = json.loads(gj_path.read_text())
#     n_obs_patched = 0
#     n_features_decid = 0
#     n_features_ever = 0
#     for feat in gj.get("features", []):
#         props = feat.get("properties", {})
#         cid = props.get("ids", {}).get("chain_id")
#         if cid is None:
#             continue
#         cid = int(cid)

#         if cid in evergreen_ids:
#             # evergreen: classifier doesn't apply — assign leaf_on to every OM
#             n_features_ever += 1
#             for obs in props.get("observations", []):
#                 obs.setdefault("phenology", {})
#                 obs["phenology"]["phenophase"] = "leaf_on"
#                 obs["phenology"]["phenophase_source"] = "evergreen_assigned"
#                 n_obs_patched += 1
#             continue

#         chain_map = smoothed_by_chain.get(cid)
#         if not chain_map:
#             continue
#         n_features_decid += 1
#         for obs in props.get("observations", []):
#             om = obs.get("om_id")
#             if om is None:
#                 continue
#             new_ph = chain_map.get(int(om))
#             if new_ph is None:
#                 continue
#             obs.setdefault("phenology", {})
#             obs["phenology"]["phenophase"] = new_ph
#             obs["phenology"]["phenophase_source"] = "gb_classifier_smoothed"
#             n_obs_patched += 1
#         # refresh event OMs in classification
#         if not args.no_events and cid in events_by_chain:
#             cls = props.setdefault("classification", {})
#             for k, v in events_by_chain[cid].items():
#                 cls[k] = v

#     gj.setdefault("phenology_config", {})["phenophase_method"] = {
#         "type": "gb_classifier",
#         "smoothing": (None if args.no_smooth else
#                       {"min_run": args.min_run, "flip_below": args.flip_below}),
#         "features": list(FEATURE_COLS),
#     }

#     if args.in_place:
#         out_path = gj_path
#     else:
#         out_path = gj_path.with_name(gj_path.stem + "_phenoclf.geojson")
#     out_path.write_text(json.dumps(gj, indent=2, default=str))

#     # --- report ---
#     rep = []
#     rep.append("=" * 60)
#     rep.append("PHENOPHASE PATCH REPORT")
#     rep.append("=" * 60)
#     rep.append(f"GeoJSON in : {gj_path}")
#     rep.append(f"GeoJSON out: {out_path}")
#     rep.append(f"Chains total:          {n_total}")
#     rep.append(f"  Deciduous (classifier): {len(deciduous_ids)}")
#     rep.append(f"  Evergreen (leaf_on):    {len(evergreen_ids)}")
#     rep.append(f"Features patched:      {n_features_decid + n_features_ever} "
#                f"({n_features_decid} deciduous, {n_features_ever} evergreen)")
#     rep.append(f"Observations patched:  {n_obs_patched}")
#     if not args.no_smooth:
#         rep.append(f"Labels flipped by smoothing: {n_flipped_total} "
#                    f"(min_run={args.min_run}, flip_below={args.flip_below})")
#     rep.append("")
#     rep.append("Smoothed phenophase distribution:")
#     for st, c in pred_df["pred_smoothed"].value_counts().items():
#         rep.append(f"  {st:<16}{c}")
#     report = "\n".join(rep)
#     print("\n" + report)
#     (phenology_dir / "phenophase_patch_report.txt").write_text(report, encoding="utf-8")
#     print(f"\nWrote: {out_path}")
#     print(f"Wrote: {phenology_dir / 'phenophase_predictions_all.csv'}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
#!/usr/bin/env python3
"""
Apply the trained phenophase classifier to tree_master_geojson.geojson.

What it does
------------
1. Loads a model saved by 10_phenophase_classifier.py --save-model
   (or trains one on the spot from the label files if none is given).
2. Rebuilds the SAME window-free local-trough features for EVERY crown in the
   dataset and predicts per-OM phenophase with the trained model.
3. Applies temporal smoothing (the isolated-label correction your reviewer asked
   for): a run of a class shorter than --min-run OMs, sandwiched between a
   different class, is flipped to its neighbours. Optionally confidence-aware:
   a label is only flipped if the model was less than --flip-below confident.
4. Rewrites each observation's phenology.phenophase in the GeoJSON with the
   smoothed prediction, and recomputes classification.leaf_off_start_om /
   full_leaf_off_om / leaf_on_return_om to stay consistent. Everything else in
   the file is preserved byte-for-byte in structure.

The original file is never overwritten unless --in-place is given; by default
the result is written to tree_master_geojson_phenoclf.geojson beside it.

Usage
-----
    # using a saved model (recommended: train once, reuse)
    python 12_apply_phenophase_to_geojson.py --config /path/pipeline_config.json \\
        --model /path/validation/phenophase_clf/models/phenophase_gb.joblib \\
        [--min-run 2] [--flip-below 0.90] [--leafoff-span 0]

    # or train on the fly from the label files (no saved model)
    python 12_apply_phenophase_to_geojson.py --config /path/pipeline_config.json \\
        --train-labels /path/leaf_leafoff_validation.xlsx \\
        --test-labels  /path/test_leafonoff.xlsx --model-type gb

Outputs (under <phenology_dir>/)
--------------------------------
    tree_master_geojson_phenoclf.geojson   patched copy (default)
    phenophase_predictions_all.csv         chain_id, om_id, raw pred, smoothed,
                                           prob per class, whether it was flipped
    phenophase_patch_report.txt            counts: predicted, flipped, per class
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

# reuse feature engineering + labeling from script 10
def _load_clf():
    import importlib.util
    here = Path(__file__).resolve().parent
    for cand in ("10_phenophase_classifier.py", "phenophase_classifier.py"):
        p = here / cand
        if p.exists():
            spec = importlib.util.spec_from_file_location("clf10", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise ImportError("Need 10_phenophase_classifier.py next to this script for feature code.")

from phenology_validation_common import (  # noqa: E402
    load_config,
    load_features_df,
    om_ids_from_features,
    setup_app_dir,
)


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------
def smooth_sequence(states: List[str], probs: Optional[np.ndarray],
                    state_list: List[str], min_run: int,
                    flip_below: float) -> Tuple[List[str], List[bool]]:
    """Flip ISOLATED single OMs, judged over a 5-OM window (with a 3-OM fallback
    at the near-edge positions).

    Rule (pure isolation, no grammar exceptions):
      * Interior OM i (has 2 neighbours each side): flagged if the FOUR
        surrounding OMs are all the same label and differ from OM i:
            states[i-2]==states[i-1]==states[i+1]==states[i+2] != states[i]
        e.g.  on on TR on on   -> TR flagged
      * 2nd OM (i=1) and 2nd-to-last (i=n-2): only one neighbour on the short
        side, so fall back to a 3-OM window -- flagged if the single left/right
        neighbour and the opposite neighbour(s) are all the same label and
        differ from OM i:
            on OFF on on ...   -> OFF (i=1) flagged
            ... on on on TR on -> TR (i=n-2) flagged
      * 1st (i=0) and last (i=n-1) OMs: never flagged.

    A flagged OM is flipped to the surrounding label ONLY IF the model's
    confidence at that OM is < flip_below (a confident isolated label is kept).
    `min_run` is ignored (kept for signature compatibility).
    Returns (smoothed_states, flipped_flags).
    """
    n = len(states)
    out = list(states)
    flipped = [False] * n
    if n < 3:
        return out, flipped

    orig = list(states)  # decide against original labels; flips don't chain

    def maybe_flip(i: int, target: str):
        if probs is not None and flip_below < 1.0:
            if float(np.max(probs[i])) >= flip_below:
                return  # confident enough to keep
        out[i] = target
        flipped[i] = True

    for i in range(1, n - 1):
        mid = orig[i]
        have_left2 = i - 2 >= 0
        have_right2 = i + 2 <= n - 1

        if have_left2 and have_right2:
            # full 5-window: all four neighbours identical and != mid
            neigh = {orig[i - 2], orig[i - 1], orig[i + 1], orig[i + 2]}
            if len(neigh) == 1 and mid not in neigh:
                maybe_flip(i, next(iter(neigh)))
        else:
            # near-edge (i==1 or i==n-2): 3-window fallback, 1 neighbour each side
            left, right = orig[i - 1], orig[i + 1]
            if left == right and mid != left:
                maybe_flip(i, left)
    return out, flipped


def events_from_states(oms: List[int], states: List[str]) -> Dict[str, Optional[int]]:
    """Derive leaf_off_start / full_leaf_off / leaf_on_return OMs from a smoothed
    per-OM state sequence (first cycle)."""
    off_idx = [i for i, s in enumerate(states) if s == "leaf_off"]
    res = {"leaf_off_start_om": None, "full_leaf_off_om": None, "leaf_on_return_om": None}
    if not off_idx:
        return res
    first_off = off_idx[0]
    # leaf_off_start = first transitioning/off after the last leaf_on before first_off
    start = first_off
    for i in range(first_off - 1, -1, -1):
        if states[i] == "leaf_on":
            start = i + 1
            break
        start = i
    res["leaf_off_start_om"] = int(oms[start])
    res["full_leaf_off_om"] = int(oms[off_idx[len(off_idx) // 2]])  # middle of the off run
    # leaf_on_return = first leaf_on at/after the last off in this first cluster
    last_off = first_off
    for i in range(first_off, len(states)):
        if states[i] == "leaf_off":
            last_off = i
        elif states[i] == "leaf_on" and i > last_off:
            res["leaf_on_return_om"] = int(oms[i])
            break
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Apply trained phenophase model to tree_master_geojson")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="path to a .joblib saved by script 10")
    ap.add_argument("--geojson", default=None,
                    help="path to tree_master_geojson.geojson (default: <phenology_dir>/tree_master_geojson.geojson)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the input GeoJSON instead of writing a _phenoclf copy")
    # smoothing controls
    ap.add_argument("--min-run", type=int, default=2,
                    help="(DEPRECATED / ignored) smoothing now flags only single "
                         "isolated OMs whose two neighbours share a different label. "
                         "Kept for backward-compatible command lines.")
    ap.add_argument("--flip-below", type=float, default=1.0,
                    help="only flip a label if model confidence < this (default 1.0 = always flip)")
    ap.add_argument("--no-smooth", action="store_true", help="skip temporal smoothing")
    ap.add_argument("--no-events", action="store_true",
                    help="do not recompute classification event OMs from smoothed labels")
    # on-the-fly training fallback
    ap.add_argument("--eval-test-labels", default=None,
                    help="Score raw vs smoothed predictions against this hand-labeled "
                         "test file (xlsx/csv). Reports per-OM accuracy/precision/recall/F1 "
                         "BEFORE and AFTER smoothing, so you can see if smoothing helped.")
    ap.add_argument("--eval-sheet", default=0)
    ap.add_argument("--eval-drop-overlap-with", default=None,
                    help="Optional training-label file; test chains also present there "
                         "are dropped from the evaluation for a clean held-out score.")
    ap.add_argument("--train-labels", default=None)
    ap.add_argument("--test-labels", default=None)
    ap.add_argument("--train-sheet", default=0)
    ap.add_argument("--test-sheet", default=0)
    ap.add_argument("--model-type", choices=["gb", "logreg"], default="gb")
    ap.add_argument("--leafoff-span", type=int, default=0)
    ap.add_argument("--veg-min", type=float, default=0.45)
    ap.add_argument("--ds-thresh", type=float, default=-0.145,
                    help="DS threshold (default -0.145 from fitted logistic regression)")
    ap.add_argument("--w-veg-amp", type=float, default=-0.4772)
    ap.add_argument("--w-depth", type=float, default=0.7921)
    ap.add_argument("--w-gcc-amp", type=float, default=-0.6147)
    ap.add_argument("--w-tex", type=float, default=0.3949)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clf = _load_clf()
    config = load_config(Path(args.config).resolve())
    project_root = Path(config["project_root"])
    phenology_dir = Path(config["phenology_dir"])

    gj_path = Path(args.geojson) if args.geojson else phenology_dir / "tree_master_geojson.geojson"
    if not gj_path.exists():
        print(f"ERROR: GeoJSON not found: {gj_path}")
        return 1

    # --- compute DS scores and identify deciduous chains ---
    setup_app_dir(project_root)
    from phenology_leafshed import LeafShedConfig, compute_leafshed_scores
    features_df = load_features_df(phenology_dir)
    om_ids = om_ids_from_features(features_df)
    cfg = LeafShedConfig(veg_min_threshold=args.veg_min, ds_threshold=args.ds_thresh,
                         w_veg_amp=args.w_veg_amp, w_depth=args.w_depth,
                         w_gcc_amp=args.w_gcc_amp, w_tex=args.w_tex)
    scores_df, pp_df, _n = compute_leafshed_scores(features_df, om_ids=om_ids, cfg=cfg)

    scores_df["chain_id"] = scores_df["chain_id"].astype(int)
    deciduous_ids = set(scores_df.loc[scores_df["is_deciduous"], "chain_id"].tolist())
    evergreen_ids = set(scores_df["chain_id"].tolist()) - deciduous_ids
    n_total = len(deciduous_ids) + len(evergreen_ids)
    print(f"DS filter: {len(deciduous_ids)}/{n_total} deciduous "
          f"(DS >= {args.ds_thresh}), {len(evergreen_ids)} evergreen "
          f"→ classifier runs only on deciduous; evergreen OMs assigned leaf_on.")

    # --- curves for DECIDUOUS chains only ---
    pp_df["chain_id"] = pp_df["chain_id"].astype(int)
    pp_df["om_id"] = pp_df["om_id"].astype(int)
    pp_df_decid = pp_df[pp_df["chain_id"].isin(deciduous_ids)].copy()
    curves = {}
    for cid, sub in pp_df_decid.groupby("chain_id"):
        sub = sub.sort_values("om_id")
        curves[int(cid)] = (sub["om_id"].to_numpy(float),
                            sub["veg_fraction_hsv_norm"].to_numpy(float),
                            sub["veg_fraction_hsv_hat"].to_numpy(float))

    # also build a simple om_list for evergreen chains (for GeoJSON patching)
    evergreen_oms: Dict[int, List[int]] = {}
    for cid, sub in pp_df[pp_df["chain_id"].isin(evergreen_ids)].groupby("chain_id"):
        evergreen_oms[int(cid)] = sorted(sub["om_id"].astype(int).tolist())

    # --- get the model ---
    if args.model:
        import joblib
        bundle = joblib.load(args.model)
        model = bundle["model"]
        FEATURE_COLS = bundle["feature_cols"]
        STATES = bundle["states"]
        print(f"Loaded model: {args.model}  (trained on {bundle.get('trained_on_rows','?')} rows)")
    else:
        if not (args.train_labels and args.test_labels):
            print("ERROR: give --model, or --train-labels and --test-labels to train on the fly.")
            return 1
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        FEATURE_COLS = clf.FEATURE_COLS
        STATES = clf.STATES
        STI = clf.STATE_TO_INT
        train_lab = clf.load_labels(Path(args.train_labels).resolve(), args.train_sheet)
        train_df = clf.assemble(train_lab, curves, args.leafoff_span, "train")
        tr = train_df.dropna(subset=["true_state"]).copy()
        Xtr = tr[FEATURE_COLS].to_numpy(float)
        ytr = tr["true_state"].map(STI).to_numpy()
        if args.model_type == "gb":
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                   max_depth=3, l2_regularization=1.0,
                                                   random_state=args.seed)
        else:
            model = Pipeline([("scale", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        model.fit(Xtr, ytr)
        print(f"Trained {args.model_type} on {len(tr)} rows (on the fly).")

    has_proba = hasattr(model, "predict_proba")

    # --- predict + smooth for every chain ---
    pred_rows = []
    smoothed_by_chain: Dict[int, Dict[int, str]] = {}
    events_by_chain: Dict[int, Dict[str, Optional[int]]] = {}
    n_flipped_total = 0

    for cid, (oms, vn, vh) in curves.items():
        feats = clf.build_features_for_chain(oms.copy(), vn.copy(), vh.copy())
        X = feats[FEATURE_COLS].to_numpy(float)
        yi = model.predict(X)
        raw_states = [STATES[i] for i in yi]
        probs = model.predict_proba(X) if has_proba else None

        if args.no_smooth:
            sm_states, flipped = list(raw_states), [False] * len(raw_states)
        else:
            sm_states, flipped = smooth_sequence(raw_states, probs, STATES,
                                                 args.min_run, args.flip_below)
        n_flipped_total += sum(flipped)

        om_list = [int(o) for o in feats["om_id"].tolist()]
        smoothed_by_chain[int(cid)] = {om: st for om, st in zip(om_list, sm_states)}
        if not args.no_events:
            events_by_chain[int(cid)] = events_from_states(om_list, sm_states)

        for k, om in enumerate(om_list):
            row = {"chain_id": int(cid), "om_id": om,
                   "pred_raw": raw_states[k], "pred_smoothed": sm_states[k],
                   "flipped": bool(flipped[k])}
            if probs is not None:
                for ci, st in enumerate(STATES):
                    row[f"p_{st}"] = float(probs[k][ci])
            pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(phenology_dir / "phenophase_predictions_all.csv", index=False)

    # --- evaluate raw vs smoothed on a hand-labeled test set ---
    if args.eval_test_labels:
        eval_path = Path(args.eval_test_labels)
        if not eval_path.exists():
            print(f"[warn] eval labels not found: {eval_path}; skipping evaluation.")
        else:
            test_lab = clf.load_labels(eval_path, args.eval_sheet)
            # optional overlap drop
            if args.eval_drop_overlap_with:
                trp = Path(args.eval_drop_overlap_with)
                if trp.exists():
                    tr_ids = set(clf.load_labels(trp, 0)["chain_id"].astype(int))
                    ov = sorted(set(test_lab["chain_id"].astype(int)) & tr_ids)
                    if ov:
                        test_lab = test_lab[~test_lab["chain_id"].isin(ov)].copy()
                        print(f"[eval] dropped {len(ov)} train/test overlap chains: {ov}")
            # true per-OM states via the same rule script 10 uses
            test_feat = clf.assemble(test_lab, curves, args.leafoff_span, "test")
            truth = test_feat.dropna(subset=["true_state"])[["chain_id", "om_id", "true_state"]].copy()
            truth["chain_id"] = truth["chain_id"].astype(int)
            truth["om_id"] = truth["om_id"].astype(int)

            merged = truth.merge(pred_df[["chain_id", "om_id", "pred_raw", "pred_smoothed"]],
                                 on=["chain_id", "om_id"], how="inner")
            n = len(merged)
            if n == 0:
                print("[eval] no overlapping (chain, om) rows between test labels and predictions.")
            else:
                STI = clf.STATE_TO_INT
                yt = merged["true_state"].map(STI).to_numpy()
                yr = merged["pred_raw"].map(STI).to_numpy()
                ys = merged["pred_smoothed"].map(STI).to_numpy()
                acc_raw = float((yr == yt).mean())
                acc_sm = float((ys == yt).mean())
                m_raw = clf.per_class_metrics(yt, yr)
                m_sm = clf.per_class_metrics(yt, ys)
                cm_raw = clf.confusion(yt, yr)
                cm_sm = clf.confusion(yt, ys)

                merged.to_csv(phenology_dir / "phenophase_test_eval_perom.csv", index=False)
                m_raw.to_csv(phenology_dir / "phenophase_test_metrics_raw.csv", index=False)
                m_sm.to_csv(phenology_dir / "phenophase_test_metrics_smoothed.csv", index=False)

                # how many test OMs did smoothing change, and net effect
                changed = int((merged["pred_raw"] != merged["pred_smoothed"]).sum())
                fixed = int(((yr != yt) & (ys == yt)).sum())     # was wrong, now right
                broke = int(((yr == yt) & (ys != yt)).sum())     # was right, now wrong

                el = []
                el.append("=" * 60)
                el.append("TEST-SET EVALUATION: raw vs smoothed")
                el.append("=" * 60)
                el.append(f"Scored (chain x OM) rows: {n}")
                el.append(f"Smoothing changed {changed} test labels: "
                          f"{fixed} fixed (wrong->right), {broke} broke (right->wrong), "
                          f"net {fixed - broke:+d}")
                el.append("")
                el.append(f"Overall accuracy BEFORE smoothing: {acc_raw:.4f} ({acc_raw:.1%})")
                el.append(f"Overall accuracy AFTER  smoothing: {acc_sm:.4f} ({acc_sm:.1%})")
                el.append(f"Change: {acc_sm - acc_raw:+.4f} ({acc_sm - acc_raw:+.1%})")
                el.append("")
                el.append("Confusion BEFORE (rows=true, cols=pred):")
                el += ["  " + ln for ln in cm_raw.to_string().splitlines()]
                el.append("")
                el.append("Confusion AFTER (rows=true, cols=pred):")
                el += ["  " + ln for ln in cm_sm.to_string().splitlines()]
                el.append("")
                el.append("Per-class F1  (before -> after):")
                for st in clf.STATES:
                    f_b = m_raw.loc[m_raw["phenophase"] == st, "f1"]
                    f_a = m_sm.loc[m_sm["phenophase"] == st, "f1"]
                    if len(f_b) and len(f_a):
                        el.append(f"  {st:<16}{float(f_b.iloc[0]):.3f}  ->  {float(f_a.iloc[0]):.3f}")
                mac_b = float(m_raw.loc[m_raw['phenophase']=='macro avg','f1'].iloc[0])
                mac_a = float(m_sm.loc[m_sm['phenophase']=='macro avg','f1'].iloc[0])
                el.append(f"  {'macro avg':<16}{mac_b:.3f}  ->  {mac_a:.3f}")
                eval_report = "\n".join(el)
                print("\n" + eval_report)
                (phenology_dir / "phenophase_test_eval_report.txt").write_text(
                    eval_report, encoding="utf-8")

    # --- patch the GeoJSON ---
    gj = json.loads(gj_path.read_text())
    n_obs_patched = 0
    n_features_decid = 0
    n_features_ever = 0
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        cid = props.get("ids", {}).get("chain_id")
        if cid is None:
            continue
        cid = int(cid)

        if cid in evergreen_ids:
            # evergreen: classifier doesn't apply — assign leaf_on to every OM
            n_features_ever += 1
            for obs in props.get("observations", []):
                obs.setdefault("phenology", {})
                obs["phenology"]["phenophase"] = "leaf_on"
                obs["phenology"]["phenophase_source"] = "evergreen_assigned"
                n_obs_patched += 1
            continue

        chain_map = smoothed_by_chain.get(cid)
        if not chain_map:
            continue
        n_features_decid += 1
        for obs in props.get("observations", []):
            om = obs.get("om_id")
            if om is None:
                continue
            new_ph = chain_map.get(int(om))
            if new_ph is None:
                continue
            obs.setdefault("phenology", {})
            obs["phenology"]["phenophase"] = new_ph
            obs["phenology"]["phenophase_source"] = "gb_classifier_smoothed"
            n_obs_patched += 1
        # refresh event OMs in classification
        if not args.no_events and cid in events_by_chain:
            cls = props.setdefault("classification", {})
            for k, v in events_by_chain[cid].items():
                cls[k] = v

    gj.setdefault("phenology_config", {})["phenophase_method"] = {
        "type": "gb_classifier",
        "smoothing": (None if args.no_smooth else
                      {"min_run": args.min_run, "flip_below": args.flip_below}),
        "features": list(FEATURE_COLS),
    }

    if args.in_place:
        out_path = gj_path
    else:
        out_path = gj_path.with_name(gj_path.stem + "_phenoclf.geojson")
    out_path.write_text(json.dumps(gj, indent=2, default=str))

    # --- report ---
    rep = []
    rep.append("=" * 60)
    rep.append("PHENOPHASE PATCH REPORT")
    rep.append("=" * 60)
    rep.append(f"GeoJSON in : {gj_path}")
    rep.append(f"GeoJSON out: {out_path}")
    rep.append(f"Chains total:          {n_total}")
    rep.append(f"  Deciduous (classifier): {len(deciduous_ids)}")
    rep.append(f"  Evergreen (leaf_on):    {len(evergreen_ids)}")
    rep.append(f"Features patched:      {n_features_decid + n_features_ever} "
               f"({n_features_decid} deciduous, {n_features_ever} evergreen)")
    rep.append(f"Observations patched:  {n_obs_patched}")
    if not args.no_smooth:
        rep.append(f"Labels flipped by smoothing: {n_flipped_total} "
                   f"(min_run={args.min_run}, flip_below={args.flip_below})")
    rep.append("")
    rep.append("Smoothed phenophase distribution:")
    for st, c in pred_df["pred_smoothed"].value_counts().items():
        rep.append(f"  {st:<16}{c}")
    report = "\n".join(rep)
    print("\n" + report)
    (phenology_dir / "phenophase_patch_report.txt").write_text(report, encoding="utf-8")
    print(f"\nWrote: {out_path}")
    print(f"Wrote: {phenology_dir / 'phenophase_predictions_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())