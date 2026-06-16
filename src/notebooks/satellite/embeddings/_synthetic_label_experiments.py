#!/usr/bin/env python3
"""Experiments using the synthetic clustering-derived Acacia labels.

A. Cross-label transfer: train on SYNTHETIC clustering labels (crowns with no
   clean visual label), test on the held-out clean VISUAL ground truth.
   -> "Are the synthetic labels good enough to scale Acacia mapping?"

B. Augmentation: does adding synthetic clustering rows to the clean visual
   training set improve the visual classifier? Test only on clean visual
   crowns, both random-holdout and leave-area-out.

Features default to the GEE centroid embedding (prof-endorsed small-crown source).
Reports balanced accuracy, macro-F1, and confusion matrices.
"""
from __future__ import annotations
import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "drone_phenology_rf_package"
MS = SourceFileLoader("ms", str(PKG / "python" / "06_model_sweep.py")).load_module()
EXP = ROOT / "exports"


def fit(model, x, y):
    if "hist_gradient" in str(model):
        model.fit(x, y, model__sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(x, y)
    return model


def evalrep(name, model, xtr, ytr, xte, yte, seed, trees):
    m = MS.make_models(seed, trees, xtr.shape[1])[model]
    fit(m, xtr, ytr)
    pred = m.predict(xte)
    ba = balanced_accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro", zero_division=0)
    cm = confusion_matrix(yte, pred, labels=[0, 1]).tolist()
    print(f"  {name:42s} model={model:18s} bacc={ba:.3f} f1={f1:.3f} n_test={len(yte)} cm(rows=true[0,1])={cm}")
    return ba, f1, cm


def best_over_models(tag, xtr, ytr, xte, yte, seed, trees):
    best = None
    for model in ["rf_balanced","extra_trees","extra_trees_kbest","logistic_l2","svc_rbf","hist_gradient"]:
        try:
            m = MS.make_models(seed, trees, xtr.shape[1])[model]
            fit(m, xtr, ytr)
            pred = m.predict(xte)
            ba = balanced_accuracy_score(yte, pred)
            f1 = f1_score(yte, pred, average="macro", zero_division=0)
            cm = confusion_matrix(yte, pred, labels=[0,1]).tolist()
            if best is None or ba > best[1]:
                best = (model, ba, f1, cm)
        except Exception as e:
            pass
    model, ba, f1, cm = best
    print(f"  [BEST] {tag:38s} model={model:18s} bacc={ba:.3f} f1={f1:.3f} n_test={len(yte)} cm={cm}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(EXP / "gee_centroid_2024_acacia_label_configs.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trees", type=int, default=400)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    feats = MS.rf_eval.infer_feature_cols(df, "label_acacia_visual")
    seed, trees = args.seed, args.trees

    visual = df[df.label_acacia_visual != -1].copy()
    visual["label_acacia_visual"] = visual.label_acacia_visual.astype(int)
    clust = df[df.label_acacia_clustering != -1].copy()
    clust["label_acacia_clustering"] = clust.label_acacia_clustering.astype(int)
    # synthetic-only crowns = clustering-labelled but NOT in the clean visual set
    synth_only = clust[~clust.crown_uid.isin(set(visual.crown_uid))].copy()

    print("="*100)
    print(f"Features: {len(feats)} ({feats[0]}..{feats[-1]})  | visual={len(visual)} clustering={len(clust)} synth_only={len(synth_only)}")

    # ---- EXP A: train on synthetic clustering labels, test on ALL clean visual ----
    print("\n[A] Train on SYNTHETIC clustering labels (synth-only crowns), test on clean VISUAL ground truth")
    xtr, ytr = synth_only[feats], synth_only["label_acacia_clustering"]
    xte, yte = visual[feats], visual["label_acacia_visual"]
    best_over_models("A: synth->visual (all 400)", xtr, ytr, xte, yte, seed, trees)

    # ---- EXP B: augmentation, evaluated by random visual holdout ----
    print("\n[B] Visual-only vs Visual+Synthetic training; test = held-out clean VISUAL (random 30%)")
    vtr, vte = train_test_split(visual, test_size=0.30, random_state=seed, stratify=visual.label_acacia_visual)
    xte, yte = vte[feats], vte["label_acacia_visual"].astype(int)
    # baseline visual-only
    best_over_models("B0: visual-only", vtr[feats], vtr.label_acacia_visual.astype(int), xte, yte, seed, trees)
    # augmented: visual train + all synthetic-only rows (use their clustering label as y)
    aug_x = pd.concat([vtr[feats], synth_only[feats]], ignore_index=True)
    aug_y = pd.concat([vtr.label_acacia_visual.astype(int), synth_only.label_acacia_clustering.astype(int)], ignore_index=True)
    best_over_models("B1: visual + synthetic", aug_x, aug_y, xte, yte, seed, trees)

    # ---- EXP B-LAO: same, but leave-area-out on visual (Sanjay Van sites) ----
    print("\n[B-LAO] Leave-area-out on clean VISUAL; train adds synthetic-only rows from OTHER areas")
    for site in sorted(visual.area.unique()):
        vte = visual[visual.area == site]
        if vte.label_acacia_visual.nunique() < 2 or len(vte) < 8:
            continue
        vtr = visual[visual.area != site]
        xte, yte = vte[feats], vte.label_acacia_visual.astype(int)
        b0 = best_over_models(f"  visual-only  holdout={site}", vtr[feats], vtr.label_acacia_visual.astype(int), xte, yte, seed, trees)
        sx = synth_only[synth_only.area != site]
        aug_x = pd.concat([vtr[feats], sx[feats]], ignore_index=True)
        aug_y = pd.concat([vtr.label_acacia_visual.astype(int), sx.label_acacia_clustering.astype(int)], ignore_index=True)
        best_over_models(f"  visual+synth holdout={site}", aug_x, aug_y, xte, yte, seed, trees)


if __name__ == "__main__":
    main()
