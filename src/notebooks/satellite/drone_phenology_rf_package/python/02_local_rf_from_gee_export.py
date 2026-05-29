#!/usr/bin/env python3
"""
Train/evaluate Random Forest locally from a GEE-exported feature CSV.

Typical use:
  python python/02_local_rf_from_gee_export.py \
    --csv exports/crown_rf_export_features_label_esd_buffer_2025.csv \
    --label label_esd \
    --split random

  python python/02_local_rf_from_gee_export.py \
    --csv exports/crown_rf_export_features_label_acacia_buffer_2025.csv \
    --label label_acacia \
    --split leave_area_out \
    --holdout SV_S4

The CSV should contain predictor columns exported by gee/rf_crown_classifiers.js.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

NON_FEATURE_COLS = {
    'system:index', '.geo', 'geo', 'geometry', 'crown_uid', 'area', 'source_file', 'source_index',
    'orig_crown_id', 'crown_num', 'species_raw', 'species_clean', 'species_status', 'tree_type_raw',
    'health_class', 'field_status', 'field_description', 'lon', 'lat', 'classification', 'random',
    'stac_year', 'stac_item_count', 'buffer_meters', 's1_item_count'
}
LABEL_COLS = {
    'label_esd', 'label_deciduous', 'label_acacia', 'label_yellow_strict', 'label_yellow_broad',
    'label_red_showy', 'label_showy_flower'
}


def infer_feature_cols(df: pd.DataFrame, label: str) -> list[str]:
    drop = NON_FEATURE_COLS | LABEL_COLS | {label}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def make_split(df: pd.DataFrame, label: str, split: str, holdout: str | None, seed: int):
    usable = df[df[label] != -1].copy()
    usable[label] = usable[label].astype(int)
    if split == 'leave_area_out':
        if not holdout:
            raise ValueError('--holdout area is required for leave_area_out')
        train = usable[usable['area'] != holdout]
        test = usable[usable['area'] == holdout]
    elif split == 'leave_species_out':
        if not holdout:
            raise ValueError('--holdout species_clean is required for leave_species_out')
        train = usable[usable['species_clean'] != holdout]
        test = usable[usable['species_clean'] == holdout]
    else:
        train, test = train_test_split(
            usable, test_size=0.30, random_state=seed, stratify=usable[label]
        )
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV exported from GEE')
    ap.add_argument('--label', required=True, choices=sorted(LABEL_COLS))
    ap.add_argument('--split', default='random', choices=['random', 'leave_area_out', 'leave_species_out'])
    ap.add_argument('--holdout', default=None, help='Area or species to hold out, depending on split mode')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trees', type=int, default=500)
    ap.add_argument('--outdir', default='outputs/local_rf')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    train, test = make_split(df, args.label, args.split, args.holdout, args.seed)
    feature_cols = infer_feature_cols(df, args.label)
    if not feature_cols:
        raise ValueError('No numeric predictor columns found. Did you export features from GEE?')

    X_train = train[feature_cols]
    y_train = train[args.label].astype(int)
    X_test = test[feature_cols]
    y_test = test[args.label].astype(int)

    model = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('rf', RandomForestClassifier(
            n_estimators=args.trees,
            random_state=args.seed,
            class_weight='balanced_subsample',
            min_samples_leaf=2,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = {
        'csv': args.csv,
        'label': args.label,
        'split': args.split,
        'holdout': args.holdout,
        'n_train': int(len(train)),
        'n_test': int(len(test)),
        'train_label_counts': y_train.value_counts().sort_index().to_dict(),
        'test_label_counts': y_test.value_counts().sort_index().to_dict(),
        'accuracy': float(accuracy_score(y_test, pred)) if len(test) else None,
        'balanced_accuracy': float(balanced_accuracy_score(y_test, pred)) if len(test) else None,
        'macro_f1': float(f1_score(y_test, pred, average='macro')) if len(test) else None,
        'confusion_matrix': confusion_matrix(y_test, pred).tolist() if len(test) else [],
        'classification_report': classification_report(y_test, pred, zero_division=0, output_dict=True) if len(test) else {},
    }

    rf = model.named_steps['rf']
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_,
    }).sort_values('importance', ascending=False)

    stem = f'{args.label}_{args.split}' + (f'_{args.holdout}' if args.holdout else '')
    with open(outdir / f'{stem}_metrics.json', 'w') as f:
        json.dump(report, f, indent=2)
    importance.to_csv(outdir / f'{stem}_feature_importance.csv', index=False)

    print(json.dumps(report, indent=2))
    print('\nTop 25 features:')
    print(importance.head(25).to_string(index=False))


if __name__ == '__main__':
    main()
