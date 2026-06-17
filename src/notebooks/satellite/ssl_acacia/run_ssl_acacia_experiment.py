#!/usr/bin/env python3
"""Principled semi-supervised Acacia/non-Acacia experiments.

This script compares:
  1. supervised-only models trained on trusted labels,
  2. self-training, where confident predictions on unlabeled crowns are added,
  3. graph label spreading, where nearby crowns in feature space share label signal,
  4. the old hard clustering-label augmentation baseline.

The held-out test set always uses trusted labels only.
"""
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "drone_phenology_rf_package"
RF_EVAL = SourceFileLoader(
    "rf_eval",
    str(PACKAGE / "python" / "02_local_rf_from_gee_export.py"),
).load_module()

DEFAULT_CSV = ROOT / "embeddings" / "exports" / "gee_original_acacia_label_configs.csv"
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "outputs"
LABEL_DEFAULT = "label_acacia"
CLUSTER_LABEL = "label_acacia_clustering"
ID_COL = "crown_uid"


@dataclass(frozen=True)
class SplitSpec:
    split: str
    holdout: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--label-col", default=LABEL_DEFAULT)
    parser.add_argument("--cluster-label-col", default=CLUSTER_LABEL)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-holdouts", type=int, default=4)
    parser.add_argument("--self-training-threshold", type=float, default=0.85)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--random-only", action="store_true")
    return parser.parse_args()


def normalize_labels(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].fillna(-1).astype(int)
    return out


def choose_holdouts(df: pd.DataFrame, label_col: str, max_holdouts: int) -> list[str]:
    rows = []
    labeled = df[df[label_col] != -1].copy()
    for area, group in labeled.groupby("area"):
        counts = group[label_col].value_counts()
        if len(counts) < 2:
            continue
        rows.append((str(area), int(len(group)), int(counts.min())))
    rows = sorted(rows, key=lambda x: (x[2], x[1]), reverse=True)
    return [area for area, _, _ in rows[:max_holdouts]]


def build_split(df: pd.DataFrame, label_col: str, spec: SplitSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = df[df[label_col] != -1].copy()
    labeled[label_col] = labeled[label_col].astype(int)

    if spec.split == "random":
        train, test = train_test_split(
            labeled,
            test_size=0.30,
            random_state=spec.seed,
            stratify=labeled[label_col],
        )
        unlabeled = df[df[label_col] == -1].copy()
    elif spec.split == "leave_area_out":
        train = labeled[labeled["area"] != spec.holdout].copy()
        test = labeled[labeled["area"] == spec.holdout].copy()
        unlabeled = df[(df[label_col] == -1) & (df["area"] != spec.holdout)].copy()
    else:
        raise ValueError(f"Unsupported split: {spec.split}")

    if train[label_col].nunique() < 2:
        raise ValueError(f"Training set has fewer than two classes for {spec}")
    if test.empty or test[label_col].nunique() < 2:
        raise ValueError(f"Test set is empty or single-class for {spec}")
    return train.reset_index(drop=True), test.reset_index(drop=True), unlabeled.reset_index(drop=True)


def make_supervised_models(seed: int, trees: int) -> dict[str, Pipeline]:
    return {
        "supervised_logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        C=0.5,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "supervised_extra_trees": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=trees,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def make_ssl_models(seed: int, threshold: float, trees: int) -> dict[str, Pipeline]:
    logistic = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    C=0.5,
                    random_state=seed,
                ),
            ),
        ]
    )
    trees_model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=trees,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {
        "ssl_self_training_logistic": SelfTrainingClassifier(
            base_estimator=logistic,
            threshold=threshold,
            criterion="threshold",
            max_iter=10,
            verbose=False,
        ),
        "ssl_self_training_extra_trees": SelfTrainingClassifier(
            base_estimator=trees_model,
            threshold=threshold,
            criterion="threshold",
            max_iter=10,
            verbose=False,
        ),
    }


def score_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    labels = sorted(set(y_true.astype(int).tolist()) | set(pd.Series(y_pred).astype(int).tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "cm_labels": labels,
    }


def ssl_stats(model, n_labeled_train: int, n_unlabeled: int) -> dict:
    stats = {
        "n_pseudo_labeled": 0,
        "pseudo_label_counts": {},
        "ssl_iterations": None,
        "ssl_termination": "",
    }
    fitted = model
    if isinstance(model, Pipeline):
        fitted = model.named_steps["model"]

    transduction = getattr(fitted, "transduction_", None)
    if transduction is not None and n_unlabeled:
        pseudo = np.asarray(transduction)[n_labeled_train:]
        pseudo = pseudo[pseudo != -1]
        stats["n_pseudo_labeled"] = int(len(pseudo))
        stats["pseudo_label_counts"] = pd.Series(pseudo).astype(int).value_counts().sort_index().to_dict()

    labeled_iter = getattr(fitted, "labeled_iter_", None)
    if labeled_iter is not None:
        stats["ssl_iterations"] = int(np.max(labeled_iter)) if len(labeled_iter) else 0
    termination = getattr(fitted, "termination_condition_", None)
    if termination is not None:
        stats["ssl_termination"] = str(termination)
    return stats


def fit_and_score(
    method_name: str,
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    unlabeled: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    spec: SplitSpec,
) -> dict:
    x_test = test[feature_cols]
    y_test = test[label_col].astype(int)

    if method_name.startswith("ssl_"):
        train_ssl = pd.concat([train, unlabeled], ignore_index=True)
        x_train = train_ssl[feature_cols]
        y_train = np.r_[
            train[label_col].astype(int).to_numpy(),
            np.full(len(unlabeled), -1, dtype=int),
        ]
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        extra = ssl_stats(model, len(train), len(unlabeled))
    else:
        x_train = train[feature_cols]
        y_train = train[label_col].astype(int)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        extra = {}

    row = {
        "method": method_name,
        "split": spec.split,
        "holdout": spec.holdout,
        "seed": spec.seed,
        "n_labeled_train": int(len(train)),
        "n_unlabeled_train": int(len(unlabeled)) if method_name.startswith("ssl_") else 0,
        "n_test": int(len(test)),
        "train_counts": train[label_col].astype(int).value_counts().sort_index().to_dict(),
        "test_counts": test[label_col].astype(int).value_counts().sort_index().to_dict(),
        **score_predictions(y_test, pred),
        **extra,
    }
    return row


def fit_cluster_hard_label_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    unlabeled: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cluster_label_col: str,
    spec: SplitSpec,
    seed: int,
    trees: int,
) -> dict | None:
    if cluster_label_col not in train.columns and cluster_label_col not in unlabeled.columns:
        return None
    cluster_extra = unlabeled[unlabeled[cluster_label_col] != -1].copy()
    if cluster_extra.empty:
        return None
    cluster_extra[label_col] = cluster_extra[cluster_label_col].astype(int)
    augmented = pd.concat([train, cluster_extra], ignore_index=True)

    model = make_supervised_models(seed, trees)["supervised_extra_trees"]
    model.fit(augmented[feature_cols], augmented[label_col].astype(int))
    pred = model.predict(test[feature_cols])
    y_test = test[label_col].astype(int)
    return {
        "method": "hard_cluster_labels_extra_trees",
        "split": spec.split,
        "holdout": spec.holdout,
        "seed": spec.seed,
        "n_labeled_train": int(len(train)),
        "n_unlabeled_train": 0,
        "n_test": int(len(test)),
        "n_cluster_labeled_added": int(len(cluster_extra)),
        "cluster_label_counts_added": cluster_extra[label_col].astype(int).value_counts().sort_index().to_dict(),
        "train_counts": train[label_col].astype(int).value_counts().sort_index().to_dict(),
        "test_counts": test[label_col].astype(int).value_counts().sort_index().to_dict(),
        **score_predictions(y_test, pred),
    }


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = np.sum(a * a, axis=1, keepdims=True) + np.sum(b * b, axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def fit_numpy_label_spreading_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    unlabeled: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    spec: SplitSpec,
    n_neighbors: int = 15,
    alpha: float = 0.20,
    max_iter: int = 100,
    confidence_threshold: float = 0.85,
) -> dict:
    """Small dependency-light kNN label spreading.

    The graph is built over labeled training crowns plus eligible unlabeled crowns.
    Held-out test crowns are not included in the graph for leave-area-out runs.
    Test predictions are made by interpolating from the learned graph probabilities.
    """
    graph_df = pd.concat([train, unlabeled], ignore_index=True)
    y_graph = np.r_[
        train[label_col].astype(int).to_numpy(),
        np.full(len(unlabeled), -1, dtype=int),
    ]
    classes = np.array(sorted(train[label_col].astype(int).unique().tolist()), dtype=int)
    class_to_pos = {int(c): i for i, c in enumerate(classes)}
    labeled_mask = y_graph != -1

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_graph = imputer.fit_transform(graph_df[feature_cols])
    x_graph = scaler.fit_transform(x_graph)
    x_test = scaler.transform(imputer.transform(test[feature_cols]))

    n = x_graph.shape[0]
    k = min(n_neighbors, max(1, n - 1))
    d = squared_distances(x_graph, x_graph)
    np.fill_diagonal(d, np.inf)
    nn_idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    nn_dist = np.take_along_axis(d, nn_idx, axis=1)
    nn_weights = 1.0 / (np.sqrt(nn_dist) + 1e-6)

    w = np.zeros((n, n), dtype=np.float32)
    rows = np.arange(n)[:, None]
    w[rows, nn_idx] = nn_weights.astype(np.float32)
    w = np.maximum(w, w.T)
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    s = w / row_sums

    y_seed = np.zeros((n, len(classes)), dtype=np.float32)
    for i, label in enumerate(y_graph):
        if label != -1:
            y_seed[i, class_to_pos[int(label)]] = 1.0

    f = y_seed.copy()
    for _ in range(max_iter):
        f_next = alpha * (s @ f) + (1.0 - alpha) * y_seed
        f_next[labeled_mask] = y_seed[labeled_mask]
        delta = float(np.max(np.abs(f_next - f)))
        f = f_next
        if delta < 1e-5:
            break

    f_sum = f.sum(axis=1, keepdims=True)
    f_sum[f_sum == 0] = 1.0
    f = f / f_sum

    d_test = squared_distances(x_test, x_graph)
    test_k = min(n_neighbors, n)
    test_idx = np.argpartition(d_test, kth=test_k - 1, axis=1)[:, :test_k]
    test_dist = np.take_along_axis(d_test, test_idx, axis=1)
    test_weights = 1.0 / (np.sqrt(test_dist) + 1e-6)
    test_weights = test_weights / test_weights.sum(axis=1, keepdims=True)
    proba = np.einsum("ij,ijk->ik", test_weights, f[test_idx])
    pred = classes[np.argmax(proba, axis=1)]

    unlabeled_probs = f[len(train):] if len(unlabeled) else np.empty((0, len(classes)))
    confident = unlabeled_probs.max(axis=1) >= confidence_threshold if len(unlabeled_probs) else np.array([], dtype=bool)
    confident_labels = classes[np.argmax(unlabeled_probs[confident], axis=1)] if confident.any() else np.array([], dtype=int)

    return {
        "method": "ssl_numpy_label_spreading_knn",
        "split": spec.split,
        "holdout": spec.holdout,
        "seed": spec.seed,
        "n_labeled_train": int(len(train)),
        "n_unlabeled_train": int(len(unlabeled)),
        "n_test": int(len(test)),
        "n_pseudo_labeled": int(confident.sum()),
        "pseudo_label_counts": pd.Series(confident_labels).astype(int).value_counts().sort_index().to_dict(),
        "train_counts": train[label_col].astype(int).value_counts().sort_index().to_dict(),
        "test_counts": test[label_col].astype(int).value_counts().sort_index().to_dict(),
        **score_predictions(test[label_col].astype(int), pred),
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["method", "split", "holdout"], dropna=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            positive_f1_mean=("positive_f1", "mean"),
            positive_f1_std=("positive_f1", "std"),
            n_runs=("balanced_accuracy", "count"),
            n_labeled_train_mean=("n_labeled_train", "mean"),
            n_unlabeled_train_mean=("n_unlabeled_train", "mean"),
            n_pseudo_labeled_mean=("n_pseudo_labeled", "mean"),
        )
        .reset_index()
        .sort_values(["split", "holdout", "balanced_accuracy_mean", "macro_f1_mean"], ascending=[True, True, False, False])
    )


def write_brief(outdir: Path, args: argparse.Namespace, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    best_random = summary[summary["split"] == "random"].head(6)
    holdout_summary = summary[summary["split"] == "leave_area_out"].copy()
    best_holdout = (
        holdout_summary.groupby("method", as_index=False)
        .agg(
            mean_lao_bal_acc=("balanced_accuracy_mean", "mean"),
            mean_lao_macro_f1=("macro_f1_mean", "mean"),
            holdout_runs=("holdout", "nunique"),
        )
        .sort_values(["mean_lao_bal_acc", "mean_lao_macro_f1"], ascending=[False, False])
    )
    label_counts = df[args.label_col].value_counts().sort_index().to_dict()
    unlabeled = int((df[args.label_col] == -1).sum())

    def plain_table(table: pd.DataFrame) -> str:
        if table.empty:
            return "_No rows produced._"
        return table.to_string(index=False)

    lines = [
        "# SSL Acacia Experiment Brief",
        "",
        "## Setup",
        "",
        f"- Feature table: `{args.csv}`",
        f"- Trusted label column: `{args.label_col}`",
        f"- Trusted label counts: `{label_counts}`",
        f"- Unlabeled crowns available to SSL: `{unlabeled}`",
        f"- Hard clustering comparison column: `{args.cluster_label_col}`",
        "",
        "## Methods Compared",
        "",
        "- `supervised_*`: train only on trusted labels.",
        "- `ssl_self_training_*`: train on trusted labels, add only unlabeled crowns whose predicted class probability crosses the confidence threshold, then refit iteratively.",
        "- `ssl_numpy_label_spreading_knn`: build a nearest-neighbor graph in satellite feature space and diffuse trusted labels over the graph.",
        "- `hard_cluster_labels_extra_trees`: old comparison where clustering labels are treated as if they were ground truth.",
        "",
        "## Best Random-Split Rows",
        "",
        "```text",
        plain_table(best_random),
        "```",
        "",
        "## Mean Leave-Area-Out Ranking",
        "",
        "```text",
        plain_table(best_holdout),
        "```",
        "",
        "## Interpretation",
        "",
        "Use leave-area-out as the professor-facing result. Random split is a quick sanity check, but it can overstate performance because nearby crowns from the same area can appear in both train and test.",
        "",
        "The SSL methods are useful only if they improve held-out trusted-label performance or produce a small high-confidence review queue. If they pseudo-label many crowns but hurt leave-area-out balanced accuracy, the unlabeled distribution is probably not aligned enough with the trusted labels yet.",
        "",
    ]
    (outdir / "SSL_EXPERIMENT_BRIEF.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = normalize_labels(df, [args.label_col, args.cluster_label_col])
    feature_cols = RF_EVAL.infer_feature_cols(df, args.label_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns were found.")

    splits = [SplitSpec("random", "", seed) for seed in args.seeds]
    if not args.random_only:
        for holdout in choose_holdouts(df, args.label_col, args.max_holdouts):
            for seed in args.seeds:
                splits.append(SplitSpec("leave_area_out", holdout, seed))

    rows = []
    errors = []
    for spec in splits:
        print(f"\n== {spec.split} {spec.holdout or ''} seed={spec.seed} ==")
        try:
            train, test, unlabeled = build_split(df, args.label_col, spec)
            print(
                f"trusted train={len(train)} test={len(test)} unlabeled_for_ssl={len(unlabeled)} "
                f"train_counts={train[args.label_col].value_counts().sort_index().to_dict()} "
                f"test_counts={test[args.label_col].value_counts().sort_index().to_dict()}"
            )
            for name, model in make_supervised_models(spec.seed, args.trees).items():
                try:
                    row = fit_and_score(name, clone(model), train, test, unlabeled, feature_cols, args.label_col, spec)
                    rows.append(row)
                    print(f"{name}: bal_acc={row['balanced_accuracy']:.3f} macro_f1={row['macro_f1']:.3f}")
                except Exception as exc:
                    errors.append(
                        {
                            "method": name,
                            "split": spec.split,
                            "holdout": spec.holdout,
                            "seed": spec.seed,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"{name}: ERROR {exc!r}")

            for name, model in make_ssl_models(spec.seed, args.self_training_threshold, args.trees).items():
                try:
                    row = fit_and_score(name, clone(model), train, test, unlabeled, feature_cols, args.label_col, spec)
                    rows.append(row)
                    print(
                        f"{name}: bal_acc={row['balanced_accuracy']:.3f} macro_f1={row['macro_f1']:.3f} "
                        f"pseudo={row.get('n_pseudo_labeled', 0)}"
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "method": name,
                            "split": spec.split,
                            "holdout": spec.holdout,
                            "seed": spec.seed,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"{name}: ERROR {exc!r}")

            try:
                row = fit_numpy_label_spreading_and_score(
                    train,
                    test,
                    unlabeled,
                    feature_cols,
                    args.label_col,
                    spec,
                    confidence_threshold=args.self_training_threshold,
                )
                rows.append(row)
                print(
                    f"ssl_numpy_label_spreading_knn: bal_acc={row['balanced_accuracy']:.3f} "
                    f"macro_f1={row['macro_f1']:.3f} pseudo={row.get('n_pseudo_labeled', 0)}"
                )
            except Exception as exc:
                errors.append(
                    {
                        "method": "ssl_numpy_label_spreading_knn",
                        "split": spec.split,
                        "holdout": spec.holdout,
                        "seed": spec.seed,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"ssl_numpy_label_spreading_knn: ERROR {exc!r}")

            cluster_row = fit_cluster_hard_label_baseline(
                train,
                test,
                unlabeled,
                feature_cols,
                args.label_col,
                args.cluster_label_col,
                spec,
                spec.seed,
                args.trees,
            )
            if cluster_row is not None:
                rows.append(cluster_row)
                print(
                    "hard_cluster_labels_extra_trees: "
                    f"bal_acc={cluster_row['balanced_accuracy']:.3f} "
                    f"macro_f1={cluster_row['macro_f1']:.3f} "
                    f"added={cluster_row['n_cluster_labeled_added']}"
                )
        except Exception as exc:
            errors.append({"split": spec.split, "holdout": spec.holdout, "seed": spec.seed, "error": repr(exc)})
            print(f"ERROR: {exc!r}")

    if not rows:
        raise RuntimeError("No successful runs produced results.")

    results = pd.DataFrame(rows)
    summary = summarize(results).round(4)
    results.to_csv(outdir / "raw_results.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    if errors:
        (outdir / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    else:
        stale_errors = outdir / "errors.json"
        if stale_errors.exists():
            stale_errors.unlink()

    metadata = {
        "csv": str(args.csv),
        "label_col": args.label_col,
        "cluster_label_col": args.cluster_label_col,
        "feature_cols": feature_cols,
        "n_rows": int(len(df)),
        "label_counts": df[args.label_col].value_counts().sort_index().to_dict(),
        "seeds": args.seeds,
        "self_training_threshold": args.self_training_threshold,
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_brief(outdir, args, df, summary)

    print("\nWrote:")
    print(outdir / "raw_results.csv")
    print(outdir / "summary.csv")
    print(outdir / "SSL_EXPERIMENT_BRIEF.md")
    print("\nTop summary rows:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
