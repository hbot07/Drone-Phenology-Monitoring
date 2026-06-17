# Semi-Supervised Acacia Labelling

This folder is a clean experiment lane for replacing the clustering-as-labels shortcut with semi-supervised learning (SSL).

## Why The Clustering Hack Hurt

The previous augmentation treated cluster membership as ground truth. That is risky because clustering answers a different question: "which crowns look similar in satellite feature space?" It does not know the botanical class boundary. If a cluster mixes Acacia and non-Acacia, hard labels inject systematic noise and the classifier learns the wrong boundary with high confidence.

SSL is more cautious. Trusted labels remain the only ground truth. Unlabeled crowns help only by shaping the feature-space geometry or by being added when the model is sufficiently confident.

## What We Have Here

The current GEE original-crown embedding table has 3,212 crowns:

- `label_acacia`: 326 non-Acacia, 65 Acacia, 2,821 unlabeled.
- `label_acacia_visual_or_species`: more trusted labels if visual labels are accepted, with 516 non-Acacia, 251 Acacia, 2,445 unlabeled.
- `label_acacia_clustering`: the old cluster-derived labels, used here only as a comparison baseline.

The default script uses `label_acacia` because it matches the original "~2,800 unlabeled crowns" setting.

## Methods

`run_ssl_acacia_experiment.py` compares:

- `supervised_logistic` and `supervised_extra_trees`: trusted labels only.
- `ssl_self_training_logistic` and `ssl_self_training_extra_trees`: train on trusted labels, pseudo-label only high-confidence unlabeled crowns, then iterate.
- `ssl_numpy_label_spreading_knn`: build a nearest-neighbor graph over satellite features and diffuse labels through that graph.
- `hard_cluster_labels_extra_trees`: old baseline where cluster labels are promoted to hard labels.

Every method is evaluated only on held-out trusted labels.

## Run

From the repository root:

```bash
python src/notebooks/satellite/ssl_acacia/run_ssl_acacia_experiment.py
```

For the stronger visual/species label set:

```bash
python src/notebooks/satellite/ssl_acacia/run_ssl_acacia_experiment.py \
  --label-col label_acacia_visual_or_species \
  --outdir src/notebooks/satellite/ssl_acacia/outputs_visual_or_species
```

For a faster smoke test:

```bash
python src/notebooks/satellite/ssl_acacia/run_ssl_acacia_experiment.py --random-only --seeds 42
```

## Outputs

The script writes:

- `outputs/raw_results.csv`: one row per method/split/seed.
- `outputs/summary.csv`: averaged metrics.
- `outputs/SSL_EXPERIMENT_BRIEF.md`: compact professor-facing summary.
- `outputs/run_metadata.json`: exact input, label counts, feature columns, and parameters.

## How To Present This

The headline should be:

1. We do not use unlabeled crowns as truth.
2. We keep manual/visual/species labels as anchors.
3. We test whether unlabeled crowns improve generalization on held-out trusted labels.
4. We report leave-area-out results first, because random split can be spatially optimistic.
5. If SSL does not improve held-out performance, the useful output is still a ranked high-confidence review queue, not automatic labels for all crowns.

## Practical Next Step

Use the winning SSL model only to propose labels for human review:

- high-confidence Acacia candidates,
- high-confidence non-Acacia candidates,
- uncertain crowns near the boundary.

That gives a principled active-learning loop: label the most informative crowns, retrain, and repeat.
