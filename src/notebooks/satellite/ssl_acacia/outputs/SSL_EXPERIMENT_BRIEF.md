# SSL Acacia Experiment Brief

## Setup

- Feature table: `/Users/hbot07/VS Code/Drone-Phenology-Monitoring/src/notebooks/satellite/embeddings/exports/gee_original_acacia_label_configs.csv`
- Trusted label column: `label_acacia`
- Trusted label counts: `{-1: 2821, 0: 326, 1: 65}`
- Unlabeled crowns available to SSL: `2821`
- Hard clustering comparison column: `label_acacia_clustering`

## Methods Compared

- `supervised_*`: train only on trusted labels.
- `ssl_self_training_*`: train on trusted labels, add only unlabeled crowns whose predicted class probability crosses the confidence threshold, then refit iteratively.
- `ssl_numpy_label_spreading_knn`: build a nearest-neighbor graph in satellite feature space and diffuse trusted labels over the graph.
- `hard_cluster_labels_extra_trees`: old comparison where clustering labels are treated as if they were ground truth.

## Best Random-Split Rows

```text
                         method  split holdout  balanced_accuracy_mean  balanced_accuracy_std  macro_f1_mean  macro_f1_std  positive_f1_mean  positive_f1_std  n_runs  n_labeled_train_mean  n_unlabeled_train_mean  n_pseudo_labeled_mean
  ssl_self_training_extra_trees random                          0.9393                    NaN         0.8920           NaN            0.8261              NaN       1                 273.0                  2821.0                 2161.0
     ssl_self_training_logistic random                          0.9143                    NaN         0.8764           NaN            0.8000              NaN       1                 273.0                  2821.0                 2584.0
            supervised_logistic random                          0.9092                    NaN         0.8650           NaN            0.7826              NaN       1                 273.0                     0.0                    NaN
         supervised_extra_trees random                          0.8745                    NaN         0.8672           NaN            0.7805              NaN       1                 273.0                     0.0                    NaN
hard_cluster_labels_extra_trees random                          0.8398                    NaN         0.8689           NaN            0.7778              NaN       1                 273.0                     0.0                    NaN
  ssl_numpy_label_spreading_knn random                          0.8296                    NaN         0.8432           NaN            0.7368              NaN       1                 273.0                  2821.0                 2314.0
```

## Mean Leave-Area-Out Ranking

```text
                         method  mean_lao_bal_acc  mean_lao_macro_f1  holdout_runs
         supervised_extra_trees          0.602900           0.599700             4
  ssl_numpy_label_spreading_knn          0.596175           0.588875             4
hard_cluster_labels_extra_trees          0.592175           0.532900             4
     ssl_self_training_logistic          0.529850           0.430275             4
            supervised_logistic          0.509700           0.430625             4
  ssl_self_training_extra_trees          0.500000           0.370525             4
```

## Interpretation

Use leave-area-out as the professor-facing result. Random split is a quick sanity check, but it can overstate performance because nearby crowns from the same area can appear in both train and test.

The SSL methods are useful only if they improve held-out trusted-label performance or produce a small high-confidence review queue. If they pseudo-label many crowns but hurt leave-area-out balanced accuracy, the unlabeled distribution is probably not aligned enough with the trusted labels yet.
