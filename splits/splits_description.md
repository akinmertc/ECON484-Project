# Splits Description

## Split 01 — Stratified Train/Validation

| File | Rows | Ratio | Description |
|------|------|-------|-------------|
| `split_01_train_80pct.csv` | 344 | 80% | Training set |
| `split_01_val_20pct.csv` | 86 | 20% | Validation set |

**Method:** `sklearn.model_selection.train_test_split` with `stratify=Sector` to preserve sector proportions in both subsets.

**Random seed:** 42 (ensures reproducibility)

**Rationale:** DBSCAN and Isolation Forest are unsupervised models — there are no ground-truth labels to train against. This split is used for validation purposes only: anomaly rates computed on the training set are compared against the validation set to confirm that the detected anomalies are consistent across different subsets of the data and not artifacts of any particular company cluster.
