# Datasets

CAR-MNAR is evaluated on three public clinical datasets, bundled here under
`raw/` so the framework runs offline. Each is a small, widely used benchmark.

| File | Dataset | Source | Samples | Variables (excl. target) |
|------|---------|--------|---------|--------------------------|
| `raw/Diabetes.csv`  | Pima Indians Diabetes | UCI / Kaggle (`uciml/pima-indians-diabetes-database`) | 768 | 8 |
| `raw/Heart.csv`     | Heart Disease (Cleveland) | UCI Heart Disease | 303 | 13 |
| `raw/Hepatitis.csv` | Hepatitis | UCI Hepatitis | 142 | 19 |

A non-clinical row-index column present in `Heart.csv` (`no`) is dropped
automatically by the loader so that it does not enter the causal graph.

## Provenance / re-download

The originals are available from the UCI Machine Learning Repository
(<https://archive.ics.uci.edu/>) and Kaggle. To re-fetch them into `raw/`:

```bash
python -m scripts.download_data
```

Please consult the original repositories for dataset licences and terms of use;
they are redistributed here only for reproducibility of the published results.

## Synthetic data

Synthetic datasets with known ground-truth DAGs are generated on demand by the
synthetic dataset factory (`carmnar.data_generation`) and are not stored here.
