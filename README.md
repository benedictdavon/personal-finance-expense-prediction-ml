# Personal Finance Expense Prediction ML

This is a synthetic-data ML project for daily expense prediction and person-level spending-behavior clustering. It is an offline research/coursework prototype, not a personal-finance product and not trained on real financial records.

Small Python ML project for synthetic personal finance data

The project generates transactions for 50 student personas, builds daily spending features, trains Random Forest expense predictors, and clusters people by spending behavior

## Latest Results

Default run: `python scripts/run_all.py`

Dataset settings: 50 synthetic people, 2025-01-01 to 2025-12-31, seed 42

| Metric | Value |
| --- | ---: |
| Generated transactions | 51,005 |
| Spending profiles | 7 |
| Best supervised feature set | `advanced_behavioral` |
| Best test RMSE | 237.34 NTD |
| Best test MAE | 107.23 NTD |
| Best test R2 | 0.655 |
| Best CV RMSE mean | 234.22 NTD |
| Selected K-Means k | 6 |
| Best silhouette | 0.581 |
| Best Davies-Bouldin | 0.542 |

Supervised comparison:

| Feature set | Test RMSE | Test MAE | Test R2 |
| --- | ---: | ---: | ---: |
| `advanced_behavioral` | 237.34 | 107.23 | 0.655 |
| `person_weekday_weekend_baseline` | 375.74 | 171.31 | 0.136 |
| `person_expanding_mean_baseline` | 398.51 | 203.18 | 0.028 |
| `calendar_lag` | 434.47 | 207.31 | -0.155 |
| `calendar_only` | 437.99 | 209.97 | -0.174 |
| `full_behavioral` | 456.56 | 209.21 | -0.275 |
| `calendar_rolling` | 458.74 | 210.51 | -0.287 |

The advanced feature set improves the earlier score by adding historical person behavior, category-level rolling spend, income timing, and calendar features

## Dataset Comparison

Full comparison command:

```bash
python scripts/run_dataset_comparison.py --num-people 50 --seed 42 --annual-inflation-rate 0.025
```

Output:

```text
results/metrics/dataset_comparison.csv
```

| Dataset | Years | Rows | Missing days | Best RMSE | Best MAE | Best R2 | Selected k | Silhouette |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_synthetic` | 1 | 51,005 | 0 | 237.34 | 107.23 | 0.655 | 6 | 0.581 |
| `simple_rules_1y` | 1 | 42,691 | 101 | 411.55 | 201.04 | 0.762 | 2 | 0.364 |
| `realistic_mixed_1y` | 1 | 41,235 | 277 | 1024.76 | 249.70 | 0.351 | 7 | 0.364 |
| `realistic_mixed_2y` | 2 | 81,777 | 594 | 616.37 | 227.49 | 0.633 | 2 | 0.354 |
| `realistic_mixed_3y` | 3 | 122,322 | 850 | 553.56 | 213.29 | 0.730 | 3 | 0.370 |
| `realistic_mixed_5y` | 5 | 202,807 | 1,478 | 1007.97 | 258.27 | 0.280 | 4 | 0.402 |
| `stress_test_1y` | 1 | 38,141 | 508 | 5244.42 | 653.15 | -0.055 | 2 | 0.714 |

The realistic and stress-test datasets are much harder than the baseline data

See [`docs/evaluation-provenance.md`](docs/evaluation-provenance.md) for the split and synthetic-data limitations.

Longer realistic histories help the model learn more stable patterns for 2-year and 3-year runs, while the 5-year run is harder because inflation, shocks, and missing logs accumulate

## Problem

Predict daily spending for synthetic students and group people with similar spending habits

The project focuses on two tasks:

- Daily expense prediction from past behavior and calendar context
- Person-level clustering from spending summaries

## Synthetic Data

No real financial data is used

`scripts/generate_data.py` creates transaction data for these profile types:

- `frugal_student`
- `food_heavy_spender`
- `commuter`
- `social_weekend_spender`
- `irregular_high_variance_spender`
- `part_time_worker`
- `lifestyle_gym_spender`

Generated fields:

- `person_id`
- `profile_type`
- `date`
- `category`
- `transaction_type`
- `amount_ntd`
- `day_of_week`
- `is_weekend`
- `month`
- `payday_distance`
- `income_source`
- `payment_method`
- `description`

Generated files:

```text
data/raw/synthetic_expense_transactions.csv
data/processed/daily_person_features.csv
```

Generated data is ignored by git and can be recreated from the scripts

Two generators are available:

- `scripts/generate_data.py` for the original baseline synthetic data
- `scripts/generate_realistic_data.py` for scenario-based data with longer histories, inflation, heavy-tailed purchases, balance effects, and missed log days

See `scripts/README.md` for details on the data generation process

## Modeling

Supervised model:

- `RandomForestRegressor`
- Time-based holdout split
- `TimeSeriesSplit` cross-validation
- RMSE, MAE, and R2 reporting

Feature groups:

- Basic calendar features
- Lag and rolling total spend
- Rolling 3/7/14/30-day spend statistics
- Rolling category spend for food, coffee, transport, groceries, social food, gym, and irregular purchases
- Income timing features
- Expanding person/profile baselines
- Weekend and profile interactions

Baseline rows are included in the metrics table so the model can be compared against simple historical averages

Clustering:

- K-Means from k=2 to k=10
- Gaussian Mixture Model using the selected k
- PCA projection for plotting
- Silhouette, Davies-Bouldin, Calinski-Harabasz, and inertia metrics

## Project Structure

```text
AI-Capstone-Homework-1/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/
|   `-- processed/
|-- results/
|   |-- figures/
|   |-- tables/
|   `-- metrics/
|-- src/
|   `-- expense_modeling/
|       |-- config.py
|       |-- data_generator.py
|       |-- preprocessing.py
|       |-- features.py
|       |-- supervised.py
|       |-- clustering.py
|       |-- evaluation.py
|       `-- visualization.py
|-- scripts/
|   |-- generate_data.py
|   |-- generate_realistic_data.py
|   |-- run_supervised.py
|   |-- run_clustering.py
|   |-- run_dataset_comparison.py
|   `-- run_all.py
`-- tests/
    |-- test_data_generator.py
    `-- test_features.py
```

## Setup

```bash
git clone https://github.com/benedictdavon/personal-finance-expense-prediction-ml.git
cd personal-finance-expense-prediction-ml
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Full pipeline:

```bash
python scripts/run_all.py
```

Individual steps:

```bash
python scripts/generate_data.py --num-people 50 --seed 42
python scripts/run_supervised.py
python scripts/run_clustering.py
```

Tests:

```bash
python -m pytest tests
```

Main outputs:

```text
results/figures/clustering_metrics.png
results/figures/feature_importance_full_behavioral.png
results/figures/spending_clusters_pca.png
results/figures/supervised_rmse_by_feature_set.png
results/metrics/dataset_comparison.csv
results/metrics/supervised_metrics.csv
results/metrics/clustering_metrics.csv
results/tables/feature_importance.csv
results/tables/cluster_assignments.csv
```

## Notes

- Synthetic data makes the workflow reproducible, but it does not prove performance on real bank data
- The profiles are hand-built, so clustering is easier than it would be with real people
- The model predicts total daily spend, not category-level spend
- Irregular high-value purchases remain the hardest days to predict
- Notebooks are available for reference, while scripts are the reproducible path
