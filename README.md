# Urban Expansion vs Economic Activity

This repository studies whether **satellite-observed urban change** can help predict **future metro-level economic activity**. The current deliverables already cover the data pipeline, exploratory data analysis, and a compact but time-aware baseline-model notebook.

The baseline-model milestone deliverable is:

- [`Jenny_baseline_model_selection_and_justification.ipynb`](Jenny_baseline_model_selection_and_justification.ipynb)

## 1. Project At a Glance

| Item | Current status |
| --- | --- |
| **Core question** | Do lagged satellite signals help predict future metro-level GDP, employment, and permits? |
| **Geography** | 14 U.S. metros |
| **Panel span** | 2013-2023 |
| **Satellite inputs** | MODIS RGB summaries and VIIRS night-light summaries |
| **Economic inputs** | BEA GDP, BLS employment / unemployment, Census permits |
| **Completed stages** | Data pipeline, unified panel construction, EDA, baseline modeling |
| **Baseline target in the notebook** | `employment_thousands_growth` |
| **Selected reporting baseline** | **Ridge Regression on the expanded lagged panel** |
| **Strongest official holdout performer** | **Gradient Boosting Regressor** |
| **Next scientific milestone** | Add GHSL / built-up spatial features and compare them against the raw-summary baseline |

## 2. Repository Guide

| Artifact | What it contains | Why it matters |
| --- | --- | --- |
| [`00_Final_EDA_Merged_finalized.ipynb`](00_Final_EDA_Merged_finalized.ipynb) | Main exploratory analysis notebook | Establishes the empirical motivation for time-aware modeling |
| [`Jenny_baseline_model_selection_and_justification.ipynb`](Jenny_baseline_model_selection_and_justification.ipynb) | Executed baseline-model milestone notebook | Runs model selection, tuning, evaluation, and interpretation |
| [`scripts/build_baseline_model_notebook.py`](scripts/build_baseline_model_notebook.py) | Notebook generator | Rebuilds the notebook and exports the baseline figures |
| [`MODELING_NEXT_STEPS.md`](MODELING_NEXT_STEPS.md) | Modeling roadmap | Defines the GHSL / spatial-feature stage and planned ablations |
| [`01_gibs_tile_fetcher_v5.ipynb`](01_gibs_tile_fetcher_v5.ipynb) | Satellite data acquisition | Downloads and mosaics NASA GIBS imagery |
| [`02_economic_data_downloader_v6.ipynb`](02_economic_data_downloader_v6.ipynb) | Economic data construction | Builds the metro-year target panel |
| [`03_raster_preprocessing.ipynb`](03_raster_preprocessing.ipynb) | Raster preprocessing pipeline | Produces cleaned, aligned satellite tensors |
| [`figures/`](figures/) | Presentation-ready exported visuals | Stores EDA and baseline-model figures used across the project |

## 3. Baseline Modeling Setup

### 3.1 Effective sample and split

| Component | Value |
| --- | --- |
| **Target** | `employment_thousands_growth` |
| **Raw panel rows** | 140 metro-year observations |
| **Rows used for modeling** | 126 observations with a defined growth target |
| **Training years** | `2014-2018` |
| **Validation year** | `2019` |
| **Held-out test years** | `2021-2023` |
| **Excluded year** | `2020`, treated as a COVID structural break |
| **Leakage control** | All predictive features are lagged; all splits are time-based |

### 3.2 Baseline models in the notebook

| Model | Why it is included | Role |
| --- | --- | --- |
| **Linear Regression with metro fixed effects** | Simplest interpretable benchmark | Transparent reference model |
| **Ridge Regression on an expanded lagged panel** | Keeps linear interpretability while handling a small, collinear feature space | **Selected reporting baseline** |
| **Gradient Boosting Regressor** | Standard Stat 109B nonlinear comparison that can capture interactions and thresholds | Strong nonlinear comparator |

## 4. Time-Aware Cross-Validation Workflow

The baseline notebook separates **tuning** from **official evaluation**.

| Stage | Years used | Purpose |
| --- | --- | --- |
| **Rolling-origin CV fold 1** | train on years before `2016`, validate on `2016` | Hyperparameter tuning |
| **Rolling-origin CV fold 2** | train on years before `2017`, validate on `2017` | Hyperparameter tuning |
| **Rolling-origin CV fold 3** | train on years before `2018`, validate on `2018` | Hyperparameter tuning |
| **Official validation** | `2019` | Model reporting only, not tuning |
| **Held-out test** | `2021-2023` | Final evaluation only |

The notebook uses **rolling-CV mean MAE** as the primary tuning criterion and reports rolling-CV mean `R^2` as secondary context.

### 4.1 Split structure

![Official split and rolling-origin CV protocol](figures/15_baseline_cv_protocol.png)

This visual makes the evaluation design explicit:

- the top row is the official train / validation / test split used for reporting;
- the lower rows are the historical rolling-origin folds used only for tuning;
- `2020` is excluded throughout.

### 4.2 Compact tuning search

![Compact hyperparameter tuning summary](figures/16_baseline_tuning_summary.png)

The search is intentionally light:

| Model | Search space |
| --- | --- |
| **Linear Regression** | no tuning |
| **Ridge Regression** | `alpha ∈ {0.01, 0.1, 1, 10, 100}` |
| **Gradient Boosting** | small grid over `n_estimators`, `learning_rate`, `max_depth`, and `min_samples_leaf` |

This keeps the notebook defensible as a **baseline study** rather than a large optimization exercise.

## 5. Baseline Selection and Official Results

### 5.1 Selection rule

| Decision question | Answer |
| --- | --- |
| **Primary selection rule** | Lowest rolling-origin CV mean MAE |
| **Selected reporting baseline** | **Ridge Regression (expanded lagged panel)** with `alpha = 100` |
| **Why Ridge is selected** | It has the best average rolling-CV MAE under the pre-specified rule |
| **Strongest official validation / test performer** | **Gradient Boosting Regressor** with `100 trees`, `lr = 0.03`, `depth = 2`, `leaf = 1` |
| **Interpretation** | The small panel does not yield a one-number ranking, so the notebook reports both the selection rule and the holdout winner clearly |

### 5.2 Final model comparison

| Model | Selected hyperparameters | Rolling CV Mean MAE | Rolling CV Mean R^2 | Validation MAE | Test MAE | Test R^2 | Role |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Linear Regression (fixed effects)** | default | 1.129 | -2.122 | 1.605 | 2.100 | 0.142 | Simple reference model |
| **Ridge Regression (expanded lagged panel)** | `alpha = 100` | **0.848** | -0.706 | 0.744 | 2.007 | 0.115 | **Selected reporting baseline** |
| **Gradient Boosting Regressor** | `100 trees`, `lr = 0.03`, `depth = 2`, `leaf = 1` | 0.856 | **-0.162** | **0.617** | **1.944** | **0.167** | Strongest nonlinear comparison |

## 6. Essential Modeling Visuals

### 6.1 Rolling-CV stability vs official holdout comparison

<p align="center">
  <img src="figures/08_rolling_validation_stability.png" alt="Rolling-origin validation stability" width="49%" />
  <img src="figures/07_baseline_model_comparison.png" alt="Official validation and held-out test comparison" width="49%" />
</p>

Read these two figures together:

- the left panel shows how each tuned model behaves across the historical rolling holdout years;
- the right panel shows official validation and held-out test performance after refitting on the full training window.

### 6.2 Selected baseline diagnostics

<p align="center">
  <img src="figures/10_benchmark_yearwise_performance.png" alt="Selected baseline performance by held-out year" width="49%" />
  <img src="figures/09_baseline_feature_importance.png" alt="Selected baseline feature interpretation" width="49%" />
</p>

These figures answer two practical follow-up questions:

- how the selected baseline behaves across `2021`, `2022`, and `2023`;
- which lagged feature groups contribute most to its predictions.

## 7. EDA Anchor

![Cross-correlation heatmap](figures/06_cross_correlation_heatmap.png)

The EDA already suggested that **within-metro temporal relationships are more useful than pooled cross-city relationships**. That is exactly why the baseline-model notebook uses lagged predictors, time-based splits, and rolling-origin validation instead of random cross-validation.

## 8. Reading Order

| If you want to understand... | Start here |
| --- | --- |
| **The exploratory evidence behind the project** | [`00_Final_EDA_Merged_finalized.ipynb`](00_Final_EDA_Merged_finalized.ipynb) |
| **The baseline-model milestone deliverable** | [`Jenny_baseline_model_selection_and_justification.ipynb`](Jenny_baseline_model_selection_and_justification.ipynb) |
| **What the next modeling stage should do** | [`MODELING_NEXT_STEPS.md`](MODELING_NEXT_STEPS.md) |

## 9. Reproducibility

To regenerate the notebook and the exported baseline figures:

```bash
python3 scripts/build_baseline_model_notebook.py
```

The generator writes directly to:

```text
Jenny_baseline_model_selection_and_justification.ipynb
```
