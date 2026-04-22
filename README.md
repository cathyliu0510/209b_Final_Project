# Urban Expansion vs Economic Activity

This project studies whether satellite-observed urban change can help explain or predict **future metro-level economic activity**. The workflow combines NASA Worldview / GIBS imagery, official U.S. economic indicators, exploratory data analysis, and baseline predictive modeling before the planned GHSL-derived spatial features are added.

## Project Snapshot

| Item | Current project status |
| --- | --- |
| **Core question** | Do urban expansion signals and night-light changes help predict future GDP, employment, and permits? |
| **Current scope** | 14 U.S. metros, annual panel from **2013-2023** |
| **Satellite data** | MODIS RGB composites and VIIRS night-lights summaries |
| **Economic data** | BEA GDP, BLS employment / unemployment, Census building permits |
| **Completed stages** | Data pipeline, unified panel construction, EDA, raw-pixel baseline modeling |
| **Current modeling milestone** | [`Jenny_baseline_model_selection_and_justification.ipynb`](Jenny_baseline_model_selection_and_justification.ipynb) |
| **Most important next step** | Generate GHSL / built-up spatial features and compare them against the stronger raw-pixel baseline |

## Repository Guide

| Artifact | What it does | Why it matters |
| --- | --- | --- |
| [`01_gibs_tile_fetcher_v5.ipynb`](01_gibs_tile_fetcher_v5.ipynb) | Downloads and mosaics MODIS / VIIRS imagery from NASA GIBS | Builds the satellite-image foundation |
| [`02_economic_data_downloader_v6.ipynb`](02_economic_data_downloader_v6.ipynb) | Pulls BEA, BLS, and Census indicators and merges them into a metro-year panel | Builds the economic target dataset |
| [`03_raster_preprocessing.ipynb`](03_raster_preprocessing.ipynb) | Reprojects, normalizes, cloud-masks, and stacks raster data | Prepares model-ready image tensors |
| [`00_Final_EDA_Merged_finalized.ipynb`](00_Final_EDA_Merged_finalized.ipynb) | Main EDA notebook | Establishes the statistical patterns that motivate modeling |
| [`Jenny_baseline_model_selection_and_justification.ipynb`](Jenny_baseline_model_selection_and_justification.ipynb) | Executed baseline-model notebook | Runs and explains the milestone baseline models end to end |
| [`scripts/build_baseline_model_notebook.py`](scripts/build_baseline_model_notebook.py) | Regenerates the baseline-model notebook | Keeps the milestone notebook reproducible |
| [`MODELING_NEXT_STEPS.md`](MODELING_NEXT_STEPS.md) | Modeling roadmap for later milestones | Defines the GHSL / spatial-feature plan and future ablations |
| [`figures/`](figures/) and [`EDA_Figures/`](EDA_Figures/) | Exported figures used across the project | Stores presentation-ready visuals |

## Current Modeling Panel

| Component | Value |
| --- | --- |
| **Metros** | 14: Atlanta, Austin, Charlotte, Dallas, Denver, Houston, Jacksonville, Las Vegas, Nashville, Orlando, Phoenix, Raleigh, San Antonio, Tampa |
| **Panel rows** | 140 metro-year observations |
| **Time split** | Train: 84 rows (`2013-2018`), Validation: 14 rows (`2019`), Test: 42 rows (`2021-2023`) |
| **Excluded year** | `2020`, treated as a COVID structural break |
| **Current target in the baseline notebook** | `employment_thousands_growth` |
| **Current predictors** | Conservative lagged raw-pixel summaries for the baseline family, plus a pruned expanded lagged panel for the regularized benchmark |
| **Why this is still only a baseline** | The project's central built-up / urban-form features are not yet in the panel |

## Baseline Modeling Results

The current baseline notebook is intentionally stronger than a superficial benchmark. It first explains why the original held-out `R²` looked weak, then compares a transparent fixed-effects-style model, a stronger regularized linear benchmark, and a nonlinear 109B baseline.

| Model | Test R^2 | Test MAE | Interpretation |
| --- | ---: | ---: | --- |
| **Linear Regression (fixed effects)** | 0.142 | 2.100 | Transparent benchmark closest to the planned panel-regression stage |
| **Ridge Regression (expanded lagged panel)** | 0.233 | 1.843 | **Highlighted current raw-pixel benchmark** after adding a broader lagged panel and regularization |
| **Gradient Boosting Regressor** | 0.180 | 1.936 | Most stable nonlinear 109B baseline across the pre-period checks |

| Sanity check | Validation MAE | Test MAE | Why it matters |
| --- | ---: | ---: | --- |
| **Naive persistence benchmark** | 0.272 | 3.645 | Shows why `2019` alone is too fragile to trust as the only model-selection criterion |

## What The EDA Already Established

| EDA takeaway | Why it matters for modeling |
| --- | --- |
| Raw pooled pixel summaries are weak across cities | Cross-city pooling alone is not enough |
| Within-metro temporal structure is much stronger | Time-aware panel modeling is the right direction |
| GHSL built-up change looks more promising than raw pixel summaries | Later spatial-feature stages are scientifically important |

## Essential Visuals

### 1. EDA signal before modeling

![Cross-correlation heatmap](figures/06_cross_correlation_heatmap.png)

This figure summarizes the strongest current EDA message: **within-metro temporal relationships are more informative than pooled cross-city comparisons**, which is exactly why the baseline notebook uses lagged features and a time-aware split.

### 2. Baseline model stability and held-out comparison

<p align="center">
  <img src="figures/08_rolling_validation_stability.png" alt="Rolling validation stability" width="49%" />
  <img src="figures/07_baseline_model_comparison.png" alt="Final baseline model comparison" width="49%" />
</p>

The left panel shows why model selection should not depend on a single validation year. The right panel shows the final held-out comparison across the three revised baseline models.

### 3. What the highlighted benchmark is doing by test year

![Year-by-year benchmark performance](figures/10_benchmark_yearwise_performance.png)

This figure replaces the earlier pooled scatter. It makes the key point much clearer: the current benchmark struggles most in `2021`, improves in `2022`, and is noticeably tighter in `2023`. That is a more informative summary of the held-out behavior than one dense identity-line plot.

### 4. What the highlighted benchmark is actually using

![Selected-model feature importance](figures/09_baseline_feature_importance.png)

The Ridge benchmark still draws substantial signal from **lagged satellite-change summaries plus lagged economic context**. That is an important research finding because it sets a clearer and harder-to-beat bar for the future GHSL-derived feature set.

## Current Conclusion

1. The project already has a clean, reproducible baseline modeling milestone.
2. The current notebook answers the narrower question, "How far can raw satellite summaries take us before richer spatial features are ready?"
3. The next real scientific test is whether **built-up / urban-form features** outperform this stronger raw-pixel benchmark.

## Recommended Reading Order

1. Start with [`00_Final_EDA_Merged_finalized.ipynb`](00_Final_EDA_Merged_finalized.ipynb) for the full EDA story.
2. Read [`Jenny_baseline_model_selection_and_justification.ipynb`](Jenny_baseline_model_selection_and_justification.ipynb) for the current modeling milestone.
3. Use [`MODELING_NEXT_STEPS.md`](MODELING_NEXT_STEPS.md) to see exactly how the GHSL / spatial-feature stage will extend this work.

## Reproducibility

To regenerate the baseline notebook file:

```bash
python3 scripts/build_baseline_model_notebook.py
```

The notebook generator writes directly to:

```text
Jenny_baseline_model_selection_and_justification.ipynb
```
