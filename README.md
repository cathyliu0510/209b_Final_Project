# Urban Expansion vs. Economic Activity

**AC209b / CS1090B Final Project, Canvas Group 12**

This project studies whether satellite imagery can support economic monitoring when official economic data are delayed, sparse, or incomplete. The final modeling task is **satellite-only economic analogue retrieval**: given a city-year satellite image, retrieve historical city-years with similar economic structure and use those analogues to estimate GDP growth.

## Final Deliverables

| Deliverable | File | Purpose |
| --- | --- | --- |
| Final report | [`cs1090b_ms4_report_group12.pdf`](cs1090b_ms4_report_group12.pdf) | Paper-style MS4 report with motivation, data, methods, results, limitations, broader impact, and references |
| Final presentation deck | [`cs1090b_ms4_presentation_group12.pptx`](cs1090b_ms4_presentation_group12.pptx) | Slide deck used for the final video presentation |
| Main final notebook | [`cs1090b_ms4_main_group12.ipynb`](cs1090b_ms4_main_group12.ipynb) | End-to-end final modeling notebook for the 30-metro satellite-only retrieval pipeline |
| Final EDA notebook | [`final_eda_data_pipeline.ipynb`](final_eda_data_pipeline.ipynb) | Data construction, preprocessing checks, EDA, and exported figures used to motivate the modeling setup |
| Baseline-model notebook | [`baseline_model_selection_and_justification.ipynb`](baseline_model_selection_and_justification.ipynb) | MS3 baseline model selection and justification notebook used as the reference baseline stage |

## Project Question

Can satellite imagery alone retrieve historical city-years with similar economic conditions, after the image representation has been aligned with economic indicators during training?

The final system does not claim that raw imagery directly determines GDP. Instead, it learns a shared multimodal representation during training and then uses satellite imagery as a query for economically meaningful nearest-neighbor retrieval at inference time.

## Data Summary

| Item | Final setup |
| --- | --- |
| Geographic scope | 30 U.S. metropolitan areas |
| Time span | 2013-2023 |
| Unit of observation | Metro-year |
| Satellite inputs | MODIS RGB and VIIRS nighttime radiance |
| Built-up supervision | GHSL built-up surface masks |
| Economic indicators | BEA GDP, BLS employment / unemployment, Census building permits |
| Target for final benchmark | Year-over-year real GDP growth |
| Train split | 2013-2018 |
| Validation split | 2019 |
| Test split | 2021-2023 |
| Excluded year | 2020, due to the COVID structural break |

## Final Modeling Pipeline

| Stage | Method | Role |
| --- | --- | --- |
| Image encoder | ResNet-18 fine-tuned with GHSL built-up supervision | Learns urban-aware satellite embeddings from MODIS RGB and VIIRS nightlights |
| Economic encoder | MLP autoencoder selected over GRU and LSTM | Compresses economic indicators into a compact city-year economic state |
| Multimodal alignment | Contrastive VAE with modality dropout | Aligns satellite and economic embeddings in a shared 16-dimensional latent space |
| Satellite-only inference | k-nearest-neighbor retrieval in the learned latent space | Retrieves historical analogue city-years and averages their GDP growth |

## Final Results

The selected retrieval rule is **Scaled Euclidean k=8**, chosen on the 2019 validation set and then frozen before final testing.

| Method | 2021-2023 test MAE on GDP growth |
| --- | ---: |
| Random retrieval, averaged over 100 draws | 2.923 |
| Training-set mean | 2.558 |
| Best plain-cosine retrieval | 2.504 |
| **Scaled Euclidean k=8, selected** | **2.435** |
| Previous-year economic value | 2.358 |

The final satellite-only retrieval model outperforms the naive no-economics baselines while using no current economic indicators at inference time. It remains slightly behind the previous-year economic baseline, which is expected because that baseline uses economic data unavailable in the satellite-only setting.

## Key Figures

| Figure | File |
| --- | --- |
| Satellite imagery overview | [`figures/01_satellite_imagery_grid.png`](figures/01_satellite_imagery_grid.png) |
| COVID structural break | [`figures/10_covid_impact_indexed.png`](figures/10_covid_impact_indexed.png) |
| GHSL built-up change vs. GDP growth | [`figures/17_ghsl_vs_gdp_growth.png`](figures/17_ghsl_vs_gdp_growth.png) |
| Latent economic structure | [`figures/09b_umap_economic.png`](figures/09b_umap_economic.png) |
| Retrieval tuning summary | [`figures/minh_stage4_tuning_summary.png`](figures/minh_stage4_tuning_summary.png) |
| Final GDP-growth benchmark | [`figures/minh_stage4_gdp_retrieval.png`](figures/minh_stage4_gdp_retrieval.png) |

## Reproducibility Notes

The main final notebook is designed to load the cached 30-city artifacts and checkpoints that are already included in the repository:

- `data/tensors/`
- `data/modeling/panel_features.csv`
- `data/ghsl/`
- `checkpoints_30city_v2/`
- `deliverables/minh_stage4_benchmark.csv`
- `deliverables/minh_stage4_retrieval_tuning.csv`

To rerun the final modeling workflow, open:

```text
cs1090b_ms4_main_group12.ipynb
```

The notebook documents all required paths, split logic, model stages, training settings, validation tuning, and final test results.

## Generative AI Tooling Disclosure

During project development, the team used **Claude Code** and **OpenAI Codex** as coding and formatting tools for debugging, refactoring, notebook organization, and writing polish. These tools supported implementation workflow only. The research question, modeling choices, evaluation design, interpretation of results, and final scientific judgments were made and reviewed by the project team.
