# 209b
Final Project

Part 1 -- Cathy

**NASA Worldview / GIBS satellite imagery → Notebook 01**

  - Fetches MODIS RGB and VIIRS night-lights for all 5 metros × all years

  - Handles tile mosaicking, georeferencing, and saves as GeoTIFFs
    
**Metro-level economic indicators + merge by city and year → Notebook 02**

  - BEA county-aggregated GDP
    
  - BLS employment and unemployment rate
    
  - Census BPS building permits
    
  - All merged into a single panel.csv indexed by (metro, year)
    
**Initial organization and preprocessing → Notebook 03**

  - Reprojects VIIRS to match MODIS grid
    
  - Normalizes bands consistently across cities and years
    
  - Cloud masks MODIS RGB
    
  - Stacks everything into (T, H, W, C) tensors per metro
    
  - Applies train/val/test split by year
    
  - Saves to data/tensors/ ready for model input

---

Part 2 -- Jenny

**Final EDA Merged → `Final_EDA_Merged.ipynb`**

  - Merges all three upstream notebooks into one coherent end-to-end pipeline

  - Loads satellite imagery and economic panel, builds 4-channel (RGB + night-light) tensors per metro

  - Cloud masking, VIIRS reprojection to MODIS grid, percentile-based normalization

  - Extracts 20+ satellite-derived features per (metro, year): band statistics, NDVI proxy, lit-pixel fraction, Gini concentration, temporal deltas/growth rates, lag features

  - Merges satellite features with economic panel; audits and resolves missingness across cities and years

  - Applies z-score normalization using training-set statistics only (no data leakage)

  - Produces 14 polished EDA visualizations saved to `EDA_Figures/`:
    - Satellite imagery grid, economic time series, missingness heatmap
    - Feature distributions, satellite feature trends, cross-correlation heatmap
    - Scatter plots with regression, within-metro temporal correlations
    - Night-light change maps, COVID-19 impact analysis, growth rate comparison
    - Pixel-level distributions, pairplots, train/val/test split

  - Exports modeling-ready datasets to `data/modeling/`:
    - `panel_features.csv` — full merged panel (un-normalized)
    - `panel_normalized.csv` — z-score standardized for modeling
    - `normalization_stats.json` — training-set means/stds for reproducibility
