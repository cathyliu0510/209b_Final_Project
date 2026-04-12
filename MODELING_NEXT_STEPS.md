# Modeling Next Steps — Milestone 3

## Overview

Our EDA established that:
- Raw pixel statistics do **not** predict economic activity across cities (pooled r = 0.01)
- **Within-metro temporal correlations** are strong (e.g., Denver VIIRS vs GDP r = 0.72)
- **GHSL built-up growth** correlates with GDP growth (r = 0.28), far outperforming raw pixels
- **2020 is excluded** as a structural break (COVID); temporal split: train 2013--2018, val 2019, test 2021--2023

The modeling plan has two stages: (1) generate spatial features via U-Net segmentation, and (2) use those features for economic prediction.

---

## Stage 1: U-Net Segmentation (Minh)

**Goal:** Train a U-Net to predict GHSL built-up masks from MODIS RGB imagery, then generate annual masks for all metros and years (2013--2023).

### 1.1 Data Preparation
- **Input:** MODIS RGB tiles (3 channels, 250m resolution) per metro per GHSL epoch
- **Target:** Binarized GHSL built-up masks (~90m, resampled to MODIS grid)
- **Training pairs:** 70 image-mask pairs (14 metros x 5 epochs: 2000, 2005, 2010, 2015, 2020)
- **Split:** Train on 2000/2005/2010 epochs, validate on 2015, test on 2020

### 1.2 Model Architecture
- U-Net with encoder pretrained on satellite imagery (e.g., ResNet-18 or EfficientNet backbone)
- Binary segmentation: built-up vs. not-built-up
- Loss: Binary cross-entropy + Dice loss (handles class imbalance)

### 1.3 Evaluation
- **Metrics:** IoU (Intersection over Union), Dice coefficient, pixel accuracy
- **Held-out evaluation:** Performance on 2015 and 2020 GHSL epochs for unseen metros
- **Key question:** Is 70 training pairs sufficient, or do we need augmentation?

### 1.4 Mask Generation
- Apply trained U-Net to all MODIS tiles (14 metros x 11 years = 154 predictions)
- Output: annual built-up probability maps and binary masks

### 1.5 Key Assumption
We assume **temporal stability** -- that the visual-to-built-up relationship learned from 2000--2010 transfers to 2013--2023. This is reasonable for gradual urban expansion but should be validated by checking mask quality on 2015/2020 held-out epochs.

---

## Stage 2: Spatial Feature Engineering (Jenny)

**Goal:** Extract interpretable spatial features from U-Net-predicted masks to replace raw pixel statistics.

### 2.1 Features to Extract (per metro, per year)
| Feature | Description | Why it matters |
|---------|-------------|----------------|
| **Built-up area (km^2)** | Total pixels classified as built-up | Direct measure of urban footprint |
| **Built-up fraction** | Built-up pixels / total pixels | Density measure |
| **Compactness** | Perimeter^2 / (4pi x Area) | Compact vs. sprawling growth |
| **Infill vs. sprawl ratio** | New built-up inside vs. outside existing footprint | Growth pattern characterization |
| **Built-up change (delta)** | Year-over-year change in built-up area | Expansion rate |
| **Edge density** | Built-up boundary length / area | Urban fragmentation |
| **Largest patch index** | Largest contiguous built-up cluster / total | Monocentric vs. polycentric |

### 2.2 Integration with Economic Panel
- Merge spatial features with existing `panel_features.csv` (140 rows, 47 columns)
- Add 1-year and 2-year lag features for spatial metrics
- Result: enriched panel with ~55--60 features

---

## Stage 3: Panel Regression Baseline (Jenny)

**Goal:** Establish an interpretable baseline using fixed-effects panel regression.

### 3.1 Model Specification
```
y_{it} = alpha_i + beta * X_{i,t-k} + epsilon_{it}
```
- `y_{it}`: economic outcome (GDP growth, employment growth, permits) for metro i in year t
- `alpha_i`: metro fixed effects (absorb cross-city level differences)
- `X_{i,t-k}`: satellite-derived spatial features with k-year lag (k = 1, 2, 3)
- Addresses the key EDA finding: within-metro temporal signal matters, cross-city pooling fails

### 3.2 Feature Sets to Compare
1. **Raw pixel stats only** (current features: VIIRS mean, lit fraction, Gini, etc.)
2. **GHSL-derived spatial features only** (built-up area, compactness, etc.)
3. **Combined** (raw pixels + spatial features)
4. **Spatial features + economic lags** (full model)

### 3.3 Evaluation
- **Train:** 2013--2018 (6 years x 14 metros = 84 obs)
- **Val:** 2019 (14 obs) -- used for lag selection and hyperparameter tuning
- **Test:** 2021--2023 (3 years x 14 metros = 42 obs)
- **Metrics:** R^2, RMSE, MAE on test set
- **2020 excluded** to avoid COVID distortion

---

## Stage 4: LSTM Temporal Prediction (Maddy)

**Goal:** Capture nonlinear temporal dynamics using an LSTM with multi-year lag structure.

### 4.1 Input Structure
- Sequence of spatial + economic features per metro, with lookback window of 1--3 years
- Each time step: vector of ~55--60 features
- Target: next-year economic outcome (GDP growth, employment change, permits)

### 4.2 Architecture
- LSTM with 1--2 hidden layers (start small given 140-row panel)
- Regularization: dropout, early stopping on validation loss
- Consider: GRU as simpler alternative if LSTM overfits

### 4.3 Comparison
- LSTM vs. panel regression on identical train/val/test splits
- Does temporal modeling add predictive power beyond fixed-effects regression?
- Per-target comparison: GDP growth, employment growth, building permits

### 4.4 Overfitting Risk
- 140 observations (10 years x 14 metros) is small for deep learning
- Mitigation: aggressive regularization, leave-one-metro-out cross-validation, feature selection

---

## Stage 5: Dataset Extension (Cathy)

**Goal:** Broaden geographic and temporal coverage.

### 5.1 Temporal Extension
- Extend economic panel back to 2001 (BEA GDP data available)
- More pre-period data strengthens temporal patterns and lag analysis

### 5.2 Geographic Extension
- Add Rust Belt metros (Detroit, Cleveland, Pittsburgh) -- different growth trajectory
- Add Northeast metros (Boston, Philadelphia) -- mature urban areas
- Tests whether findings generalize beyond Sun Belt growth cities

### 5.3 Documentation
- Finalize all data sources and pipeline documentation
- Ensure full reproducibility of the unified notebook

---

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | U-Net training + evaluation on held-out GHSL epochs |
| 2 | Generate annual masks, extract spatial features, merge with panel |
| 3 | Panel regression baseline + LSTM implementation |
| 4 | Model comparison, ablation studies, final evaluation on test years |
| 5 | Write-up and final presentation |

---

## Open Design Decisions

1. **U-Net data sufficiency:** Is 70 image-mask pairs enough? Consider data augmentation (flips, rotations, crops) and transfer learning from pretrained satellite models.
2. **GHSL epoch interpolation:** Should we interpolate GHSL between 5-year epochs to create denser supervision, or keep sparse labels and rely on U-Net generalization?
3. **Lag selection:** Which lag (1, 2, or 3 years) gives best predictive signal? Use validation set (2019) for selection.
4. **COVID handling:** 2020 is excluded from training/prediction. Consider adding a binary COVID indicator for 2021 if recovery effects persist.
5. **Generalizability:** Will patterns from Sun Belt metros transfer to Rust Belt / Northeast cities with declining or stable populations?
