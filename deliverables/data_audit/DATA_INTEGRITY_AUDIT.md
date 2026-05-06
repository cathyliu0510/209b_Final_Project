# Data Integrity and MODIS Quality Audit

This audit was generated from the current working tree to explain the project's actual data state, not just the intended state described in the notebooks.

## Executive Summary

- The local working tree currently contains the restored **14-metro** data inventory, but `upstream/main` still exposes only **5 metros** and **5 imagery folders**.
- The most complete published branch is `upstream/rename-add-prefix`, which exposes **14 metros** in the economic panel and **14 metro imagery folders**.
- The fixed `08-01` MODIS acquisition date is a heuristic, not a guarantee of low cloud cover for every metro-year. A broader diffuse-cloud proxy shows multiple high-risk frames even when the notebook's strict near-white cloud mask stays low.
- The current notebook cloud mask likely **underestimates** cloud contamination because it only flags nearly pure white pixels. The audit therefore logs both the notebook-compatible strict metric and a broader diffuse-cloud risk proxy.
- Varying image dimensions are stable within each metro and remain exact multiples of 512 pixels, which is consistent with full GIBS tile mosaics. Rectangular rasters therefore do not automatically mean a city was cut in half.
- What is still unresolved is semantic bbox quality: stable tile geometry is reassuring, but a proper overlay review against metro boundaries is still needed before making strong urban-expansion claims.

## 1. Branch and Inventory Status

| Ref | Commit | Panel rows | Panel metros | Imagery metros |
| --- | --- | --- | --- | --- |
| working-tree | 99e8c9e | 154 | 14 | 14 |
| upstream/main | e7c0b9a | 55 | 5 | 5 |
| upstream/rename-add-prefix | 50fdca2 | 154 | 14 | 14 |
| origin/main | 9aa50cc | 55 | 5 | 5 |
| origin/rename-add-prefix | 54ffb47 | 55 | 5 | 5 |

### Rationale

- The modeling and EDA notebooks describe a 14-metro project, so any 5-metro branch is an inconsistent project state, not just a smaller sample choice.
- The safest branch-level source of truth for the restored data is currently `upstream/rename-add-prefix`, not `upstream/main`.

## 2. Metro-Level MODIS / VIIRS Inventory and Quality

| Metro | MODIS years | VIIRS years | Strict mean cloud % | Diffuse mean cloud % | Worst diffuse cloud % | Years >= 40% diffuse cloud | MODIS dims | Geometry check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| las_vegas | 11 | 7 | 3.98 | 55.43 | 80.07 | 10 | 1024x512 | stable full-tile mosaic |
| phoenix | 11 | 7 | 2.94 | 54.66 | 83.91 | 9 | 512x1024 | stable full-tile mosaic |
| denver | 11 | 7 | 2.54 | 17.67 | 57.37 | 2 | 512x512 | stable full-tile mosaic |
| dallas | 11 | 7 | 0.16 | 15.89 | 47.16 | 1 | 512x512 | stable full-tile mosaic |
| raleigh | 11 | 7 | 0.06 | 12.81 | 31.91 | 0 | 1024x512 | stable full-tile mosaic |
| tampa | 11 | 7 | 0.09 | 11.73 | 30.42 | 0 | 512x512 | stable full-tile mosaic |
| jacksonville | 11 | 7 | 0.10 | 10.08 | 16.91 | 0 | 1024x512 | stable full-tile mosaic |
| orlando | 11 | 7 | 0.09 | 9.04 | 18.37 | 0 | 512x512 | stable full-tile mosaic |

The full metro-level audit table is saved as `deliverables/data_audit/metro_imagery_audit.csv`.

## 3. Worst Cloud Cases

The current MODIS inventory contains **22 metro-year frames** with **diffuse-cloud risk at or above 40%**. This is a more realistic indicator for gray or hazy cloud scenes than the notebook's strict near-white mask.

| Metro | Year | Strict cloud % | Diffuse cloud % | Dark border % | Width | Height |
| --- | --- | --- | --- | --- | --- | --- |
| phoenix | 2018 | 4.04 | 83.91 | 12.05 | 512 | 1024 |
| las_vegas | 2021 | 10.05 | 80.07 | 0.00 | 1024 | 512 |
| phoenix | 2015 | 9.04 | 79.16 | 0.00 | 512 | 1024 |
| las_vegas | 2016 | 6.20 | 77.78 | 0.00 | 1024 | 512 |
| las_vegas | 2015 | 5.88 | 65.40 | 0.00 | 1024 | 512 |
| phoenix | 2022 | 7.98 | 65.27 | 0.00 | 512 | 1024 |
| phoenix | 2016 | 2.10 | 61.09 | 0.00 | 512 | 1024 |
| phoenix | 2019 | 0.45 | 60.68 | 23.36 | 512 | 1024 |
| las_vegas | 2019 | 3.50 | 59.24 | 0.00 | 1024 | 512 |
| las_vegas | 2014 | 8.05 | 58.47 | 0.00 | 1024 | 512 |
| phoenix | 2013 | 2.97 | 58.05 | 0.00 | 512 | 1024 |
| phoenix | 2021 | 0.60 | 57.57 | 0.00 | 512 | 1024 |

### Rationale

- This directly supports the teammate concern that some MODIS frames are not visually reliable for interpreting urban expansion.
- The fact that the diffuse metric is often much larger than the notebook-compatible strict metric is itself a research problem worth documenting: the current cloud filter is probably too permissive for downstream interpretation.

## 4. Dimension and Cropping Check

| Metro | Expected bbox aspect | Observed image aspect | Aspect error % | MODIS dims | Tile grid | Status |
| --- | --- | --- | --- | --- | --- | --- |
| dallas | 1.121 | 1.000 | 10.79 | 512x512 | 1x1 tiles | stable full-tile mosaic |
| houston | 1.310 | 1.000 | 23.69 | 512x512 | 1x1 tiles | stable full-tile mosaic |
| jacksonville | 1.007 | 2.000 | 98.55 | 1024x512 | 2x1 tiles | stable full-tile mosaic |
| las_vegas | 1.138 | 2.000 | 75.79 | 1024x512 | 2x1 tiles | stable full-tile mosaic |
| orlando | 1.238 | 1.000 | 19.25 | 512x512 | 1x1 tiles | stable full-tile mosaic |
| phoenix | 1.061 | 0.500 | 52.86 | 512x1024 | 1x2 tiles | stable full-tile mosaic |
| raleigh | 0.980 | 2.000 | 104.18 | 1024x512 | 2x1 tiles | stable full-tile mosaic |
| san_antonio | 1.315 | 2.000 | 52.09 | 1024x512 | 2x1 tiles | stable full-tile mosaic |
| tampa | 0.889 | 1.000 | 12.50 | 512x512 | 1x1 tiles | stable full-tile mosaic |

### Rationale

- This check compares the configured bbox shape against the saved raster shape, but it also records the GIBS tile grid implied by the image dimensions.
- Because the fetch notebook saves full 512-pixel tiles rather than exact geographic crops, aspect mismatch alone is not strong evidence of clipping. Stable `1x2` or `2x1` tile grids can still be expected outcomes.
- This audit does **not** prove that every bbox is semantically correct; it only shows that the saved geometry is stable and tile-aligned rather than obviously broken.

## 5. Why August 1 Was Used, and Why It Is Not Enough

- `01_gibs_tile_fetcher_v5.ipynb` sets `MONTH_DAY = "08-01"` and comments that it is the default because August 1 often gives relatively low cloud cover in CONUS.
- That choice is a practical starting heuristic, not a validated per-metro or per-year optimum.
- The cloud audit above shows why the current README should state this explicitly: some metro-years still have severe cloud obstruction even after choosing the August 1 default.

## 6. Candidate-Date Search Follow-Up

A compact candidate-date search was run for the highest-risk metros (`phoenix`, `las_vegas`, `denver`, `dallas`) across `2013-2023`, comparing:

- `07-01`
- `07-15`
- `08-01`
- `08-15`
- `09-01`

Result summary:

- in **34 of 44** searched metro-years, `08-01` was **not** the best candidate
- some of the largest improvements were:
  - `phoenix 2018`: `08-01` diffuse cloud `83.91%` → `2018-09-01` diffuse cloud `15.93%`
  - `denver 2018`: `08-01` diffuse cloud `57.37%` → `2018-09-01` diffuse cloud `1.87%`
  - `las_vegas 2021`: `08-01` diffuse cloud `80.07%` → `2021-07-15` diffuse cloud `39.03%`
  - `dallas 2022`: `08-01` diffuse cloud `47.16%` → `2022-08-15` diffuse cloud `7.15%`

Artifacts:

- `deliverables/data_audit/modis_date_search/modis_date_search_summary.md`
- `deliverables/data_audit/modis_date_search/modis_date_candidates.csv`
- `scripts/search_modis_candidate_dates.py`

## 7. Recommended Immediate Actions

1. Restore the 14-metro dataset onto the team-facing `main` branch before anyone continues modeling from the published repo.
2. Regenerate the worst MODIS metros first using the searched candidate dates, starting with `Phoenix`, `Las Vegas`, `Denver`, and `Dallas`.
3. Replace the fixed-date MODIS fetch rule in the acquisition notebook with the new candidate-date search workflow, using a broader diffuse-cloud score rather than only the strict near-white mask.
4. Run a manual GIS overlay review for metro bboxes before using imagery as evidence of urban expansion in the final report.
