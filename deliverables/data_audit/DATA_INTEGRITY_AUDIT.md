# Data Integrity and MODIS Quality Audit

This audit was generated from the current working tree to explain the project's actual data state, not just the intended state described in the notebooks.

## Executive Summary

- The local working tree currently contains the restored **14-metro** data inventory, but `upstream/main` still exposes only **5 metros** and **5 imagery folders**.
- The most complete published branch is `upstream/rename-add-prefix`, which exposes **14 metros** in the economic panel and **14 metro imagery folders**.
- The MODIS acquisition workflow is now driven by an audited per-metro-year manifest: **136 of 154** metro-years moved away from the old `08-01` heuristic.
- The final selection rule now prioritizes full coverage, then center-region visibility, and only then whole-frame cloud minimization. In the refreshed imagery inventory there are **0** frames with missing tiles and **0** selected dates with large dark-gap coverage.
- Residual center-region cloud risk is now concentrated in **6** refreshed metro-years with core diffuse-cloud score above 12%.
- Varying image dimensions are stable within each metro and remain exact multiples of 512 pixels, which is consistent with full GIBS tile mosaics. Rectangular rasters therefore do not automatically mean a city was cut in half.
- Stable tile geometry does not by itself prove semantic bbox correctness, so imagery should still be interpreted as raster-aligned metro views rather than exact legal boundaries.

## 1. Branch and Inventory Status

| Ref | Commit | Panel rows | Panel metros | Imagery metros |
| --- | --- | --- | --- | --- |
| working-tree | 4567e02 | 154 | 14 | 14 |
| upstream/main | e7c0b9a | 55 | 5 | 5 |
| upstream/rename-add-prefix | 50fdca2 | 154 | 14 | 14 |
| origin/main | 9aa50cc | 55 | 5 | 5 |
| origin/rename-add-prefix | 54ffb47 | 55 | 5 | 5 |

### Rationale

- The modeling and EDA notebooks describe a 14-metro project, so any 5-metro branch is an inconsistent project state, not just a smaller sample choice.
- The safest branch-level source of truth for the restored data is currently `upstream/rename-add-prefix`, not `upstream/main`.

## 2. Metro-Level MODIS / VIIRS Inventory and Quality

| Metro | MODIS years | VIIRS years | Strict mean cloud % | Core mean cloud % | Worst core cloud % | Years >= 12% core cloud | MODIS dims | Geometry check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| las_vegas | 11 | 7 | 1.73 | 15.10 | 28.45 | 6 | 1024x512 | stable full-tile mosaic |
| phoenix | 11 | 7 | 0.17 | 4.74 | 11.59 | 0 | 512x1024 | stable full-tile mosaic |
| raleigh | 11 | 7 | 0.14 | 1.83 | 6.44 | 0 | 1024x512 | stable full-tile mosaic |
| jacksonville | 11 | 7 | 0.11 | 1.61 | 3.01 | 0 | 1024x512 | stable full-tile mosaic |
| dallas | 11 | 7 | 0.04 | 1.54 | 4.29 | 0 | 512x512 | stable full-tile mosaic |
| atlanta | 11 | 7 | 0.18 | 1.00 | 3.32 | 0 | 1024x1024 | stable full-tile mosaic |
| tampa | 11 | 7 | 0.03 | 0.99 | 2.19 | 0 | 512x512 | stable full-tile mosaic |
| denver | 11 | 7 | 0.05 | 0.97 | 1.78 | 0 | 512x512 | stable full-tile mosaic |

The full metro-level audit table is saved as `deliverables/data_audit/metro_imagery_audit.csv`.

## 3. Worst Cloud Cases

The current MODIS inventory contains **1 metro-year frames** with **core diffuse-cloud risk at or above 20%**. This center-weighted score is more aligned with city visibility than a whole-frame average.

| Metro | Year | Strict cloud % | Core diffuse % | Diffuse cloud % | Dark border % | Width | Height |
| --- | --- | --- | --- | --- | --- | --- | --- |
| las_vegas | 2015 | 1.75 | 28.45 | 23.62 | 0.00 | 1024 | 512 |
| las_vegas | 2013 | 6.73 | 19.97 | 23.66 | 0.00 | 1024 | 512 |
| las_vegas | 2019 | 0.29 | 18.71 | 25.81 | 2.30 | 1024 | 512 |
| las_vegas | 2018 | 1.92 | 16.86 | 25.59 | 0.00 | 1024 | 512 |
| las_vegas | 2021 | 3.32 | 16.64 | 26.87 | 0.00 | 1024 | 512 |
| las_vegas | 2014 | 0.71 | 14.92 | 20.79 | 0.00 | 1024 | 512 |
| phoenix | 2022 | 0.10 | 11.59 | 14.74 | 0.00 | 512 | 1024 |
| las_vegas | 2017 | 0.57 | 10.84 | 13.87 | 0.00 | 1024 | 512 |
| las_vegas | 2020 | 1.52 | 10.71 | 23.43 | 1.94 | 1024 | 512 |
| las_vegas | 2016 | 0.39 | 10.65 | 9.17 | 0.00 | 1024 | 512 |
| phoenix | 2018 | 1.06 | 10.24 | 10.30 | 0.01 | 512 | 1024 |
| las_vegas | 2023 | 1.10 | 10.01 | 13.19 | 0.00 | 1024 | 512 |

### Rationale

- This directly supports the teammate concern that some MODIS frames are not visually reliable for interpreting urban expansion.
- The center-weighted metric is intentionally stricter about city-core visibility than the older whole-frame rule, so it better matches the actual modeling use case.

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

## 5. How MODIS Dates Are Selected Now

- The original workflow used `08-01` as a practical starting heuristic. That rule is no longer treated as the source of truth for refreshed imagery.
- The final date-selection rule now prefers **complete frames** first, then minimizes **center-region cloud risk**, and only then uses whole-frame cloud metrics as tie-breakers.
- This ordering is intentional. A frame that preserves the city core and avoids black wedges is safer for downstream feature extraction than one that only looks cleaner in peripheral tiles.
- The refreshed acquisition manifest lives at `data/imagery/modis_acquisition_manifest.csv`, and the residual QA contact sheet is saved at `deliverables/data_audit/modis_residual_qa.png`.

## 6. Final Interpretation Notes

1. The refreshed MODIS acquisition manifest, tensors, and modeling tables should be treated as the current pre-final-model source of truth.
2. Residual high-cloud cases are now explicit and bounded; they remain usable for numeric summaries, but should be used cautiously as qualitative visual evidence.
3. Rectangular MODIS rasters are expected tile mosaics in this pipeline, so they should not be interpreted as accidental cropping by default.
4. The repo is now internally consistent around the restored 14-metro state even though the public `main` branch may lag that state.
