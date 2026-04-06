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
    


