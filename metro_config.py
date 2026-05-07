"""Shared metro configuration for the urban-expansion project.

This module centralizes the geography metadata that was previously hard-coded
across multiple notebooks and scripts. Expanding from 14 to 30 metros should
only require editing this file, not every preprocessing notebook.
"""

from __future__ import annotations

from dataclasses import dataclass


GHSL_TILE_LEFT_ORIGIN = -180.00791620855503
GHSL_TILE_TOP_ORIGIN = 89.09958337887517
GHSL_TILE_SIZE_DEG = 10.0


@dataclass(frozen=True)
class MetroConfig:
    label: str
    cbsa: str
    bbox: tuple[float, float, float, float]


# Bboxes follow the same project convention as the original 14-city setup:
# a hand-tuned metro-core rectangle that captures the dominant built-up area
# used for MODIS / VIIRS / GHSL alignment.
METRO_CONFIGS: dict[str, MetroConfig] = {
    "atlanta": MetroConfig("Atlanta, GA", "12060", (-84.55, 33.65, -84.25, 33.90)),
    "austin": MetroConfig("Austin, TX", "12420", (-97.94, 30.10, -97.50, 30.52)),
    "boston": MetroConfig("Boston, MA", "14460", (-71.20, 42.22, -70.92, 42.48)),
    "charlotte": MetroConfig("Charlotte, NC", "16740", (-81.00, 35.10, -80.70, 35.35)),
    "chicago": MetroConfig("Chicago, IL", "16980", (-88.10, 41.70, -87.50, 42.10)),
    "cleveland": MetroConfig("Cleveland, OH", "17410", (-81.85, 41.35, -81.50, 41.62)),
    "columbus": MetroConfig("Columbus, OH", "18140", (-83.20, 39.84, -82.82, 40.14)),
    "dallas": MetroConfig("Dallas, TX", "19100", (-97.08, 32.62, -96.55, 33.02)),
    "denver": MetroConfig("Denver, CO", "19740", (-105.10, 39.60, -104.75, 39.85)),
    "detroit": MetroConfig("Detroit, MI", "19820", (-83.35, 42.20, -82.95, 42.55)),
    "houston": MetroConfig("Houston, TX", "26420", (-95.60, 29.65, -95.15, 29.95)),
    "indianapolis": MetroConfig("Indianapolis, IN", "26900", (-86.33, 39.64, -85.98, 39.88)),
    "jacksonville": MetroConfig("Jacksonville, FL", "27260", (-81.84, 30.10, -81.33, 30.54)),
    "kansas_city": MetroConfig("Kansas City, MO", "28140", (-94.75, 38.95, -94.30, 39.20)),
    "las_vegas": MetroConfig("Las Vegas, NV", "29820", (-115.35, 36.05, -115.00, 36.30)),
    "miami": MetroConfig("Miami, FL", "33100", (-80.42, 25.55, -80.05, 25.90)),
    "minneapolis": MetroConfig("Minneapolis, MN", "33460", (-93.45, 44.85, -92.95, 45.15)),
    "nashville": MetroConfig("Nashville, TN", "34980", (-87.05, 35.96, -86.52, 36.35)),
    "new_orleans": MetroConfig("New Orleans, LA", "35380", (-90.25, 29.82, -89.95, 30.10)),
    "oklahoma_city": MetroConfig("Oklahoma City, OK", "36420", (-97.65, 35.35, -97.25, 35.65)),
    "orlando": MetroConfig("Orlando, FL", "36740", (-81.55, 28.40, -81.20, 28.65)),
    "philadelphia": MetroConfig("Philadelphia, PA", "37980", (-75.35, 39.85, -74.95, 40.10)),
    "phoenix": MetroConfig("Phoenix, AZ", "38060", (-112.32, 33.29, -111.65, 33.82)),
    "pittsburgh": MetroConfig("Pittsburgh, PA", "38300", (-80.10, 40.30, -79.80, 40.55)),
    "raleigh": MetroConfig("Raleigh, NC", "39580", (-78.80, 35.70, -78.50, 35.95)),
    "sacramento": MetroConfig("Sacramento, CA", "40900", (-121.65, 38.45, -121.30, 38.68)),
    "salt_lake_city": MetroConfig("Salt Lake City, UT", "41620", (-112.08, 40.62, -111.75, 40.86)),
    "san_antonio": MetroConfig("San Antonio, TX", "41700", (-98.65, 29.35, -98.35, 29.55)),
    "st_louis": MetroConfig("St. Louis, MO", "41180", (-90.35, 38.52, -90.00, 38.75)),
    "tampa": MetroConfig("Tampa, FL", "45300", (-82.55, 27.90, -82.35, 28.10)),
}


METROS = list(METRO_CONFIGS.keys())
METRO_LABELS = {metro: cfg.label for metro, cfg in METRO_CONFIGS.items()}
BBOXES = {metro: cfg.bbox for metro, cfg in METRO_CONFIGS.items()}
CBSA_CODES = {metro: cfg.cbsa for metro, cfg in METRO_CONFIGS.items()}

ALL_YEARS = list(range(2013, 2024))
MODIS_YEARS = list(range(2013, 2024))
VIIRS_YEARS = list(range(2017, 2024))
GHSL_EPOCHS = [2000, 2005, 2010, 2015, 2020]

EXCLUDE_YEAR = 2020
TRAIN_YEARS = list(range(2013, 2019))
VAL_YEARS = [2019]
TEST_YEARS = list(range(2021, 2024))
PANEL_YEARS = [year for year in ALL_YEARS if year != EXCLUDE_YEAR]


def ghsl_tiles_for_bbox(
    bbox: tuple[float, float, float, float], eps: float = 1e-9
) -> tuple[tuple[int, int], ...]:
    """Return every GHSL 10-degree tile intersecting a bbox."""

    min_lon, min_lat, max_lon, max_lat = bbox
    top_row = int((GHSL_TILE_TOP_ORIGIN - max_lat) // GHSL_TILE_SIZE_DEG) + 1
    bottom_row = int(
        (GHSL_TILE_TOP_ORIGIN - (min_lat + eps)) // GHSL_TILE_SIZE_DEG
    ) + 1
    left_col = int(
        ((min_lon - GHSL_TILE_LEFT_ORIGIN) // GHSL_TILE_SIZE_DEG)
    ) + 1
    right_col = int(
        (((max_lon - eps) - GHSL_TILE_LEFT_ORIGIN) // GHSL_TILE_SIZE_DEG)
    ) + 1
    tiles = []
    for row in range(top_row, bottom_row + 1):
        for col in range(left_col, right_col + 1):
            tiles.append((row, col))
    return tuple(tiles)


GHSL_TILE_SPANS = {
    metro: ghsl_tiles_for_bbox(cfg.bbox) for metro, cfg in METRO_CONFIGS.items()
}
GHSL_TILES = {metro: tiles[0] for metro, tiles in GHSL_TILE_SPANS.items()}
