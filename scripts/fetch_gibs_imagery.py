#!/usr/bin/env python3
"""Fetch MODIS RGB and VIIRS night-light mosaics for configured metros.

MODIS dates come from the audited acquisition manifest when available; VIIRS
uses the same fixed-date rule as the existing notebook workflow, including the
project's known 2022/2023 overrides.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metro_config import BBOXES, METROS, MODIS_YEARS, VIIRS_YEARS


GIBS_BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best"
TILE_SIZE_PX = 512
OUTPUT_DIR = REPO_ROOT / "data" / "imagery"
MODIS_MANIFEST_PATH = OUTPUT_DIR / "modis_acquisition_manifest.csv"
DEFAULT_MONTH_DAY = "08-01"
VIIRS_DATE_OVERRIDES = {2022: "09-01", 2023: "07-01"}

LAYERS = {
    "modis_rgb": {
        "name": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "matrix": "250m",
        "ext": "jpg",
        "bands": 3,
        "zoom": 6,
    },
    "viirs_night": {
        "name": "VIIRS_SNPP_DayNightBand_ENCC",
        "matrix": "500m",
        "ext": "png",
        "bands": 1,
        "zoom": 5,
    },
}


def deg_to_tile_4326(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n_cols = 2 ** (zoom + 1)
    n_rows = 2 ** zoom
    col = int((lon + 180.0) / 360.0 * n_cols)
    row = int((90.0 - lat) / 180.0 * n_rows)
    return max(0, min(col, n_cols - 1)), max(0, min(row, n_rows - 1))


def tile_to_deg_4326(col: int, row: int, zoom: int) -> tuple[float, float]:
    n_cols = 2 ** (zoom + 1)
    n_rows = 2 ** zoom
    return 90.0 - row / n_rows * 180.0, col / n_cols * 360.0 - 180.0


def fetch_tile(layer_name: str, matrix_set: str, ext: str, date_str: str, zoom: int, row: int, col: int, n_bands: int) -> np.ndarray | None:
    url = (
        f"{GIBS_BASE}/{layer_name}/default/{date_str}/"
        f"{matrix_set}/{zoom}/{row}/{col}.{ext}"
    )
    try:
        req = Request(url, headers={"User-Agent": "focused-cray-gibs-fetch/1.0"})
        with urlopen(req, timeout=30) as resp:
            content = resp.read()
        mode = "RGB" if n_bands == 3 else "L"
        arr = np.array(Image.open(io.BytesIO(content)).convert(mode))
        if n_bands == 1:
            arr = arr[:, :, np.newaxis]
        return arr
    except HTTPError as exc:
        if exc.code in (400, 404):
            return None
        raise
    except (URLError, TimeoutError, OSError):
        return None


def load_modis_dates() -> dict[tuple[str, int], str]:
    if not MODIS_MANIFEST_PATH.exists():
        return {}
    with MODIS_MANIFEST_PATH.open() as fh:
        rows = list(csv.DictReader(fh))
    return {
        (row["metro"], int(row["year"])): row["selected_date"]
        for row in rows
        if row.get("selected_date")
    }


def layer_date(layer_key: str, metro: str, year: int, modis_dates: dict[tuple[str, int], str]) -> str:
    if layer_key == "modis_rgb":
        return modis_dates.get((metro, year), f"{year}-{DEFAULT_MONTH_DAY}")
    if year in VIIRS_DATE_OVERRIDES:
        return f"{year}-{VIIRS_DATE_OVERRIDES[year]}"
    return f"{year}-{DEFAULT_MONTH_DAY}"


def write_geotiff(arr: np.ndarray, out_path: Path, zoom: int, col_min: int, col_max: int, row_min: int, row_max: int) -> None:
    nw_lat, nw_lon = tile_to_deg_4326(col_min, row_min, zoom)
    se_lat, se_lon = tile_to_deg_4326(col_max + 1, row_max + 1, zoom)
    transform = from_bounds(
        west=nw_lon,
        east=se_lon,
        north=nw_lat,
        south=se_lat,
        width=arr.shape[1],
        height=arr.shape[0],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 1 if arr.ndim == 2 else arr.shape[2]
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=count,
        dtype=rasterio.uint8,
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="lzw",
    ) as dst:
        if count == 1:
            dst.write(arr if arr.ndim == 2 else arr[:, :, 0], 1)
        else:
            for band_idx in range(count):
                dst.write(arr[:, :, band_idx], band_idx + 1)


def mosaic_and_save(metro: str, layer_key: str, year: int, overwrite: bool, modis_dates: dict[tuple[str, int], str]) -> Path | None:
    cfg = LAYERS[layer_key]
    min_lon, min_lat, max_lon, max_lat = BBOXES[metro]
    out_path = OUTPUT_DIR / metro / layer_key / f"{year}.tif"
    if out_path.exists() and not overwrite:
        return None

    col_min, row_max = deg_to_tile_4326(min_lat, min_lon, cfg["zoom"])
    col_max, row_min = deg_to_tile_4326(max_lat, max_lon, cfg["zoom"])
    n_tile_cols = col_max - col_min + 1
    n_tile_rows = row_max - row_min + 1

    canvas = np.zeros(
        (n_tile_rows * TILE_SIZE_PX, n_tile_cols * TILE_SIZE_PX, cfg["bands"]),
        dtype=np.uint8,
    )
    date_str = layer_date(layer_key, metro, year, modis_dates)
    found_any = False
    for r_idx, row in enumerate(range(row_min, row_max + 1)):
        for c_idx, col in enumerate(range(col_min, col_max + 1)):
            tile = fetch_tile(
                cfg["name"], cfg["matrix"], cfg["ext"], date_str, cfg["zoom"], row, col, cfg["bands"]
            )
            if tile is None:
                continue
            found_any = True
            if tile.shape[:2] != (TILE_SIZE_PX, TILE_SIZE_PX):
                mode = "RGB" if cfg["bands"] == 3 else "L"
                resized = Image.fromarray(tile.squeeze() if cfg["bands"] == 1 else tile, mode=mode).resize((TILE_SIZE_PX, TILE_SIZE_PX))
                tile = np.array(resized)
                if cfg["bands"] == 1:
                    tile = tile[:, :, np.newaxis]
            y0 = r_idx * TILE_SIZE_PX
            x0 = c_idx * TILE_SIZE_PX
            canvas[y0:y0 + TILE_SIZE_PX, x0:x0 + TILE_SIZE_PX] = tile

    if not found_any:
        raise RuntimeError(f"No tiles found for {metro} {layer_key} {year} ({date_str})")

    if cfg["bands"] == 1:
        canvas = canvas[:, :, 0]
    write_geotiff(canvas, out_path, cfg["zoom"], col_min, col_max, row_min, row_max)
    return out_path


def parse_csv_arg(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metros", default=",".join(METROS))
    parser.add_argument("--layers", default="modis_rgb,viirs_night")
    parser.add_argument("--years", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metros = parse_csv_arg(args.metros)
    layers = parse_csv_arg(args.layers)
    years_filter = {int(y) for y in parse_csv_arg(args.years)} if args.years else None
    modis_dates = load_modis_dates()

    for metro in metros:
        if metro not in BBOXES:
            raise ValueError(f"Unknown metro: {metro}")
        for layer_key in layers:
            if layer_key not in LAYERS:
                raise ValueError(f"Unknown layer: {layer_key}")
            years = MODIS_YEARS if layer_key == "modis_rgb" else VIIRS_YEARS
            if years_filter is not None:
                years = [year for year in years if year in years_filter]
            for year in years:
                result = mosaic_and_save(metro, layer_key, year, args.overwrite, modis_dates)
                if result is not None:
                    print(f"saved {result.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
