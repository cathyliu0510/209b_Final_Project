#!/usr/bin/env python3
"""Refresh MODIS GeoTIFFs from the selected acquisition-date manifest.

This script is the deterministic bridge between the candidate-date audit and
the imagery actually used by downstream notebooks. It reads the selected dates
produced by `search_modis_candidate_dates.py`, rebuilds every requested metro-
year mosaic from NASA GIBS, overwrites the corresponding GeoTIFF, and logs the
full provenance in CSV form.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from search_modis_candidate_dates import (
    EXT,
    LAYER_NAME,
    MATRIX_SET,
    METROS,
    OUT_DIR as SEARCH_OUT_DIR,
    TILE_SIZE_PX,
    ZOOM,
    deg_to_tile_4326,
    fetch_tile,
    score_rgb,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTED_DATES_PATH = SEARCH_OUT_DIR / "modis_selected_dates.csv"
IMAGERY_ROOT = REPO_ROOT / "data" / "imagery"
MANIFEST_PATH = IMAGERY_ROOT / "modis_acquisition_manifest.csv"
REFRESH_LOG_PATH = REPO_ROOT / "deliverables" / "data_audit" / "modis_refresh_log.csv"


def tile_to_deg_4326(col: int, row: int, zoom: int) -> tuple[float, float]:
    n_cols = 2 ** (zoom + 1)
    n_rows = 2 ** zoom
    return 90.0 - row / n_rows * 180.0, col / n_cols * 360.0 - 180.0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_mosaic_for_date(metro: str, date_str: str) -> tuple[np.ndarray | None, dict[str, int]]:
    min_lon, min_lat, max_lon, max_lat = METROS[metro]
    col_min, row_max = deg_to_tile_4326(min_lat, min_lon, ZOOM)
    col_max, row_min = deg_to_tile_4326(max_lat, max_lon, ZOOM)
    n_tile_cols = col_max - col_min + 1
    n_tile_rows = row_max - row_min + 1

    canvas = np.zeros(
        (n_tile_rows * TILE_SIZE_PX, n_tile_cols * TILE_SIZE_PX, 3),
        dtype=np.uint8,
    )
    requested_tiles = n_tile_cols * n_tile_rows
    fetched_tiles = 0

    for r_idx, row in enumerate(range(row_min, row_max + 1)):
        for c_idx, col in enumerate(range(col_min, col_max + 1)):
            tile = fetch_tile(date_str, row, col)
            if tile is None:
                continue
            fetched_tiles += 1
            if tile.shape[:2] != (TILE_SIZE_PX, TILE_SIZE_PX):
                tile = np.array(Image.fromarray(tile).resize((TILE_SIZE_PX, TILE_SIZE_PX)))
            y0 = r_idx * TILE_SIZE_PX
            x0 = c_idx * TILE_SIZE_PX
            canvas[y0:y0 + TILE_SIZE_PX, x0:x0 + TILE_SIZE_PX] = tile

    if fetched_tiles == 0:
        return None, {
            "requested_tiles": requested_tiles,
            "fetched_tiles": 0,
            "missing_tiles": requested_tiles,
            "n_tile_cols": n_tile_cols,
            "n_tile_rows": n_tile_rows,
            "col_min": col_min,
            "col_max": col_max,
            "row_min": row_min,
            "row_max": row_max,
        }

    return canvas, {
        "requested_tiles": requested_tiles,
        "fetched_tiles": fetched_tiles,
        "missing_tiles": requested_tiles - fetched_tiles,
        "n_tile_cols": n_tile_cols,
        "n_tile_rows": n_tile_rows,
        "col_min": col_min,
        "col_max": col_max,
        "row_min": row_min,
        "row_max": row_max,
    }


def write_geotiff(arr: np.ndarray, out_path: Path, metro: str, tile_meta: dict[str, int]) -> None:
    col_min = tile_meta["col_min"]
    col_max = tile_meta["col_max"]
    row_min = tile_meta["row_min"]
    row_max = tile_meta["row_max"]

    nw_lat, nw_lon = tile_to_deg_4326(col_min, row_min, ZOOM)
    se_lat, se_lon = tile_to_deg_4326(col_max + 1, row_max + 1, ZOOM)
    transform = from_bounds(
        west=nw_lon,
        east=se_lon,
        north=nw_lat,
        south=se_lat,
        width=arr.shape[1],
        height=arr.shape[0],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=3,
        dtype=rasterio.uint8,
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="lzw",
    ) as dst:
        for band_idx in range(3):
            dst.write(arr[:, :, band_idx], band_idx + 1)


def filter_rows(rows: list[dict[str, str]], metros: set[str] | None, years: set[int] | None) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        metro = row["metro"]
        year = int(row["year"])
        if metros and metro not in metros:
            continue
        if years and year not in years:
            continue
        filtered.append(row)
    return filtered


def parse_optional_set(raw: str | None, cast=str) -> set | None:
    if not raw:
        return None
    return {cast(part.strip()) for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metros", default=None, help="Optional comma-separated metro subset")
    parser.add_argument("--years", default=None, help="Optional comma-separated year subset")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional pause between metro-years")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not SELECTED_DATES_PATH.exists():
        raise FileNotFoundError(
            f"Selected-date manifest not found: {SELECTED_DATES_PATH}. "
            "Run scripts/search_modis_candidate_dates.py first."
        )

    metro_filter = parse_optional_set(args.metros, cast=str)
    year_filter = parse_optional_set(args.years, cast=int)
    selected_rows = filter_rows(load_rows(SELECTED_DATES_PATH), metro_filter, year_filter)

    refresh_rows: list[dict[str, str]] = []
    for row in selected_rows:
        metro = row["metro"]
        year = int(row["year"])
        date_str = row["selected_date"]
        out_path = IMAGERY_ROOT / metro / "modis_rgb" / f"{year}.tif"
        print(f"[refresh] {metro} {year} <- {date_str}")

        arr, tile_meta = build_mosaic_for_date(metro, date_str)
        if arr is None:
            refresh_rows.append(
                {
                    "metro": metro,
                    "year": str(year),
                    "selected_date": date_str,
                    "selected_quality_flag": row["selected_quality_flag"],
                    "baseline_08_01_date": row["baseline_08_01_date"],
                    "baseline_08_01_core_diffuse_cloud_pct": row["baseline_08_01_core_diffuse_cloud_pct"],
                    "baseline_08_01_diffuse_cloud_pct": row["baseline_08_01_diffuse_cloud_pct"],
                    "selected_core_diffuse_cloud_pct": row["selected_core_diffuse_cloud_pct"],
                    "selected_diffuse_cloud_pct": row["selected_diffuse_cloud_pct"],
                    "selected_core_dark_or_empty_pct": row["selected_core_dark_or_empty_pct"],
                    "selected_dark_or_empty_pct": row["selected_dark_or_empty_pct"],
                    "status": "missing",
                    "requested_tiles": str(tile_meta["requested_tiles"]),
                    "fetched_tiles": str(tile_meta["fetched_tiles"]),
                    "missing_tiles": str(tile_meta["missing_tiles"]),
                    "width": "0",
                    "height": "0",
                    "actual_core_diffuse_cloud_pct": "nan",
                    "actual_diffuse_cloud_pct": "nan",
                    "actual_core_dark_or_empty_pct": "nan",
                    "actual_dark_or_empty_pct": "nan",
                    "actual_core_mean_brightness": "nan",
                    "actual_mean_brightness": "nan",
                    "output_path": str(out_path.relative_to(REPO_ROOT)),
                    "refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            continue

        scores = score_rgb(arr)
        if not args.dry_run:
            write_geotiff(arr, out_path, metro, tile_meta)

        refresh_rows.append(
            {
                "metro": metro,
                "year": str(year),
                "selected_date": date_str,
                "selected_quality_flag": row["selected_quality_flag"],
                "baseline_08_01_date": row["baseline_08_01_date"],
                "baseline_08_01_core_diffuse_cloud_pct": row["baseline_08_01_core_diffuse_cloud_pct"],
                "baseline_08_01_diffuse_cloud_pct": row["baseline_08_01_diffuse_cloud_pct"],
                "selected_core_diffuse_cloud_pct": row["selected_core_diffuse_cloud_pct"],
                "selected_diffuse_cloud_pct": row["selected_diffuse_cloud_pct"],
                "selected_core_dark_or_empty_pct": row["selected_core_dark_or_empty_pct"],
                "selected_dark_or_empty_pct": row["selected_dark_or_empty_pct"],
                "status": "written" if not args.dry_run else "dry_run",
                "requested_tiles": str(tile_meta["requested_tiles"]),
                "fetched_tiles": str(tile_meta["fetched_tiles"]),
                "missing_tiles": str(tile_meta["missing_tiles"]),
                "width": str(arr.shape[1]),
                "height": str(arr.shape[0]),
                "actual_core_diffuse_cloud_pct": f"{scores['core_diffuse_cloud_pct']:.2f}",
                "actual_diffuse_cloud_pct": f"{scores['diffuse_cloud_pct']:.2f}",
                "actual_core_dark_or_empty_pct": f"{scores['core_dark_or_empty_pct']:.2f}",
                "actual_dark_or_empty_pct": f"{scores['dark_or_empty_pct']:.2f}",
                "actual_core_mean_brightness": f"{scores['core_mean_brightness']:.4f}",
                "actual_mean_brightness": f"{scores['mean_brightness']:.4f}",
                "output_path": str(out_path.relative_to(REPO_ROOT)),
                "refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    fieldnames = [
        "metro",
        "year",
        "selected_date",
        "selected_quality_flag",
        "baseline_08_01_date",
        "baseline_08_01_core_diffuse_cloud_pct",
        "baseline_08_01_diffuse_cloud_pct",
        "selected_core_diffuse_cloud_pct",
        "selected_diffuse_cloud_pct",
        "selected_core_dark_or_empty_pct",
        "selected_dark_or_empty_pct",
        "status",
        "requested_tiles",
        "fetched_tiles",
        "missing_tiles",
        "width",
        "height",
        "actual_core_diffuse_cloud_pct",
        "actual_diffuse_cloud_pct",
        "actual_core_dark_or_empty_pct",
        "actual_dark_or_empty_pct",
        "actual_core_mean_brightness",
        "actual_mean_brightness",
        "output_path",
        "refreshed_at_utc",
    ]
    write_csv(MANIFEST_PATH, refresh_rows, fieldnames)
    write_csv(REFRESH_LOG_PATH, refresh_rows, fieldnames)
    print(f"Wrote refresh manifest to {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote refresh log to {REFRESH_LOG_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
