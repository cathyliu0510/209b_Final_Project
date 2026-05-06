#!/usr/bin/env python3
"""Search for lower-cloud MODIS candidate dates for selected metros/years.

This script is meant to replace the current fixed `08-01` heuristic with a
compact, reproducible candidate-date search. It reuses the same NASA GIBS WMTS
layer and tile logic as the project fetch notebook, but only downloads small
candidate mosaics for scoring.

Default use case:
    python3 scripts/search_modis_candidate_dates.py

By default, it audits the highest-risk metros from the current data audit:
Phoenix, Las Vegas, Denver, and Dallas.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "deliverables" / "data_audit" / "modis_date_search"
GIBS_BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best"
LAYER_NAME = "MODIS_Terra_CorrectedReflectance_TrueColor"
MATRIX_SET = "250m"
EXT = "jpg"
ZOOM = 6
TILE_SIZE_PX = 512

DEFAULT_METROS = ["phoenix", "las_vegas", "denver", "dallas"]
DEFAULT_YEARS = list(range(2013, 2024))
DEFAULT_DATES = ["07-01", "07-15", "08-01", "08-15", "09-01"]

METROS = {
    "atlanta": (-84.55, 33.65, -84.25, 33.90),
    "austin": (-97.94, 30.10, -97.50, 30.52),
    "charlotte": (-81.00, 35.10, -80.70, 35.35),
    "dallas": (-97.08, 32.62, -96.55, 33.02),
    "denver": (-105.10, 39.60, -104.75, 39.85),
    "houston": (-95.60, 29.65, -95.15, 29.95),
    "jacksonville": (-81.84, 30.10, -81.33, 30.54),
    "las_vegas": (-115.35, 36.05, -115.00, 36.30),
    "nashville": (-87.05, 35.96, -86.52, 36.35),
    "orlando": (-81.55, 28.40, -81.20, 28.65),
    "phoenix": (-112.32, 33.29, -111.65, 33.82),
    "raleigh": (-78.80, 35.70, -78.50, 35.95),
    "san_antonio": (-98.65, 29.35, -98.35, 29.55),
    "tampa": (-82.55, 27.90, -82.35, 28.10),
}


def deg_to_tile_4326(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n_cols = 2 ** (zoom + 1)
    n_rows = 2 ** zoom
    col = int((lon + 180.0) / 360.0 * n_cols)
    row = int((90.0 - lat) / 180.0 * n_rows)
    return max(0, min(col, n_cols - 1)), max(0, min(row, n_rows - 1))


def fetch_tile(date_str: str, row: int, col: int) -> np.ndarray | None:
    url = (
        f"{GIBS_BASE}/{LAYER_NAME}/default/{date_str}"
        f"/{MATRIX_SET}/{ZOOM}/{row}/{col}.{EXT}"
    )
    try:
        req = Request(url, headers={"User-Agent": "focused-cray-modis-audit/1.0"})
        with urlopen(req, timeout=25) as resp:
            content = resp.read()
        arr = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
        return arr
    except HTTPError as exc:
        if exc.code in (400, 404):
            return None
        return None
    except (URLError, TimeoutError, OSError):
        return None


def build_mosaic(metro: str, date_str: str) -> np.ndarray | None:
    min_lon, min_lat, max_lon, max_lat = METROS[metro]
    col_min, row_max = deg_to_tile_4326(min_lat, min_lon, ZOOM)
    col_max, row_min = deg_to_tile_4326(max_lat, max_lon, ZOOM)
    n_tile_cols = col_max - col_min + 1
    n_tile_rows = row_max - row_min + 1

    canvas = np.zeros(
        (n_tile_rows * TILE_SIZE_PX, n_tile_cols * TILE_SIZE_PX, 3),
        dtype=np.uint8,
    )
    found_any = False
    for r_idx, row in enumerate(range(row_min, row_max + 1)):
        for c_idx, col in enumerate(range(col_min, col_max + 1)):
            tile = fetch_tile(date_str, row, col)
            if tile is None:
                continue
            found_any = True
            if tile.shape[:2] != (TILE_SIZE_PX, TILE_SIZE_PX):
                tile = np.array(Image.fromarray(tile).resize((TILE_SIZE_PX, TILE_SIZE_PX)))
            y0 = r_idx * TILE_SIZE_PX
            x0 = c_idx * TILE_SIZE_PX
            canvas[y0:y0 + TILE_SIZE_PX, x0:x0 + TILE_SIZE_PX] = tile
    return canvas if found_any else None


def score_rgb(arr_uint8: np.ndarray) -> dict[str, float]:
    arr = arr_uint8.astype(np.float32) / 255.0
    brightness = arr.mean(axis=2)
    spread = arr.max(axis=2) - arr.min(axis=2)

    strict_cloud = np.all(arr > 0.95, axis=2)
    diffuse_cloud = (brightness > 0.65) & (spread < 0.12)
    dark_or_empty = np.max(arr, axis=2) < 0.02

    return {
        "strict_white_cloud_pct": float(strict_cloud.mean() * 100.0),
        "diffuse_cloud_pct": float(diffuse_cloud.mean() * 100.0),
        "dark_or_empty_pct": float(dark_or_empty.mean() * 100.0),
        "mean_brightness": float(arr.mean()),
    }


def save_preview(arr_uint8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_uint8).save(path)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(
    rows: list[dict[str, str]],
    metros: list[str],
    years: list[int],
    date_options: list[str],
) -> None:
    path = OUT_DIR / "modis_date_search_summary.md"
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["metro"], int(row["year"])), []).append(row)

    lines = [
        "# MODIS Candidate Date Search",
        "",
        "This artifact ranks compact candidate dates for high-risk metro-years using the same diffuse-cloud proxy introduced in the data audit.",
        "",
        f"- Metros searched: {', '.join(metros)}",
        f"- Years searched: {years[0]}-{years[-1]}",
        f"- Candidate month-days: {', '.join(date_options)}",
        "",
        "## Best candidate by metro-year",
        "",
        "| Metro | Year | Best date | Diffuse cloud % | Strict cloud % | Dark/empty % | Mean brightness |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for metro in metros:
        for year in years:
            candidates = grouped.get((metro, year), [])
            if not candidates:
                continue
            best = sorted(
                candidates,
                key=lambda r: (
                    float(r["diffuse_cloud_pct"]),
                    float(r["strict_white_cloud_pct"]),
                    float(r["dark_or_empty_pct"]),
                ),
            )[0]
            lines.append(
                f"| {metro} | {year} | {best['date']} | {best['diffuse_cloud_pct']} | "
                f"{best['strict_white_cloud_pct']} | {best['dark_or_empty_pct']} | {best['mean_brightness']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metros", default=",".join(DEFAULT_METROS))
    parser.add_argument("--years", default=",".join(map(str, DEFAULT_YEARS)))
    parser.add_argument("--dates", default=",".join(DEFAULT_DATES))
    parser.add_argument("--save-previews", action="store_true")
    args = parser.parse_args()

    metros = [m.strip() for m in args.metros.split(",") if m.strip()]
    years = [int(y) for y in args.years.split(",") if y.strip()]
    date_options = [d.strip() for d in args.dates.split(",") if d.strip()]
    out_rows: list[dict[str, str]] = []

    for metro in metros:
        if metro not in METROS:
            raise ValueError(f"Unknown metro: {metro}")
        for year in years:
            print(f"[search] {metro} {year}")
            for month_day in date_options:
                date_str = f"{year}-{month_day}"
                arr = build_mosaic(metro, date_str)
                if arr is None:
                    out_rows.append(
                        {
                            "metro": metro,
                            "year": str(year),
                            "date": date_str,
                            "status": "missing",
                            "strict_white_cloud_pct": "nan",
                            "diffuse_cloud_pct": "nan",
                            "dark_or_empty_pct": "nan",
                            "mean_brightness": "nan",
                        }
                    )
                    continue

                scores = score_rgb(arr)
                out_rows.append(
                    {
                        "metro": metro,
                        "year": str(year),
                        "date": date_str,
                        "status": "ok",
                        "strict_white_cloud_pct": f"{scores['strict_white_cloud_pct']:.2f}",
                        "diffuse_cloud_pct": f"{scores['diffuse_cloud_pct']:.2f}",
                        "dark_or_empty_pct": f"{scores['dark_or_empty_pct']:.2f}",
                        "mean_brightness": f"{scores['mean_brightness']:.4f}",
                    }
                )

                if args.save_previews:
                    preview_path = OUT_DIR / "previews" / metro / f"{date_str}.png"
                    save_preview(arr, preview_path)

    write_csv(
        OUT_DIR / "modis_date_candidates.csv",
        out_rows,
        [
            "metro",
            "year",
            "date",
            "status",
            "strict_white_cloud_pct",
            "diffuse_cloud_pct",
            "dark_or_empty_pct",
            "mean_brightness",
        ],
    )
    write_markdown_summary(out_rows, metros, years, date_options)
    print(f"Wrote date-search artifacts to {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
