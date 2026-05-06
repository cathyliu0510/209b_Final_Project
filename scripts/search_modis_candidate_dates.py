#!/usr/bin/env python3
"""Search for lower-cloud MODIS candidate dates for selected metros/years.

This script is meant to replace the current fixed `08-01` heuristic with a
compact, reproducible candidate-date search. It reuses the same NASA GIBS WMTS
layer and tile logic as the project fetch notebook, but only downloads small
candidate mosaics for scoring.

Default use case:
    python3 scripts/search_modis_candidate_dates.py

By default, it audits all 14 metros across the full 2013-2023 span.
"""

from __future__ import annotations

import argparse
import csv
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
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

DEFAULT_METROS = [
    "atlanta", "austin", "charlotte", "dallas", "denver", "houston",
    "jacksonville", "las_vegas", "nashville", "orlando", "phoenix",
    "raleigh", "san_antonio", "tampa",
]
DEFAULT_YEARS = list(range(2013, 2024))
DEFAULT_DATES = [
    "04-01",
    "04-15",
    "05-01",
    "05-15",
    "06-01",
    "06-15",
    "07-01",
    "07-15",
    "08-01",
    "08-15",
    "09-01",
    "09-15",
    "10-01",
    "10-15",
]
CORE_FRACTION = 0.6

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


def build_mosaic(metro: str, date_str: str) -> tuple[np.ndarray | None, dict[str, int]]:
    min_lon, min_lat, max_lon, max_lat = METROS[metro]
    col_min, row_max = deg_to_tile_4326(min_lat, min_lon, ZOOM)
    col_max, row_min = deg_to_tile_4326(max_lat, max_lon, ZOOM)
    n_tile_cols = col_max - col_min + 1
    n_tile_rows = row_max - row_min + 1
    requested_tiles = n_tile_cols * n_tile_rows

    canvas = np.zeros(
        (n_tile_rows * TILE_SIZE_PX, n_tile_cols * TILE_SIZE_PX, 3),
        dtype=np.uint8,
    )
    found_any = False
    fetched_tiles = 0
    for r_idx, row in enumerate(range(row_min, row_max + 1)):
        for c_idx, col in enumerate(range(col_min, col_max + 1)):
            tile = fetch_tile(date_str, row, col)
            if tile is None:
                continue
            found_any = True
            fetched_tiles += 1
            if tile.shape[:2] != (TILE_SIZE_PX, TILE_SIZE_PX):
                tile = np.array(Image.fromarray(tile).resize((TILE_SIZE_PX, TILE_SIZE_PX)))
            y0 = r_idx * TILE_SIZE_PX
            x0 = c_idx * TILE_SIZE_PX
            canvas[y0:y0 + TILE_SIZE_PX, x0:x0 + TILE_SIZE_PX] = tile
    meta = {
        "requested_tiles": requested_tiles,
        "fetched_tiles": fetched_tiles,
        "missing_tiles": requested_tiles - fetched_tiles,
    }
    return (canvas if found_any else None), meta


def center_crop(arr: np.ndarray, fraction: float = CORE_FRACTION) -> np.ndarray:
    height, width = arr.shape[:2]
    crop_height = max(1, int(round(height * fraction)))
    crop_width = max(1, int(round(width * fraction)))
    y0 = max(0, (height - crop_height) // 2)
    x0 = max(0, (width - crop_width) // 2)
    return arr[y0:y0 + crop_height, x0:x0 + crop_width]


def score_region(arr: np.ndarray) -> dict[str, float]:
    brightness = arr.mean(axis=2)
    spread = arr.max(axis=2) - arr.min(axis=2)

    strict_cloud = np.all(arr > 0.95, axis=2)
    diffuse_cloud = (brightness > 0.72) & (spread < 0.10)
    dark_or_empty = np.max(arr, axis=2) < 0.02

    return {
        "strict_white_cloud_pct": float(strict_cloud.mean() * 100.0),
        "diffuse_cloud_pct": float(diffuse_cloud.mean() * 100.0),
        "dark_or_empty_pct": float(dark_or_empty.mean() * 100.0),
        "mean_brightness": float(arr.mean()),
    }


def score_rgb(arr_uint8: np.ndarray) -> dict[str, float]:
    arr = arr_uint8.astype(np.float32) / 255.0
    global_scores = score_region(arr)
    core_scores = score_region(center_crop(arr))
    return {
        **global_scores,
        "core_strict_white_cloud_pct": core_scores["strict_white_cloud_pct"],
        "core_diffuse_cloud_pct": core_scores["diffuse_cloud_pct"],
        "core_dark_or_empty_pct": core_scores["dark_or_empty_pct"],
        "core_mean_brightness": core_scores["mean_brightness"],
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


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def candidate_fieldnames() -> list[str]:
    return [
        "metro",
        "year",
        "date",
        "status",
        "requested_tiles",
        "fetched_tiles",
        "missing_tiles",
        "strict_white_cloud_pct",
        "diffuse_cloud_pct",
        "dark_or_empty_pct",
        "mean_brightness",
        "core_strict_white_cloud_pct",
        "core_diffuse_cloud_pct",
        "core_dark_or_empty_pct",
        "core_mean_brightness",
    ]


def candidate_row(metro: str, year: int, date_str: str) -> tuple[dict[str, str], np.ndarray | None]:
    arr, tile_meta = build_mosaic(metro, date_str)
    if arr is None:
        return (
            {
                "metro": metro,
                "year": str(year),
                "date": date_str,
                "status": "missing",
                "requested_tiles": str(tile_meta["requested_tiles"]),
                "fetched_tiles": str(tile_meta["fetched_tiles"]),
                "missing_tiles": str(tile_meta["missing_tiles"]),
                "strict_white_cloud_pct": "nan",
                "diffuse_cloud_pct": "nan",
                "dark_or_empty_pct": "nan",
                "mean_brightness": "nan",
                "core_strict_white_cloud_pct": "nan",
                "core_diffuse_cloud_pct": "nan",
                "core_dark_or_empty_pct": "nan",
                "core_mean_brightness": "nan",
            },
            None,
        )

    scores = score_rgb(arr)
    return (
        {
            "metro": metro,
            "year": str(year),
            "date": date_str,
            "status": "ok",
            "requested_tiles": str(tile_meta["requested_tiles"]),
            "fetched_tiles": str(tile_meta["fetched_tiles"]),
            "missing_tiles": str(tile_meta["missing_tiles"]),
            "strict_white_cloud_pct": f"{scores['strict_white_cloud_pct']:.2f}",
            "diffuse_cloud_pct": f"{scores['diffuse_cloud_pct']:.2f}",
            "dark_or_empty_pct": f"{scores['dark_or_empty_pct']:.2f}",
            "mean_brightness": f"{scores['mean_brightness']:.4f}",
            "core_strict_white_cloud_pct": f"{scores['core_strict_white_cloud_pct']:.2f}",
            "core_diffuse_cloud_pct": f"{scores['core_diffuse_cloud_pct']:.2f}",
            "core_dark_or_empty_pct": f"{scores['core_dark_or_empty_pct']:.2f}",
            "core_mean_brightness": f"{scores['core_mean_brightness']:.4f}",
        },
        arr,
    )


def sort_candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (row["metro"], int(row["year"]), row["date"]))


def selection_key(row: dict[str, str]) -> tuple[float, ...]:
    diffuse = float(row["diffuse_cloud_pct"])
    strict = float(row["strict_white_cloud_pct"])
    dark = float(row["dark_or_empty_pct"])
    core_diffuse = float(row["core_diffuse_cloud_pct"])
    core_strict = float(row["core_strict_white_cloud_pct"])
    core_dark = float(row["core_dark_or_empty_pct"])
    missing_tiles = int(row["missing_tiles"])

    # Final selection rule:
    # 1. Prefer complete mosaics with a clean center region. This prioritizes
    #    city-core visibility over peripheral tiles while still blocking large
    #    dark gaps.
    # 2. Among complete frames, minimize center-region diffuse cloud before
    #    using the whole-frame cloud score as a tiebreaker.
    # 3. Fall back to higher-dark or incomplete candidates only when every
    #    complete option still looks compromised.
    if missing_tiles == 0 and core_dark <= 2.0 and dark <= 5.0:
        return (0.0, core_diffuse, diffuse, core_strict, strict, core_dark, dark)
    if missing_tiles == 0 and dark <= 5.0:
        return (1.0, core_dark, core_diffuse, diffuse, core_strict, strict, dark)
    if missing_tiles == 0:
        return (2.0, dark, core_dark, core_diffuse, diffuse, core_strict, strict)
    return (3.0, float(missing_tiles), dark, core_dark, core_diffuse, diffuse, core_strict, strict)


def quality_flag(row: dict[str, str]) -> str:
    missing_tiles = int(row["missing_tiles"])
    diffuse = float(row["diffuse_cloud_pct"])
    dark = float(row["dark_or_empty_pct"])
    core_diffuse = float(row["core_diffuse_cloud_pct"])
    core_dark = float(row["core_dark_or_empty_pct"])
    if missing_tiles == 0 and dark <= 5.0 and core_dark <= 2.0 and core_diffuse <= 10.0 and diffuse <= 18.0:
        return "clean"
    if missing_tiles == 0 and dark <= 5.0 and core_dark <= 2.0:
        return "cloudy_but_complete"
    if missing_tiles == 0 and dark <= 5.0:
        return "core_gap_or_haze_fallback"
    if missing_tiles == 0:
        return "dark_gap_fallback"
    return "incomplete_mosaic_fallback"


def write_markdown_summary(
    rows: list[dict[str, str]],
    metros: list[str],
    years: list[int],
    date_options: list[str],
) -> None:
    path = OUT_DIR / "modis_date_search_summary.md"
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        grouped.setdefault((row["metro"], int(row["year"])), []).append(row)

    total_cases = 0
    improved_cases = 0
    biggest_improvements: list[tuple[str, int, float, str, float, float]] = []
    for metro in metros:
        for year in years:
            candidates = grouped.get((metro, year), [])
            if not candidates:
                continue
            total_cases += 1
            best = min(candidates, key=selection_key)
            aug = next((r for r in candidates if r["date"].endswith("08-01")), None)
            if aug and aug["date"] != best["date"]:
                improvement = float(aug["diffuse_cloud_pct"]) - float(best["diffuse_cloud_pct"])
                improved_cases += 1
                biggest_improvements.append(
                    (
                        metro,
                        year,
                        float(aug["diffuse_cloud_pct"]),
                        best["date"],
                        float(best["diffuse_cloud_pct"]),
                        improvement,
                    )
                )

    lines = [
        "# MODIS Candidate Date Search",
        "",
        "This artifact ranks candidate dates for every metro-year using a center-weighted cloud score plus a full-coverage guardrail.",
        "",
        f"- Metros searched: {', '.join(metros)}",
        f"- Years searched: {years[0]}-{years[-1]}",
        f"- Candidate month-days: {', '.join(date_options)}",
        f"- Metro-year cases with at least one valid candidate: {total_cases}",
        f"- Cases where `08-01` was not the best candidate: {improved_cases}",
        (
            "- Cases where the selected date still has one or more missing tiles "
            f"(meaning every candidate looked incomplete): "
            f"{sum(1 for metro in metros for year in years if grouped.get((metro, year)) and quality_flag(min(grouped[(metro, year)], key=selection_key)) == 'incomplete_mosaic_fallback')}"
        ),
        "",
        "## Largest improvements over `08-01`",
        "",
        "| Metro | Year | `08-01` diffuse cloud % | Best date | Best diffuse cloud % | Core diffuse cloud % | Improvement |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for metro, year, aug_diffuse, best_date, best_diffuse, improvement in sorted(biggest_improvements, key=lambda x: x[-1], reverse=True)[:25]:
        best_row = min(grouped[(metro, year)], key=selection_key)
        lines.append(
            f"| {metro} | {year} | {aug_diffuse:.2f} | {best_date} | {best_diffuse:.2f} | {best_row['core_diffuse_cloud_pct']} | {improvement:.2f} |"
        )
    lines.extend([
        "",
        "## Best candidate by metro-year",
        "",
        "| Metro | Year | Best date | Quality flag | Missing tiles | Core diffuse % | Diffuse cloud % | Core dark % | Dark/empty % | Mean brightness |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for metro in metros:
        for year in years:
            candidates = grouped.get((metro, year), [])
            if not candidates:
                continue
            best = min(candidates, key=selection_key)
            lines.append(
                f"| {metro} | {year} | {best['date']} | {quality_flag(best)} | {best['missing_tiles']} | {best['core_diffuse_cloud_pct']} | {best['diffuse_cloud_pct']} | "
                f"{best['core_dark_or_empty_pct']} | {best['dark_or_empty_pct']} | {best['mean_brightness']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def write_selected_dates(rows: list[dict[str, str]], metros: list[str], years: list[int]) -> None:
    selected_rows: list[dict[str, str]] = []
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        grouped.setdefault((row["metro"], int(row["year"])), []).append(row)

    for metro in metros:
        for year in years:
            candidates = grouped.get((metro, year), [])
            if not candidates:
                continue
            best = min(candidates, key=selection_key)
            baseline = next((r for r in candidates if r["date"].endswith("08-01")), best)
            selected_rows.append(
                {
                    "metro": metro,
                    "year": str(year),
                    "selected_date": best["date"],
                    "selected_quality_flag": quality_flag(best),
                    "selected_missing_tiles": best["missing_tiles"],
                    "selected_core_diffuse_cloud_pct": best["core_diffuse_cloud_pct"],
                    "selected_diffuse_cloud_pct": best["diffuse_cloud_pct"],
                    "selected_core_strict_white_cloud_pct": best["core_strict_white_cloud_pct"],
                    "selected_strict_white_cloud_pct": best["strict_white_cloud_pct"],
                    "selected_core_dark_or_empty_pct": best["core_dark_or_empty_pct"],
                    "selected_dark_or_empty_pct": best["dark_or_empty_pct"],
                    "baseline_08_01_date": baseline["date"],
                    "baseline_08_01_missing_tiles": baseline["missing_tiles"],
                    "baseline_08_01_core_diffuse_cloud_pct": baseline["core_diffuse_cloud_pct"],
                    "baseline_08_01_diffuse_cloud_pct": baseline["diffuse_cloud_pct"],
                    "baseline_08_01_dark_or_empty_pct": baseline["dark_or_empty_pct"],
                    "improvement_vs_08_01": f"{float(baseline['diffuse_cloud_pct']) - float(best['diffuse_cloud_pct']):.2f}",
                }
            )

    write_csv(
        OUT_DIR / "modis_selected_dates.csv",
        selected_rows,
        [
            "metro",
            "year",
            "selected_date",
            "selected_quality_flag",
            "selected_missing_tiles",
            "selected_core_diffuse_cloud_pct",
            "selected_diffuse_cloud_pct",
            "selected_core_strict_white_cloud_pct",
            "selected_strict_white_cloud_pct",
            "selected_core_dark_or_empty_pct",
            "selected_dark_or_empty_pct",
            "baseline_08_01_date",
            "baseline_08_01_missing_tiles",
            "baseline_08_01_core_diffuse_cloud_pct",
            "baseline_08_01_diffuse_cloud_pct",
            "baseline_08_01_dark_or_empty_pct",
            "improvement_vs_08_01",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metros", default=",".join(DEFAULT_METROS))
    parser.add_argument("--years", default=",".join(map(str, DEFAULT_YEARS)))
    parser.add_argument("--dates", default=",".join(DEFAULT_DATES))
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    metros = [m.strip() for m in args.metros.split(",") if m.strip()]
    years = [int(y) for y in args.years.split(",") if y.strip()]
    date_options = [d.strip() for d in args.dates.split(",") if d.strip()]
    candidate_csv = OUT_DIR / "modis_date_candidates.csv"
    out_rows: list[dict[str, str]] = load_existing_rows(candidate_csv) if args.resume else []
    completed = {(row["metro"], row["year"], row["date"]) for row in out_rows}

    for metro in metros:
        if metro not in METROS:
            raise ValueError(f"Unknown metro: {metro}")
        for year in years:
            print(f"[search] {metro} {year}")
            pending_dates = [
                f"{year}-{month_day}"
                for month_day in date_options
                if (metro, str(year), f"{year}-{month_day}") not in completed
            ]
            if not pending_dates:
                continue

            max_workers = max(1, min(args.max_workers, len(pending_dates)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(candidate_row, metro, year, date_str): date_str
                    for date_str in pending_dates
                }
                for future in as_completed(future_map):
                    date_str = future_map[future]
                    row, arr = future.result()
                    out_rows.append(row)
                    out_rows = sort_candidate_rows(out_rows)
                    write_csv(candidate_csv, out_rows, candidate_fieldnames())
                    if args.save_previews and arr is not None:
                        preview_path = OUT_DIR / "previews" / metro / f"{date_str}.png"
                        save_preview(arr, preview_path)

    write_csv(
        candidate_csv,
        sort_candidate_rows(out_rows),
        candidate_fieldnames(),
    )
    out_rows = sort_candidate_rows(out_rows)
    all_metros = sorted({row["metro"] for row in out_rows})
    all_years = sorted({int(row["year"]) for row in out_rows})
    write_selected_dates(out_rows, all_metros, all_years)
    write_markdown_summary(out_rows, all_metros, all_years, sorted({row["date"][5:] for row in out_rows}))
    print(f"Wrote date-search artifacts to {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
