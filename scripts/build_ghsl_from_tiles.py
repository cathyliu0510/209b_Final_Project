#!/usr/bin/env python3
"""Download the minimal GHSL tile subset and build metro-level masks.

This script reconstructs the GHSL artifacts used by the project notebooks:

- data/ghsl/{metro}/{epoch}.tif
- data/ghsl/built_up_summary.csv

It intentionally downloads only the 7 tile zips needed for the 14 metros in
this repo, rather than the full global GHSL archives.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import requests
import rasterio
from rasterio.enums import Resampling


EPOCHS = [2000, 2005, 2010, 2015, 2020]
BASE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_S_GLOBE_R2023A"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGERY_DIR = REPO_ROOT / "data" / "imagery"
GHSL_RAW_DIR = REPO_ROOT / "data" / "raw" / "ghsl" / "tiles"
GHSL_OUT_DIR = REPO_ROOT / "data" / "ghsl"


@dataclass(frozen=True)
class MetroConfig:
    bbox: Tuple[float, float, float, float]
    tile: Tuple[int, int]


METROS: Dict[str, MetroConfig] = {
    "atlanta": MetroConfig((-84.55, 33.65, -84.25, 33.90), (6, 10)),
    "austin": MetroConfig((-97.94, 30.10, -97.50, 30.52), (6, 9)),
    "charlotte": MetroConfig((-81.00, 35.10, -80.70, 35.35), (6, 10)),
    "dallas": MetroConfig((-97.08, 32.62, -96.55, 33.02), (6, 9)),
    "denver": MetroConfig((-105.10, 39.60, -104.75, 39.85), (5, 8)),
    "houston": MetroConfig((-95.60, 29.65, -95.15, 29.95), (6, 9)),
    "jacksonville": MetroConfig((-81.84, 30.10, -81.33, 30.54), (6, 10)),
    "las_vegas": MetroConfig((-115.35, 36.05, -115.00, 36.30), (6, 7)),
    "nashville": MetroConfig((-87.05, 35.96, -86.52, 36.35), (6, 10)),
    "orlando": MetroConfig((-81.55, 28.40, -81.20, 28.65), (7, 10)),
    "phoenix": MetroConfig((-112.32, 33.29, -111.65, 33.82), (6, 7)),
    "raleigh": MetroConfig((-78.80, 35.70, -78.50, 35.95), (6, 11)),
    "san_antonio": MetroConfig((-98.65, 29.35, -98.35, 29.55), (6, 9)),
    "tampa": MetroConfig((-82.55, 27.90, -82.35, 28.10), (7, 10)),
}


def zip_name(epoch: int, row: int, col: int) -> str:
    return f"GHS_BUILT_S_E{epoch}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"


def tif_name(epoch: int, row: int, col: int) -> str:
    return f"GHS_BUILT_S_E{epoch}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.tif"


def zip_url(epoch: int, row: int, col: int) -> str:
    return (
        f"{BASE_URL}/GHS_BUILT_S_E{epoch}_GLOBE_R2023A_4326_3ss/V1-0/tiles/"
        f"{zip_name(epoch, row, col)}"
    )


def local_zip_path(epoch: int, row: int, col: int) -> Path:
    return GHSL_RAW_DIR / str(epoch) / zip_name(epoch, row, col)


def vsizip_tif_path(epoch: int, row: int, col: int) -> str:
    zpath = local_zip_path(epoch, row, col)
    return f"/vsizip/{zpath}/{tif_name(epoch, row, col)}"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def pixel_area_km2(transform: rasterio.Affine, center_lat_deg: float) -> float:
    x_res = abs(transform.a)
    y_res = abs(transform.e)
    return (
        x_res * 111.320 * math.cos(math.radians(center_lat_deg))
        * y_res * 110.574
    )


def modis_reference(metro: str) -> dict:
    ref_path = IMAGERY_DIR / metro / "modis_rgb" / "2013.tif"
    with rasterio.open(ref_path) as src:
        return {
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
        }


def ensure_needed_tiles() -> None:
    needed = sorted({cfg.tile for cfg in METROS.values()})
    print("Downloading required GHSL tiles...")
    for epoch in EPOCHS:
        for row, col in needed:
            dest = local_zip_path(epoch, row, col)
            url = zip_url(epoch, row, col)
            download_file(url, dest)
            print(f"  ready: {dest.relative_to(REPO_ROOT)}")


def build_outputs() -> pd.DataFrame:
    GHSL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for metro, cfg in METROS.items():
        min_lon, min_lat, max_lon, max_lat = cfg.bbox
        center_lat = (min_lat + max_lat) / 2.0
        row, col = cfg.tile
        metro_dir = GHSL_OUT_DIR / metro
        metro_dir.mkdir(parents=True, exist_ok=True)
        ref = modis_reference(metro)

        for epoch in EPOCHS:
            tile_path = vsizip_tif_path(epoch, row, col)
            out_path = metro_dir / f"{epoch}.tif"

            with rasterio.open(tile_path) as src:
                bounds = src.bounds
                if not (
                    bounds.left <= min_lon <= bounds.right
                    and bounds.left <= max_lon <= bounds.right
                    and bounds.bottom <= min_lat <= bounds.top
                    and bounds.bottom <= max_lat <= bounds.top
                ):
                    raise ValueError(
                        f"{metro} bbox is not fully inside R{row}_C{col} for {epoch}: "
                        f"{bounds}"
                    )

                window = src.window(min_lon, min_lat, max_lon, max_lat)
                src_arr = src.read(
                    1,
                    window=window,
                    out_shape=(ref["height"], ref["width"]),
                    resampling=Resampling.nearest,
                )
                native_arr = src.read(1, window=window)
                nodata = src.nodata
                native_px_area = pixel_area_km2(src.transform, center_lat)

            if nodata is not None:
                binary = ((src_arr >= 1000) & (src_arr != nodata)).astype("uint8")
                metro_binary = (
                    (native_arr >= 1000) & (native_arr != nodata)
                ).astype("uint8")
            else:
                binary = (src_arr >= 1000).astype("uint8")
                metro_binary = (native_arr >= 1000).astype("uint8")

            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=ref["height"],
                width=ref["width"],
                count=1,
                dtype=rasterio.uint8,
                crs=ref["crs"],
                transform=ref["transform"],
                compress="lzw",
                nodata=255,
            ) as dst:
                dst.write(binary, 1)

            built_km2 = float(metro_binary.sum()) * native_px_area
            records.append(
                {"metro": metro, "epoch": epoch, "built_up_km2": built_km2}
            )
            print(
                f"  built: {out_path.relative_to(REPO_ROOT)} "
                f"({built_km2:.3f} km^2)"
            )

    summary = (
        pd.DataFrame(records)
        .sort_values(["metro", "epoch"])
        .reset_index(drop=True)
    )
    summary.to_csv(GHSL_OUT_DIR / "built_up_summary.csv", index=False)
    return summary


def compare_existing(existing: pd.DataFrame | None, summary: pd.DataFrame) -> None:
    if existing is None:
        return
    existing = existing.sort_values(["metro", "epoch"]).reset_index(drop=True)
    if list(existing.columns) != list(summary.columns) or len(existing) != len(summary):
        return
    max_abs_diff = (existing["built_up_km2"] - summary["built_up_km2"]).abs().max()
    print(f"max abs diff vs existing summary: {max_abs_diff:.12f}")


def main() -> None:
    existing_summary = None
    existing_path = GHSL_OUT_DIR / "built_up_summary.csv"
    if existing_path.exists():
        existing_summary = pd.read_csv(existing_path)
    ensure_needed_tiles()
    summary = build_outputs()
    compare_existing(existing_summary, summary)
    print(f"saved summary: {(GHSL_OUT_DIR / 'built_up_summary.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
