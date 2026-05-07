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
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject

BASE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_S_GLOBE_R2023A"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metro_config import BBOXES, GHSL_EPOCHS, GHSL_TILE_SPANS, METRO_CONFIGS


EPOCHS = GHSL_EPOCHS
IMAGERY_DIR = REPO_ROOT / "data" / "imagery"
GHSL_RAW_DIR = REPO_ROOT / "data" / "raw" / "ghsl" / "tiles"
GHSL_OUT_DIR = REPO_ROOT / "data" / "ghsl"


@dataclass(frozen=True)
class MetroConfig:
    bbox: Tuple[float, float, float, float]
    tiles: Tuple[Tuple[int, int], ...]


METROS: Dict[str, MetroConfig] = {
    metro: MetroConfig(BBOXES[metro], GHSL_TILE_SPANS[metro]) for metro in METRO_CONFIGS
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


def download_file(url: str, dest: Path, attempts: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return

    last_err: Exception | None = None
    tmp = dest.with_suffix(dest.suffix + ".partial")
    for attempt in range(1, attempts + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            with requests.get(url, stream=True, timeout=(30, 180)) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(dest)
            return
        except Exception as err:
            last_err = err
            print(
                f"  retry {attempt}/{attempts}: {dest.name} failed with {err}",
                flush=True,
            )
            if tmp.exists():
                tmp.unlink()
            if attempt < attempts:
                time.sleep(min(10 * attempt, 30))
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts") from last_err


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
    needed = sorted({tile for cfg in METROS.values() for tile in cfg.tiles})
    print("Downloading required GHSL tiles...", flush=True)
    for epoch in EPOCHS:
        for row, col in needed:
            dest = local_zip_path(epoch, row, col)
            url = zip_url(epoch, row, col)
            download_file(url, dest)
            print(f"  ready: {dest.relative_to(REPO_ROOT)}", flush=True)


def build_outputs() -> pd.DataFrame:
    GHSL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for metro, cfg in METROS.items():
        min_lon, min_lat, max_lon, max_lat = cfg.bbox
        center_lat = (min_lat + max_lat) / 2.0
        metro_dir = GHSL_OUT_DIR / metro
        metro_dir.mkdir(parents=True, exist_ok=True)
        ref = modis_reference(metro)

        for epoch in EPOCHS:
            out_path = metro_dir / f"{epoch}.tif"

            tile_paths = [vsizip_tif_path(epoch, row, col) for row, col in cfg.tiles]
            with ExitStack() as stack:
                srcs = [stack.enter_context(rasterio.open(tile_path)) for tile_path in tile_paths]
                mosaic, mosaic_transform = merge(
                    srcs,
                    bounds=(min_lon, min_lat, max_lon, max_lat),
                    res=(abs(srcs[0].transform.a), abs(srcs[0].transform.e)),
                    nodata=srcs[0].nodata,
                    resampling=Resampling.nearest,
                    method="first",
                )
                native_arr = mosaic[0]
                nodata = srcs[0].nodata
                src_arr = np.full(
                    (ref["height"], ref["width"]),
                    nodata if nodata is not None else 0,
                    dtype=native_arr.dtype,
                )
                reproject(
                    source=native_arr,
                    destination=src_arr,
                    src_transform=mosaic_transform,
                    src_crs=srcs[0].crs,
                    dst_transform=ref["transform"],
                    dst_crs=ref["crs"],
                    src_nodata=nodata,
                    dst_nodata=nodata if nodata is not None else 0,
                    resampling=Resampling.nearest,
                )
                native_px_area = pixel_area_km2(mosaic_transform, center_lat)

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
                f"({built_km2:.3f} km^2)",
                flush=True,
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
    print(
        f"saved summary: {(GHSL_OUT_DIR / 'built_up_summary.csv').relative_to(REPO_ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
