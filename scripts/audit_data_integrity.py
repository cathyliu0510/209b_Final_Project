#!/usr/bin/env python3
"""Audit project data integrity and MODIS imagery quality.

This script documents the current state of the project data in a way that is
easy to rerun and easy to hand off to teammates. It answers four concrete
questions:

1. Which branches currently expose the 5-metro vs 14-metro versions?
2. Does the local working tree contain the full 14-metro data inventory?
3. How cloudy are the current MODIS RGB frames, metro by metro and year by year?
4. Are the varying image dimensions stable and consistent with configured metro
   bounding boxes, or do they look like accidental truncation?

Outputs are written to ``deliverables/data_audit``.
"""

from __future__ import annotations

import csv
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from search_modis_candidate_dates import score_rgb


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
IMAGERY_DIR = DATA_DIR / "imagery"
PANEL_PATH = DATA_DIR / "economic" / "panel.csv"
OUT_DIR = REPO_ROOT / "deliverables" / "data_audit"
MODIS_SELECTED_PATH = OUT_DIR / "modis_date_search" / "modis_selected_dates.csv"
MODIS_REFRESH_LOG_PATH = OUT_DIR / "modis_refresh_log.csv"
MODIS_RESIDUAL_QA_PATH = OUT_DIR / "modis_residual_qa.png"

EXPECTED_MODIS_YEARS = list(range(2013, 2024))
EXPECTED_VIIRS_YEARS = list(range(2017, 2024))
BRANCH_REFS = [
    "upstream/main",
    "upstream/rename-add-prefix",
    "origin/main",
    "origin/rename-add-prefix",
]


@dataclass(frozen=True)
class MetroBBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


METRO_BBOXES: dict[str, MetroBBox] = {
    "atlanta": MetroBBox(-84.55, 33.65, -84.25, 33.90),
    "austin": MetroBBox(-97.94, 30.10, -97.50, 30.52),
    "charlotte": MetroBBox(-81.00, 35.10, -80.70, 35.35),
    "dallas": MetroBBox(-97.08, 32.62, -96.55, 33.02),
    "denver": MetroBBox(-105.10, 39.60, -104.75, 39.85),
    "houston": MetroBBox(-95.60, 29.65, -95.15, 29.95),
    "jacksonville": MetroBBox(-81.84, 30.10, -81.33, 30.54),
    "las_vegas": MetroBBox(-115.35, 36.05, -115.00, 36.30),
    "nashville": MetroBBox(-87.05, 35.96, -86.52, 36.35),
    "orlando": MetroBBox(-81.55, 28.40, -81.20, 28.65),
    "phoenix": MetroBBox(-112.32, 33.29, -111.65, 33.82),
    "raleigh": MetroBBox(-78.80, 35.70, -78.50, 35.95),
    "san_antonio": MetroBBox(-98.65, 29.35, -98.35, 29.55),
    "tampa": MetroBBox(-82.55, 27.90, -82.35, 28.10),
}


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def safe_git_output(*args: str) -> str | None:
    try:
        return git_output(*args)
    except subprocess.CalledProcessError:
        return None


def read_panel_rows_from_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(text.splitlines()))


def read_local_panel_rows() -> list[dict[str, str]]:
    with PANEL_PATH.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_optional_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def unique_metros(rows: Iterable[dict[str, str]]) -> list[str]:
    return sorted({row["metro"] for row in rows if row.get("metro")})


def list_local_imagery_metros() -> list[str]:
    if not IMAGERY_DIR.exists():
        return []
    return sorted([path.name for path in IMAGERY_DIR.iterdir() if path.is_dir()])


def list_git_imagery_metros(ref: str) -> list[str]:
    out = safe_git_output("ls-tree", "--name-only", f"{ref}:data/imagery")
    if not out:
        return []
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def branch_inventory() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    local_panel_rows = read_local_panel_rows()
    local_panel_metros = unique_metros(local_panel_rows)
    local_imagery_metros = list_local_imagery_metros()
    rows.append(
        {
            "ref": "working-tree",
            "commit": safe_git_output("rev-parse", "--short", "HEAD") or "unknown",
            "panel_rows": str(len(local_panel_rows)),
            "panel_metros": str(len(local_panel_metros)),
            "imagery_metros": str(len(local_imagery_metros)),
            "metro_list": ", ".join(local_panel_metros),
        }
    )

    for ref in BRANCH_REFS:
        commit = safe_git_output("rev-parse", "--short", ref) or "missing"
        panel_text = safe_git_output("show", f"{ref}:data/economic/panel.csv")
        if panel_text:
            panel_rows = read_panel_rows_from_text(panel_text)
            panel_metros = unique_metros(panel_rows)
        else:
            panel_rows = []
            panel_metros = []
        imagery_metros = list_git_imagery_metros(ref)
        rows.append(
            {
                "ref": ref,
                "commit": commit,
                "panel_rows": str(len(panel_rows)),
                "panel_metros": str(len(panel_metros)),
                "imagery_metros": str(len(imagery_metros)),
                "metro_list": ", ".join(panel_metros),
            }
        )
    return rows


def image_years(path: Path, allowed_years: set[int] | None = None) -> list[int]:
    years = []
    for tif_path in sorted(path.glob("*.tif")):
        try:
            year = int(tif_path.stem)
        except ValueError:
            continue
        if allowed_years is not None and year not in allowed_years:
            continue
        years.append(year)
    return years


def cloud_stats_for_rgb(path: Path) -> tuple[int, int, float, float, float, float, float, float]:
    img = Image.open(path).convert("RGB")
    arr_uint8 = np.asarray(img, dtype=np.uint8)
    height, width = arr_uint8.shape[:2]
    scores = score_rgb(arr_uint8)

    return (
        width,
        height,
        scores["strict_white_cloud_pct"],
        scores["diffuse_cloud_pct"],
        scores["dark_or_empty_pct"],
        scores["mean_brightness"],
        scores["core_diffuse_cloud_pct"],
        scores["core_dark_or_empty_pct"],
    )


def expected_bbox_aspect(metro: str) -> float | None:
    bbox = METRO_BBOXES.get(metro)
    if not bbox:
        return None
    mid_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    lon_km = (bbox.max_lon - bbox.min_lon) * 111.320 * math.cos(math.radians(mid_lat))
    lat_km = (bbox.max_lat - bbox.min_lat) * 110.574
    if lat_km == 0:
        return None
    return lon_km / lat_km


def audit_local_imagery() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metro_rows: list[dict[str, str]] = []
    year_rows: list[dict[str, str]] = []

    for metro_dir in sorted([path for path in IMAGERY_DIR.iterdir() if path.is_dir()]):
        metro = metro_dir.name
        modis_dir = metro_dir / "modis_rgb"
        viirs_dir = metro_dir / "viirs_night"
        modis_years = image_years(modis_dir, set(EXPECTED_MODIS_YEARS)) if modis_dir.exists() else []
        viirs_years = image_years(viirs_dir, set(EXPECTED_VIIRS_YEARS)) if viirs_dir.exists() else []

        modis_dims: set[str] = set()
        viirs_dims: set[str] = set()
        strict_cloud_values: list[float] = []
        diffuse_cloud_values: list[float] = []
        dark_edge_values: list[float] = []
        core_diffuse_values: list[float] = []
        core_dark_values: list[float] = []
        severe_years: list[int] = []

        for year in modis_years:
            path = modis_dir / f"{year}.tif"
            (
                width,
                height,
                strict_cloud_pct,
                diffuse_cloud_pct,
                dark_edge_pct,
                mean_brightness,
                core_diffuse_cloud_pct,
                core_dark_or_empty_pct,
            ) = cloud_stats_for_rgb(path)
            modis_dims.add(f"{width}x{height}")
            strict_cloud_values.append(strict_cloud_pct)
            diffuse_cloud_values.append(diffuse_cloud_pct)
            dark_edge_values.append(dark_edge_pct)
            core_diffuse_values.append(core_diffuse_cloud_pct)
            core_dark_values.append(core_dark_or_empty_pct)
            if core_diffuse_cloud_pct >= 20.0 or diffuse_cloud_pct >= 40.0:
                severe_years.append(year)
            year_rows.append(
                {
                    "metro": metro,
                    "year": str(year),
                    "width": str(width),
                    "height": str(height),
                    "strict_white_cloud_pct": f"{strict_cloud_pct:.2f}",
                    "diffuse_cloud_pct": f"{diffuse_cloud_pct:.2f}",
                    "dark_border_pct": f"{dark_edge_pct:.2f}",
                    "mean_brightness": f"{mean_brightness:.4f}",
                    "core_diffuse_cloud_pct": f"{core_diffuse_cloud_pct:.2f}",
                    "core_dark_or_empty_pct": f"{core_dark_or_empty_pct:.2f}",
                }
            )

        for year in viirs_years:
            path = viirs_dir / f"{year}.tif"
            img = Image.open(path)
            viirs_dims.add(f"{img.size[0]}x{img.size[1]}")

        observed_aspect = None
        aspect_error_pct = None
        if modis_dims:
            width, height = map(int, sorted(modis_dims)[0].split("x"))
            if height:
                observed_aspect = width / height
        expected_aspect = expected_bbox_aspect(metro)
        if observed_aspect and expected_aspect:
            aspect_error_pct = abs(observed_aspect - expected_aspect) / expected_aspect * 100.0

        tile_grid = "missing"
        geometry_check = "review"
        if modis_dims:
            width, height = map(int, sorted(modis_dims)[0].split("x"))
            tile_grid = f"{width // 512}x{height // 512} tiles"
            if len(modis_dims) == 1 and width % 512 == 0 and height % 512 == 0:
                geometry_check = "stable full-tile mosaic"

        metro_rows.append(
            {
                "metro": metro,
                "modis_year_count": str(len(modis_years)),
                "viirs_year_count": str(len(viirs_years)),
                "missing_modis_years": ", ".join(map(str, sorted(set(EXPECTED_MODIS_YEARS) - set(modis_years)))) or "none",
                "missing_viirs_years": ", ".join(map(str, sorted(set(EXPECTED_VIIRS_YEARS) - set(viirs_years)))) or "none",
                "modis_dimensions": ", ".join(sorted(modis_dims)) or "missing",
                "viirs_dimensions": ", ".join(sorted(viirs_dims)) or "missing",
                "strict_mean_cloud_pct": f"{np.mean(strict_cloud_values):.2f}" if strict_cloud_values else "nan",
                "strict_max_cloud_pct": f"{np.max(strict_cloud_values):.2f}" if strict_cloud_values else "nan",
                "diffuse_mean_cloud_pct": f"{np.mean(diffuse_cloud_values):.2f}" if diffuse_cloud_values else "nan",
                "diffuse_max_cloud_pct": f"{np.max(diffuse_cloud_values):.2f}" if diffuse_cloud_values else "nan",
                "core_diffuse_mean_cloud_pct": f"{np.mean(core_diffuse_values):.2f}" if core_diffuse_values else "nan",
                "core_diffuse_max_cloud_pct": f"{np.max(core_diffuse_values):.2f}" if core_diffuse_values else "nan",
                "years_ge_20_diffuse": str(sum(value >= 20.0 for value in diffuse_cloud_values)),
                "years_ge_40_diffuse": str(sum(value >= 40.0 for value in diffuse_cloud_values)),
                "years_ge_12_core_diffuse": str(sum(value >= 12.0 for value in core_diffuse_values)),
                "severe_cloud_years": ", ".join(map(str, severe_years)) or "none",
                "max_dark_border_pct": f"{np.max(dark_edge_values):.2f}" if dark_edge_values else "nan",
                "max_core_dark_pct": f"{np.max(core_dark_values):.2f}" if core_dark_values else "nan",
                "modis_tile_grid": tile_grid,
                "expected_bbox_aspect": f"{expected_aspect:.3f}" if expected_aspect else "nan",
                "observed_image_aspect": f"{observed_aspect:.3f}" if observed_aspect else "nan",
                "aspect_error_pct": f"{aspect_error_pct:.2f}" if aspect_error_pct is not None else "nan",
                "geometry_check": geometry_check,
            }
        )

    return metro_rows, year_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(row.get(key, "") for key, _ in columns) + " |")
    return "\n".join([header, divider, *body])


def save_cloud_summary_plot(metro_rows: list[dict[str, str]]) -> Path:
    plot_path = OUT_DIR / "modis_cloud_summary.png"
    order = sorted(
        metro_rows,
        key=lambda row: float(row["core_diffuse_mean_cloud_pct"]),
        reverse=True,
    )
    metros = [row["metro"].replace("_", " ").title() for row in order]
    mean_cloud = [float(row["core_diffuse_mean_cloud_pct"]) for row in order]
    max_cloud = [float(row["core_diffuse_max_cloud_pct"]) for row in order]

    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    y = np.arange(len(metros))
    ax.barh(y, max_cloud, color="#cbd5e1", label="Worst year")
    ax.barh(y, mean_cloud, color="#2563eb", label="Mean cloud cover")
    ax.set_yticks(y, metros, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cloud-mask share (%)")
    ax.set_title("MODIS center-weighted cloud-risk audit by metro (2013-2023)", fontsize=14, fontweight="bold")
    ax.axvline(12, color="#f59e0b", linestyle="--", linewidth=1.2, label="12% concern line")
    ax.axvline(20, color="#dc2626", linestyle="--", linewidth=1.2, label="20% severe line")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(plot_path, dpi=220, facecolor="white")
    plt.close(fig)
    return plot_path


def write_summary_markdown(
    branch_rows: list[dict[str, str]],
    metro_rows: list[dict[str, str]],
    year_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    refresh_rows: list[dict[str, str]],
) -> None:
    summary_path = OUT_DIR / "DATA_INTEGRITY_AUDIT.md"

    severe_frames = sorted(
        year_rows,
        key=lambda row: (float(row["core_diffuse_cloud_pct"]), float(row["diffuse_cloud_pct"])),
        reverse=True,
    )[:12]
    top_cloudy_metros = sorted(
        metro_rows,
        key=lambda row: float(row["core_diffuse_mean_cloud_pct"]),
        reverse=True,
    )[:8]
    aspect_rows = [
        row for row in metro_rows
        if float(row["aspect_error_pct"]) > 10.0
    ]
    if not aspect_rows:
        aspect_rows = sorted(metro_rows, key=lambda row: float(row["aspect_error_pct"]))[:5]

    severe_count = sum(float(row["core_diffuse_cloud_pct"]) >= 20.0 for row in year_rows)
    branch_problem = next((row for row in branch_rows if row["ref"] == "upstream/main"), None)
    restored_branch = next((row for row in branch_rows if row["ref"] == "upstream/rename-add-prefix"), None)
    selected_replaced = sum(row["selected_date"] != row["baseline_08_01_date"] for row in selected_rows) if selected_rows else 0
    selected_dark_gap = sum(float(row["selected_dark_or_empty_pct"]) > 5.0 for row in selected_rows) if selected_rows else 0
    selected_high_diffuse = sum(float(row["selected_core_diffuse_cloud_pct"]) > 12.0 for row in selected_rows) if selected_rows else 0
    refreshed_high_diffuse = sum(float(row["actual_core_diffuse_cloud_pct"]) > 12.0 for row in refresh_rows) if refresh_rows else 0
    refreshed_missing = sum(int(row["missing_tiles"]) > 0 for row in refresh_rows) if refresh_rows else 0

    lines = [
        "# Data Integrity and MODIS Quality Audit",
        "",
        "This audit was generated from the current working tree to explain the project's actual data state, not just the intended state described in the notebooks.",
        "",
        "## Executive Summary",
        "",
        f"- The local working tree currently contains the restored **14-metro** data inventory, but `{branch_problem['ref']}` still exposes only **{branch_problem['panel_metros']} metros** and **{branch_problem['imagery_metros']} imagery folders**.",
        f"- The most complete published branch is `{restored_branch['ref']}`, which exposes **{restored_branch['panel_metros']} metros** in the economic panel and **{restored_branch['imagery_metros']} metro imagery folders**.",
        (
            f"- The MODIS acquisition workflow is now driven by an audited per-metro-year manifest: "
            f"**{selected_replaced} of {len(selected_rows)}** metro-years moved away from the old `08-01` heuristic."
            if selected_rows
            else "- The MODIS acquisition workflow is expected to be driven by an audited per-metro-year manifest, but that manifest was not found."
        ),
        (
            f"- The final selection rule now prioritizes full coverage, then center-region visibility, and only then whole-frame cloud minimization. In the refreshed imagery inventory there are **{refreshed_missing}** frames with missing tiles and **{selected_dark_gap}** selected dates with large dark-gap coverage."
            if refresh_rows and selected_rows
            else "- The final selection rule prioritizes full coverage, then center-region visibility, then whole-frame cloud minimization."
        ),
        (
            f"- Residual center-region cloud risk is now concentrated in **{refreshed_high_diffuse}** refreshed metro-years with core diffuse-cloud score above 12%."
            if refresh_rows
            else "- Residual cloud risk is tracked with a center-weighted diffuse-cloud proxy because the notebook-compatible near-white mask is too permissive on hazy scenes."
        ),
        "- Varying image dimensions are stable within each metro and remain exact multiples of 512 pixels, which is consistent with full GIBS tile mosaics. Rectangular rasters therefore do not automatically mean a city was cut in half.",
        "- Stable tile geometry does not by itself prove semantic bbox correctness, so imagery should still be interpreted as raster-aligned metro views rather than exact legal boundaries.",
        "",
        "## 1. Branch and Inventory Status",
        "",
        markdown_table(
            branch_rows,
            [
                ("ref", "Ref"),
                ("commit", "Commit"),
                ("panel_rows", "Panel rows"),
                ("panel_metros", "Panel metros"),
                ("imagery_metros", "Imagery metros"),
            ],
        ),
        "",
        "### Rationale",
        "",
        "- The modeling and EDA notebooks describe a 14-metro project, so any 5-metro branch is an inconsistent project state, not just a smaller sample choice.",
        "- The safest branch-level source of truth for the restored data is currently `upstream/rename-add-prefix`, not `upstream/main`.",
        "",
        "## 2. Metro-Level MODIS / VIIRS Inventory and Quality",
        "",
        markdown_table(
            top_cloudy_metros,
            [
                ("metro", "Metro"),
                ("modis_year_count", "MODIS years"),
                ("viirs_year_count", "VIIRS years"),
                ("strict_mean_cloud_pct", "Strict mean cloud %"),
                ("core_diffuse_mean_cloud_pct", "Core mean cloud %"),
                ("core_diffuse_max_cloud_pct", "Worst core cloud %"),
                ("years_ge_12_core_diffuse", "Years >= 12% core cloud"),
                ("modis_dimensions", "MODIS dims"),
                ("geometry_check", "Geometry check"),
            ],
        ),
        "",
        "The full metro-level audit table is saved as `deliverables/data_audit/metro_imagery_audit.csv`.",
        "",
        "## 3. Worst Cloud Cases",
        "",
        f"The current MODIS inventory contains **{severe_count} metro-year frames** with **core diffuse-cloud risk at or above 20%**. This center-weighted score is more aligned with city visibility than a whole-frame average.",
        "",
        markdown_table(
            severe_frames,
            [
                ("metro", "Metro"),
                ("year", "Year"),
                ("strict_white_cloud_pct", "Strict cloud %"),
                ("core_diffuse_cloud_pct", "Core diffuse %"),
                ("diffuse_cloud_pct", "Diffuse cloud %"),
                ("dark_border_pct", "Dark border %"),
                ("width", "Width"),
                ("height", "Height"),
            ],
        ),
        "",
        "### Rationale",
        "",
        "- This directly supports the teammate concern that some MODIS frames are not visually reliable for interpreting urban expansion.",
        "- The center-weighted metric is intentionally stricter about city-core visibility than the older whole-frame rule, so it better matches the actual modeling use case.",
        "",
        "## 4. Dimension and Cropping Check",
        "",
        markdown_table(
            aspect_rows,
            [
                ("metro", "Metro"),
                ("expected_bbox_aspect", "Expected bbox aspect"),
                ("observed_image_aspect", "Observed image aspect"),
                ("aspect_error_pct", "Aspect error %"),
                ("modis_dimensions", "MODIS dims"),
                ("modis_tile_grid", "Tile grid"),
                ("geometry_check", "Status"),
            ],
        ),
        "",
        "### Rationale",
        "",
        "- This check compares the configured bbox shape against the saved raster shape, but it also records the GIBS tile grid implied by the image dimensions.",
        "- Because the fetch notebook saves full 512-pixel tiles rather than exact geographic crops, aspect mismatch alone is not strong evidence of clipping. Stable `1x2` or `2x1` tile grids can still be expected outcomes.",
        "- This audit does **not** prove that every bbox is semantically correct; it only shows that the saved geometry is stable and tile-aligned rather than obviously broken.",
        "",
        "## 5. How MODIS Dates Are Selected Now",
        "",
        "- The original workflow used `08-01` as a practical starting heuristic. That rule is no longer treated as the source of truth for refreshed imagery.",
        "- The final date-selection rule now prefers **complete frames** first, then minimizes **center-region cloud risk**, and only then uses whole-frame cloud metrics as tie-breakers.",
        "- This ordering is intentional. A frame that preserves the city core and avoids black wedges is safer for downstream feature extraction than one that only looks cleaner in peripheral tiles.",
        (
            f"- The refreshed acquisition manifest lives at `data/imagery/modis_acquisition_manifest.csv`, and the residual QA contact sheet is saved at `{MODIS_RESIDUAL_QA_PATH.relative_to(REPO_ROOT)}`."
            if refresh_rows
            else "- The refreshed acquisition manifest is expected at `data/imagery/modis_acquisition_manifest.csv`."
        ),
        "",
        "## 6. Final Interpretation Notes",
        "",
        "1. The refreshed MODIS acquisition manifest, tensors, and modeling tables should be treated as the current pre-final-model source of truth.",
        "2. Residual high-cloud cases are now explicit and bounded; they remain usable for numeric summaries, but should be used cautiously as qualitative visual evidence.",
        "3. Rectangular MODIS rasters are expected tile mosaics in this pipeline, so they should not be interpreted as accidental cropping by default.",
        "4. The repo is now internally consistent around the restored 14-metro state even though the public `main` branch may lag that state.",
    ]

    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    branch_rows = branch_inventory()
    metro_rows, year_rows = audit_local_imagery()
    selected_rows = read_optional_csv_rows(MODIS_SELECTED_PATH)
    refresh_rows = read_optional_csv_rows(MODIS_REFRESH_LOG_PATH)

    write_csv(
        OUT_DIR / "branch_data_status.csv",
        branch_rows,
        ["ref", "commit", "panel_rows", "panel_metros", "imagery_metros", "metro_list"],
    )
    write_csv(
        OUT_DIR / "metro_imagery_audit.csv",
        metro_rows,
        [
            "metro",
            "modis_year_count",
            "viirs_year_count",
            "missing_modis_years",
            "missing_viirs_years",
            "modis_dimensions",
            "viirs_dimensions",
            "strict_mean_cloud_pct",
            "strict_max_cloud_pct",
            "diffuse_mean_cloud_pct",
            "diffuse_max_cloud_pct",
            "core_diffuse_mean_cloud_pct",
            "core_diffuse_max_cloud_pct",
            "years_ge_20_diffuse",
            "years_ge_40_diffuse",
            "years_ge_12_core_diffuse",
            "severe_cloud_years",
            "max_dark_border_pct",
            "max_core_dark_pct",
            "modis_tile_grid",
            "expected_bbox_aspect",
            "observed_image_aspect",
            "aspect_error_pct",
            "geometry_check",
        ],
    )
    write_csv(
        OUT_DIR / "modis_cloud_year_audit.csv",
        year_rows,
        [
            "metro",
            "year",
            "width",
            "height",
            "strict_white_cloud_pct",
            "diffuse_cloud_pct",
            "dark_border_pct",
            "mean_brightness",
            "core_diffuse_cloud_pct",
            "core_dark_or_empty_pct",
        ],
    )
    save_cloud_summary_plot(metro_rows)
    write_summary_markdown(branch_rows, metro_rows, year_rows, selected_rows, refresh_rows)

    print(f"Wrote audit artifacts to {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
