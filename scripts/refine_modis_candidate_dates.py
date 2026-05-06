#!/usr/bin/env python3
"""Run a second-pass MODIS date search for problematic metro-years.

The first-pass compact search is intentionally small. This script only expands
the calendar window for the cases that still look weak after that first pass:
either the best candidate is incomplete (`missing_tiles > 0`) or the selected
frame is still too cloudy.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTED_PATH = REPO_ROOT / "deliverables" / "data_audit" / "modis_date_search" / "modis_selected_dates.csv"
SUMMARY_PATH = REPO_ROOT / "deliverables" / "data_audit" / "modis_date_search" / "modis_refinement_targets.csv"
DEFAULT_DIFFUSE_THRESHOLD = 20.0
DEFAULT_CORE_DIFFUSE_THRESHOLD = 12.0
DEFAULT_DARK_THRESHOLD = 5.0
DEFAULT_CORE_DARK_THRESHOLD = 2.0


def build_dense_dates() -> list[str]:
    out = []
    current = date(2001, 3, 15)
    end = date(2001, 11, 15)
    while current <= end:
        out.append(current.strftime("%m-%d"))
        current += timedelta(days=7)
    return out


def build_rescue_dates() -> list[str]:
    out = []
    current = date(2001, 1, 1)
    end = date(2001, 12, 15)
    while current <= end:
        out.append(current.strftime("%m-%d"))
        current += timedelta(days=7)
    return out


def flag_rows(
    rows: list[dict[str, str]],
    diffuse_threshold: float,
    core_diffuse_threshold: float,
    dark_threshold: float,
    core_dark_threshold: float,
) -> list[dict[str, str]]:
    flagged = []
    for row in rows:
        missing_tiles = int(row["selected_missing_tiles"])
        diffuse = float(row["selected_diffuse_cloud_pct"])
        core_diffuse = float(row["selected_core_diffuse_cloud_pct"])
        dark = float(row["selected_dark_or_empty_pct"])
        core_dark = float(row["selected_core_dark_or_empty_pct"])
        if (
            missing_tiles > 0
            or diffuse > diffuse_threshold
            or core_diffuse > core_diffuse_threshold
            or dark > dark_threshold
            or core_dark > core_dark_threshold
        ):
            reason = []
            if missing_tiles > 0:
                reason.append("incomplete_mosaic")
            if diffuse > diffuse_threshold:
                reason.append("high_diffuse_cloud")
            if core_diffuse > core_diffuse_threshold:
                reason.append("high_core_diffuse_cloud")
            if dark > dark_threshold:
                reason.append("high_dark_gap")
            if core_dark > core_dark_threshold:
                reason.append("high_core_dark_gap")
            flagged.append(
                {
                    "metro": row["metro"],
                    "year": row["year"],
                    "selected_date": row["selected_date"],
                    "selected_quality_flag": row["selected_quality_flag"],
                    "selected_missing_tiles": row["selected_missing_tiles"],
                    "selected_core_diffuse_cloud_pct": row["selected_core_diffuse_cloud_pct"],
                    "selected_diffuse_cloud_pct": row["selected_diffuse_cloud_pct"],
                    "selected_core_dark_or_empty_pct": row["selected_core_dark_or_empty_pct"],
                    "selected_dark_or_empty_pct": row["selected_dark_or_empty_pct"],
                    "reason": "+".join(reason),
                }
            )
    return flagged


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffuse-threshold", type=float, default=DEFAULT_DIFFUSE_THRESHOLD)
    parser.add_argument("--core-diffuse-threshold", type=float, default=DEFAULT_CORE_DIFFUSE_THRESHOLD)
    parser.add_argument("--dark-threshold", type=float, default=DEFAULT_DARK_THRESHOLD)
    parser.add_argument("--core-dark-threshold", type=float, default=DEFAULT_CORE_DARK_THRESHOLD)
    parser.add_argument("--dates", default=",".join(build_dense_dates()))
    parser.add_argument("--rescue-dates", default=",".join(build_rescue_dates()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not SELECTED_PATH.exists():
        raise FileNotFoundError(f"Missing selected-date manifest: {SELECTED_PATH}")

    rows = load_rows(SELECTED_PATH)
    flagged = flag_rows(
        rows,
        args.diffuse_threshold,
        args.core_diffuse_threshold,
        args.dark_threshold,
        args.core_dark_threshold,
    )

    write_rows(
        SUMMARY_PATH,
        flagged,
        [
            "metro",
            "year",
            "selected_date",
            "selected_quality_flag",
            "selected_missing_tiles",
            "selected_core_diffuse_cloud_pct",
            "selected_diffuse_cloud_pct",
            "selected_core_dark_or_empty_pct",
            "selected_dark_or_empty_pct",
            "reason",
        ],
    )
    print(f"Flagged {len(flagged)} problematic metro-years")
    print(f"Wrote refinement target list to {SUMMARY_PATH.relative_to(REPO_ROOT)}")

    if args.dry_run or not flagged:
        return

    expanded_dates = args.dates
    for row in flagged:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "search_modis_candidate_dates.py"),
            "--metros",
            row["metro"],
            "--years",
            row["year"],
            "--dates",
            expanded_dates,
            "--resume",
        ]
        print("[refine]", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    refreshed_rows = load_rows(SELECTED_PATH)
    remaining_flagged = flag_rows(
        refreshed_rows,
        args.diffuse_threshold,
        args.core_diffuse_threshold,
        args.dark_threshold,
        args.core_dark_threshold,
    )
    if remaining_flagged:
        rescue_dates = args.rescue_dates
        print(f"Flagged {len(remaining_flagged)} metro-years after warm-season refinement; running year-round rescue search")
        for row in remaining_flagged:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "search_modis_candidate_dates.py"),
                "--metros",
                row["metro"],
                "--years",
                row["year"],
                "--dates",
                rescue_dates,
                "--resume",
            ]
            print("[rescue]", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)

        refreshed_rows = load_rows(SELECTED_PATH)
        remaining_flagged = flag_rows(
            refreshed_rows,
            args.diffuse_threshold,
            args.core_diffuse_threshold,
            args.dark_threshold,
            args.core_dark_threshold,
        )

    write_rows(
        SUMMARY_PATH,
        remaining_flagged,
        [
            "metro",
            "year",
            "selected_date",
            "selected_quality_flag",
            "selected_missing_tiles",
            "selected_core_diffuse_cloud_pct",
            "selected_diffuse_cloud_pct",
            "selected_core_dark_or_empty_pct",
            "selected_dark_or_empty_pct",
            "reason",
        ],
    )
    print(f"Remaining flagged metro-years after refinement: {len(remaining_flagged)}")


if __name__ == "__main__":
    main()
