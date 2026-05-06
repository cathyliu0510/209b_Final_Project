#!/usr/bin/env python3
"""Run the full pre-final-model refresh pipeline.

This is the reproducible umbrella command for the data/EDA/baseline refresh:
1. Search compact MODIS candidate dates for all metros/years.
2. Expand the search only for problematic metro-years.
3. Refresh MODIS GeoTIFFs from the selected-date manifest.
4. Re-audit data integrity and imagery quality.
5. Re-execute the key preprocessing / EDA / baseline artifacts.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--skip-notebooks", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_search and not args.baseline_only:
        run([py, str(REPO_ROOT / "scripts" / "search_modis_candidate_dates.py")])
        run([py, str(REPO_ROOT / "scripts" / "refine_modis_candidate_dates.py")])

    if not args.skip_refresh and not args.baseline_only:
        run([py, str(REPO_ROOT / "scripts" / "refresh_modis_from_selected_dates.py")])
        run([py, str(REPO_ROOT / "scripts" / "audit_data_integrity.py")])

    if not args.skip_notebooks:
        preferred_jupyter = Path(sys.executable).with_name("jupyter")
        if preferred_jupyter.exists():
            jupyter = str(preferred_jupyter)
        else:
            jupyter = shutil.which("jupyter")
        if jupyter is None:
            raise RuntimeError("jupyter not found in PATH")

        if not args.baseline_only:
            run(
                [
                    jupyter,
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "--ExecutePreprocessor.timeout=-1",
                    "03_raster_preprocessing.ipynb",
                ]
            )
            run(
                [
                    jupyter,
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "--ExecutePreprocessor.timeout=-1",
                    "00_Final_EDA_Merged.ipynb",
                ]
            )
            for target in [
                REPO_ROOT / "00_Final_EDA_Merged_finalized.ipynb",
                REPO_ROOT / "Cathy_Comprehensive_EDA.ipynb",
            ]:
                shutil.copy2(REPO_ROOT / "00_Final_EDA_Merged.ipynb", target)

        run([py, str(REPO_ROOT / "scripts" / "build_baseline_model_notebook.py")])


if __name__ == "__main__":
    main()
