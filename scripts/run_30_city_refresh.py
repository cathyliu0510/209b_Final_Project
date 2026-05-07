#!/usr/bin/env python3
"""Rebuild the 30-city preprocessing and final-model artifacts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
JUPYTER = shutil.which("jupyter") or str(Path(sys.executable).with_name("jupyter"))

NEW_METROS = [
    "boston", "chicago", "cleveland", "columbus", "detroit", "indianapolis",
    "kansas_city", "miami", "minneapolis", "new_orleans", "oklahoma_city",
    "philadelphia", "pittsburgh", "sacramento", "salt_lake_city", "st_louis",
]


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=merged_env)


def main() -> None:
    metros_csv = ",".join(NEW_METROS)

    run([PY, "scripts/sync_30_city_notebooks.py"])
    run([PY, "scripts/build_economic_panel.py"])
    run([PY, "scripts/search_modis_candidate_dates.py", "--metros", metros_csv, "--resume"])
    run([PY, "scripts/refine_modis_candidate_dates.py"])
    run([PY, "scripts/fetch_gibs_imagery.py", "--metros", metros_csv, "--layers", "modis_rgb,viirs_night", "--overwrite"])
    run([PY, "scripts/build_ghsl_from_tiles.py"])

    run([
        JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=-1", "03_raster_preprocessing.ipynb",
    ])
    run([
        JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=-1", "00_Final_EDA_Merged.ipynb",
    ])
    shutil.copy2(REPO_ROOT / "00_Final_EDA_Merged.ipynb", REPO_ROOT / "00_Final_EDA_Merged_finalized.ipynb")
    shutil.copy2(REPO_ROOT / "00_Final_EDA_Merged.ipynb", REPO_ROOT / "Cathy_Comprehensive_EDA.ipynb")

    run(
        [
            JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=-1", "V2_Minh_Final_Model_Pipeline_Cleaned_14City.ipynb",
        ],
        env={"FC_RETRAIN_ALL": "1"},
    )
    run([
        JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=-1", "V2_Minh_Final_Model_Pipeline_Cleaned_14City.ipynb",
    ])
    shutil.copy2(
        REPO_ROOT / "V2_Minh_Final_Model_Pipeline_Cleaned_14City.ipynb",
        REPO_ROOT / "Minh_Final_Model_Pipeline_Cleaned.ipynb",
    )


if __name__ == "__main__":
    main()
