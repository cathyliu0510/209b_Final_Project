#!/usr/bin/env python3
"""Render a contact sheet for the most visually uncertain refreshed MODIS frames."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rasterio


REPO_ROOT = Path(__file__).resolve().parents[1]
REFRESH_LOG_PATH = REPO_ROOT / "deliverables" / "data_audit" / "modis_refresh_log.csv"
OUT_PATH = REPO_ROOT / "deliverables" / "data_audit" / "modis_residual_qa.png"


def main() -> None:
    df = pd.read_csv(REFRESH_LOG_PATH)
    df["residual_risk_score"] = (
        df["actual_core_diffuse_cloud_pct"].fillna(0) * 2.0
        + df["actual_core_dark_or_empty_pct"].fillna(0) * 1.5
        + df["actual_diffuse_cloud_pct"].fillna(0)
    )
    df = df.sort_values("residual_risk_score", ascending=False).head(8).reset_index(drop=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    for ax, row in zip(axes, df.to_dict(orient="records")):
        path = REPO_ROOT / row["output_path"]
        with rasterio.open(path) as src:
            arr = src.read([1, 2, 3]).transpose(1, 2, 0)
        ax.imshow(arr)
        ax.set_title(
            (
                f"{row['metro']} {row['year']}\n"
                f"{row['selected_date']} | core {row['actual_core_diffuse_cloud_pct']:.1f}%"
                f" | full {row['actual_diffuse_cloud_pct']:.1f}%"
            ),
            fontsize=11,
        )
        ax.axis("off")

    for ax in axes[len(df):]:
        ax.axis("off")

    fig.suptitle(
        "Residual MODIS QA: highest center-weighted cloud-risk scores after refresh",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
