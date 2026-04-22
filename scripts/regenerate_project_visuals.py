#!/usr/bin/env python3
"""
Regenerate the project's EDA figures with a cleaner, more consistent visual style.

This script overwrites the saved PNG assets in both:
  - figures/
  - EDA_Figures/

It intentionally focuses on visual clarity: fewer overlapping labels, cleaner
legends, better spacing, more direct titles, and simpler missingness messaging.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRS = [ROOT / "figures", ROOT / "EDA_Figures"]
for figure_dir in FIGURE_DIRS:
    figure_dir.mkdir(exist_ok=True)


METRO_LABELS = {
    "atlanta": "Atlanta, GA",
    "austin": "Austin, TX",
    "charlotte": "Charlotte, NC",
    "dallas": "Dallas, TX",
    "denver": "Denver, CO",
    "houston": "Houston, TX",
    "jacksonville": "Jacksonville, FL",
    "las_vegas": "Las Vegas, NV",
    "nashville": "Nashville, TN",
    "orlando": "Orlando, FL",
    "phoenix": "Phoenix, AZ",
    "raleigh": "Raleigh, NC",
    "san_antonio": "San Antonio, TX",
    "tampa": "Tampa, FL",
}

SHORT_LABELS = {metro: label.split(",")[0] for metro, label in METRO_LABELS.items()}


@dataclass
class TensorStack:
    metro: str
    tensor: np.ndarray
    years: np.ndarray


@dataclass
class Context:
    panel: pd.DataFrame
    econ5: pd.DataFrame
    metros: List[str]
    metro_colors: Dict[str, tuple]
    years_panel: List[int]
    years_full: List[int]
    tensors: Dict[str, TensorStack]
    tensor_metros: List[str]
    tensor_years: List[int]
    viirs_from_tensors: pd.DataFrame


def set_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "axes.titlesize": 14.5,
            "axes.labelsize": 12.5,
            "figure.titlesize": 18.5,
            "legend.fontsize": 10,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.titlepad": 8,
            "grid.alpha": 0.18,
            "grid.color": "#b8c2cc",
            "axes.edgecolor": "#c8c8c8",
            "font.family": "DejaVu Sans",
            "savefig.dpi": 190,
        },
    )
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"


def load_context() -> Context:
    panel = pd.read_csv(ROOT / "data/modeling/panel_features.csv").sort_values(
        ["metro", "year"]
    )
    econ5 = pd.read_csv(ROOT / "data/economic/panel.csv").sort_values(["metro", "year"])

    metros = sorted(panel["metro"].unique())
    years_panel = sorted(panel["year"].unique())

    palette = sns.color_palette("tab20", len(metros))
    metro_colors = {metro: palette[i] for i, metro in enumerate(metros)}

    tensors: Dict[str, TensorStack] = {}
    for path in sorted((ROOT / "data/tensors").glob("*_stack.npz")):
        metro = path.stem.replace("_stack", "")
        archive = np.load(path)
        tensors[metro] = TensorStack(
            metro=metro,
            tensor=archive["tensor"],
            years=archive["years"],
        )

    tensor_metros = sorted(tensors)
    tensor_years = sorted({int(y) for stack in tensors.values() for y in stack.years})

    viirs_rows = []
    for metro, stack in tensors.items():
        for idx, year in enumerate(stack.years):
            year = int(year)
            if year < 2017:
                continue
            viirs_rows.append(
                {
                    "metro": metro,
                    "year": year,
                    "viirs_mean_tensor": float(stack.tensor[idx, :, :, 3].mean()),
                }
            )

    viirs_from_tensors = pd.DataFrame(viirs_rows).sort_values(["metro", "year"])

    years_full = sorted(
        set(years_panel).union(set(econ5["year"].unique())).union(set(tensor_years))
    )

    return Context(
        panel=panel,
        econ5=econ5,
        metros=metros,
        metro_colors=metro_colors,
        years_panel=years_panel,
        years_full=years_full,
        tensors=tensors,
        tensor_metros=tensor_metros,
        tensor_years=tensor_years,
        viirs_from_tensors=viirs_from_tensors,
    )


def save_figure(fig: mpl.figure.Figure, filename: str, close: bool = True) -> None:
    for figure_dir in FIGURE_DIRS:
        fig.savefig(figure_dir / filename, bbox_inches="tight", facecolor="white")
    print(f"Saved {filename} to {', '.join(str(d) for d in FIGURE_DIRS)}")
    if close:
        plt.close(fig)


def metro_name(metro: str) -> str:
    return METRO_LABELS.get(metro, metro.replace("_", " ").title())


def short_name(metro: str) -> str:
    return SHORT_LABELS.get(metro, metro.replace("_", " ").title())


def title_case(text: str) -> str:
    return text.replace("_", " ").title()


def human_num(value: float) -> str:
    return f"{int(round(value)):,}"


def crop_bbox_from_rgb(rgb: np.ndarray, pad_ratio: float = 0.02) -> tuple[int, int, int, int]:
    mask = np.isfinite(rgb).all(axis=-1) & (rgb.sum(axis=-1) > 1e-4)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        h, w = rgb.shape[:2]
        return 0, h, 0, w

    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1

    h, w = rgb.shape[:2]
    pad_r = int((r1 - r0) * pad_ratio)
    pad_c = int((c1 - c0) * pad_ratio)
    return (
        max(0, r0 - pad_r),
        min(h, r1 + pad_r),
        max(0, c0 - pad_c),
        min(w, c1 + pad_c),
    )


def apply_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bbox
    if arr.ndim == 3:
        return arr[r0:r1, c0:c1, :]
    return arr[r0:r1, c0:c1]


def thousands_formatter(_: float, __: int) -> str:
    return ""


def comma_fmt(x: float, _: int) -> str:
    return f"{int(x):,}"


def make_figure_01_satellite_imagery_grid(ctx: Context) -> None:
    representative = ["atlanta", "austin", "dallas", "denver", "phoenix", "tampa"]
    representative = [metro for metro in representative if metro in ctx.tensors]

    fig, axes = plt.subplots(
        len(representative),
        3,
        figsize=(14, 3.25 * len(representative)),
        constrained_layout=True,
    )
    if len(representative) == 1:
        axes = np.array([axes])

    viirs_for_cbar = None
    for row_idx, metro in enumerate(representative):
        stack = ctx.tensors[metro]
        years = list(stack.years.astype(int))
        idx_2015 = years.index(2015)
        idx_2023 = years.index(2023)

        rgb_2015 = stack.tensor[idx_2015, :, :, :3]
        rgb_2023 = stack.tensor[idx_2023, :, :, :3]
        night_2023 = stack.tensor[idx_2023, :, :, 3]

        bbox = crop_bbox_from_rgb(rgb_2023)
        panels = [
            apply_bbox(rgb_2015, bbox),
            apply_bbox(rgb_2023, bbox),
            apply_bbox(night_2023, bbox),
        ]

        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if col_idx < 2:
                ax.imshow(np.clip(panel, 0, 1))
            else:
                viirs_for_cbar = ax.imshow(panel, cmap="magma", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row_idx, 0].text(
            -0.06,
            0.5,
            short_name(metro),
            transform=axes[row_idx, 0].transAxes,
            fontsize=13,
            fontweight="bold",
            ha="right",
            va="center",
        )

    col_titles = ["MODIS RGB (2015)", "MODIS RGB (2023)", "VIIRS Night-Lights (2023)"]
    for ax, col_title in zip(axes[0], col_titles):
        ax.set_title(col_title, fontsize=15, fontweight="bold")

    if viirs_for_cbar is not None:
        cbar = fig.colorbar(viirs_for_cbar, ax=axes[:, 2], fraction=0.025, pad=0.02)
        cbar.set_label("Normalized VIIRS intensity")

    fig.suptitle("Satellite Imagery Overview", fontweight="bold", y=1.01)
    save_figure(fig, "01_satellite_imagery_grid.png")


def make_figure_02_economic_timeseries(ctx: Context) -> None:
    metrics = [
        ("gdp_millions", "Real GDP", comma_fmt),
        ("employment_thousands", "Employment (thousands)", comma_fmt),
        ("unemployment_rate", "Unemployment Rate (%)", None),
        ("total_permits", "Building Permits", comma_fmt),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes = axes.ravel()

    for ax, (column, title, formatter) in zip(axes, metrics):
        for metro in ctx.metros:
            sub = ctx.panel[ctx.panel["metro"] == metro]
            ax.plot(
                sub["year"],
                sub[column],
                color=ctx.metro_colors[metro],
                linewidth=2.2,
                marker="o",
                markersize=3.5,
                alpha=0.9,
            )
        ax.axvspan(2019.6, 2020.4, color="#ef4444", alpha=0.08)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Year")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        if formatter is not None:
            ax.yaxis.set_major_formatter(FuncFormatter(formatter))

    handles = [
        Line2D([0], [0], color=ctx.metro_colors[metro], lw=3, label=metro_name(metro))
        for metro in ctx.metros
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.suptitle("Economic Indicators by Metro Area (2020 Omitted from Modeling Panel)", fontweight="bold")
    save_figure(fig, "02_economic_timeseries.png")


def make_missingness_figure(ctx: Context) -> None:
    panel = ctx.panel.copy()
    years = ctx.years_panel
    metros = ctx.metros

    viirs_cols = [c for c in panel.columns if c.startswith("viirs_") and c != "viirs_available"]
    lag_cols = [
        c
        for c in panel.columns
        if c.endswith("_lag1") or c.endswith("_delta") or c.endswith("_growth")
    ]

    viirs_missing_by_year = {
        year: (100.0 if year < 2017 else 0.0) for year in years
    }
    first_year = min(years)
    lag_missing_by_year = {
        year: (len(metros) * len(lag_cols) if year == first_year else 0) for year in years
    }

    interp = (
        panel.loc[panel["interpolated"].astype(bool), ["metro", "year"]]
        .assign(flag=1)
        .pivot(index="metro", columns="year", values="flag")
        .reindex(index=["charlotte", "tampa"])
        .reindex(columns=years)
        .fillna(0)
    )

    viirs_missing_cells = int(sum(year < 2017 for year in years) * len(metros) * len(viirs_cols))
    lag_missing_cells = int(len(metros) * len(lag_cols))
    permits_interp_cells = int(panel["interpolated"].sum())

    fig = plt.figure(figsize=(15, 9.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.15, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    colors_a = ["#dc2626" if year < 2017 else "#16a34a" for year in years]
    ax_a.bar(years, [viirs_missing_by_year[year] for year in years], color=colors_a, width=0.72)
    ax_a.axvline(2016.5, color="#334155", linestyle="--", linewidth=1.6)
    ax_a.text(
        2016.5,
        52,
        "VIIRS coverage begins\nin 2017",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#94a3b8"),
    )
    ax_a.set_ylim(0, 115)
    ax_a.set_ylabel("Share of VIIRS features missing")
    ax_a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax_a.set_title(
        "(A) Structural VIIRS gap by year",
        loc="left",
        fontweight="bold",
        fontsize=14,
    )

    colors_b = ["#f59e0b" if year == first_year else "#d1d5db" for year in years]
    ax_b.bar(years, [lag_missing_by_year[year] for year in years], color=colors_b, width=0.72)
    ax_b.set_ylim(0, lag_missing_cells * 1.18)
    ax_b.set_title(
        "(B) First-year derived-feature gap",
        loc="left",
        fontweight="bold",
        fontsize=14,
    )
    ax_b.set_ylabel("Missing cells")
    ax_b.text(
        0.98,
        0.98,
        f"{lag_missing_cells:,} cells\n({len(metros)} metros x {len(lag_cols)} derived features)",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#92400e",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff7ed", edgecolor="#f59e0b"),
    )

    sns.heatmap(
        interp,
        ax=ax_c,
        cmap=sns.light_palette("#ea580c", as_cmap=True),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=years,
        yticklabels=[metro_name(metro) for metro in interp.index],
    )
    ax_c.set_title(
        "(C) Interpolated permit gaps only where needed",
        loc="left",
        fontweight="bold",
        fontsize=14,
    )
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("")
    ax_c.tick_params(axis="x", labelrotation=0, labelsize=10)
    ax_c.tick_params(axis="y", labelsize=10.5)
    for i, metro in enumerate(interp.index):
        row_total = int(interp.loc[metro].sum())
        ax_c.text(
            len(years) + 0.15,
            i + 0.5,
            f"{row_total}",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color="#9a3412",
        )

    categories = pd.DataFrame(
        {
            "category": [
                "VIIRS pre-2017",
                "Lag / delta / growth\nfirst year",
                "Permits interpolation",
            ],
            "cells": [viirs_missing_cells, lag_missing_cells, permits_interp_cells],
            "resolution": [
                "Zero-fill plus `viirs_available` flag",
                "Leave as expected first-year NaN",
                "Linear interpolation plus flag",
            ],
        }
    )
    bars = ax_d.barh(
        categories["category"],
        categories["cells"],
        color=["#dc2626", "#f59e0b", "#fb923c"],
        edgecolor="none",
    )
    ax_d.invert_yaxis()
    ax_d.set_title(
        "(D) Missingness is mostly structural",
        loc="left",
        fontweight="bold",
        fontsize=14,
    )
    ax_d.set_xlabel("Affected cells before resolution")
    ax_d.xaxis.set_major_formatter(FuncFormatter(comma_fmt))
    ax_d.tick_params(axis="y", labelsize=11)
    for bar, cells, resolution in zip(bars, categories["cells"], categories["resolution"]):
        y_mid = bar.get_y() + bar.get_height() / 2
        ax_d.text(
            bar.get_width() + max(categories["cells"]) * 0.02,
            y_mid,
            f"{cells:,}\n{resolution}",
            va="center",
            ha="left",
            fontsize=10.5,
        )

    fig.suptitle("Missingness Audit: Structural Causes and Handling", fontweight="bold")
    save_figure(fig, "03_missingness_heatmap.png")


def make_figure_04_feature_distributions(ctx: Context) -> None:
    features = [
        ("gdp_millions", "GDP ($M)"),
        ("employment_thousands", "Employment (thousands)"),
        ("total_permits", "Building permits"),
        ("viirs_mean", "VIIRS mean intensity"),
        ("modis_brightness_mean", "MODIS brightness"),
        ("viirs_lit_frac", "VIIRS lit-pixel fraction"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 18), constrained_layout=True)
    axes = axes.ravel()

    for ax, (column, title) in zip(axes, features):
        data = ctx.panel[["metro", column]].dropna().copy()
        order = (
            data.groupby("metro")[column].median().sort_values(ascending=False).index.tolist()
        )
        sns.boxplot(
            data=data,
            x=column,
            y="metro",
            order=order,
            ax=ax,
            color="#dbeafe",
            fliersize=0,
            linewidth=1.1,
        )
        sns.stripplot(
            data=data,
            x=column,
            y="metro",
            order=order,
            ax=ax,
            color="#2563eb",
            size=2.8,
            alpha=0.35,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([short_name(metro) for metro in order], fontsize=11)
        if column in {"gdp_millions", "employment_thousands", "total_permits"}:
            ax.xaxis.set_major_formatter(FuncFormatter(comma_fmt))

    fig.suptitle("Feature Distributions by Metro", fontweight="bold")
    save_figure(fig, "04_feature_distributions.png")


def make_figure_05_satellite_feature_trends(ctx: Context) -> None:
    features = [
        ("viirs_mean", "VIIRS mean intensity", "mako"),
        ("viirs_lit_frac", "VIIRS lit-pixel fraction", "rocket"),
        ("viirs_gini", "VIIRS spatial concentration (Gini)", "flare"),
        ("modis_brightness_mean", "MODIS mean brightness", "crest"),
        ("modis_ndvi_proxy_mean", "MODIS NDVI proxy", "viridis"),
        ("modis_dark_frac", "MODIS dark-pixel fraction", "magma_r"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.5), constrained_layout=True)
    axes = axes.ravel()

    for ax, (column, title, cmap) in zip(axes, features):
        matrix = (
            ctx.panel.pivot(index="metro", columns="year", values=column)
            .reindex(index=ctx.metros)
            .reindex(columns=ctx.years_panel)
        )
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            cbar=True,
            linewidths=0.5,
            linecolor="white",
            yticklabels=[short_name(metro) for metro in ctx.metros],
            xticklabels=ctx.years_panel,
            cbar_kws={"shrink": 0.66, "pad": 0.01},
        )
        ax.set_title(title, fontweight="bold", fontsize=13.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=90, labelsize=9.5)
        ax.tick_params(axis="y", labelsize=9.5)

    fig.suptitle("Satellite Feature Trends by Metro and Year", fontweight="bold")
    save_figure(fig, "05_satellite_feature_trends.png")


def make_figure_06_cross_correlation_heatmap(ctx: Context) -> None:
    corr_sat = [
        "viirs_mean",
        "viirs_lit_frac",
        "viirs_bright_frac",
        "viirs_gini",
        "modis_brightness_mean",
        "modis_ndvi_proxy_mean",
        "modis_dark_frac",
    ]
    corr_econ = [
        "gdp_millions",
        "employment_thousands",
        "unemployment_rate",
        "total_permits",
        "gdp_per_employee",
        "permits_per_1k_emp",
    ]

    viirs_panel = ctx.panel[ctx.panel["viirs_available"].astype(bool)].copy()
    cross_corr = viirs_panel[corr_sat + corr_econ].corr().loc[corr_sat, corr_econ]

    flat = cross_corr.stack().sort_values(key=lambda s: s.abs(), ascending=False)
    strongest_name = flat.index[0]
    strongest_value = flat.iloc[0]
    strongest_text = (
        f"Strongest pooled association: {title_case(strongest_name[0])} vs "
        f"{title_case(strongest_name[1])} (r = {strongest_value:+.2f})."
    )

    fig, ax = plt.subplots(figsize=(11.5, 8.5), constrained_layout=True)
    sns.heatmap(
        cross_corr.rename(index=title_case, columns=title_case),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        annot_kws={"fontsize": 12},
        cbar_kws={"shrink": 0.88, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Cross-Metro Satellite vs Economic Correlations (2017–2023)", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=90, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.text(
        0,
        1.03,
        strongest_text,
        transform=ax.transAxes,
        fontsize=10,
        color="#475569",
        va="bottom",
    )
    save_figure(fig, "06_cross_correlation_heatmap.png")


def make_figure_07_scatter(ctx: Context) -> None:
    viirs_panel = ctx.panel[ctx.panel["viirs_available"].astype(bool)].copy()
    scatter_pairs = [
        ("viirs_mean", "gdp_millions", "VIIRS mean", "GDP ($M)"),
        ("viirs_lit_frac", "employment_thousands", "VIIRS lit-pixel fraction", "Employment (thousands)"),
        ("viirs_bright_frac", "total_permits", "VIIRS bright-pixel fraction", "Building permits"),
        ("modis_brightness_mean", "gdp_per_employee", "MODIS brightness", "GDP per employee ($K)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12), constrained_layout=True)
    axes = axes.ravel()

    for ax, (x_col, y_col, x_label, y_label) in zip(axes, scatter_pairs):
        for metro in ctx.metros:
            sub = viirs_panel[viirs_panel["metro"] == metro]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                color=ctx.metro_colors[metro],
                s=42,
                alpha=0.75,
                edgecolors="white",
                linewidth=0.4,
            )

        valid = viirs_panel[[x_col, y_col]].dropna()
        slope, intercept, r_value, p_value, _ = stats.linregress(valid[x_col], valid[y_col])
        x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 200)
        ax.plot(x_line, slope * x_line + intercept, linestyle="--", color="#6b7280", linewidth=2.0)
        ax.text(
            0.03,
            0.97,
            f"r = {r_value:+.2f}\np = {p_value:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db"),
        )
        ax.set_title(f"{x_label} vs {y_label}", fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if y_col in {"gdp_millions", "employment_thousands", "total_permits"}:
            ax.yaxis.set_major_formatter(FuncFormatter(comma_fmt))

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ctx.metro_colors[metro],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
            label=short_name(metro),
        )
        for metro in ctx.metros
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.045),
    )
    fig.suptitle("Satellite Features vs Economic Indicators", fontweight="bold")
    save_figure(fig, "07_satellite_vs_economic_scatter.png")


def make_figure_08_within_metro_correlations(ctx: Context) -> None:
    viirs_panel = ctx.panel[ctx.panel["viirs_available"].astype(bool)].copy()
    pairs = [
        ("viirs_mean", "gdp_millions", "VIIRS mean vs GDP"),
        ("viirs_mean", "employment_thousands", "VIIRS mean vs employment"),
        ("viirs_lit_frac", "total_permits", "VIIRS lit frac vs permits"),
        ("viirs_mean", "unemployment_rate", "VIIRS mean vs unemployment"),
    ]

    rows = []
    for metro in ctx.metros:
        sub = viirs_panel[viirs_panel["metro"] == metro]
        for sat_col, econ_col, label in pairs:
            valid = sub[[sat_col, econ_col]].dropna()
            if len(valid) < 4:
                rows.append({"metro": metro, "pair": label, "r": np.nan})
                continue
            r_value, _ = stats.pearsonr(valid[sat_col], valid[econ_col])
            rows.append({"metro": metro, "pair": label, "r": r_value})

    display_df = (
        pd.DataFrame(rows)
        .pivot(index="metro", columns="pair", values="r")
        .reindex(index=ctx.metros)
    )

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    sns.heatmap(
        display_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"shrink": 0.84, "label": "Pearson r"},
        xticklabels=[label.replace(" vs ", "\nvs\n") for label in display_df.columns],
        yticklabels=[metro_name(metro) for metro in display_df.index],
        annot_kws={"fontsize": 11},
        ax=ax,
    )
    ax.set_title("Within-Metro Temporal Correlations", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, "08_within_metro_correlations.png")


def make_figure_09_nightlight_change_maps(ctx: Context) -> None:
    ordered = ctx.metros
    n_cols = 7
    n_rows = math.ceil(len(ordered) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(23, 9.75), constrained_layout=True)
    axes = np.atleast_2d(axes)

    changes = {}
    vmax_candidates = []
    for metro in ordered:
        if metro not in ctx.tensors:
            continue
        stack = ctx.tensors[metro]
        years = list(stack.years.astype(int))
        if 2017 not in years or 2023 not in years:
            continue
        start_idx = years.index(2017)
        end_idx = years.index(2023)
        rgb_2023 = stack.tensor[end_idx, :, :, :3]
        bbox = crop_bbox_from_rgb(rgb_2023)
        start = apply_bbox(stack.tensor[start_idx, :, :, 3], bbox)
        end = apply_bbox(stack.tensor[end_idx, :, :, 3], bbox)
        change = end - start
        changes[metro] = change
        vmax_candidates.append(np.nanpercentile(np.abs(change), 99))

    global_vmax = max(0.1, float(np.nanmax(vmax_candidates))) if vmax_candidates else 0.5
    image = None

    for idx, metro in enumerate(ordered):
        ax = axes[idx // n_cols, idx % n_cols]
        if metro not in changes:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"{short_name(metro)}\nTensor stack not available",
                ha="center",
                va="center",
                fontsize=11,
                color="#6b7280",
            )
            continue
        image = ax.imshow(changes[metro], cmap="RdBu_r", vmin=-global_vmax, vmax=global_vmax)
        ax.set_title(short_name(metro), fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    total_axes = n_rows * n_cols
    for idx in range(len(ordered), total_axes):
        axes[idx // n_cols, idx % n_cols].axis("off")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, orientation="horizontal")
        cbar.set_label("2023 minus 2017 (red = brighter, blue = dimmer)")

    fig.suptitle("Night-Light Change from 2017 to 2023", fontweight="bold")
    save_figure(fig, "09_nightlight_change_maps.png")


def make_figure_10_covid_impact(ctx: Context) -> None:
    metros5 = sorted(set(ctx.econ5["metro"]).intersection(set(ctx.viirs_from_tensors["metro"])))
    econ_indexed = ctx.econ5.copy()
    base_2019 = econ_indexed[econ_indexed["year"] == 2019].set_index("metro")
    for column in ["gdp_millions", "employment_thousands", "total_permits"]:
        econ_indexed[column] = econ_indexed.apply(
            lambda row: row[column] / base_2019.loc[row["metro"], column] * 100,
            axis=1,
        )

    viirs_indexed = ctx.viirs_from_tensors.copy()
    viirs_base = viirs_indexed[viirs_indexed["year"] == 2019].set_index("metro")
    viirs_indexed["viirs_mean_tensor"] = viirs_indexed.apply(
        lambda row: row["viirs_mean_tensor"] / viirs_base.loc[row["metro"], "viirs_mean_tensor"] * 100,
        axis=1,
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=True)
    axes = axes.ravel()

    panels = [
        (econ_indexed, "gdp_millions", "GDP (2019 = 100)"),
        (econ_indexed, "employment_thousands", "Employment (2019 = 100)"),
        (econ_indexed, "total_permits", "Building permits (2019 = 100)"),
        (viirs_indexed, "viirs_mean_tensor", "VIIRS mean (2019 = 100)"),
    ]

    for ax, (frame, column, title) in zip(axes, panels):
        for metro in metros5:
            sub = frame[frame["metro"] == metro]
            ax.plot(
                sub["year"],
                sub[column],
                color=ctx.metro_colors[metro],
                linewidth=2.5,
                marker="o",
                markersize=4,
            )
        ax.axhline(100, color="#6b7280", linestyle="--", linewidth=1.3)
        ax.axvspan(2019.5, 2020.5, color="#ef4444", alpha=0.12)
        ax.set_title(title, fontweight="bold", fontsize=13.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Index")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=10.5)

    handles = [
        Line2D([0], [0], color=ctx.metro_colors[metro], lw=3, label=metro_name(metro))
        for metro in metros5
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(metros5), frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("COVID-19 Structural Break Comparison", fontweight="bold")
    save_figure(fig, "10_covid_impact_indexed.png")


def cagr(start_value: float, end_value: float, n_years: int) -> float:
    if pd.isna(start_value) or pd.isna(end_value) or start_value <= 0:
        return np.nan
    return ((end_value / start_value) ** (1 / n_years) - 1) * 100


def make_figure_11_growth_rates(ctx: Context) -> None:
    rows = []
    for metro in ctx.metros:
        sub = ctx.panel[ctx.panel["metro"] == metro].set_index("year")
        row = {"metro": metro_name(metro)}
        row["GDP\n2013-2023"] = cagr(sub.loc[2013, "gdp_millions"], sub.loc[2023, "gdp_millions"], 10)
        row["Employment\n2013-2023"] = cagr(
            sub.loc[2013, "employment_thousands"],
            sub.loc[2023, "employment_thousands"],
            10,
        )
        row["Permits\n2013-2023"] = cagr(sub.loc[2013, "total_permits"], sub.loc[2023, "total_permits"], 10)

        viirs_sub = ctx.viirs_from_tensors[ctx.viirs_from_tensors["metro"] == metro].set_index("year")
        if {2017, 2023}.issubset(set(viirs_sub.index)):
            row["VIIRS\n2017-2023"] = cagr(
                viirs_sub.loc[2017, "viirs_mean_tensor"],
                viirs_sub.loc[2023, "viirs_mean_tensor"],
                6,
            )
        else:
            row["VIIRS\n2017-2023"] = np.nan
        rows.append(row)

    growth_df = pd.DataFrame(rows).set_index("metro")

    fig, ax = plt.subplots(figsize=(9.5, 10.5), constrained_layout=True)
    sns.heatmap(
        growth_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "CAGR (%)", "shrink": 0.82},
        annot_kws={"fontsize": 11},
        ax=ax,
    )
    ax.set_title("Compound Annual Growth Rates by Metro", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, "11_growth_rates_cagr.png")


def make_figure_12_pixel_distributions(ctx: Context) -> None:
    sample_metro = "austin"
    if sample_metro not in ctx.tensors:
        return

    stack = ctx.tensors[sample_metro]
    year_to_idx = {int(year): idx for idx, year in enumerate(stack.years)}
    compare_years = [2017, 2020, 2023]
    channels = [
        (0, "MODIS Red", "#ef4444"),
        (1, "MODIS Green", "#22c55e"),
        (2, "MODIS Blue", "#3b82f6"),
        (3, "VIIRS Night", "#f59e0b"),
    ]
    year_colors = {2017: "#0f766e", 2020: "#7c3aed", 2023: "#dc2626"}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)

    for ax, (channel_idx, title, base_color) in zip(axes, channels):
        for year in compare_years:
            idx = year_to_idx[year]
            values = stack.tensor[idx, :, :, channel_idx].ravel()
            values = values[np.isfinite(values)]
            zero_share = (values == 0).mean() * 100
            ax.hist(
                values,
                bins=80,
                density=True,
                histtype="step",
                linewidth=2.2,
                color=year_colors[year],
                alpha=0.95,
                label=f"{year}  (zeros {zero_share:.0f}%)",
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Pixel value")
        if ax is axes[0]:
            ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=10, loc="upper right")

    fig.suptitle(f"Pixel Value Distributions — {metro_name(sample_metro)}", fontweight="bold")
    save_figure(fig, "12_pixel_distributions.png")


def make_figure_13_pairplot(ctx: Context) -> None:
    pair_display = {
        "viirs_mean": "VIIRS mean",
        "modis_brightness_mean": "MODIS brightness",
        "gdp_millions": "GDP ($M)",
        "total_permits": "Permits",
    }
    pair_cols = list(pair_display)
    pair_data = (
        ctx.panel.loc[ctx.panel["viirs_available"].astype(bool), ["metro"] + pair_cols]
        .dropna()
        .copy()
    )
    pair_data["Metro"] = pair_data["metro"].map(short_name)
    pair_data = pair_data.drop(columns="metro").rename(columns=pair_display)

    palette = {short_name(metro): ctx.metro_colors[metro] for metro in ctx.metros}
    grid = sns.PairGrid(pair_data, hue="Metro", palette=palette, corner=True, height=2.55)
    grid.map_lower(sns.scatterplot, s=34, alpha=0.72, edgecolor="white", linewidth=0.4)
    grid.map_diag(sns.histplot, bins=16, element="step", fill=False, linewidth=1.7)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ctx.metro_colors[metro],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7.5,
            label=short_name(metro),
        )
        for metro in ctx.metros
    ]
    legend = grid.figure.legend(
        handles=legend_handles,
        title="Metro",
        loc="upper left",
        bbox_to_anchor=(0.69, 0.80),
        frameon=False,
        ncol=2,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    if legend is not None and legend._legend_box is not None:
        legend._legend_box.align = "left"
    for ax in grid.figure.axes:
        ax.tick_params(axis="both", labelsize=9.5)
    grid.figure.suptitle("Key Feature Pairplot", fontweight="bold", y=0.98)
    save_figure(grid.figure, "13_pairplot_key_features.png")


def make_figure_14_split(ctx: Context) -> None:
    split_colors = {"train": "#22c55e", "val": "#f59e0b", "test": "#ef4444"}
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)

    ordered = list(reversed(ctx.metros))
    for metro_idx, metro in enumerate(ordered):
        sub = ctx.panel[ctx.panel["metro"] == metro]
        for _, row in sub.iterrows():
            ax.barh(
                metro_idx,
                0.8,
                left=row["year"] - 0.4,
                color=split_colors[row["split"]],
                edgecolor="white",
                linewidth=0.8,
            )

        ax.barh(
            metro_idx,
            width=0.8,
            left=2019.6,
            height=0.8,
            color="#e5e7eb",
            edgecolor="white",
            hatch="//",
            linewidth=0.8,
        )

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([metro_name(metro) for metro in ordered])
    ax.set_xlabel("Year")
    ax.set_title("Temporal Train / Validation / Test Split", fontweight="bold")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    handles = [
        Patch(facecolor=split_colors["train"], label="Train (6 yrs)"),
        Patch(facecolor=split_colors["val"], label="Val (1 yr)"),
        Patch(facecolor=split_colors["test"], label="Test (3 yrs)"),
        Patch(facecolor="#e5e7eb", hatch="//", label="Excluded (COVID)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)
    save_figure(fig, "14_train_val_test_split.png")


def main() -> None:
    set_theme()
    ctx = load_context()

    make_figure_01_satellite_imagery_grid(ctx)
    make_figure_02_economic_timeseries(ctx)
    make_missingness_figure(ctx)
    make_figure_04_feature_distributions(ctx)
    make_figure_05_satellite_feature_trends(ctx)
    make_figure_06_cross_correlation_heatmap(ctx)
    make_figure_07_scatter(ctx)
    make_figure_08_within_metro_correlations(ctx)
    make_figure_09_nightlight_change_maps(ctx)
    make_figure_10_covid_impact(ctx)
    make_figure_11_growth_rates(ctx)
    make_figure_12_pixel_distributions(ctx)
    make_figure_13_pairplot(ctx)
    make_figure_14_split(ctx)
    print("All audited visual assets regenerated.")


if __name__ == "__main__":
    main()
