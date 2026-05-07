#!/usr/bin/env python3
"""Build clean summary visuals for Minh's final-model notebook.

These figures provide a compact, presentation-ready layer on top of the
notebook's native diagnostics. They are rebuilt from the executed notebook
outputs plus the Stage 4 CSV summaries written during notebook execution.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Minh_Final_Model_Pipeline_Cleaned.ipynb"
FIG_DIR = ROOT / "figures"
DELIVERABLES_DIR = ROOT / "deliverables"

SNS_RC = {
    "figure.dpi": 160,
    "savefig.dpi": 180,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def iter_stream_texts(nb: dict):
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text = out.get("text", "")
                yield "".join(text) if isinstance(text, list) else text


def stream_text_matching(nb: dict, marker: str) -> str:
    for text in iter_stream_texts(nb):
        if marker in text:
            return text
    raise ValueError(f"Could not find stream output containing marker: {marker}")


def savefig(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_stage1_city_chart(nb: dict) -> None:
    text = stream_text_matching(nb, "Per-city segmentation results on 2020 holdout:")
    pattern = re.compile(r"^\s*([a-z_]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", re.MULTILINE)
    rows = []
    for metro, iou, dice in pattern.findall(text):
        rows.append(
            {
                "metro": metro.replace("_", " ").title(),
                "IoU": float(iou),
                "Dice": float(dice),
            }
        )
    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("IoU", ascending=True)
    sns.set_theme(style="whitegrid", rc=SNS_RC)
    fig, ax = plt.subplots(figsize=(8.8, 6.3))
    ax.barh(df["metro"], df["IoU"], color="#0f766e", edgecolor="white", height=0.72)
    for y, v in enumerate(df["IoU"]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", ha="left", fontsize=9)
    ax.set_xlim(0, max(0.75, df["IoU"].max() + 0.07))
    ax.set_xlabel("IoU on 2020 GHSL holdout")
    ax.set_ylabel("")
    ax.set_title("Stage 1 Summary: Per-City Segmentation Performance", fontweight="bold")
    fig.subplots_adjust(bottom=0.16)
    fig.text(
        0.5,
        0.03,
        "Higher is better. Built-up signal is real across the 14-city holdout, but performance varies by metro.",
        fontsize=9.5,
        color="#334155",
        ha="center",
    )
    sns.despine(ax=ax)
    savefig(fig, "minh_stage1_segmentation_summary.png")


def build_stage2_model_chart(nb: dict) -> None:
    text = stream_text_matching(nb, "Test reconstruction MSE (2021-2023 holdout)")
    pattern = re.compile(r"^\s*(MLP|GRU|LSTM) autoencoder\s*:\s*([0-9.]+)", re.MULTILINE)
    rows = [(name, float(val)) for name, val in pattern.findall(text)]
    if not rows:
        return

    df = pd.DataFrame(rows, columns=["Model", "Test MSE"]).sort_values("Test MSE", ascending=False)
    palette = {"MLP": "#0f766e", "GRU": "#94a3b8", "LSTM": "#cbd5e1"}

    sns.set_theme(style="whitegrid", rc=SNS_RC)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh(df["Model"], df["Test MSE"], color=[palette[m] for m in df["Model"]], edgecolor="white")
    for y, v in enumerate(df["Test MSE"]):
        ax.text(v + 0.03, y, f"{v:.3f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("2021-2023 reconstruction MSE")
    ax.set_ylabel("")
    ax.set_title("Stage 2 Summary: Economic Encoder Comparison", fontweight="bold")
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5,
        0.04,
        "Lower is better. The static MLP encoder clearly outperforms GRU and LSTM on the post-COVID holdout.",
        fontsize=9.5,
        color="#334155",
        ha="center",
    )
    sns.despine(ax=ax)
    savefig(fig, "minh_stage2_autoencoder_comparison.png")


def build_stage3_validation_chart(nb: dict) -> None:
    text = stream_text_matching(nb, "Nowcasting evaluation (2019 validation)")
    pattern = re.compile(
        r"^\s*(Per-city mean baseline|Joint z \(both modalities\)|Image-only nowcast)\s*:\s*([0-9.]+)",
        re.MULTILINE,
    )
    label_map = {
        "Per-city mean baseline": "Per-city mean",
        "Joint z (both modalities)": "Joint latent decode",
        "Image-only nowcast": "Image-only decode",
    }
    rows = [{"Method": label_map[label], "Validation MSE": float(value)} for label, value in pattern.findall(text)]
    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("Validation MSE", ascending=False)
    colors = {
        "Per-city mean": "#64748b",
        "Joint latent decode": "#7c3aed",
        "Image-only decode": "#dc2626",
    }

    sns.set_theme(style="whitegrid", rc=SNS_RC)
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ax.barh(df["Method"], df["Validation MSE"], color=[colors[m] for m in df["Method"]], edgecolor="white")
    for y, v in enumerate(df["Validation MSE"]):
        ax.text(v + 0.02, y, f"{v:.3f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("2019 validation MSE")
    ax.set_ylabel("")
    ax.set_title("Stage 3 Summary: Direct Nowcasting Check", fontweight="bold")
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5,
        0.04,
        "Lower is better. The per-city mean baseline still wins, so the decoder should not be the main claim.",
        fontsize=9.5,
        color="#334155",
        ha="center",
    )
    sns.despine(ax=ax)
    savefig(fig, "minh_stage3_nowcast_validation.png")


def build_stage3_reconstruction_chart(nb: dict) -> None:
    text = stream_text_matching(nb, "Test reconstruction MSE vs per-city mean baseline (2021-2023)")
    pattern = re.compile(
        r"^\s*(Image|Econ)\s+VAE=([0-9.]+)\s+per-city=([0-9.]+)\s+improvement=([+-]?[0-9.]+)%",
        re.MULTILINE,
    )
    rows = []
    for label, vae, baseline, improvement in pattern.findall(text):
        rows.append(
            {
                "Modality": "Image embeddings" if label == "Image" else "Economic embeddings",
                "VAE reconstruction": float(vae),
                "Per-city mean baseline": float(baseline),
                "Improvement (%)": float(improvement),
            }
        )
    if not rows:
        return

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.34

    sns.set_theme(style="whitegrid", rc=SNS_RC)
    fig, ax = plt.subplots(figsize=(8.1, 4.9))
    ax.bar(
        x - width / 2,
        df["Per-city mean baseline"],
        width,
        label="Per-city mean baseline",
        color="#cbd5e1",
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        df["VAE reconstruction"],
        width,
        label="VAE reconstruction",
        color="#dc2626",
        edgecolor="white",
    )

    for xpos, val in zip(x - width / 2, df["Per-city mean baseline"]):
        ax.text(xpos, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    group_max = np.maximum(df["VAE reconstruction"].to_numpy(), df["Per-city mean baseline"].to_numpy())
    for xpos, val in zip(
        x + width / 2,
        df["VAE reconstruction"],
    ):
        ax.text(xpos, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Modality"])
    ax.set_ylabel("2021-2023 test reconstruction MSE")
    ax.set_title("Stage 3 Summary: Held-Out Reconstruction Check", fontweight="bold")
    ax.set_ylim(0, float(group_max.max()) + 0.55)
    ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(bottom=0.20)
    fig.text(
        0.5,
        0.04,
        "Lower is better. On the 14-city rerun, the per-city baseline still outperforms the VAE decoder on both modalities.",
        fontsize=9.5,
        color="#334155",
        ha="center",
    )
    sns.despine(ax=ax)
    savefig(fig, "minh_stage3_reconstruction_check.png")


def build_stage4_tuning_chart() -> None:
    tuning_path = DELIVERABLES_DIR / "minh_stage4_retrieval_tuning.csv"
    if not tuning_path.exists():
        return

    df = pd.read_csv(tuning_path).sort_values(["val_mae", "test_mae"]).head(6).copy()
    df["Display"] = df["model"] + df["test_mae"].map(lambda v: f"  |  test {v:.3f}")

    sns.set_theme(style="whitegrid", rc=SNS_RC)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    colors = ["#0f766e" if "Scaled Manhattan k=3" in model else "#cbd5e1" for model in df["model"]]
    ax.barh(df["Display"], df["val_mae"], color=colors, edgecolor="white")
    for y, v in enumerate(df["val_mae"]):
        ax.text(v + 0.015, y, f"{v:.3f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("2019 validation MAE on GDP growth")
    ax.set_ylabel("")
    ax.set_title("Stage 4 Summary: Compact Retrieval Tuning Screen", fontweight="bold")
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5,
        0.04,
        "Lower is better. A small validation search selects Scaled Manhattan k=3 without adding a more complex model.",
        fontsize=9.5,
        color="#334155",
        ha="center",
    )
    sns.despine(ax=ax)
    savefig(fig, "minh_stage4_tuning_summary.png")


def build_stage4_retrieval_chart() -> None:
    benchmark_path = DELIVERABLES_DIR / "minh_stage4_benchmark.csv"
    if not benchmark_path.exists():
        return

    df = pd.read_csv(benchmark_path)
    df = df.iloc[::-1].reset_index(drop=True)

    colors = []
    for method in df["Method"]:
        if method == "Scaled Manhattan k=3 (selected)":
            colors.append("#0f766e")
        elif str(method).startswith("Cosine"):
            colors.append("#94a3b8")
        else:
            colors.append("#cbd5e1")

    sns.set_theme(style="whitegrid", rc=SNS_RC)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(df["Method"], df["test_mae"], color=colors, edgecolor="white")
    for y, v in enumerate(df["test_mae"]):
        ax.text(v + 0.03, y, f"{v:.3f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("2021-2023 test MAE on GDP growth")
    ax.set_ylabel("")
    ax.set_title("Stage 4 Summary: Official GDP-Growth Benchmark", fontweight="bold")
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5,
        0.04,
        "Lower is better. The selected scale-aware retrieval rule is modestly stronger than the plain-cosine baseline.",
        fontsize=9.5,
        color="#334155",
        ha="center",
    )
    sns.despine(ax=ax)
    savefig(fig, "minh_stage4_gdp_retrieval.png")


def build_stage4_baseline_progress_chart() -> None:
    progress_path = DELIVERABLES_DIR / "minh_baseline_progress.csv"
    if not progress_path.exists():
        return

    df = pd.read_csv(progress_path)

    sns.set_theme(style="whitegrid", rc=SNS_RC)
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.axis("off")

    table_rows = []
    for _, row in df.iterrows():
        table_rows.append(
            [
                textwrap.fill(str(row["Dimension"]), width=24),
                textwrap.fill(str(row["MS3 baseline notebook"]), width=38),
                textwrap.fill(str(row["Final modeling notebook"]), width=42),
            ]
        )
    table = ax.table(
        cellText=table_rows,
        colLabels=["Dimension", "MS3 baseline notebook", "Final modeling notebook"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.20, 0.36, 0.44],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    table.scale(1, 2.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#0f172a")
        elif col == 2:
            cell.set_facecolor("#ecfdf5")
        else:
            cell.set_facecolor("white")

    fig.suptitle("Stage 4 Summary: Improvement Over the MS3 Baseline", fontsize=18, fontweight="bold", y=0.96)
    fig.text(
        0.5,
        0.05,
        "The final notebook moves from lagged tabular forecasting to image-only economic analogue retrieval.",
        fontsize=10.5,
        color="#334155",
        ha="center",
    )
    savefig(fig, "minh_stage4_baseline_progress.png")


def main() -> None:
    nb = load_notebook(NOTEBOOK)
    build_stage1_city_chart(nb)
    build_stage2_model_chart(nb)
    build_stage3_validation_chart(nb)
    build_stage3_reconstruction_chart(nb)
    build_stage4_tuning_chart()
    build_stage4_retrieval_chart()
    build_stage4_baseline_progress_chart()
    print("Wrote Minh pipeline visual summaries to figures/")


if __name__ == "__main__":
    main()
