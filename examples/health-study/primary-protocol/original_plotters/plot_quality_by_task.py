#!/usr/bin/env python3
"""Regenerate Figure 3 for the manuscript.

The plot uses the primary, non-pseudo deployment-shift runs so that the
distribution matches the main manuscript tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "exp/tasks/v3_benchmark/results/public_benchmark_v3_conformal_results.csv"
TARGET_COVERAGE = 0.90
PRIMARY_TARGET_FRACTION = 0.10
ZOOM_YLIM = (0.89, 0.91)
ZOOM_YTICKS = [0.89, 0.90, 0.91]
CALLOUT_YLIM = (0.865, 0.945)
CALLOUT_WIDTH = 0.90

COLORS = ["#0072B2", "#D55E00", "#009E73"]


TEXT = {
    "en": {
        "title": "Coverage across deployment-shift benchmark",
        "ylabel": "Empirical coverage",
        "legend": "Dataset",
        "random_zoom": "Random source zoom",
        "target_zoom": "Shifted target 10% zoom",
        "conditions": {
            "random": "Random source",
            "shift_source": "Shifted source",
            "shift_target": "Shifted target 10%",
        },
        "datasets": {
            "brfss_multiyear_diabetes": "BRFSS diabetes",
            "nhanes_multicycle_diabetes": "NHANES diabetes",
            "uci_diabetes_readmission": "UCI readmission",
        },
        "dataset_short": {
            "brfss_multiyear_diabetes": "BRFSS",
            "nhanes_multicycle_diabetes": "NHANES",
            "uci_diabetes_readmission": "UCI",
        },
    },
    "cn": {
        "title": "部署偏移基准中的覆盖率",
        "ylabel": "经验覆盖率",
        "legend": "数据集",
        "random_zoom": "随机源域放大",
        "target_zoom": "偏移目标域 10% 放大",
        "conditions": {
            "random": "随机源域",
            "shift_source": "偏移源域",
            "shift_target": "偏移目标域 10%",
        },
        "datasets": {
            "brfss_multiyear_diabetes": "BRFSS 糖尿病",
            "nhanes_multicycle_diabetes": "NHANES 糖尿病",
            "uci_diabetes_readmission": "UCI 再入院",
        },
        "dataset_short": {
            "brfss_multiyear_diabetes": "BRFSS",
            "nhanes_multicycle_diabetes": "NHANES",
            "uci_diabetes_readmission": "UCI",
        },
    },
}


def configure_style(language: str) -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.titlesize": 16,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if language == "cn":
        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/mnt/c/Windows/Fonts/msyh.ttc",
            "/mnt/c/Windows/Fonts/simhei.ttf",
        ]
        for font_path in font_candidates:
            path = Path(font_path)
            if path.exists():
                font_manager.fontManager.addfont(str(path))
                font_name = font_manager.FontProperties(fname=str(path)).get_name()
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [
                    font_name,
                    "Noto Sans CJK SC",
                    "Microsoft YaHei",
                    "SimHei",
                    "DejaVu Sans",
                ]
                break


def load_plot_data(language: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    primary = df[(df["experiment"] == "primary") & (df["ablation"] == "none")].copy()
    non_pseudo = primary["shift_name"] != "pseudo_random"

    masks = {
        "random": (
            (primary["setting"] == "random_exchangeable")
            & (primary["workflow"] == "source_split")
            & np.isclose(primary["target_cal_fraction"], 0.0)
        ),
        "shift_source": (
            (primary["setting"] == "covariate_shift")
            & non_pseudo
            & (primary["workflow"] == "source_split")
            & np.isclose(primary["target_cal_fraction"], 0.0)
        ),
        "shift_target": (
            (primary["setting"] == "covariate_shift")
            & non_pseudo
            & (primary["workflow"] == "target_recalibrated_split")
            & np.isclose(primary["target_cal_fraction"], PRIMARY_TARGET_FRACTION)
        ),
    }

    parts = []
    labels = TEXT[language]
    for key, mask in masks.items():
        part = primary.loc[mask, ["dataset", "coverage"]].copy()
        part["condition"] = labels["conditions"][key]
        parts.append(part)

    plot_df = pd.concat(parts, ignore_index=True)
    plot_df["dataset_label"] = plot_df["dataset"].map(labels["datasets"])
    plot_df["dataset_short"] = plot_df["dataset"].map(labels["dataset_short"])
    return plot_df


def unique_legend(handles: list, labels: list[str]) -> tuple[list, list[str]]:
    seen = set()
    out_handles = []
    out_labels = []
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in seen:
            seen.add(label)
            out_handles.append(handle)
            out_labels.append(label)
    return out_handles, out_labels


def draw_panel(ax, data: pd.DataFrame, condition_order: list[str], hue_order: list[str], labels: dict) -> tuple[list, list[str]]:
    palette = dict(zip(hue_order, COLORS, strict=True))
    sns.boxplot(
        data=data,
        x="condition",
        y="coverage",
        hue="dataset_label",
        order=condition_order,
        hue_order=hue_order,
        palette=palette,
        width=0.58,
        linewidth=1.2,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="condition",
        y="coverage",
        hue="dataset_label",
        order=condition_order,
        hue_order=hue_order,
        palette=palette,
        dodge=True,
        alpha=0.58,
        jitter=0.18,
        size=3.6,
        linewidth=0.25,
        edgecolor="white",
        ax=ax,
    )
    handles, legend_labels = unique_legend(*ax.get_legend_handles_labels())
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.axhline(TARGET_COVERAGE, color="black", ls="--", lw=1.0, zorder=1)
    ax.set_xlabel("")
    ax.set_ylabel(labels["ylabel"])
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0.30, 1.02)
    ax.set_yticks(np.arange(0.30, 1.01, 0.10))
    ax.set_yticks(np.arange(0.35, 1.01, 0.05), minor=True)
    ax.grid(axis="y", which="major", color="#d9d9d9", linewidth=0.7, alpha=0.65)
    ax.grid(axis="y", which="minor", color="#eeeeee", linewidth=0.5, alpha=0.55)
    sns.despine(ax=ax)
    return handles, legend_labels


def draw_zoom_inset(
    ax,
    data: pd.DataFrame,
    condition: str,
    labels: dict,
    title: str,
    ylim: tuple[float, float],
    yticks: list[float],
) -> None:
    subset = data[data["condition"] == condition].copy()
    short_order = list(labels["dataset_short"].values())
    short_palette = dict(zip(short_order, COLORS, strict=True))

    sns.boxplot(
        data=subset,
        x="dataset_short",
        y="coverage",
        hue="dataset_short",
        order=short_order,
        hue_order=short_order,
        palette=short_palette,
        width=0.52,
        linewidth=0.9,
        showfliers=False,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=subset,
        x="dataset_short",
        y="coverage",
        hue="dataset_short",
        order=short_order,
        hue_order=short_order,
        palette=short_palette,
        alpha=0.64,
        jitter=0.12,
        size=2.4,
        linewidth=0.2,
        edgecolor="white",
        legend=False,
        ax=ax,
    )
    ax.axhline(TARGET_COVERAGE, color="black", ls="--", lw=0.8, zorder=1)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def add_callout(
    fig,
    ax,
    inset_ax,
    x_center: float,
    y_min: float,
    y_max: float,
    width: float,
    connector_start_offset: float,
    connector_end: tuple[float, float],
) -> None:
    rect = Rectangle(
        (x_center - width / 2, y_min),
        width,
        y_max - y_min,
        fill=False,
        edgecolor="#404040",
        linewidth=1.15,
        linestyle="-",
        zorder=8,
    )
    ax.add_patch(rect)
    connector = ConnectionPatch(
        xyA=(x_center + connector_start_offset * width, y_max),
        coordsA=ax.transData,
        xyB=connector_end,
        coordsB=inset_ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=11,
        color="#4a4a4a",
        linewidth=1.0,
        shrinkA=3,
        shrinkB=4,
        zorder=7,
    )
    fig.add_artist(connector)


def render(language: str, output: Path) -> None:
    configure_style(language)
    labels = TEXT[language]
    plot_df = load_plot_data(language)
    condition_order = [
        labels["conditions"]["random"],
        labels["conditions"]["shift_source"],
        labels["conditions"]["shift_target"],
    ]
    hue_order = list(labels["datasets"].values())

    fig = plt.figure(figsize=(8.2, 6.9))
    ax = fig.add_axes([0.10, 0.22, 0.86, 0.55])
    handles, legend_labels = draw_panel(ax, plot_df, condition_order, hue_order, labels)

    random_inset = fig.add_axes([0.18, 0.79, 0.28, 0.13])
    target_inset = fig.add_axes([0.58, 0.79, 0.28, 0.13])
    draw_zoom_inset(
        random_inset,
        plot_df,
        labels["conditions"]["random"],
        labels,
        labels["random_zoom"],
        ZOOM_YLIM,
        ZOOM_YTICKS,
    )
    draw_zoom_inset(
        target_inset,
        plot_df,
        labels["conditions"]["shift_target"],
        labels,
        labels["target_zoom"],
        ZOOM_YLIM,
        ZOOM_YTICKS,
    )
    add_callout(
        fig,
        ax,
        random_inset,
        x_center=0,
        y_min=CALLOUT_YLIM[0],
        y_max=CALLOUT_YLIM[1],
        width=CALLOUT_WIDTH,
        connector_start_offset=-0.38,
        connector_end=(0.03, 0.03),
    )
    add_callout(
        fig,
        ax,
        target_inset,
        x_center=2,
        y_min=CALLOUT_YLIM[0],
        y_max=CALLOUT_YLIM[1],
        width=CALLOUT_WIDTH,
        connector_start_offset=0.38,
        connector_end=(0.97, 0.03),
    )

    fig.suptitle(labels["title"], y=0.98)
    fig.legend(
        handles,
        legend_labels,
        title=labels["legend"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    targets = [
        ("en", ROOT / "writing/paper/figures/quality_by_task.png"),
        ("cn", ROOT / "writing/paper_cn/figures/quality_by_task.png"),
        ("en", ROOT / "writing/paper_cn/figures/quality_by_task.en.png"),
        ("en", ROOT / "benchmark_release/public_health_shift_conformal_benchmark_v3/figures/quality_by_task.png"),
    ]
    for language, output in targets:
        render(language, output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
