#!/usr/bin/env python3
"""Regenerate manuscript empirical figures from archived benchmark outputs.

This script intentionally regenerates only data-driven result figures. The
graphical abstract and workflow schematic in the manuscript are explanatory
artwork and are not reproduced here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
TARGET_COVERAGE = 0.90
PRIMARY_TARGET_FRACTION = 0.10
ZOOM_YLIM = (0.89, 0.91)
ZOOM_YTICKS = [0.89, 0.90, 0.91]
CALLOUT_YLIM = (0.865, 0.945)
CALLOUT_WIDTH = 0.90

COLORS = ["#0072B2", "#D55E00", "#009E73"]
DATASET_LABELS = {
    "brfss_multiyear_diabetes": "BRFSS diabetes",
    "nhanes_multicycle_diabetes": "NHANES diabetes",
    "uci_diabetes_readmission": "UCI readmission",
}
DATASET_SHORT = {
    "brfss_multiyear_diabetes": "BRFSS",
    "nhanes_multicycle_diabetes": "NHANES",
    "uci_diabetes_readmission": "UCI",
}


def configure_style() -> None:
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_coverage_plot_data(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    primary = df[(df["experiment"] == "primary") & (df["ablation"] == "none")].copy()
    non_pseudo = primary["shift_name"] != "pseudo_random"
    masks = {
        "Random source": (
            (primary["setting"] == "random_exchangeable")
            & (primary["workflow"] == "source_split")
            & np.isclose(primary["target_cal_fraction"], 0.0)
        ),
        "Shifted source": (
            (primary["setting"] == "covariate_shift")
            & non_pseudo
            & (primary["workflow"] == "source_split")
            & np.isclose(primary["target_cal_fraction"], 0.0)
        ),
        "Shifted target 10%": (
            (primary["setting"] == "covariate_shift")
            & non_pseudo
            & (primary["workflow"] == "target_recalibrated_split")
            & np.isclose(primary["target_cal_fraction"], PRIMARY_TARGET_FRACTION)
        ),
    }
    parts = []
    for condition, mask in masks.items():
        part = primary.loc[mask, ["dataset", "coverage"]].copy()
        part["condition"] = condition
        parts.append(part)
    plot_df = pd.concat(parts, ignore_index=True)
    plot_df["dataset_label"] = plot_df["dataset"].map(DATASET_LABELS)
    plot_df["dataset_short"] = plot_df["dataset"].map(DATASET_SHORT)
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


def draw_coverage_panel(ax, data: pd.DataFrame) -> tuple[list, list[str]]:
    condition_order = ["Random source", "Shifted source", "Shifted target 10%"]
    hue_order = list(DATASET_LABELS.values())
    palette = dict(zip(hue_order, COLORS))
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
    ax.set_ylabel("Empirical coverage")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0.30, 1.02)
    ax.set_yticks(np.arange(0.30, 1.01, 0.10))
    ax.set_yticks(np.arange(0.35, 1.01, 0.05), minor=True)
    ax.grid(axis="y", which="major", color="#d9d9d9", linewidth=0.7, alpha=0.65)
    ax.grid(axis="y", which="minor", color="#eeeeee", linewidth=0.5, alpha=0.55)
    sns.despine(ax=ax)
    return handles, legend_labels


def draw_zoom_inset(ax, data: pd.DataFrame, condition: str, title: str) -> None:
    subset = data[data["condition"] == condition].copy()
    short_order = list(DATASET_SHORT.values())
    short_palette = dict(zip(short_order, COLORS))
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
    ax.set_ylim(*ZOOM_YLIM)
    ax.set_yticks(ZOOM_YTICKS)
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def add_callout(
    fig: plt.Figure,
    ax,
    inset_ax,
    x_center: float,
    connector_start_offset: float,
    connector_end: tuple[float, float],
) -> None:
    rect = Rectangle(
        (x_center - CALLOUT_WIDTH / 2, CALLOUT_YLIM[0]),
        CALLOUT_WIDTH,
        CALLOUT_YLIM[1] - CALLOUT_YLIM[0],
        fill=False,
        edgecolor="#404040",
        linewidth=1.15,
        linestyle="-",
        zorder=8,
    )
    ax.add_patch(rect)
    connector = ConnectionPatch(
        xyA=(x_center + connector_start_offset * CALLOUT_WIDTH, CALLOUT_YLIM[1]),
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


def plot_quality_by_task(results_path: Path, out_dir: Path) -> None:
    plot_df = load_coverage_plot_data(results_path)
    fig = plt.figure(figsize=(8.2, 6.9))
    ax = fig.add_axes([0.10, 0.22, 0.86, 0.55])
    handles, legend_labels = draw_coverage_panel(ax, plot_df)

    random_inset = fig.add_axes([0.18, 0.79, 0.28, 0.13])
    target_inset = fig.add_axes([0.58, 0.79, 0.28, 0.13])
    draw_zoom_inset(random_inset, plot_df, "Random source", "Random source zoom")
    draw_zoom_inset(target_inset, plot_df, "Shifted target 10%", "Shifted target 10% zoom")
    add_callout(fig, ax, random_inset, x_center=0, connector_start_offset=-0.38, connector_end=(0.03, 0.03))
    add_callout(fig, ax, target_inset, x_center=2, connector_start_offset=0.38, connector_end=(0.97, 0.03))

    fig.suptitle("Coverage across deployment-shift benchmark", y=0.98)
    fig.legend(
        handles,
        legend_labels,
        title="Dataset",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    fig.savefig(out_dir / "quality_by_task.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_threshold_mismatch(run_level_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(run_level_path)
    df["dataset_label"] = df["dataset"].map(
        {
            "brfss_multiyear_diabetes": "BRFSS",
            "nhanes_multicycle_diabetes": "NHANES",
            "uci_diabetes_readmission": "UCI",
        }
    )
    palette = {"BRFSS": COLORS[0], "NHANES": COLORS[1], "UCI": COLORS[2]}
    markers = {"BRFSS": "o", "NHANES": "X", "UCI": "s"}
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    sns.regplot(
        data=df,
        x="oracle_minus_source_qhat",
        y="source_shortfall",
        scatter=False,
        color="black",
        line_kws={"linewidth": 1.4},
        ax=ax,
    )
    sns.scatterplot(
        data=df,
        x="oracle_minus_source_qhat",
        y="source_shortfall",
        hue="dataset_label",
        style="dataset_label",
        palette=palette,
        markers=markers,
        s=55,
        edgecolor="white",
        linewidth=0.35,
        ax=ax,
    )
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_title("Score-threshold mismatch tracks undercoverage")
    ax.set_xlabel("Target oracle q90 - source qhat")
    ax.set_ylabel("Source split coverage shortfall")
    ax.legend(title="Dataset", loc="upper left", frameon=True)
    ax.grid(color="#d0d0d0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "score_threshold_mismatch.png", dpi=300)
    plt.close(fig)


def plot_mechanism_diagnostics(budget_path: Path, out_dir: Path) -> None:
    budget = pd.read_csv(budget_path)
    label_map = {
        "target_recalibrated_split": "Target split",
        "target_recalibrated_mondrian": "Mondrian target",
    }
    budget = budget[budget["workflow"].isin(label_map)].copy()
    budget["workflow_label"] = budget["workflow"].map(label_map)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.lineplot(
        data=budget,
        x="target_cal_fraction",
        y="coverage_shortfall",
        hue="workflow_label",
        style="workflow_label",
        markers=True,
        dashes=False,
        palette=["#009E73", "#CC79A7"],
        linewidth=1.8,
        ax=ax,
    )
    ax.set_title("Dense calibration-budget curve")
    ax.set_xlabel("Target-domain labeled calibration fraction")
    ax.set_ylabel("Coverage shortfall")
    ax.set_ylim(bottom=0)
    ax.legend(title="Workflow", loc="upper right", frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "mechanism_diagnostics.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures", help="Directory for regenerated PNG files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_quality_by_task(ROOT / "results/public_benchmark_conformal_results.csv", args.out_dir)
    plot_score_threshold_mismatch(ROOT / "tables/mechanism_score_diagnostics_run_level.csv", args.out_dir)
    plot_mechanism_diagnostics(ROOT / "tables/target_calibration_budget_table.csv", args.out_dir)
    print(f"Wrote manuscript empirical figures to {args.out_dir}")


if __name__ == "__main__":
    main()
