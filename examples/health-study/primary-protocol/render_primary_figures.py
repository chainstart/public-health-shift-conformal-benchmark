"""Render archived primary-protocol observations with recovered plotting functions.

The archived source files are unmodified. This adapter only supplies corrected
data, fixes the RNG for categorical jitter, restores Figure 3's submitted palette,
and replaces an inappropriate iid confidence band by condition-omission sensitivity.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(output):
    output.mkdir(parents=True, exist_ok=True)
    originals = ROOT / "original_plotters"
    f2 = load_module("original_coverage", originals / "plot_quality_by_task.py")
    f3 = load_module("original_empirical", originals / "create_manuscript_figures.py")
    inputs = [ROOT / "plot-inputs/restored_original_budgets/budget_metrics.csv",
              ROOT / "plot-inputs/restored_random_controls/control_metrics.csv",
              ROOT / "plot-inputs/restored_threshold_diagnostic/threshold_diagnostic_by_seed.csv"]
    budget = pd.read_csv(inputs[0])
    seed = budget.groupby(["dataset", "shift", "seed", "fraction", "method"]).coverage.mean().reset_index()
    primary = seed[np.isclose(seed.fraction, .1)]
    controls = pd.read_csv(inputs[1])
    assert controls.groupby("dataset").size().eq(5).all()
    parts = []
    for condition, frame in [("random", controls),
                             ("shift_source", primary[primary.method == "source_split"]),
                             ("shift_target", primary[primary.method == "target_split"])]:
        part = frame[["dataset", "coverage"]].copy()
        part["condition"] = f2.TEXT["en"]["conditions"][condition]
        parts.append(part)
    plot_df = pd.concat(parts, ignore_index=True)
    assert [len(x) for x in parts] == [15, 60, 60]
    plot_df["dataset_label"] = plot_df.dataset.map(f2.TEXT["en"]["datasets"])
    plot_df["dataset_short"] = plot_df.dataset.map(f2.TEXT["en"]["dataset_short"])
    assert not plot_df.isna().any().any()
    plot_df.to_csv(output / "figure2-plotted-data.csv", index=False)
    f2.load_plot_data = lambda language: plot_df.copy()
    original_configure = f2.configure_style

    def readable_style(language):
        original_configure(language)
        plt.rcParams.update({"legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})

    f2.configure_style = readable_style
    original_zoom = f2.draw_zoom_inset

    def readable_zoom(ax, *args, **kwargs):
        original_zoom(ax, *args, **kwargs)
        # Original 7 pt shrinks below 6 pt in the manuscript; preserve geometry.
        ax.tick_params(axis="both", labelsize=11)
        ax.title.set_fontsize(11)

    f2.draw_zoom_inset = readable_zoom
    np.random.seed(20261017)
    f2.render("en", output / "quality_by_task.pdf")
    np.random.seed(20261017)
    f2.render("en", output / "quality_by_task.png")

    diagnostic = pd.read_csv(inputs[2])
    assert len(diagnostic) == 60
    assert len(diagnostic[["dataset", "shift", "seed"]].drop_duplicates()) == 60
    diagnostic["oracle_minus_source_qhat"] = diagnostic.threshold_gap
    diagnostic["source_shortfall"] = diagnostic.shortfall
    diagnostic.to_csv(output / "figure3-plotted-data.csv", index=False)
    f3.configure_style()
    sns.set_style("whitegrid")  # Submitted Figure 3 has light gray full spines.
    plt.rcParams.update({"legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
    # The release script shares Figure 2's palette; the actual submitted Figure 3
    # used seaborn's deep palette, as shown by the byte-identical archived image.
    f3.COLORS = ["#4C72B0", "#DD8452", "#55A868"]
    band_record = {}

    def descriptive_regplot(*args, **kwargs):
        kwargs["ci"] = None
        ax = sns.regplot(*args, **kwargs)
        grid = ax.lines[-1].get_xdata()
        x, y = diagnostic.threshold_gap.to_numpy(), diagnostic.shortfall.to_numpy()
        fits = []
        for (dataset, shift), _ in diagnostic.groupby(["dataset", "shift"]):
            keep = ~((diagnostic.dataset == dataset) & (diagnostic["shift"] == shift))
            assert keep.sum() == 55
            fits.append(np.polyval(np.polyfit(x[keep], y[keep], 1), grid))
        assert len(fits) == 12
        fits = np.asarray(fits)
        ax.fill_between(grid, fits.min(axis=0), fits.max(axis=0), color="black", alpha=.15, zorder=1)
        band_record.update(n_conditions=12, ols_coefficients=np.polyfit(x, y, 1).tolist(),
                           band="range of leave-one-condition-out OLS fits; not a confidence interval")
        return ax

    f3.sns = SimpleNamespace(regplot=descriptive_regplot, scatterplot=sns.scatterplot)
    original_savefig, original_close = plt.Figure.savefig, plt.close

    def save_figure3(fig, filename, *args, **kwargs):
        # Preserve the original title, markers, legend, dimensions, grid and line.
        fig.axes[0].set_xlabel("Diagnostic target-test q90 - source qhat")
        original_savefig(fig, output / "score_threshold_mismatch.pdf")
        return original_savefig(fig, filename, *args, **kwargs)

    plt.Figure.savefig = save_figure3
    try:
        f3.plot_score_threshold_mismatch(output / "figure3-plotted-data.csv", output)
    finally:
        plt.Figure.savefig = original_savefig
        plt.close = original_close
    receipt = {"source": "Recovered original plotting functions, with explicit data/statistical adapters",
               "versions": {"matplotlib": matplotlib.__version__, "seaborn": sns.__version__, "pandas": pd.__version__},
               "figure2_counts": [15, 60, 60], "figure3_count": 60, "figure3": band_record,
               "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
               "original_scripts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals.glob("*.py")},
               "output_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.pdf")}}
    (output / "figure-validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    render(parser.parse_args().output_dir)
