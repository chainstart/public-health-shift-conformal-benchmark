"""Generate empirical manuscript artwork and tables solely from verified results."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]
PAIRS = [("heart", c) for c in ["cleveland", "hungarian", "long_beach", "switzerland"]] + [("sepsis", c) for c in ["B", "A"]]
LABELS = ["CLE", "HUN", "VA", "SWI", "A to B", "B to A"]
METHODS = ["source_split", "source_mondrian", "target_split", "target_mondrian"]
NAMES = {"source_split": "Source split", "source_mondrian": "Source Mondrian",
         "target_split": "Target split", "target_mondrian": "Target Mondrian",
         "available_split": "Available split", "available_mondrian": "Available Mondrian",
         "sentinel_split": "Uniform split", "sentinel_mondrian": "Uniform Mondrian"}
COLORS = ["#0072B2", "#CC79A7", "#D55E00", "#009E73"]
MARKERS = ["o", "D", "^", "s"]
MODELS = [("elastic_net", "Elastic-net logistic"), ("lightgbm", "LightGBM")]


def display(f):
    f = f.copy()
    for name in ["dataset", "backbone", "method", "regime", "condition"]:
        if name not in f:
            continue
        mappings = {"dataset": {"heart": "Heart", "sepsis": "Sepsis"},
                    "backbone": {"elastic_net": "Logistic", "lightgbm": "LightGBM"},
                    "method": NAMES, "regime": {"uniform": "Uniform", "outcome_depleted": "Outcome-only", "risk_priority": "Risk-priority"},
                    "condition": {"cleveland": "CLE", "hungarian": "HUN", "long_beach": "VA", "switzerland": "SWI", "A": "B to A", "B": "A to B", "random_control": "Random"}}
        f[name] = f[name].map(mappings[name])
    return f


def save_table(f, path, long=False, caption=None, label=None):
    text = f.to_latex(index=False, escape=True, longtable=long, float_format=lambda x: f"{x:.4f}",
                      caption=caption, label=label)
    path.write_text(text)


def save(fig, directory, name):
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(directory / (name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert json.loads((args.results / "independent-verification.json").read_text())["status"] == "pass"
    assert json.loads((args.results / "cohort-audit.json").read_text())["status"] == "pass"
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(args.results / "budget_by_condition.csv")
    bd = pd.read_csv(args.results / "budget_by_dataset.csv")
    c = pd.read_csv(args.results / "channel_by_dataset.csv")
    raw_b = pd.read_csv(args.results / "budget_metrics.csv")
    raw_c = pd.read_csv(args.results / "channel_metrics.csv")
    # Small reproducible count summaries preserve every budget and mechanism.
    # Outcome classes are never pooled to conceal finite-rank overflow.
    for name, frame, keys in [
        ("budget_label_counts", raw_b, ["dataset", "backbone", "condition", "fraction", "method"]),
        ("channel_label_counts", raw_c, ["dataset", "backbone", "condition", "regime", "method"]),
    ]:
        frame = frame.copy()
        frame["class0_overflow"] = frame.n_class0 < 9
        frame["class1_overflow"] = frame.n_class1 < 9
        counts_summary = frame.groupby(keys).agg(
            planned=("target_budget", "first"), actual_min=("n_calibration", "min"),
            actual_max=("n_calibration", "max"), class0_min=("n_class0", "min"),
            class0_max=("n_class0", "max"), class1_min=("n_class1", "min"),
            class1_max=("n_class1", "max"), class0_overflow_fraction=("class0_overflow", "mean"),
            class1_overflow_fraction=("class1_overflow", "mean")).reset_index()
        counts_summary.to_csv(args.results / (name + ".csv"), index=False)
    plt.rcParams.update({"font.size": 14.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.titlesize": 15, "axes.labelsize": 14.5, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 14})
    selected = b[np.isclose(b.fraction, .1)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ri, (model, label) in enumerate(MODELS):
        for ci, (metric, ylabel) in enumerate([("coverage", "Marginal coverage"), ("mean_set_size", "Mean set size")]):
            ax = axes[ri, ci]
            for mi, method in enumerate(METHODS):
                vals = [selected[(selected.dataset == d) & (selected.condition == cond) & (selected.backbone == model) & (selected.method == method)].iloc[0][metric] for d, cond in PAIRS]
                ax.scatter(np.arange(6) + (mi - 1.5) * .15, vals, label=NAMES[method], color=COLORS[mi], marker=MARKERS[mi], s=43, zorder=3)
            ax.set_xticks(np.arange(6), LABELS, rotation=25)
            ax.set_ylabel(ylabel)
            ax.set_title(label)
            ax.grid(axis="y", alpha=.25)
            ax.axvline(3.5, color=".6", lw=.8)
            if metric == "coverage":
                ax.axhline(.9, color="black", ls="--", lw=1)
                ax.set_ylim(min(.7, selected.coverage.min() - .04), 1.025)
            else:
                ax.set_ylim(max(0, selected.mean_set_size.min() - .08), 2.04)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.035), frameon=False)
    fig.tight_layout(rect=[0, .09, 1, 1])
    save(fig, out, "external_institutional_validation")
    risk = c[c.regime == "risk_priority"]
    combos = [(d, m) for d in ["heart", "sepsis"] for m, _ in MODELS]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.7))
    for ax, metric, title in zip(axes, ["coverage", "class_0_coverage", "class_1_coverage"], ["Marginal", "Outcome class 0", "Outcome class 1"]):
        for mi, method in enumerate(["available_split", "available_mondrian", "sentinel_split", "sentinel_mondrian"]):
            vals = [risk[(risk.dataset == d) & (risk.backbone == m) & (risk.method == method)].iloc[0][metric] for d, m in combos]
            ax.scatter(np.arange(4) + (mi - 1.5) * .15, vals, label=NAMES[method], color=COLORS[mi], marker=MARKERS[mi], s=43, zorder=3)
        ax.set_xticks(np.arange(4), ["Heart/EN", "Heart/GB", "Sepsis/EN", "Sepsis/GB"], rotation=50, ha="right")
        ax.set_title(title)
        ax.axhline(.9, color="black", ls="--", lw=1)
        ax.set_ylim(min(.7, risk[["coverage", "class_0_coverage", "class_1_coverage"]].min().min() - .04), 1.025)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Prediction-set coverage")
    h, labels = axes[0].get_legend_handles_labels()
    fig.legend(h, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.08), frameon=False)
    fig.tight_layout(rect=[0, .19, 1, 1])
    save(fig, out, "external_label_observation")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ri, dataset in enumerate(["heart", "sepsis"]):
        for ci, (model, title) in enumerate(MODELS):
            ax = axes[ri, ci]
            for mi, method in enumerate(METHODS):
                f = bd[(bd.dataset == dataset) & (bd.backbone == model) & (bd.method == method)].sort_values("fraction")
                ax.plot(100 * f.fraction, f.coverage, marker=MARKERS[mi], color=COLORS[mi], label=NAMES[method])
            ax.set_xscale("log")
            ax.set_xticks([.5, 1, 2, 3, 5, 10, 15, 20], [".5", "1", "2", "3", "5", "10", "15", "20"], rotation=30)
            ax.set_xlabel("Labels (% of full target)")
            ax.set_ylabel("Marginal coverage")
            ax.set_title(dataset.capitalize() + ": " + title)
            ax.axhline(.9, color="black", ls="--", lw=1)
            ax.grid(alpha=.2)
    h, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.035), frameon=False)
    fig.tight_layout(rect=[0, .09, 1, 1])
    save(fig, out, "external_budget_curves")
    cohorts = []
    for dataset, condition in PAIRS:
        f = pd.read_parquet(args.artifacts / (dataset + ".parquet"))
        target, source = f[f.center == condition], f[f.center != condition]
        draws = raw_b[(raw_b.dataset == dataset) & (raw_b.condition == condition) & (raw_b.backbone == "elastic_net") & (raw_b.method == "target_mondrian") & np.isclose(raw_b.fraction, .1)]
        cohorts.append({"Dataset": dataset.capitalize(), "Target": LABELS[PAIRS.index((dataset, condition))],
                        "Source n": len(source), "Target n": len(target), "Positive (\\%)": 100 * target.label.mean(),
                        "Labels": int(draws.target_budget.iloc[0]),
                        "Class 0": f"{draws.n_class0.min()}--{draws.n_class0.max()}",
                        "Class 1": f"{draws.n_class1.min()}--{draws.n_class1.max()}"})
    cohort_table = pd.DataFrame(cohorts)
    cohort_table["Positive (\\%)"] = cohort_table["Positive (\\%)"].map(lambda x: f"{x:.2f}")
    # Plain percent in headers must be escaped once by the table exporter.
    cohort_table = cohort_table.rename(columns={"Positive (\\%)": "Positive (%)"})
    save_table(cohort_table, out / "external_cohorts.tex")
    main = display(bd[np.isclose(bd.fraction, .1)][["dataset", "backbone", "method", "coverage", "class_0_coverage", "class_1_coverage", "mean_set_size"]])
    main.columns = ["Dataset", "Model", "Calibration", "Coverage", "$C_0$", "$C_1$", "Size"]
    main = main.rename(columns={"$C_0$": "Class 0", "$C_1$": "Class 1"})
    save_table(main, out / "external_summary.tex")
    long = display(selected[["dataset", "condition", "backbone", "method", "coverage", "class_0_coverage", "class_1_coverage", "mean_set_size"]])
    long.columns = ["Data", "Target", "Model", "Calibration", "Coverage", "Class 0", "Class 1", "Size"]
    save_table(long, out / "supplement_external_conditions.tex", True,
               "Every external institutional condition at the 10\\% target-label budget. Entries average draws within seed, then five seeds.", "tab:s-external-conditions")
    long = display(c[["dataset", "backbone", "regime", "method", "coverage", "class_0_coverage", "class_1_coverage", "mean_set_size"]])
    long.columns = ["Data", "Model", "Observation", "Calibration", "Coverage", "Class 0", "Class 1", "Size"]
    save_table(long, out / "supplement_external_channels.tex", True,
               "All observation channels and alternative calibration protocols in the external cohorts. Conditions receive equal weight within each collection.", "tab:s-external-channels")
    counts = raw_c.groupby(["dataset", "backbone", "regime", "method"]).agg(
        planned_min=("target_budget", "min"), planned_max=("target_budget", "max"),
        actual_min=("n_calibration", "min"), actual_max=("n_calibration", "max"),
        class0_min=("n_class0", "min"), class0_max=("n_class0", "max"),
        class1_min=("n_class1", "min"), class1_max=("n_class1", "max"),
        quota_failures=("quota_shortfall", lambda x: int((x > 0).sum()))).reset_index()
    counts = counts[counts.method.str.endswith("mondrian")]
    count_table = display(counts)[["dataset", "backbone", "regime", "method"]].copy()
    for label, left, right in [("Planned", "planned_min", "planned_max"), ("Actual", "actual_min", "actual_max"),
                               ("Class 0", "class0_min", "class0_max"), ("Class 1", "class1_min", "class1_max")]:
        count_table[label] = counts[left].astype(str) + "--" + counts[right].astype(str)
    count_table["Short quotas"] = counts.quota_failures
    save_table(count_table, out / "supplement_external_counts.tex", True,
               "External channel-experiment label counts; ranges span all conditions, seeds and draws. Mondrian and ordinary split share the same samples. Short quotas are retained.", "tab:s-external-counts")
    d = pd.read_csv(args.results / "predictor_diagnostics.csv")
    diag = d.groupby(["dataset", "backbone", "condition"]).agg(n_test=("n_test", "first"),
            positive_min=("class1_test", "min"), positive_max=("class1_test", "max"),
            auroc=("auroc", "mean"), balanced_accuracy=("balanced_accuracy", "mean")).reset_index()
    save_table(display(diag), out / "supplement_external_predictors.tex", True,
               "Untouched external-test predictor diagnostics, averaged over five source-model seeds. Random controls are contextual and not size-matched.", "tab:s-external-predictors")
    solver_dir = args.results.parent / "source-solver-results"
    if solver_dir.exists():
        assert json.loads((solver_dir / "independent-verification.json").read_text())["status"] == "pass"
        solver = pd.read_csv(solver_dir / "budget_by_condition.csv")
        diagnostics = pd.read_csv(solver_dir / "predictor_diagnostics.csv").groupby("condition")[["source_calibration_auroc", "auroc"]].mean()
        solver = solver.merge(diagnostics, on="condition")
        solver = solver[["condition", "method", "coverage", "class_0_coverage", "class_1_coverage", "mean_set_size", "source_calibration_auroc", "auroc"]]
        solver["condition"] = solver.condition.map({"A": "B to A", "B": "A to B"})
        solver["method"] = solver.method.map(NAMES)
        solver.columns = ["Target", "Calibration", "Coverage", "Class 0", "Class 1", "Size", "Source AUROC", "Target AUROC"]
        save_table(solver, out / "supplement_source_solver.tex", True,
                   "Posthoc source-only L2 logistic sensitivity at the unchanged 10\\% budget. Optimization and regularization both differ from the frozen SGD predictor. The two prespecified predictor families are reported in Table S18.", "tab:s-external-solver")
    cohort_table.to_csv(out / "external-cohort-summary.csv", index=False)
    main.to_csv(out / "external-main-summary.csv", index=False)
    receipt = {"status": "generated_from_verified_results", "source_results": str(args.results),
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != "assets-receipt.json"}}
    (out / "assets-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "files": len(receipt["outputs"])}))


if __name__ == "__main__":
    main()
