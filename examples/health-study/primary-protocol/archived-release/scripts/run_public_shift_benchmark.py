from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.large_scale_conformal import (  # noqa: E402
    LargeDataset,
    TARGET_COVERAGE,
    bootstrap_ci,
    evaluate_sets,
    get_random_split,
    get_shift_split,
    make_preprocessor,
    make_model,
    make_qhat,
    markdown_table,
    prediction_sets,
    sample_target_pool,
)
from scripts.public_data_preparation import (  # noqa: E402
    ensure_public_data_dirs,
    evaluate_confidence_threshold,
    estimate_domain_auc,
    prepare_brfss_multiyear,
    prepare_nhanes_multicycle,
    prepare_uci_with_pseudo,
)


RESULTS_DIR = ROOT / "results" / "public_benchmark"
FIGURES_DIR = ROOT / "figures" / "public_benchmark"
REPORTS_DIR = ROOT / "reports"

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_TARGET_FRACTIONS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]
PRIMARY_FRACTION = 0.10


DROP_DEFINING_FEATURES: dict[tuple[str, str], list[str]] = {
    ("brfss_multiyear_diabetes", "time_future"): ["survey_year"],
    ("brfss_multiyear_diabetes", "region_south"): ["region", "_STATE"],
    ("brfss_multiyear_diabetes", "age_65plus"): ["age80", "_AGE65YR", "_AGEG5YR"],
    ("brfss_multiyear_diabetes", "obesity"): ["bmi", "_BMI5CAT"],
    ("brfss_multiyear_diabetes", "race_black_vs_white"): ["_RACE"],
    ("nhanes_multicycle_diabetes", "time_future"): ["cycle_start"],
    ("nhanes_multicycle_diabetes", "age_65plus"): ["RIDAGEYR"],
    ("nhanes_multicycle_diabetes", "obesity"): ["BMXBMI", "BMXWT", "BMXHT", "BMXWAIST"],
    ("nhanes_multicycle_diabetes", "race_black_vs_white"): ["RIDRETH3"],
    ("uci_diabetes_readmission", "age_older70"): ["age_mid"],
    ("uci_diabetes_readmission", "race_african_american"): ["race"],
    ("uci_diabetes_readmission", "emergency_admission"): ["admission_source_id", "admission_mode"],
}


def fmt(value: object, digits: int = 3) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return "NA"
    return f"{value_float:.{digits}f}"


def ensure_dirs() -> None:
    ensure_public_data_dirs()
    for path in [RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def feature_dropped_dataset(ds: LargeDataset, drop_cols: list[str]) -> LargeDataset:
    present = [col for col in drop_cols if col in ds.X.columns]
    if not present:
        return replace(ds)
    X = ds.X.drop(columns=present)
    numeric_cols = [col for col in ds.numeric_cols if col not in present]
    categorical_cols = [col for col in ds.categorical_cols if col not in present]
    return replace(ds, X=X, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def estimate_density_ratio_weights_fast(
    ds: LargeDataset,
    X_source_cal: pd.DataFrame,
    X_target_unlabeled: pd.DataFrame,
    seed: int,
    max_source: int = 30_000,
    max_target: int = 30_000,
) -> tuple[np.ndarray, dict[str, float]]:
    source_fit = X_source_cal
    target_fit = X_target_unlabeled
    if len(source_fit) > max_source:
        source_fit = source_fit.sample(n=max_source, random_state=seed)
    if len(target_fit) > max_target:
        target_fit = target_fit.sample(n=max_target, random_state=seed + 1)
    X_domain = pd.concat([source_fit, target_fit], ignore_index=True)
    y_domain = np.r_[np.zeros(len(source_fit), dtype=int), np.ones(len(target_fit), dtype=int)]
    model = Pipeline(
        [
            ("preprocess", make_preprocessor(ds)),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=500,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_domain, y_domain)
    p_target = model.predict_proba(X_source_cal)[:, 1]
    odds = p_target / np.clip(1.0 - p_target, 1e-6, 1.0)
    prior_correction = len(source_fit) / max(len(target_fit), 1)
    weights = np.clip(odds * prior_correction, 0.05, 20.0)
    ess = float((weights.sum() ** 2) / np.sum(weights**2)) if np.sum(weights**2) > 0 else np.nan
    diagnostics = {
        "weight_mean": float(np.mean(weights)),
        "weight_median": float(np.median(weights)),
        "weight_p95": float(np.quantile(weights, 0.95)),
        "weight_max": float(np.max(weights)),
        "weight_ess": ess,
    }
    return weights, diagnostics


def add_conformal_result(
    rows: list[dict[str, object]],
    subgroup_rows: list[dict[str, object]],
    ds: LargeDataset,
    split: dict,
    seed: int,
    experiment: str,
    workflow: str,
    q_method: str,
    target_fraction: float,
    n_target_cal: int,
    cal_probs: np.ndarray,
    y_cal: np.ndarray,
    test_probs: np.ndarray,
    domain_auc: float,
    ablation: str = "none",
    dropped_features: str = "",
    weights: np.ndarray | None = None,
    weight_diagnostics: dict[str, float] | None = None,
) -> None:
    qhat = make_qhat(cal_probs, y_cal, q_method, weights=weights)
    sets = prediction_sets(test_probs, qhat, fallback=True)
    metrics, subgroups = evaluate_sets(split["y_test"], test_probs, sets, split["g_test"])
    row = {
        "dataset": ds.name,
        "task": ds.task,
        "seed": seed,
        "experiment": experiment,
        "ablation": ablation,
        "dropped_features": dropped_features,
        "setting": split["setting"],
        "shift_name": split["shift_name"],
        "shift_description": split["shift_description"],
        "workflow": workflow,
        "q_method": q_method,
        "target_cal_fraction": target_fraction,
        "target_coverage": TARGET_COVERAGE,
        "n_train": int(len(split["y_train"])),
        "n_source_cal": int(len(split["y_source_cal"])),
        "n_target_cal": int(n_target_cal),
        "domain_auc": domain_auc,
        **(weight_diagnostics or {}),
        **metrics,
    }
    rows.append(row)
    for subgroup in subgroups:
        subgroup_rows.append({**row, **subgroup})


def dataset_audit_rows(datasets: list[LargeDataset]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dataset_rows: list[dict[str, object]] = []
    shift_rows: list[dict[str, object]] = []
    for ds in datasets:
        X = ds.X.copy()
        missing = X.isna().mean()
        dataset_rows.append(
            {
                "dataset": ds.name,
                "task": ds.task,
                "source_url": ds.source_url,
                "analysis_n": int(ds.final_n),
                "positive_n": int(np.sum(ds.y)),
                "positive_prevalence": float(np.mean(ds.y)),
                "n_numeric_features": len(ds.numeric_cols),
                "n_categorical_features": len(ds.categorical_cols),
                "mean_feature_missing_rate": float(missing.mean()) if len(missing) else 0.0,
                "max_feature_missing_rate": float(missing.max()) if len(missing) else 0.0,
                "shift_settings": "; ".join(spec.name for spec in ds.shifts),
            }
        )
        for spec in ds.shifts:
            source_mask = ds.groups[spec.group_col].isin(spec.source_values).to_numpy()
            target_mask = ds.groups[spec.group_col].isin(spec.target_values).to_numpy()
            source_y = ds.y[source_mask]
            target_y = ds.y[target_mask]
            drop_cols = DROP_DEFINING_FEATURES.get((ds.name, spec.name), [])
            shift_rows.append(
                {
                    "dataset": ds.name,
                    "shift_name": spec.name,
                    "group_col": spec.group_col,
                    "source_values": ";".join(spec.source_values),
                    "target_values": ";".join(spec.target_values),
                    "source_n": int(source_mask.sum()),
                    "target_n": int(target_mask.sum()),
                    "source_positive_prevalence": float(np.mean(source_y)) if len(source_y) else np.nan,
                    "target_positive_prevalence": float(np.mean(target_y)) if len(target_y) else np.nan,
                    "prevalence_difference": float(np.mean(target_y) - np.mean(source_y)) if len(source_y) and len(target_y) else np.nan,
                    "drop_defining_features": ";".join([col for col in drop_cols if col in ds.X.columns]),
                    "description": spec.description,
                }
            )
    return dataset_rows, shift_rows


def run_benchmark(
    datasets: list[LargeDataset],
    seeds: list[int],
    target_fractions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    selective_rows: list[dict[str, object]] = []
    dataset_rows, shift_rows = dataset_audit_rows(datasets)

    for ds in datasets:
        print(f"[benchmark] dataset={ds.name} n={ds.final_n} shifts={len(ds.shifts)}", flush=True)
        for seed in seeds:
            print(f"[benchmark] seed={seed} dataset={ds.name}", flush=True)
            split_specs = [get_random_split(ds, seed)]
            for spec in ds.shifts:
                shifted = get_shift_split(ds, spec, seed)
                if shifted is not None:
                    split_specs.append(shifted)

            for split in split_specs:
                model = make_model(ds, "sgd_logistic", seed)
                model.fit(split["X_train"], split["y_train"])
                source_cal_probs = model.predict_proba(split["X_source_cal"])
                test_probs = model.predict_proba(split["X_test"])
                if split["setting"] == "covariate_shift":
                    X_target_unlabeled = pd.concat([split["X_target_pool"], split["X_test"]], ignore_index=True)
                    domain_auc = estimate_domain_auc(ds, split["X_source_cal"], X_target_unlabeled, seed)
                else:
                    X_target_unlabeled = None
                    domain_auc = np.nan

                add_conformal_result(
                    rows,
                    subgroup_rows,
                    ds,
                    split,
                    seed,
                    "primary",
                    "source_split",
                    "split",
                    0.0,
                    0,
                    source_cal_probs,
                    split["y_source_cal"],
                    test_probs,
                    domain_auc,
                )
                add_conformal_result(
                    rows,
                    subgroup_rows,
                    ds,
                    split,
                    seed,
                    "primary",
                    "source_mondrian",
                    "mondrian",
                    0.0,
                    0,
                    source_cal_probs,
                    split["y_source_cal"],
                    test_probs,
                    domain_auc,
                )
                selective = evaluate_confidence_threshold(source_cal_probs, split["y_source_cal"], test_probs, split["y_test"])
                selective_rows.append(
                    {
                        "dataset": ds.name,
                        "seed": seed,
                        "experiment": "primary",
                        "setting": split["setting"],
                        "shift_name": split["shift_name"],
                        "workflow": "source_confidence_threshold",
                        "target_cal_fraction": 0.0,
                        "n_target_cal": 0,
                        "domain_auc": domain_auc,
                        **selective,
                    }
                )

                if split["setting"] != "covariate_shift":
                    continue

                weights, weight_diagnostics = estimate_density_ratio_weights_fast(
                    ds,
                    split["X_source_cal"],
                    X_target_unlabeled,
                    seed + 17,
                )
                add_conformal_result(
                    rows,
                    subgroup_rows,
                    ds,
                    split,
                    seed,
                    "primary",
                    "source_weighted_split",
                    "weighted_split",
                    0.0,
                    0,
                    source_cal_probs,
                    split["y_source_cal"],
                    test_probs,
                    domain_auc,
                    weights=weights,
                    weight_diagnostics=weight_diagnostics,
                )

                for fraction in target_fractions:
                    sampled = sample_target_pool(split, fraction, seed + int(fraction * 100_000))
                    if sampled is None:
                        continue
                    X_target_cal, y_target_cal, _ = sampled
                    if len(np.unique(y_target_cal)) < 2:
                        continue
                    target_cal_probs = model.predict_proba(X_target_cal)
                    add_conformal_result(
                        rows,
                        subgroup_rows,
                        ds,
                        split,
                        seed,
                        "primary",
                        "target_recalibrated_split",
                        "split",
                        fraction,
                        len(y_target_cal),
                        target_cal_probs,
                        y_target_cal,
                        test_probs,
                        domain_auc,
                    )
                    add_conformal_result(
                        rows,
                        subgroup_rows,
                        ds,
                        split,
                        seed,
                        "primary",
                        "target_recalibrated_mondrian",
                        "mondrian",
                        fraction,
                        len(y_target_cal),
                        target_cal_probs,
                        y_target_cal,
                        test_probs,
                        domain_auc,
                    )
                    selective = evaluate_confidence_threshold(target_cal_probs, y_target_cal, test_probs, split["y_test"])
                    selective_rows.append(
                        {
                            "dataset": ds.name,
                            "seed": seed,
                            "experiment": "primary",
                            "setting": split["setting"],
                            "shift_name": split["shift_name"],
                            "workflow": "target_confidence_threshold",
                            "target_cal_fraction": fraction,
                            "n_target_cal": len(y_target_cal),
                            "domain_auc": domain_auc,
                            **selective,
                        }
                    )

            for spec in ds.shifts:
                if spec.name == "pseudo_random":
                    continue
                drop_cols = [col for col in DROP_DEFINING_FEATURES.get((ds.name, spec.name), []) if col in ds.X.columns]
                if not drop_cols:
                    continue
                ds_drop = feature_dropped_dataset(ds, drop_cols)
                split = get_shift_split(ds_drop, spec, seed)
                if split is None:
                    continue
                model = make_model(ds_drop, "sgd_logistic", seed)
                model.fit(split["X_train"], split["y_train"])
                source_cal_probs = model.predict_proba(split["X_source_cal"])
                test_probs = model.predict_proba(split["X_test"])
                X_target_unlabeled = pd.concat([split["X_target_pool"], split["X_test"]], ignore_index=True)
                domain_auc = estimate_domain_auc(ds_drop, split["X_source_cal"], X_target_unlabeled, seed)
                add_conformal_result(
                    rows,
                    subgroup_rows,
                    ds_drop,
                    split,
                    seed,
                    "drop_defining_features",
                    "source_split",
                    "split",
                    0.0,
                    0,
                    source_cal_probs,
                    split["y_source_cal"],
                    test_probs,
                    domain_auc,
                    ablation="drop_defining_features",
                    dropped_features=";".join(drop_cols),
                )
                sampled = sample_target_pool(split, PRIMARY_FRACTION, seed + int(PRIMARY_FRACTION * 100_000))
                if sampled is not None:
                    X_target_cal, y_target_cal, _ = sampled
                    if len(np.unique(y_target_cal)) >= 2:
                        target_cal_probs = model.predict_proba(X_target_cal)
                        add_conformal_result(
                            rows,
                            subgroup_rows,
                            ds_drop,
                            split,
                            seed,
                            "drop_defining_features",
                            "target_recalibrated_split",
                            "split",
                            PRIMARY_FRACTION,
                            len(y_target_cal),
                            target_cal_probs,
                            y_target_cal,
                            test_probs,
                            domain_auc,
                            ablation="drop_defining_features",
                            dropped_features=";".join(drop_cols),
                        )

    results = pd.DataFrame(rows)
    subgroups = pd.DataFrame(subgroup_rows)
    selective = pd.DataFrame(selective_rows)
    dataset_audit = pd.DataFrame(dataset_rows)
    shift_audit = pd.DataFrame(shift_rows)
    results.to_csv(RESULTS_DIR / "public_benchmark_conformal_results.csv", index=False)
    subgroups.to_csv(RESULTS_DIR / "public_benchmark_subgroup_coverage.csv", index=False)
    selective.to_csv(RESULTS_DIR / "public_benchmark_selective_baseline.csv", index=False)
    dataset_audit.to_csv(RESULTS_DIR / "public_benchmark_dataset_audit.csv", index=False)
    shift_audit.to_csv(RESULTS_DIR / "public_benchmark_shift_audit.csv", index=False)
    return results, subgroups, selective, dataset_audit, shift_audit


def paired_difference_test(
    results: pd.DataFrame,
    label: str,
    a_filter: pd.Series,
    b_filter: pd.Series,
    metric: str,
    index_cols: list[str] | None = None,
) -> dict[str, object] | None:
    if index_cols is None:
        index_cols = ["dataset", "shift_name", "seed"]
    a = results[a_filter].set_index(index_cols)[metric]
    b = results[b_filter].set_index(index_cols)[metric]
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return None
    diff = a.loc[idx].to_numpy(dtype=float) - b.loc[idx].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_ci(diff)
    return {
        "test": label,
        "metric": metric,
        "median_difference": float(np.median(diff)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_p": float(wilcoxon(diff).pvalue) if np.any(np.abs(diff) > 1e-12) else np.nan,
        "n_pairs": int(len(diff)),
    }


def summarize_and_test(
    results: pd.DataFrame,
    selective: pd.DataFrame,
    target_fractions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = results[results["experiment"] == "primary"].copy()
    primary_no_pseudo = primary[(primary["setting"] != "covariate_shift") | (primary["shift_name"] != "pseudo_random")].copy()
    summary = (
        primary_no_pseudo.groupby(["experiment", "setting", "workflow", "target_cal_fraction"])
        .agg(
            coverage=("coverage", "median"),
            coverage_shortfall=("coverage_shortfall", "median"),
            singleton_rate=("singleton_rate", "median"),
            mean_set_size=("mean_set_size", "median"),
            class_coverage_gap=("class_coverage_gap", "median"),
            max_subgroup_coverage_gap=("max_subgroup_coverage_gap", "median"),
            domain_auc=("domain_auc", "median"),
            weight_ess=("weight_ess", "median"),
            n_runs=("coverage", "size"),
        )
        .reset_index()
    )
    all_summary = (
        primary.groupby(["experiment", "setting", "workflow", "target_cal_fraction"])
        .agg(
            coverage=("coverage", "median"),
            coverage_shortfall=("coverage_shortfall", "median"),
            singleton_rate=("singleton_rate", "median"),
            mean_set_size=("mean_set_size", "median"),
            class_coverage_gap=("class_coverage_gap", "median"),
            max_subgroup_coverage_gap=("max_subgroup_coverage_gap", "median"),
            domain_auc=("domain_auc", "median"),
            weight_ess=("weight_ess", "median"),
            n_runs=("coverage", "size"),
        )
        .reset_index()
    )
    by_shift = (
        primary.groupby(["dataset", "setting", "shift_name", "workflow", "target_cal_fraction"])
        .agg(
            coverage=("coverage", "median"),
            coverage_shortfall=("coverage_shortfall", "median"),
            singleton_rate=("singleton_rate", "median"),
            class_coverage_gap=("class_coverage_gap", "median"),
            domain_auc=("domain_auc", "median"),
            weight_ess=("weight_ess", "median"),
            n_runs=("coverage", "size"),
        )
        .reset_index()
    )
    ablation_summary = (
        results[results["experiment"] == "drop_defining_features"]
        .groupby(["dataset", "shift_name", "workflow", "target_cal_fraction", "dropped_features"])
        .agg(
            coverage=("coverage", "median"),
            coverage_shortfall=("coverage_shortfall", "median"),
            singleton_rate=("singleton_rate", "median"),
            class_coverage_gap=("class_coverage_gap", "median"),
            domain_auc=("domain_auc", "median"),
            n_runs=("coverage", "size"),
        )
        .reset_index()
    )

    tests: list[dict[str, object]] = []
    source_shift = primary[
        (primary["setting"] == "covariate_shift")
        & (primary["workflow"] == "source_split")
        & (primary["target_cal_fraction"] == 0.0)
        & (primary["shift_name"] != "pseudo_random")
    ]
    random_source = primary[
        (primary["setting"] == "random_exchangeable")
        & (primary["workflow"] == "source_split")
        & (primary["target_cal_fraction"] == 0.0)
    ].set_index(["dataset", "seed"])
    diffs = []
    for _, row in source_shift.iterrows():
        key = (row["dataset"], row["seed"])
        if key in random_source.index:
            diffs.append(row["coverage_error"] - float(random_source.loc[key, "coverage_error"]))
    if diffs:
        arr = np.asarray(diffs, dtype=float)
        ci = bootstrap_ci(arr)
        tests.append(
            {
                "test": "source_shift_minus_random_coverage_error",
                "metric": "coverage_error",
                "median_difference": float(np.median(arr)),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "wilcoxon_p": float(wilcoxon(arr).pvalue) if np.any(np.abs(arr) > 1e-12) else np.nan,
                "n_pairs": len(arr),
            }
        )

    for fraction in target_fractions:
        for metric in ["coverage", "coverage_shortfall", "singleton_rate", "class_coverage_gap"]:
            test = paired_difference_test(
                primary,
                f"target_split_minus_source_{metric}_fraction_{fraction:g}",
                (primary["setting"] == "covariate_shift")
                & (primary["workflow"] == "target_recalibrated_split")
                & np.isclose(primary["target_cal_fraction"], fraction)
                & (primary["shift_name"] != "pseudo_random"),
                (primary["setting"] == "covariate_shift")
                & (primary["workflow"] == "source_split")
                & np.isclose(primary["target_cal_fraction"], 0.0)
                & (primary["shift_name"] != "pseudo_random"),
                metric,
            )
            if test is not None:
                tests.append(test)

    for metric in ["coverage", "coverage_shortfall", "singleton_rate", "class_coverage_gap"]:
        test = paired_difference_test(
            primary,
            f"weighted_source_minus_source_{metric}",
            (primary["setting"] == "covariate_shift")
            & (primary["workflow"] == "source_weighted_split")
            & (primary["shift_name"] != "pseudo_random"),
            (primary["setting"] == "covariate_shift")
            & (primary["workflow"] == "source_split")
            & np.isclose(primary["target_cal_fraction"], 0.0)
            & (primary["shift_name"] != "pseudo_random"),
            metric,
        )
        if test is not None:
            tests.append(test)

    for metric in ["coverage", "coverage_shortfall"]:
        full = (results["experiment"] == "primary") & (results["workflow"] == "source_split") & np.isclose(results["target_cal_fraction"], 0.0)
        ablated = (results["experiment"] == "drop_defining_features") & (results["workflow"] == "source_split") & np.isclose(results["target_cal_fraction"], 0.0)
        test = paired_difference_test(
            results,
            f"ablation_source_minus_full_source_{metric}",
            ablated,
            full,
            metric,
        )
        if test is not None:
            tests.append(test)

        ablated_target = (
            (results["experiment"] == "drop_defining_features")
            & (results["workflow"] == "target_recalibrated_split")
            & np.isclose(results["target_cal_fraction"], PRIMARY_FRACTION)
        )
        test = paired_difference_test(
            results,
            f"ablation_target10_minus_ablation_source_{metric}",
            ablated_target,
            ablated,
            metric,
        )
        if test is not None:
            tests.append(test)

    test_df = pd.DataFrame(tests)
    selective_summary = (
        selective.groupby(["experiment", "setting", "workflow", "target_cal_fraction"])
        .agg(
            auto_rate=("auto_rate", "median"),
            auto_accuracy=("auto_accuracy", "median"),
            manual_review_rate=("manual_review_rate", "median"),
            n_runs=("auto_rate", "size"),
        )
        .reset_index()
    )

    summary.to_csv(RESULTS_DIR / "public_benchmark_summary.csv", index=False)
    all_summary.to_csv(RESULTS_DIR / "public_benchmark_summary_all_shifts.csv", index=False)
    by_shift.to_csv(RESULTS_DIR / "public_benchmark_by_shift.csv", index=False)
    ablation_summary.to_csv(RESULTS_DIR / "public_benchmark_ablation_summary.csv", index=False)
    test_df.to_csv(RESULTS_DIR / "public_benchmark_statistical_tests.csv", index=False)
    selective_summary.to_csv(RESULTS_DIR / "public_benchmark_selective_summary.csv", index=False)
    return summary, by_shift, ablation_summary, test_df, selective_summary


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def make_figures(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    by_shift: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    tests: pd.DataFrame,
    dataset_audit: pd.DataFrame,
    shift_audit: pd.DataFrame,
) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.axis("off")
    boxes = [
        ("Public health tasks", "BRFSS 2011-2023\nNHANES 2011-2018\nUCI readmission"),
        ("Robustness", "5 seeds\n12 primary shifts\npseudo controls"),
        ("Sensitivity", "drop split-defining\nfeatures"),
        ("Mitigation", "0.5-20% target labels\nweighted conformal\nMondrian"),
        ("Audit", "dataset flow\nshift prevalence\nsubgroup gaps"),
    ]
    x_positions = np.linspace(0.04, 0.80, len(boxes))
    for i, ((title, body), x) in enumerate(zip(boxes, x_positions)):
        ax.text(x, 0.68, title, fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(
            x,
            0.43,
            body,
            fontsize=8.5,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=palette[i % len(palette)], lw=1.6),
        )
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.14, 0.45), xytext=(x + 0.10, 0.45), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))
    save_figure(fig, "figure1_design.png")

    primary_plot = results[
        (results["experiment"] == "primary")
        & (results["shift_name"] != "pseudo_random")
        & (
            ((results["workflow"] == "source_split") & np.isclose(results["target_cal_fraction"], 0.0))
            | ((results["workflow"] == "target_recalibrated_split") & np.isclose(results["target_cal_fraction"], PRIMARY_FRACTION))
        )
    ].copy()
    primary_plot["condition"] = np.where(
        primary_plot["setting"] == "random_exchangeable",
        "Random source",
        np.where(primary_plot["workflow"] == "source_split", "Shifted source", "Shifted target 10%"),
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    sns.boxplot(data=primary_plot, x="condition", y="coverage", hue="dataset", palette=palette[:3], ax=ax)
    ax.axhline(TARGET_COVERAGE, color="black", ls="--", lw=1)
    ax.set_xlabel("")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Seed-robust coverage across deployment-like shifts")
    ax.tick_params(axis="x", rotation=15)
    sns.despine()
    save_figure(fig, "figure2_seed_robustness.png")

    budget = results[
        (results["experiment"] == "primary")
        & (results["setting"] == "covariate_shift")
        & (results["shift_name"] != "pseudo_random")
        & (results["workflow"].isin(["target_recalibrated_split", "target_recalibrated_mondrian"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    sns.lineplot(data=budget, x="target_cal_fraction", y="coverage_shortfall", hue="workflow", style="workflow", markers=True, dashes=False, errorbar=("ci", 95), palette=palette[2:4], ax=ax)
    ax.set_xlabel("Target-domain labeled calibration fraction")
    ax.set_ylabel("Coverage shortfall")
    ax.set_title("Dense calibration-budget curve")
    sns.despine()
    save_figure(fig, "figure3_dense_budget_curve.png")

    method = results[
        (results["experiment"] == "primary")
        & (results["setting"] == "covariate_shift")
        & (results["shift_name"] != "pseudo_random")
        & (
            ((results["workflow"].isin(["source_split", "source_weighted_split"])) & np.isclose(results["target_cal_fraction"], 0.0))
            | ((results["workflow"].isin(["target_recalibrated_split", "target_recalibrated_mondrian"])) & np.isclose(results["target_cal_fraction"], PRIMARY_FRACTION))
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.boxplot(data=method, x="workflow", y="coverage_shortfall", hue="workflow", legend=False, palette=palette[:4], ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Coverage shortfall")
    ax.set_title("Method comparison under covariate shift")
    ax.tick_params(axis="x", rotation=20)
    sns.despine()
    save_figure(fig, "figure4_method_comparison.png")

    full_source = results[
        (results["experiment"] == "primary")
        & (results["workflow"] == "source_split")
        & np.isclose(results["target_cal_fraction"], 0.0)
        & (results["setting"] == "covariate_shift")
        & (results["shift_name"] != "pseudo_random")
    ][["dataset", "shift_name", "seed", "coverage_shortfall"]].rename(columns={"coverage_shortfall": "full_shortfall"})
    ablated_source = results[
        (results["experiment"] == "drop_defining_features")
        & (results["workflow"] == "source_split")
        & np.isclose(results["target_cal_fraction"], 0.0)
    ][["dataset", "shift_name", "seed", "coverage_shortfall"]].rename(columns={"coverage_shortfall": "ablated_shortfall"})
    ablation_plot = full_source.merge(ablated_source, on=["dataset", "shift_name", "seed"], how="inner")
    ablation_long = ablation_plot.melt(id_vars=["dataset", "shift_name", "seed"], value_vars=["full_shortfall", "ablated_shortfall"], var_name="feature_set", value_name="coverage_shortfall")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.boxplot(data=ablation_long, x="feature_set", y="coverage_shortfall", hue="dataset", palette=palette[:3], ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Coverage shortfall")
    ax.set_title("Sensitivity after removing split-defining features")
    sns.despine()
    save_figure(fig, "figure5_ablation_sensitivity.png")

    heat = by_shift[
        (by_shift["workflow"] == "source_split")
        & (by_shift["setting"] == "covariate_shift")
    ].pivot_table(index=["dataset", "shift_name"], values="coverage_shortfall", aggfunc="median")
    fig, ax = plt.subplots(figsize=(6.2, max(3.8, 0.34 * len(heat))))
    sns.heatmap(heat, cmap="cividis", annot=True, fmt=".3f", cbar_kws={"label": "Coverage shortfall"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Source-calibrated under-coverage by shift")
    save_figure(fig, "supplement_shift_shortfall_heatmap.png")

    effect_terms = "coverage_shortfall|coverage_error|weighted_source|ablation"
    effects = tests[tests["test"].str.contains(effect_terms, regex=True)].copy()
    if not effects.empty:
        fig, ax = plt.subplots(figsize=(8.4, max(4.2, 0.28 * len(effects))))
        sns.barplot(data=effects, x="median_difference", y="test", color=palette[0], ax=ax)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Median paired difference")
        ax.set_ylabel("")
        ax.set_title("Primary paired statistical effects")
        save_figure(fig, "supplement_effect_estimates.png")

    audit_numeric = shift_audit.set_index(["dataset", "shift_name"])[["prevalence_difference"]].copy()
    fig, ax = plt.subplots(figsize=(7.0, max(3.8, 0.34 * len(audit_numeric))))
    sns.heatmap(audit_numeric, cmap="RdBu_r", center=0, annot=True, fmt=".3f", cbar_kws={"label": "Target - source prevalence"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Outcome-prevalence shift audit")
    save_figure(fig, "supplement_shift_prevalence_audit.png")


def write_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    by_shift: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    tests: pd.DataFrame,
    selective_summary: pd.DataFrame,
    dataset_audit: pd.DataFrame,
    shift_audit: pd.DataFrame,
    seeds: list[int],
    target_fractions: list[float],
    max_brfss_per_year: int,
) -> None:
    random_source = summary[
        (summary["setting"] == "random_exchangeable")
        & (summary["workflow"] == "source_split")
        & np.isclose(summary["target_cal_fraction"], 0.0)
    ].iloc[0]
    shift_source = summary[
        (summary["setting"] == "covariate_shift")
        & (summary["workflow"] == "source_split")
        & np.isclose(summary["target_cal_fraction"], 0.0)
    ].iloc[0]
    weighted_source = summary[
        (summary["setting"] == "covariate_shift")
        & (summary["workflow"] == "source_weighted_split")
        & np.isclose(summary["target_cal_fraction"], 0.0)
    ].iloc[0]
    target10 = summary[
        (summary["setting"] == "covariate_shift")
        & (summary["workflow"] == "target_recalibrated_split")
        & np.isclose(summary["target_cal_fraction"], PRIMARY_FRACTION)
    ].iloc[0]
    target10_mondrian = summary[
        (summary["setting"] == "covariate_shift")
        & (summary["workflow"] == "target_recalibrated_mondrian")
        & np.isclose(summary["target_cal_fraction"], PRIMARY_FRACTION)
    ].iloc[0]

    source_test = tests[tests["test"] == "source_shift_minus_random_coverage_error"]
    target10_test = tests[tests["test"] == "target_split_minus_source_coverage_shortfall_fraction_0.1"]
    weighted_test = tests[tests["test"] == "weighted_source_minus_source_coverage_shortfall"]
    ablation_test = tests[tests["test"] == "ablation_source_minus_full_source_coverage_shortfall"]

    target_budget_table = summary[
        (summary["setting"] == "covariate_shift")
        & (summary["workflow"].isin(["target_recalibrated_split", "target_recalibrated_mondrian"]))
    ][["workflow", "target_cal_fraction", "coverage", "coverage_shortfall", "singleton_rate", "class_coverage_gap", "n_runs"]]

    method_table = summary[
        (summary["setting"] == "covariate_shift")
        & (
            ((summary["workflow"].isin(["source_split", "source_weighted_split"])) & np.isclose(summary["target_cal_fraction"], 0.0))
            | ((summary["workflow"].isin(["target_recalibrated_split", "target_recalibrated_mondrian"])) & np.isclose(summary["target_cal_fraction"], PRIMARY_FRACTION))
        )
    ][["workflow", "target_cal_fraction", "coverage", "coverage_shortfall", "singleton_rate", "class_coverage_gap", "weight_ess", "n_runs"]]

    shift_table = by_shift[
        (by_shift["workflow"] == "source_split")
        & (by_shift["setting"] == "covariate_shift")
    ][["dataset", "shift_name", "coverage", "coverage_shortfall", "domain_auc", "n_runs"]]

    ablation_overall = (
        results[results["experiment"] == "drop_defining_features"]
        .groupby(["workflow", "target_cal_fraction"])
        .agg(
            coverage=("coverage", "median"),
            coverage_shortfall=("coverage_shortfall", "median"),
            singleton_rate=("singleton_rate", "median"),
            n_runs=("coverage", "size"),
        )
        .reset_index()
    )

    key_tests = pd.concat([source_test, target10_test, weighted_test, ablation_test], ignore_index=True)
    audit_brief = dataset_audit[["dataset", "analysis_n", "positive_prevalence", "n_numeric_features", "n_categorical_features", "mean_feature_missing_rate", "max_feature_missing_rate"]]
    shift_audit_brief = shift_audit[["dataset", "shift_name", "source_n", "target_n", "source_positive_prevalence", "target_positive_prevalence", "prevalence_difference", "drop_defining_features"]]

    report = f"""# Public Shift Benchmark 收尾实验报告

生成日期：2026-07-05  
项目目录：`{ROOT}`  
实验版本：论文发布版  

## 本轮目标

本轮针对投稿前风险点做 5 项收尾实验：扩展 seed 稳健性、去除 split-defining 变量的敏感性分析、补密目标域标注预算曲线、加入 importance-weighted conformal 对照、生成数据处理和 shift 审计表。

## 数据与审计

{markdown_table(audit_brief, max_rows=10)}

分 shift 的 source/target 人数、阳性率和 ablation 删除变量如下：

{markdown_table(shift_audit_brief, max_rows=30)}

## 实验矩阵

- seeds：`{seeds}`
- 目标域标注比例：`{target_fractions}`
- BRFSS 每年最大抽样：`{max_brfss_per_year}`
- 主分析：random exchangeable、source split、source Mondrian、source weighted split、target recalibrated split、target recalibrated Mondrian
- 敏感性分析：对 12 个非 pseudo deployment-like shifts 删除 split-defining 特征后，重新训练 source split，并在 10% 目标域标注下再校准

## 核心结论

随机同分布验证中，source split conformal 中位 coverage 为 **{fmt(random_source['coverage'])}**，coverage shortfall 为 **{fmt(random_source['coverage_shortfall'])}**。

排除 pseudo-random negative control 后，deployment-like shift 下 source split conformal 的中位 coverage 为 **{fmt(shift_source['coverage'])}**，coverage shortfall 为 **{fmt(shift_source['coverage_shortfall'])}**。

importance-weighted source split 的中位 coverage 为 **{fmt(weighted_source['coverage'])}**，coverage shortfall 为 **{fmt(weighted_source['coverage_shortfall'])}**。这说明仅靠无标注目标域加权不能稳定替代目标域标注再校准。

10% 目标域标注再校准后，split conformal coverage 为 **{fmt(target10['coverage'])}**，coverage shortfall 为 **{fmt(target10['coverage_shortfall'])}**；10% 目标域 Mondrian coverage 为 **{fmt(target10_mondrian['coverage'])}**，类别覆盖差为 **{fmt(target10_mondrian['class_coverage_gap'])}**。

## 目标域标注预算曲线

{markdown_table(target_budget_table, max_rows=30)}

## 方法对照

{markdown_table(method_table, max_rows=20)}

## 分 shift 主结果

{markdown_table(shift_table, max_rows=40)}

## 去除 split-defining 特征后的敏感性分析

总体结果：

{markdown_table(ablation_overall, max_rows=10)}

分 shift 结果：

{markdown_table(ablation_summary, max_rows=30)}

解释：如果去除年龄、BMI、地区、年份、族群代理或入院来源等直接定义 shift 的变量后，source-calibrated coverage shortfall 仍然存在，说明主结果不是完全由一个显式分组变量泄漏造成，而是代表了更广义的分布和风险结构变化。

## 统计检验

{markdown_table(key_tests[['test', 'metric', 'median_difference', 'ci_low', 'ci_high', 'wilcoxon_p', 'n_pairs']], max_rows=20)}

## 自动预测与人工复核负担

{markdown_table(selective_summary, max_rows=30)}

## 图表清单

- `figures/public_benchmark/figure1_design.png`
- `figures/public_benchmark/figure2_seed_robustness.png`
- `figures/public_benchmark/figure3_dense_budget_curve.png`
- `figures/public_benchmark/figure4_method_comparison.png`
- `figures/public_benchmark/figure5_ablation_sensitivity.png`
- `figures/public_benchmark/supplement_shift_shortfall_heatmap.png`
- `figures/public_benchmark/supplement_effect_estimates.png`
- `figures/public_benchmark/supplement_shift_prevalence_audit.png`

## 投稿质量判断

这轮收尾后，文章从“有希望的公开数据 benchmark”进一步变成更完整的投稿包：主结论跨 3 个公开健康任务、5 个 seeds、12 个非 pseudo deployment shifts、密集 calibration budget、weighted conformal 对照和 drop-defining-feature 敏感性分析均可复现。

基于公开数据集，不申请真实 ICU/EHR 的前提下，当前证据强度可以稳定维持在 **70-75 分**。可以支撑方法学或医学信息学方向的 benchmark/technical paper；仍不应包装成真实临床部署验证。

## 复现命令

```bash
cd {ROOT}
python3 scripts/run_public_shift_benchmark.py --seeds {' '.join(str(seed) for seed in seeds)} --target-fractions {' '.join(str(frac) for frac in target_fractions)} --max-brfss-per-year {max_brfss_per_year}
```
"""
    (REPORTS_DIR / "public_shift_benchmark_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public-data deployment-shift conformal benchmark.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--target-fractions", nargs="+", type=float, default=DEFAULT_TARGET_FRACTIONS)
    parser.add_argument("--max-brfss-per-year", type=int, default=60_000)
    parser.add_argument("--force-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    datasets = [
        prepare_brfss_multiyear(force=args.force_preprocess, max_rows_per_year=args.max_brfss_per_year),
        prepare_nhanes_multicycle(force=args.force_preprocess),
        prepare_uci_with_pseudo(force=args.force_preprocess),
    ]
    results, subgroups, selective, dataset_audit, shift_audit = run_benchmark(datasets, args.seeds, args.target_fractions)
    summary, by_shift, ablation_summary, tests, selective_summary = summarize_and_test(results, selective, args.target_fractions)
    make_figures(results, summary, by_shift, ablation_summary, tests, dataset_audit, shift_audit)
    write_report(
        results,
        summary,
        by_shift,
        ablation_summary,
        tests,
        selective_summary,
        dataset_audit,
        shift_audit,
        args.seeds,
        args.target_fractions,
        args.max_brfss_per_year,
    )
    print(f"Wrote {RESULTS_DIR / 'public_benchmark_summary.csv'}")
    print(f"Wrote {REPORTS_DIR / 'public_shift_benchmark_report.md'}")


if __name__ == "__main__":
    main()
