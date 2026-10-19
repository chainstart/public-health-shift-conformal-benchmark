from __future__ import annotations

import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[1]
REFERENCE_DIR = PROJECT_ROOT / "exp" / "source_role_outcome_blind_20260725"
PRIOR_DIR = PROJECT_ROOT / "exp" / "shift_aware_baselines_20260813"
PROVENANCE_PATH = REFERENCE_DIR / "generation_provenance.json"
PRIOR_METRICS_PATH = PRIOR_DIR / "shift_aware_replicate_metrics.parquet"
ALPHA = 0.10
METHOD = "lrqr_300_300_300"
MAX_COMPONENTS = 50
RIDGE = 1.0e-8
MATCHED_NAMESPACE = "ara.shift_aware_baselines.draw.v1"
LRQR_NAMESPACE = "ara.lrqr_baseline.draw.v1"
METRICS = (
    "coverage",
    "coverage_shortfall",
    "mean_set_size",
    "empty_set_rate",
    "singleton_rate",
    "full_set_rate",
    "class_0_coverage",
    "class_1_coverage",
    "class_coverage_gap",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_hash(values: pd.Series) -> str:
    return hashlib.sha256("\x1f".join(sorted(values.astype(str))).encode()).hexdigest()


def seed_for(namespace: str, *parts: Any) -> int:
    payload = "\x1f".join((namespace, *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_roles(pair: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_path = PROJECT_ROOT / pair["prediction_path"]
    covariate_path = PROJECT_ROOT / pair["covariate_path"]
    predictions = pd.read_parquet(prediction_path)
    covariates = pd.read_parquet(covariate_path)
    joined = predictions.merge(
        covariates,
        on=["dataset", "shift", "seed", "row_id"],
        validate="one_to_one",
    )
    return tuple(
        joined.loc[joined["role"] == role].reset_index(drop=True)
        for role in ("source_calibration", "target_calibration_pool", "target_test")
    )  # type: ignore[return-value]


def reconstruct_features(
    source: pd.DataFrame,
    target: pd.DataFrame,
    test: pd.DataFrame,
    pair: dict[str, Any],
    labeled_index: np.ndarray,
    unlabeled_source_index: np.ndarray,
    target_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    numeric = list(pair["numeric_covariates"])
    categorical = list(pair["categorical_covariates"])
    columns = list(pair["covariate_columns"])
    transform = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(
                        handle_unknown="ignore", drop="first", sparse_output=False
                    ),
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )
    transform.fit(
        pd.concat(
            [
                source.iloc[labeled_index][columns],
                source.iloc[unlabeled_source_index][columns],
            ],
            ignore_index=True,
        )
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories")
        raw_labeled = np.asarray(transform.transform(source.iloc[labeled_index][columns]), dtype=float)
        raw_source = np.asarray(
            transform.transform(source.iloc[unlabeled_source_index][columns]), dtype=float
        )
        raw_target = np.asarray(transform.transform(target.iloc[target_index][columns]), dtype=float)
        raw_test = np.asarray(transform.transform(test[columns]), dtype=float)
    rank = int(np.linalg.matrix_rank(raw_source - raw_source.mean(axis=0, keepdims=True)))
    components = min(MAX_COMPONENTS, rank, raw_source.shape[1])
    pca = PCA(n_components=components, whiten=True, svd_solver="full").fit(raw_source)

    def project(raw: np.ndarray) -> np.ndarray:
        value = pca.transform(raw)
        if value.shape[1] < MAX_COMPONENTS:
            value = np.pad(value, ((0, 0), (0, MAX_COMPONENTS - value.shape[1])))
        return np.column_stack((np.ones(len(value)), value)).astype(float)

    return (
        project(raw_labeled),
        project(raw_target),
        project(raw_source),
        project(raw_test),
        raw_source.shape[1],
        components,
    )


def calculate_metrics(test: pd.DataFrame, threshold: np.ndarray) -> dict[str, float]:
    probability = test["prob_1"].to_numpy(dtype=float)
    label = test["label"].to_numpy(dtype=np.int8)
    include_zero = probability <= threshold
    include_one = 1.0 - probability <= threshold
    covered = np.where(label == 0, include_zero, include_one)
    size = include_zero.astype(int) + include_one.astype(int)
    by_class = [float(covered[label == candidate].mean()) for candidate in (0, 1)]
    coverage = float(covered.mean())
    return {
        "coverage": coverage,
        "coverage_shortfall": max(0.0, 1.0 - ALPHA - coverage),
        "mean_set_size": float(size.mean()),
        "empty_set_rate": float((size == 0).mean()),
        "singleton_rate": float((size == 1).mean()),
        "full_set_rate": float((size == 2).mean()),
        "class_0_coverage": by_class[0],
        "class_1_coverage": by_class[1],
        "class_coverage_gap": abs(by_class[1] - by_class[0]),
    }


def objective_gradient(
    theta: np.ndarray,
    beta: float,
    regularization: float,
    x_labeled: np.ndarray,
    score: np.ndarray,
    x_target: np.ndarray,
    x_source: np.ndarray,
) -> tuple[float, float]:
    h1, h2, h3 = x_labeled @ theta, x_target @ theta, x_source @ theta
    residual = score - h1
    pinball = np.where(residual >= 0, (1.0 - ALPHA) * residual, -ALPHA * residual)
    objective = float(
        pinball.mean()
        + regularization * (beta * beta * np.mean(h3 * h3) - 2.0 * beta * np.mean(h2))
        + RIDGE * np.dot(theta[1:], theta[1:])
    )
    score_gradient = np.where(residual > 0, -(1.0 - ALPHA), ALPHA)
    theta_gradient = (
        x_labeled.T @ score_gradient / len(x_labeled)
        + 2.0 * regularization * beta * beta * x_source.T @ h3 / len(x_source)
        - 2.0 * regularization * beta * x_target.mean(axis=0)
    )
    theta_gradient[1:] += 2.0 * RIDGE * theta[1:]
    beta_gradient = 2.0 * regularization * beta * np.mean(h3 * h3) - 2.0 * regularization * np.mean(h2)
    return objective, float(np.linalg.norm(np.r_[theta_gradient, beta_gradient]))


def aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = (
        frame.groupby(["dataset", "shift", "seed", "method"], observed=True)[list(METRICS)]
        .mean()
        .reset_index()
    )
    shift = (
        seed.groupby(["dataset", "shift", "method"], observed=True)[list(METRICS)]
        .mean()
        .reset_index()
    )
    dataset = (
        shift.groupby(["dataset", "method"], observed=True)[list(METRICS)]
        .mean()
        .reset_index()
    )
    equal = dataset.groupby("method", observed=True)[list(METRICS)].mean().reset_index()
    return shift, dataset, equal


def close(left: float, right: float, tolerance: float = 2.0e-7) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return abs(left - right) <= tolerance


def main() -> int:
    results = json.loads((RUN_DIR / "results.json").read_text())
    lrqr_metrics = pd.read_parquet(RUN_DIR / "lrqr_replicate_metrics.parquet")
    extended = pd.read_parquet(RUN_DIR / "extended_replicate_metrics.parquet")
    parameters = pd.read_parquet(RUN_DIR / "lrqr_parameters.parquet")
    diagnostics = pd.read_csv(RUN_DIR / "lrqr_diagnostics.csv")
    draws = pd.read_csv(RUN_DIR / "draw_audit.csv")
    cv = pd.read_csv(RUN_DIR / "lrqr_cv_trace.csv")
    provenance = json.loads(PROVENANCE_PATH.read_text())
    metric_lookup = lrqr_metrics.set_index(["dataset", "shift", "seed", "replicate"])
    diagnostic_lookup = diagnostics.set_index(["dataset", "shift", "seed", "replicate"])
    draw_lookup = draws.set_index(["dataset", "shift", "seed", "replicate"])
    parameter_groups = {
        key: subset.sort_values("coefficient_position")["coefficient"].to_numpy(dtype=float)
        for key, subset in parameters.groupby(["dataset", "shift", "seed", "replicate"])
    }

    artifact_hash_mismatches = [
        name
        for name, record in results["artifacts"].items()
        if sha256_file(RUN_DIR / name) != record["sha256"]
    ]
    input_hash_mismatches = 0
    role_overlaps = 0
    draw_mismatches = 0
    source_overlap_count = 0
    feature_mismatches = 0
    objective_mismatches = 0
    metric_mismatches = 0
    threshold_mismatches = 0
    checked_problems = 0
    mismatch_examples: list[dict[str, Any]] = []

    for pair_position, pair in enumerate(provenance["pairs"], start=1):
        prediction_path = PROJECT_ROOT / pair["prediction_path"]
        covariate_path = PROJECT_ROOT / pair["covariate_path"]
        input_hash_mismatches += int(sha256_file(prediction_path) != pair["prediction_sha256"])
        input_hash_mismatches += int(sha256_file(covariate_path) != pair["covariate_sha256"])
        source, target, test = load_roles(pair)
        role_sets = [set(part["row_id"].astype(str)) for part in (source, target, test)]
        role_overlaps += sum(
            bool(role_sets[a].intersection(role_sets[b])) for a, b in ((0, 1), (0, 2), (1, 2))
        )
        dataset, shift, seed = pair["dataset"], pair["shift"], int(pair["seed"])
        for replicate in range(5):
            key = (dataset, shift, seed, replicate)
            source_seed = seed_for(MATCHED_NAMESPACE, *key, "source")
            target_seed = seed_for(MATCHED_NAMESPACE, *key, "target")
            unlabeled_seed = seed_for(LRQR_NAMESPACE, *key, "source_unlabeled")
            labeled_index = np.random.default_rng(source_seed).choice(len(source), 300, replace=False)
            target_index = np.random.default_rng(target_seed).choice(len(target), 300, replace=False)
            remaining = np.setdiff1d(np.arange(len(source)), labeled_index, assume_unique=False)
            unlabeled_index = remaining[
                np.random.default_rng(unlabeled_seed).choice(len(remaining), 300, replace=False)
            ]
            source_overlap_count += int(bool(set(labeled_index).intersection(unlabeled_index)))
            draw = draw_lookup.loc[key]
            draw_mismatches += int(draw["source_row_ids_sha256"] != ids_hash(source.iloc[labeled_index]["row_id"]))
            draw_mismatches += int(draw["target_row_ids_sha256"] != ids_hash(target.iloc[target_index]["row_id"]))
            draw_mismatches += int(
                draw["source_unlabeled_row_ids_sha256"] != ids_hash(source.iloc[unlabeled_index]["row_id"])
            )
            x1, x2, x3, xtest, raw_count, components = reconstruct_features(
                source, target, test, pair, labeled_index, unlabeled_index, target_index
            )
            observed = metric_lookup.loc[key]
            diagnostic = diagnostic_lookup.loc[key]
            feature_mismatches += int(int(diagnostic["raw_feature_count"]) != raw_count)
            feature_mismatches += int(int(diagnostic["pca_components"]) != components)
            theta = parameter_groups[key]
            probability = source.iloc[labeled_index]["prob_1"].to_numpy(dtype=float)
            label = source.iloc[labeled_index]["label"].to_numpy(dtype=np.int8)
            score = np.where(label == 0, probability, 1.0 - probability)
            objective, gradient_norm = objective_gradient(
                theta,
                float(observed["selected_beta"]),
                float(observed["selected_lambda"]),
                x1,
                score,
                x2,
                x3,
            )
            if not close(objective, float(diagnostic["objective"])) or not close(
                gradient_norm, float(diagnostic["gradient_norm"])
            ):
                objective_mismatches += 1
            threshold = xtest @ theta
            expected_metrics = calculate_metrics(test, threshold)
            local_mismatches = [
                name for name, value in expected_metrics.items() if not close(value, float(observed[name]))
            ]
            metric_mismatches += len(local_mismatches)
            threshold_mismatches += sum(
                not close(value, float(observed[name]))
                for name, value in (
                    ("threshold_minimum", float(threshold.min())),
                    ("threshold_median", float(np.median(threshold))),
                    ("threshold_maximum", float(threshold.max())),
                )
            )
            if local_mismatches and len(mismatch_examples) < 5:
                mismatch_examples.append({"key": list(key), "metrics": local_mismatches})
            checked_problems += 1
        print(f"verified {pair_position}/60 shift-seed pairs", flush=True)

    cv_groups = cv.groupby(["dataset", "shift", "seed", "replicate"], sort=False)
    cv_selection_mismatches = 0
    for _, subset in cv_groups:
        expected = subset["mean_validation_gradient_norm"].idxmin()
        selected = subset.index[subset["selected"]]
        cv_selection_mismatches += int(len(selected) != 1 or int(selected[0]) != int(expected))

    prior = pd.read_parquet(PRIOR_METRICS_PATH)
    prior_keys = ["dataset", "shift", "seed", "replicate", "method"]
    observed_prior = (
        extended.loc[extended["method"] != METHOD]
        .sort_values(prior_keys)
        .reset_index(drop=True)[prior.columns]
    )
    expected_prior = prior.sort_values(prior_keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(observed_prior, expected_prior, check_dtype=False)
        prior_mismatch = 0
    except AssertionError:
        prior_mismatch = 1
    combined_lrqr = (
        extended.loc[extended["method"] == METHOD]
        .sort_values(prior_keys)
        .reset_index(drop=True)[lrqr_metrics.columns]
    )
    expected_lrqr = lrqr_metrics.sort_values(prior_keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(combined_lrqr, expected_lrqr, check_dtype=False)
        combined_lrqr_mismatch = 0
    except AssertionError:
        combined_lrqr_mismatch = 1

    shift, dataset, equal = aggregate(extended)
    summary_mismatches = 0
    for calculated, filename, keys in (
        (shift, "extended_shift_summary.csv", ["dataset", "shift", "method"]),
        (dataset, "extended_dataset_summary.csv", ["dataset", "method"]),
        (equal, "extended_equal_dataset_summary.csv", ["method"]),
    ):
        observed_summary = pd.read_csv(RUN_DIR / filename).sort_values(keys).reset_index(drop=True)
        calculated = calculated.sort_values(keys).reset_index(drop=True)
        summary_mismatches += int(
            not np.allclose(
                observed_summary[list(METRICS)].to_numpy(dtype=float),
                calculated[list(METRICS)].to_numpy(dtype=float),
                atol=2.0e-12,
                rtol=0.0,
            )
        )

    checks = {
        "artifact_hash_mismatches": artifact_hash_mismatches,
        "input_hash_mismatches": input_hash_mismatches,
        "role_overlaps": role_overlaps,
        "source_labeled_unlabeled_overlaps": source_overlap_count,
        "draw_mismatches": draw_mismatches,
        "feature_mismatches": feature_mismatches,
        "objective_or_gradient_mismatches": objective_mismatches,
        "threshold_mismatches": threshold_mismatches,
        "metric_mismatches": metric_mismatches,
        "cv_selection_mismatches": cv_selection_mismatches,
        "prior_metrics_copy_mismatch": prior_mismatch,
        "combined_lrqr_copy_mismatch": combined_lrqr_mismatch,
        "summary_mismatches": summary_mismatches,
        "checked_problems": checked_problems,
        "mismatch_examples": mismatch_examples,
    }
    failure_count = (
        len(artifact_hash_mismatches)
        + input_hash_mismatches
        + role_overlaps
        + source_overlap_count
        + draw_mismatches
        + feature_mismatches
        + objective_mismatches
        + threshold_mismatches
        + metric_mismatches
        + cv_selection_mismatches
        + prior_mismatch
        + combined_lrqr_mismatch
        + summary_mismatches
        + int(checked_problems != 300)
    )
    receipt = {
        "schema_version": "ara.public_shift.lrqr_verification.v1",
        "status": "PASS" if failure_count == 0 else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent reconstruction of input hashes, role/draw separation, source-supported feature maps, saved LR-QR objectives and gradients, prediction sets, metrics, aggregation, and CV selection. The optimizer trajectory itself is not rerun.",
        "checks": checks,
    }
    (RUN_DIR / "validation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
