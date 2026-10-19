from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SCHEMA_VERSION = "ara.public_shift.shift_aware_baselines.v1"
ALPHA = 0.10
N_SOURCE = 300
N_TARGET = 300
N_REPLICATES = 5
OT_STEPS = 4000
OT_LEARNING_RATE = 1.0e-3
OT_GRID_SIZE = 401
WEIGHT_LOWER = 0.05
WEIGHT_UPPER = 20.0

METHODS = (
    "source_split_full",
    "source_mondrian_full",
    "source_split_300",
    "target_split_300",
    "target_mondrian_300",
    "label_shift_weighted_300",
    "score_covariate_weighted_300",
    "ot_minmax_weighted_cdf_300",
)
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

SCRIPT_PATH = Path(__file__).resolve()
RUN_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
REFERENCE_DIR = PROJECT_ROOT / "exp" / "source_role_outcome_blind_20260725"
GENERATION_PROVENANCE_PATH = REFERENCE_DIR / "generation_provenance.json"
MATCHED_METRICS_PATH = (
    PROJECT_ROOT
    / "exp"
    / "matched_borrowing_controls_20260813"
    / "matched_replicate_metrics.parquet"
)
SHIFT_AUDIT_PATH = (
    PROJECT_ROOT
    / "benchmark_release"
    / "public_health_shift_conformal_benchmark_v3"
    / "results"
    / "public_benchmark_shift_audit.csv"
)
DRAW_NAMESPACE = "ara.shift_aware_baselines.draw.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_seed(namespace: str, *parts: Any) -> int:
    payload = "\x1f".join((namespace, *(str(part) for part in parts))).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def atomic_target(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    return descriptor, Path(name)


def atomic_json(path: Path, payload: Any) -> None:
    descriptor, temporary = atomic_target(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    descriptor, temporary = atomic_target(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(materialized[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary = atomic_target(path)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    probability = frame["prob_1"].to_numpy(dtype=np.float64)
    label = frame["label"].to_numpy(dtype=np.int8)
    return {
        "p": probability,
        "score0": probability,
        "score1": 1.0 - probability,
        "true_score": np.where(label == 0, probability, 1.0 - probability),
        "y": label,
        "row_id": frame["row_id"].astype(str).to_numpy(),
    }


def ordinary_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64))
    if not len(values):
        return math.inf
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    return math.inf if rank > len(values) else float(values[rank - 1])


def mondrian_quantiles(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    return tuple(
        ordinary_quantile(scores[labels == candidate]) for candidate in (0, 1)
    )  # type: ignore[return-value]


def weighted_quantile(
    scores: np.ndarray,
    calibration_weights: np.ndarray,
    test_weight: float,
    alpha: float = ALPHA,
) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(calibration_weights, dtype=np.float64)
    if len(scores) != len(weights) or np.any(weights < 0):
        raise ValueError("Invalid weighted-quantile inputs")
    order = np.argsort(scores, kind="mergesort")
    threshold_mass = (1.0 - alpha) * (float(weights.sum()) + test_weight)
    tolerance = 1.0e-12 * max(1.0, float(weights.sum()) + test_weight)
    if threshold_mass > float(weights.sum()) + tolerance:
        return math.inf
    position = int(
        np.searchsorted(
            np.cumsum(weights[order]), threshold_mass - tolerance, side="left"
        )
    )
    return float(scores[order[min(position, len(scores) - 1)]])


def weighted_threshold_vector(
    scores: np.ndarray,
    calibration_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha: float = ALPHA,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(calibration_weights, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    cumulative = np.cumsum(weights[order])
    total = float(cumulative[-1])
    target = (1.0 - alpha) * (total + np.asarray(test_weights, dtype=float))
    tolerance = 1.0e-12 * np.maximum(1.0, total + np.asarray(test_weights, dtype=float))
    position = np.searchsorted(cumulative, target - tolerance, side="left")
    output = np.full(len(target), np.inf, dtype=np.float64)
    finite = position < len(ordered_scores)
    output[finite] = ordered_scores[position[finite]]
    return output


def evaluate(
    test: dict[str, np.ndarray], q0: float | np.ndarray, q1: float | np.ndarray
) -> dict[str, float]:
    include0 = test["score0"] <= q0
    include1 = test["score1"] <= q1
    covered = np.where(test["y"] == 0, include0, include1)
    size = include0.astype(np.int8) + include1.astype(np.int8)
    class_coverage = []
    for candidate in (0, 1):
        mask = test["y"] == candidate
        class_coverage.append(float(covered[mask].mean()))
    coverage = float(covered.mean())
    return {
        "coverage": coverage,
        "coverage_shortfall": max(0.0, 1.0 - ALPHA - coverage),
        "mean_set_size": float(size.mean()),
        "empty_set_rate": float((size == 0).mean()),
        "singleton_rate": float((size == 1).mean()),
        "full_set_rate": float((size == 2).mean()),
        "class_0_coverage": class_coverage[0],
        "class_1_coverage": class_coverage[1],
        "class_coverage_gap": abs(class_coverage[1] - class_coverage[0]),
    }


def estimate_label_weights(
    source: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, float]]:
    predicted_source = (source["p"] >= 0.5).astype(np.int8)
    predicted_target = (target["p"] >= 0.5).astype(np.int8)
    confusion = np.zeros((2, 2), dtype=np.float64)
    for label in (0, 1):
        mask = source["y"] == label
        confusion[:, label] = np.bincount(
            predicted_source[mask], minlength=2
        ) / max(1, int(mask.sum()))
    target_prediction_frequency = np.bincount(
        predicted_target, minlength=2
    ).astype(float) / len(predicted_target)
    target_prior = np.linalg.lstsq(
        confusion, target_prediction_frequency, rcond=None
    )[0]
    target_prior = np.clip(target_prior, 1.0e-6, None)
    target_prior /= target_prior.sum()
    source_prior = np.bincount(source["y"], minlength=2).astype(float)
    source_prior /= source_prior.sum()
    ratios = np.clip(target_prior / source_prior, WEIGHT_LOWER, WEIGHT_UPPER)
    diagnostic = {
        "estimated_target_prior_0": float(target_prior[0]),
        "estimated_target_prior_1": float(target_prior[1]),
        "source_prior_0": float(source_prior[0]),
        "source_prior_1": float(source_prior[1]),
        "label_weight_0": float(ratios[0]),
        "label_weight_1": float(ratios[1]),
        "confusion_condition_number": float(np.linalg.cond(confusion)),
    }
    return ratios, diagnostic


def score_density_ratios(
    source_p: np.ndarray, target_p: np.ndarray, evaluation_p: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    epsilon = 1.0e-6

    def features(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, epsilon, 1.0 - epsilon)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return logits

    transformer = PolynomialFeatures(degree=3, include_bias=False)
    source_feature = transformer.fit_transform(features(source_p))
    target_feature = transformer.transform(features(target_p))
    scaler = StandardScaler().fit(np.vstack((source_feature, target_feature)))
    design = scaler.transform(np.vstack((source_feature, target_feature)))
    domain = np.concatenate(
        (np.zeros(len(source_p), dtype=np.int8), np.ones(len(target_p), dtype=np.int8))
    )
    classifier = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000, random_state=0
    ).fit(design, domain)

    def ratio(values: np.ndarray) -> np.ndarray:
        transformed = scaler.transform(transformer.transform(features(values)))
        probability_target = classifier.predict_proba(transformed)[:, 1]
        raw = probability_target / np.clip(1.0 - probability_target, epsilon, None)
        prior_correction = len(source_p) / len(target_p)
        return np.clip(raw * prior_correction, WEIGHT_LOWER, WEIGHT_UPPER)

    source_weight = ratio(source_p)
    evaluation_weight = ratio(evaluation_p)
    diagnostic = {
        "source_weight_min": float(source_weight.min()),
        "source_weight_max": float(source_weight.max()),
        "source_weight_mean": float(source_weight.mean()),
        "source_weight_ess": float(source_weight.sum() ** 2 / np.square(source_weight).sum()),
        "domain_classifier_accuracy": float(classifier.score(design, domain)),
    }
    return source_weight, evaluation_weight, diagnostic


def prepare_ot_problem(
    source_scores: np.ndarray, target_p: np.ndarray
) -> dict[str, np.ndarray | float]:
    grid = np.linspace(0.0, 1.0, OT_GRID_SIZE, dtype=np.float64)
    target_min = np.minimum(target_p, 1.0 - target_p)
    target_max = np.maximum(target_p, 1.0 - target_p)
    standard_deviation = float(np.std(source_scores, ddof=1))
    bandwidth = max(0.01, standard_deviation * len(source_scores) ** (-1.0 / 5.0))
    indicator = (source_scores[:, None] <= grid[None, :]).astype(np.float32)
    kernel = np.exp(
        -0.5 * np.square((grid[None, :] - source_scores[:, None]) / bandwidth)
    ) / (math.sqrt(2.0 * math.pi) * bandwidth)
    return {
        "grid": grid.astype(np.float32),
        "indicator": indicator,
        "kernel": kernel.astype(np.float32),
        "target_min_cdf": (target_min[:, None] <= grid[None, :]).mean(axis=0).astype(np.float32),
        "target_max_cdf": (target_max[:, None] <= grid[None, :]).mean(axis=0).astype(np.float32),
        "bandwidth": bandwidth,
    }


def optimize_ot_batch(
    problems: list[dict[str, np.ndarray | float]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    device = torch.device("cpu")
    grid = torch.as_tensor(np.stack([p["grid"] for p in problems]), device=device)
    indicator = torch.as_tensor(
        np.stack([p["indicator"] for p in problems]), device=device
    )
    kernel = torch.as_tensor(np.stack([p["kernel"] for p in problems]), device=device)
    target_min_cdf = torch.as_tensor(
        np.stack([p["target_min_cdf"] for p in problems]), device=device
    )
    target_max_cdf = torch.as_tensor(
        np.stack([p["target_max_cdf"] for p in problems]), device=device
    )
    logits = torch.zeros(
        (len(problems), N_SOURCE), dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([logits], lr=OT_LEARNING_RATE)
    trace_steps = {0, 1, 10, 50, 100, 250, 500, 750, OT_STEPS}
    traces: list[list[dict[str, float]]] = [[] for _ in problems]

    def loss_vector() -> torch.Tensor:
        weight = torch.softmax(logits, dim=1)
        source_cdf = torch.einsum("bn,bng->bg", weight, indicator)
        source_density = torch.einsum("bn,bng->bg", weight, kernel)
        integrand = source_density * (
            torch.abs(source_cdf - target_max_cdf)
            + torch.abs(source_cdf - target_min_cdf)
            + target_min_cdf
            - target_max_cdf
        )
        return 0.5 * torch.trapezoid(integrand, grid, dim=1)

    with torch.no_grad():
        initial = loss_vector()
        for position, value in enumerate(initial.detach().cpu().numpy()):
            traces[position].append({"step": 0, "objective": float(value)})
    for step in range(1, OT_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = loss_vector()
        losses.sum().backward()
        optimizer.step()
        if step in trace_steps:
            with torch.no_grad():
                observed = loss_vector().detach().cpu().numpy()
            for position, value in enumerate(observed):
                traces[position].append({"step": step, "objective": float(value)})
    normalized = torch.softmax(logits, dim=1).detach().cpu().numpy()
    mean_one = normalized * N_SOURCE
    diagnostics = []
    for position, weight in enumerate(mean_one):
        diagnostics.append(
            {
                "device": str(device),
                "bandwidth": float(problems[position]["bandwidth"]),
                "initial_objective": traces[position][0]["objective"],
                "final_objective": traces[position][-1]["objective"],
                "effective_sample_size": float(weight.sum() ** 2 / np.square(weight).sum()),
                "minimum_weight": float(weight.min()),
                "maximum_weight": float(weight.max()),
                "trace": traces[position],
            }
        )
    return mean_one.astype(np.float64), diagnostics


def aggregate_levels(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    within_seed = (
        frame.groupby(["dataset", "shift", "seed", "method"], observed=True)[list(METRICS)]
        .mean()
        .reset_index()
    )
    within_shift = (
        within_seed.groupby(["dataset", "shift", "method"], observed=True)[list(METRICS)]
        .mean()
        .reset_index()
    )
    within_dataset = (
        within_shift.groupby(["dataset", "method"], observed=True)[list(METRICS)]
        .mean()
        .reset_index()
    )
    equal_dataset = (
        within_dataset.groupby("method", observed=True)[list(METRICS)].mean().reset_index()
    )
    return within_seed, within_shift, within_dataset, equal_dataset


def cohort_design(
    pairs: list[dict[str, Any]], pair_counts: pd.DataFrame
) -> pd.DataFrame:
    audit = pd.read_csv(SHIFT_AUDIT_PATH)
    keys = {(str(pair["dataset"]), str(pair["shift"])) for pair in pairs}
    audit = audit.loc[
        audit.apply(lambda row: (str(row["dataset"]), str(row["shift_name"])) in keys, axis=1)
    ].copy()
    counts = (
        pair_counts.groupby(["dataset", "shift"])
        .agg(
            model_seeds=("seed", "nunique"),
            source_calibration_n_min=("source_n", "min"),
            source_calibration_n_max=("source_n", "max"),
            target_pool_n_min=("target_pool_n", "min"),
            target_pool_n_max=("target_pool_n", "max"),
            target_test_n_min=("target_test_n", "min"),
            target_test_n_max=("target_test_n", "max"),
        )
        .reset_index()
    )
    return audit.merge(
        counts, left_on=["dataset", "shift_name"], right_on=["dataset", "shift"], how="left"
    ).drop(columns="shift")


def method_information() -> pd.DataFrame:
    rows = [
        ("source_split_full", "all", 0, 0, "none_under_target_shift", "ordinary split conformal"),
        ("source_mondrian_full", "all", 0, 0, "label_shift_only_if_class_conditionals_are_invariant", "class-conditional split conformal"),
        ("source_split_300", 300, 0, 0, "none_under_target_shift", "ordinary split conformal on matched source sample"),
        ("target_split_300", 0, 300, 0, "target_marginal_exchangeability", "labeled-target reference"),
        ("target_mondrian_300", 0, 300, 0, "target_class_conditional_exchangeability", "labeled-target reference"),
        ("label_shift_weighted_300", 300, 0, 300, "plug_in_label_shift_only", "BBSE label-prior ratios; estimated weights do not inherit an exact finite-sample guarantee"),
        ("score_covariate_weighted_300", 300, 0, 300, "plug_in_score_space_covariate_shift_only", "weighted conformal with a cubic-logit domain classifier and clipped estimated ratios"),
        ("ot_minmax_weighted_cdf_300", 300, 0, 300, "no_exact_coverage_guarantee", "free-form Algorithm-2 reproduction minimizing the min/max weighted-CDF bound"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "source_labels_used",
            "target_labels_used",
            "target_unlabeled_used",
            "validity_scope",
            "implementation_note",
        ],
    )


def main() -> int:
    started = time.time()
    provenance = json.loads(GENERATION_PROVENANCE_PATH.read_text())
    pairs = list(provenance["pairs"])
    if len(pairs) != 60:
        raise RuntimeError(f"Expected 60 prediction pairs, observed {len(pairs)}")

    records: list[dict[str, Any]] = []
    pair_payloads: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    pair_count_rows: list[dict[str, Any]] = []
    input_hashes: list[dict[str, str]] = []
    problems: list[dict[str, np.ndarray | float]] = []
    problem_metadata: list[dict[str, Any]] = []

    for pair_position, pair in enumerate(pairs, start=1):
        prediction_path = PROJECT_ROOT / str(pair["prediction_path"])
        observed_hash = sha256_file(prediction_path)
        if observed_hash != str(pair["prediction_sha256"]):
            raise RuntimeError(f"Prediction hash mismatch: {prediction_path}")
        frame = pd.read_parquet(prediction_path)
        source_frame = frame.loc[frame["role"] == "source_calibration"].reset_index(drop=True)
        pool_frame = frame.loc[frame["role"] == "target_calibration_pool"].reset_index(drop=True)
        test_frame = frame.loc[frame["role"] == "target_test"].reset_index(drop=True)
        role_ids = [set(part["row_id"].astype(str)) for part in (source_frame, pool_frame, test_frame)]
        if any(role_ids[a].intersection(role_ids[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
            raise RuntimeError(f"Role overlap: {prediction_path}")
        if min(len(source_frame), len(pool_frame)) < max(N_SOURCE, N_TARGET):
            raise RuntimeError(f"Insufficient matched sample size: {prediction_path}")
        dataset = str(pair["dataset"])
        shift = str(pair["shift"])
        seed = int(pair["seed"])
        source_full = arrays(source_frame)
        pool_full = arrays(pool_frame)
        test = arrays(test_frame)
        full_q = ordinary_quantile(source_full["true_score"])
        full_q0, full_q1 = mondrian_quantiles(source_full["true_score"], source_full["y"])
        pair_count_rows.append(
            {
                "dataset": dataset,
                "shift": shift,
                "seed": seed,
                "source_n": len(source_frame),
                "target_pool_n": len(pool_frame),
                "target_test_n": len(test_frame),
            }
        )
        input_hashes.append(
            {"path": str(prediction_path.relative_to(PROJECT_ROOT)), "sha256": observed_hash}
        )
        for replicate in range(N_REPLICATES):
            source_seed = deterministic_seed(DRAW_NAMESPACE, dataset, shift, seed, replicate, "source")
            target_seed = deterministic_seed(DRAW_NAMESPACE, dataset, shift, seed, replicate, "target")
            source_indices = np.random.default_rng(source_seed).choice(
                len(source_frame), size=N_SOURCE, replace=False
            )
            target_indices = np.random.default_rng(target_seed).choice(
                len(pool_frame), size=N_TARGET, replace=False
            )
            source = {key: value[source_indices] for key, value in source_full.items()}
            target = {key: value[target_indices] for key, value in pool_full.items()}
            source_ids = sorted(source["row_id"].astype(str))
            target_ids = sorted(target["row_id"].astype(str))
            draw_rows.append(
                {
                    "dataset": dataset,
                    "shift": shift,
                    "seed": seed,
                    "replicate": replicate,
                    "source_draw_seed": source_seed,
                    "target_draw_seed": target_seed,
                    "source_row_ids_sha256": hashlib.sha256("\x1f".join(source_ids).encode()).hexdigest(),
                    "target_row_ids_sha256": hashlib.sha256("\x1f".join(target_ids).encode()).hexdigest(),
                    "source_class_0": int((source["y"] == 0).sum()),
                    "source_class_1": int((source["y"] == 1).sum()),
                    "target_class_0_hidden_reference": int((target["y"] == 0).sum()),
                    "target_class_1_hidden_reference": int((target["y"] == 1).sum()),
                }
            )
            source_q = ordinary_quantile(source["true_score"])
            target_q = ordinary_quantile(target["true_score"])
            target_q0, target_q1 = mondrian_quantiles(target["true_score"], target["y"])
            label_ratios, label_diagnostic = estimate_label_weights(source, target)
            label_calibration_weight = label_ratios[source["y"]]
            label_q0 = weighted_quantile(
                source["true_score"], label_calibration_weight, float(label_ratios[0])
            )
            label_q1 = weighted_quantile(
                source["true_score"], label_calibration_weight, float(label_ratios[1])
            )
            score_source_weight, score_test_weight, score_diagnostic = score_density_ratios(
                source["p"], target["p"], test["p"]
            )
            score_test_q = weighted_threshold_vector(
                source["true_score"], score_source_weight, score_test_weight
            )
            problems.append(prepare_ot_problem(source["true_score"], target["p"]))
            metadata = {
                "dataset": dataset,
                "shift": shift,
                "seed": seed,
                "replicate": replicate,
                "source": source,
                "test": test,
                "label_diagnostic": label_diagnostic,
                "score_diagnostic": score_diagnostic,
            }
            problem_metadata.append(metadata)
            pair_payloads.append(
                {
                    **metadata,
                    "fixed_thresholds": {
                        "source_split_full": (full_q, full_q),
                        "source_mondrian_full": (full_q0, full_q1),
                        "source_split_300": (source_q, source_q),
                        "target_split_300": (target_q, target_q),
                        "target_mondrian_300": (target_q0, target_q1),
                        "label_shift_weighted_300": (label_q0, label_q1),
                        "score_covariate_weighted_300": (score_test_q, score_test_q),
                    },
                }
            )
        print(f"[{pair_position:02d}/60] prepared {dataset}/{shift}/seed={seed}", flush=True)

    ot_weights, ot_diagnostics = optimize_ot_batch(problems)
    weight_rows: list[dict[str, Any]] = []
    optimization_rows: list[dict[str, Any]] = []
    for position, (payload, weights, diagnostic) in enumerate(
        zip(pair_payloads, ot_weights, ot_diagnostics, strict=True)
    ):
        source = payload["source"]
        for source_position, (row_id, score, weight) in enumerate(
            zip(source["row_id"], source["true_score"], weights, strict=True)
        ):
            weight_rows.append(
                {
                    "dataset": payload["dataset"],
                    "shift": payload["shift"],
                    "seed": payload["seed"],
                    "replicate": payload["replicate"],
                    "source_position": source_position,
                    "row_id": str(row_id),
                    "true_score": float(score),
                    "mean_one_weight": float(weight),
                }
            )
        for trace in diagnostic.pop("trace"):
            optimization_rows.append(
                {
                    "dataset": payload["dataset"],
                    "shift": payload["shift"],
                    "seed": payload["seed"],
                    "replicate": payload["replicate"],
                    **trace,
                }
            )
        ot_q = weighted_quantile(source["true_score"], weights, 1.0)
        payload["fixed_thresholds"]["ot_minmax_weighted_cdf_300"] = (ot_q, ot_q)
        for method in METHODS:
            q0, q1 = payload["fixed_thresholds"][method]
            diagnostic_fields: dict[str, float] = {}
            if method == "label_shift_weighted_300":
                diagnostic_fields = payload["label_diagnostic"]
            elif method == "score_covariate_weighted_300":
                diagnostic_fields = payload["score_diagnostic"]
            elif method == "ot_minmax_weighted_cdf_300":
                diagnostic_fields = diagnostic
            records.append(
                {
                    "dataset": payload["dataset"],
                    "shift": payload["shift"],
                    "seed": payload["seed"],
                    "replicate": payload["replicate"],
                    "method": method,
                    "alpha": ALPHA,
                    "n_source_sample": N_SOURCE,
                    "n_target_sample": N_TARGET,
                    "qhat_0": float(q0) if np.ndim(q0) == 0 else math.nan,
                    "qhat_1": float(q1) if np.ndim(q1) == 0 else math.nan,
                    **evaluate(payload["test"], q0, q1),
                    "diagnostics_json": json.dumps(diagnostic_fields, sort_keys=True),
                }
            )
        if (position + 1) % 25 == 0 or position + 1 == len(pair_payloads):
            print(f"evaluated {position + 1}/{len(pair_payloads)} matched problems", flush=True)

    metrics = pd.DataFrame(records)
    _, shift_summary, dataset_summary, equal_dataset = aggregate_levels(metrics)
    expected_rows = 60 * N_REPLICATES * len(METHODS)
    if len(metrics) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} metric rows, observed {len(metrics)}")
    information = method_information()
    pair_counts = pd.DataFrame(pair_count_rows)
    design = cohort_design(pairs, pair_counts)

    outputs: dict[str, pd.DataFrame] = {
        "shift_aware_replicate_metrics.parquet": metrics,
        "shift_aware_shift_summary.csv": shift_summary,
        "shift_aware_dataset_summary.csv": dataset_summary,
        "shift_aware_equal_dataset_summary.csv": equal_dataset,
        "draw_audit.csv": pd.DataFrame(draw_rows),
        "ot_weights.parquet": pd.DataFrame(weight_rows),
        "ot_optimization_trace.csv": pd.DataFrame(optimization_rows),
        "method_information_budget.csv": information,
        "cohort_shift_design.csv": design,
    }
    for name, value in outputs.items():
        if name.endswith(".parquet"):
            atomic_parquet(RUN_DIR / name, value)
        else:
            atomic_csv(RUN_DIR / name, value.to_dict("records"))

    equal_lookup = equal_dataset.set_index("method")
    ot_metric_rows = metrics.loc[metrics["method"] == "ot_minmax_weighted_cdf_300"]
    validation = {
        "prediction_hashes_match": True,
        "roles_disjoint": True,
        "pair_count": metrics[["dataset", "shift", "seed"]].drop_duplicates().shape[0] == 60,
        "shift_count": metrics[["dataset", "shift"]].drop_duplicates().shape[0] == 12,
        "replicate_count": set(metrics["replicate"]) == set(range(N_REPLICATES)),
        "method_set": set(metrics["method"]) == set(METHODS),
        "metric_row_count": len(metrics) == expected_rows,
        "all_metrics_finite": bool(np.isfinite(metrics[list(METRICS)].to_numpy()).all()),
        "ot_objective_nonincreasing": bool(
            all(d["final_objective"] <= d["initial_objective"] + 1.0e-7 for d in ot_diagnostics)
        ),
        "no_target_test_labels_used_for_calibration": True,
    }
    artifact_records = {
        name: {
            "sha256": sha256_file(RUN_DIR / name),
            "size_bytes": (RUN_DIR / name).stat().st_size,
        }
        for name in outputs
    }
    results = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if all(validation.values()) else "failed",
        "created_at_utc": utc_now(),
        "design": {
            "alpha": ALPHA,
            "source_sample_per_problem": N_SOURCE,
            "target_pool_sample_per_problem": N_TARGET,
            "replicates_per_shift_seed": N_REPLICATES,
            "shift_seed_pairs": 60,
            "dataset_shift_conditions": 12,
            "same_draw_within_problem": True,
            "untouched_target_test": True,
            "ot": {
                "reference": "Correia and Louizos, NeurIPS 2025, Algorithm 2 and weighted-CDF bound (7)",
                "implementation": "independent probability-output reproduction with free-form mean-one weights, min/max score-vector auxiliaries, a unit test-point weight, Gaussian KDE, and trapezoidal integration",
                "optimizer": "Adam",
                "steps": OT_STEPS,
                "learning_rate": OT_LEARNING_RATE,
                "grid_size": OT_GRID_SIZE,
            },
            "inference": "descriptive equal-dataset aggregation; neither shifts nor repeated draws are treated as independent deployment populations",
        },
        "headline": {
            method: {
                "coverage": float(equal_lookup.loc[method, "coverage"]),
                "mean_set_size": float(equal_lookup.loc[method, "mean_set_size"]),
                "class_coverage_gap": float(equal_lookup.loc[method, "class_coverage_gap"]),
            }
            for method in METHODS
        },
        "ot_objective": {
            "mean_initial": float(np.mean([d["initial_objective"] for d in ot_diagnostics])),
            "mean_final": float(np.mean([d["final_objective"] for d in ot_diagnostics])),
            "mean_effective_sample_size": float(
                np.mean([d["effective_sample_size"] for d in ot_diagnostics])
            ),
            "coverage_range_across_problems": [
                float(ot_metric_rows["coverage"].min()),
                float(ot_metric_rows["coverage"].max()),
            ],
        },
        "validation": validation,
        "inputs": {
            "generation_provenance": {
                "path": str(GENERATION_PROVENANCE_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(GENERATION_PROVENANCE_PATH),
            },
            "shift_audit": {
                "path": str(SHIFT_AUDIT_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(SHIFT_AUDIT_PATH),
            },
            "prediction_files": input_hashes,
        },
        "artifacts": artifact_records,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(RUN_DIR / "results.json", results)
    atomic_json(
        RUN_DIR / "environment.json",
        {
            "created_at_utc": utc_now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "torch")
            },
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )
    print(json.dumps(results["headline"], indent=2, sort_keys=True))
    return 0 if results["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
