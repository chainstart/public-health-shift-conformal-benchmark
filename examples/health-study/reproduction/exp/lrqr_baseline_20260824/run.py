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
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCHEMA_VERSION = "ara.public_shift.lrqr_baseline.v1"
ALPHA = 0.10
N_SOURCE_LABELED = 300
N_SOURCE_UNLABELED = 300
N_TARGET_UNLABELED = 300
N_REPLICATES = 5
MAX_COMPONENTS = 50
BETA_LOWER = 0.05
BETA_UPPER = 20.0
CV_STEPS = 1800
FINAL_STEPS = 8000
LEARNING_RATE = 0.02
RIDGE = 1.0e-8
CHUNK_SIZE = 20
METHOD = "lrqr_300_300_300"
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
PRIOR_DIR = PROJECT_ROOT / "exp" / "shift_aware_baselines_20260813"
GENERATION_PROVENANCE_PATH = REFERENCE_DIR / "generation_provenance.json"
PRIOR_METRICS_PATH = PRIOR_DIR / "shift_aware_replicate_metrics.parquet"
PRIOR_DRAW_PATH = PRIOR_DIR / "draw_audit.csv"
PRIOR_INFORMATION_PATH = PRIOR_DIR / "method_information_budget.csv"
MATCHED_DRAW_NAMESPACE = "ara.shift_aware_baselines.draw.v1"
LRQR_DRAW_NAMESPACE = "ara.lrqr_baseline.draw.v1"
LRQR_CV_NAMESPACE = "ara.lrqr_baseline.cv.v1"


@dataclass
class Problem:
    dataset: str
    shift: str
    seed: int
    replicate: int
    x_labeled: np.ndarray
    scores: np.ndarray
    x_target: np.ndarray
    x_source_unlabeled: np.ndarray
    source_ids_hash: str
    target_ids_hash: str
    source_unlabeled_ids_hash: str
    source_unlabeled_seed: int
    raw_feature_count: int
    pca_components: int
    pair: dict[str, Any]
    source_indices: np.ndarray
    target_indices: np.ndarray
    source_unlabeled_indices: np.ndarray
    preprocessor: ColumnTransformer
    pca: PCA


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(values: Iterable[Any]) -> str:
    return hashlib.sha256("\x1f".join(sorted(str(value) for value in values)).encode()).hexdigest()


def deterministic_seed(namespace: str, *parts: Any) -> int:
    payload = "\x1f".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def atomic_target(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
            writer = csv.DictWriter(handle, fieldnames=list(materialized[0]), lineterminator="\n")
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


def ordinary_quantile(scores: np.ndarray) -> float:
    ordered = np.sort(np.asarray(scores, dtype=float))
    rank = int(math.ceil((len(ordered) + 1) * (1.0 - ALPHA)))
    return math.inf if rank > len(ordered) else float(ordered[rank - 1])


def role_frames(pair: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_path = PROJECT_ROOT / str(pair["prediction_path"])
    covariate_path = PROJECT_ROOT / str(pair["covariate_path"])
    if sha256_file(prediction_path) != str(pair["prediction_sha256"]):
        raise RuntimeError(f"Prediction hash mismatch: {prediction_path}")
    if sha256_file(covariate_path) != str(pair["covariate_sha256"]):
        raise RuntimeError(f"Covariate hash mismatch: {covariate_path}")
    predictions = pd.read_parquet(prediction_path)
    covariates = pd.read_parquet(covariate_path)
    frame = predictions.merge(covariates, on=["dataset", "shift", "seed", "row_id"], validate="one_to_one")
    source = frame.loc[frame["role"] == "source_calibration"].reset_index(drop=True)
    target = frame.loc[frame["role"] == "target_calibration_pool"].reset_index(drop=True)
    test = frame.loc[frame["role"] == "target_test"].reset_index(drop=True)
    ids = [set(part["row_id"].astype(str)) for part in (source, target, test)]
    if any(ids[a].intersection(ids[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError(f"Role overlap for {pair['dataset']}/{pair['shift']}/{pair['seed']}")
    return source, target, test


def build_features(
    source: pd.DataFrame,
    target: pd.DataFrame,
    pair: dict[str, Any],
    source_indices: np.ndarray,
    source_unlabeled_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, PCA, int, int]:
    numeric = list(pair["numeric_covariates"])
    categorical = list(pair["categorical_covariates"])
    columns = list(pair["covariate_columns"])
    preprocessor = ColumnTransformer(
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
                        handle_unknown="ignore",
                        drop="first",
                        sparse_output=False,
                    ),
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )
    source_fit = pd.concat(
        [source.iloc[source_indices][columns], source.iloc[source_unlabeled_indices][columns]],
        ignore_index=True,
    )
    preprocessor.fit(source_fit)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories")
        raw_labeled = np.asarray(preprocessor.transform(source.iloc[source_indices][columns]), dtype=float)
        raw_source_unlabeled = np.asarray(
            preprocessor.transform(source.iloc[source_unlabeled_indices][columns]), dtype=float
        )
        raw_target = np.asarray(preprocessor.transform(target.iloc[target_indices][columns]), dtype=float)
    centered = raw_source_unlabeled - raw_source_unlabeled.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered))
    components = min(MAX_COMPONENTS, rank, raw_source_unlabeled.shape[1])
    if components < 1:
        raise RuntimeError("LR-QR feature representation has zero source-supported rank")
    pca = PCA(n_components=components, whiten=True, svd_solver="full")
    pca.fit(raw_source_unlabeled)

    def transform(raw: np.ndarray) -> np.ndarray:
        projected = pca.transform(raw)
        if projected.shape[1] < MAX_COMPONENTS:
            projected = np.pad(projected, ((0, 0), (0, MAX_COMPONENTS - projected.shape[1])))
        return np.column_stack((np.ones(len(projected)), projected)).astype(np.float64)

    return (
        transform(raw_labeled),
        transform(raw_target),
        transform(raw_source_unlabeled),
        preprocessor,
        pca,
        raw_source_unlabeled.shape[1],
        components,
    )


def prepare_problems(pairs: list[dict[str, Any]]) -> tuple[list[Problem], list[dict[str, Any]]]:
    problems: list[Problem] = []
    input_hashes: list[dict[str, Any]] = []
    for pair_position, pair in enumerate(pairs, start=1):
        source, target, _ = role_frames(pair)
        dataset, shift, seed = str(pair["dataset"]), str(pair["shift"]), int(pair["seed"])
        input_hashes.append(
            {
                "prediction_path": str(pair["prediction_path"]),
                "prediction_sha256": str(pair["prediction_sha256"]),
                "covariate_path": str(pair["covariate_path"]),
                "covariate_sha256": str(pair["covariate_sha256"]),
            }
        )
        for replicate in range(N_REPLICATES):
            source_seed = deterministic_seed(
                MATCHED_DRAW_NAMESPACE, dataset, shift, seed, replicate, "source"
            )
            target_seed = deterministic_seed(
                MATCHED_DRAW_NAMESPACE, dataset, shift, seed, replicate, "target"
            )
            source_unlabeled_seed = deterministic_seed(
                LRQR_DRAW_NAMESPACE, dataset, shift, seed, replicate, "source_unlabeled"
            )
            source_indices = np.random.default_rng(source_seed).choice(
                len(source), N_SOURCE_LABELED, replace=False
            )
            target_indices = np.random.default_rng(target_seed).choice(
                len(target), N_TARGET_UNLABELED, replace=False
            )
            remaining = np.setdiff1d(np.arange(len(source)), source_indices, assume_unique=False)
            source_unlabeled_indices = remaining[
                np.random.default_rng(source_unlabeled_seed).choice(
                    len(remaining), N_SOURCE_UNLABELED, replace=False
                )
            ]
            if set(source_indices).intersection(source_unlabeled_indices):
                raise RuntimeError("Labeled and unlabeled source draws overlap")
            x_labeled, x_target, x_source_unlabeled, preprocessor, pca, raw_count, components = (
                build_features(
                    source,
                    target,
                    pair,
                    source_indices,
                    source_unlabeled_indices,
                    target_indices,
                )
            )
            source_probability = source.iloc[source_indices]["prob_1"].to_numpy(dtype=float)
            source_label = source.iloc[source_indices]["label"].to_numpy(dtype=np.int8)
            scores = np.where(source_label == 0, source_probability, 1.0 - source_probability)
            problems.append(
                Problem(
                    dataset=dataset,
                    shift=shift,
                    seed=seed,
                    replicate=replicate,
                    x_labeled=x_labeled,
                    scores=scores,
                    x_target=x_target,
                    x_source_unlabeled=x_source_unlabeled,
                    source_ids_hash=ids_sha256(source.iloc[source_indices]["row_id"]),
                    target_ids_hash=ids_sha256(target.iloc[target_indices]["row_id"]),
                    source_unlabeled_ids_hash=ids_sha256(
                        source.iloc[source_unlabeled_indices]["row_id"]
                    ),
                    source_unlabeled_seed=source_unlabeled_seed,
                    raw_feature_count=raw_count,
                    pca_components=components,
                    pair=pair,
                    source_indices=source_indices,
                    target_indices=target_indices,
                    source_unlabeled_indices=source_unlabeled_indices,
                    preprocessor=preprocessor,
                    pca=pca,
                )
            )
        print(
            f"[{pair_position:02d}/{len(pairs)}] prepared {dataset}/{shift}/seed={seed}",
            flush=True,
        )
    return problems, input_hashes


def folds_for(problem: Problem) -> list[np.ndarray]:
    seed = deterministic_seed(
        LRQR_CV_NAMESPACE,
        problem.dataset,
        problem.shift,
        problem.seed,
        problem.replicate,
    )
    return [np.asarray(part, dtype=int) for part in np.array_split(np.random.default_rng(seed).permutation(300), 3)]


def lambda_grid() -> np.ndarray:
    optimum = N_SOURCE_LABELED ** (-1.0 / 3.0) * (
        1.0 / N_TARGET_UNLABELED + 1.0 / N_SOURCE_UNLABELED
    ) ** (-1.0 / 3.0)
    return np.linspace(optimum / 10.0, optimum, 10)


def batched_adam(
    x_labeled: np.ndarray,
    scores: np.ndarray,
    x_target: np.ndarray,
    x_source_unlabeled: np.ndarray,
    lambdas: np.ndarray,
    initial_betas: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cpu")
    x1 = torch.as_tensor(x_labeled, dtype=torch.float32, device=device)
    score = torch.as_tensor(scores, dtype=torch.float32, device=device)
    x2 = torch.as_tensor(x_target, dtype=torch.float32, device=device)
    x3 = torch.as_tensor(x_source_unlabeled, dtype=torch.float32, device=device)
    regularization = torch.as_tensor(lambdas, dtype=torch.float32, device=device)
    theta = torch.zeros(
        (len(lambdas), x_labeled.shape[-1]), dtype=torch.float32, device=device, requires_grad=True
    )
    with torch.no_grad():
        theta[:, 0] = torch.quantile(score, 1.0 - ALPHA, dim=1)
    beta = torch.as_tensor(initial_betas, dtype=torch.float32, device=device).clone().requires_grad_(True)
    optimizer = torch.optim.Adam((theta, beta), lr=LEARNING_RATE)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        h1 = (x1 * theta[:, None, :]).sum(dim=-1)
        h2 = (x2 * theta[:, None, :]).sum(dim=-1)
        h3 = (x3 * theta[:, None, :]).sum(dim=-1)
        residual = score - h1
        pinball = torch.where(
            residual >= 0,
            (1.0 - ALPHA) * residual,
            -ALPHA * residual,
        ).mean(dim=1)
        penalty = regularization * (
            beta.square() * h3.square().mean(dim=1) - 2.0 * beta * h2.mean(dim=1)
        )
        ridge = RIDGE * theta[:, 1:].square().sum(dim=1)
        (pinball + penalty + ridge).sum().backward()
        torch.nn.utils.clip_grad_norm_((theta, beta), 100.0)
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(BETA_LOWER, BETA_UPPER)
    return theta.detach().cpu().numpy().astype(np.float64), beta.detach().cpu().numpy().astype(np.float64)


def objective_and_gradient(
    theta: np.ndarray,
    beta: float,
    regularization: float,
    x_labeled: np.ndarray,
    scores: np.ndarray,
    x_target: np.ndarray,
    x_source_unlabeled: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    h1 = x_labeled @ theta
    h2 = x_target @ theta
    h3 = x_source_unlabeled @ theta
    residual = scores - h1
    pinball = np.where(
        residual >= 0,
        (1.0 - ALPHA) * residual,
        -ALPHA * residual,
    )
    value = float(
        pinball.mean()
        + regularization * (beta * beta * np.mean(h3 * h3) - 2.0 * beta * np.mean(h2))
        + RIDGE * np.dot(theta[1:], theta[1:])
    )
    pinball_gradient = np.where(residual > 0, -(1.0 - ALPHA), ALPHA)
    theta_gradient = (
        x_labeled.T @ pinball_gradient / len(x_labeled)
        + 2.0
        * regularization
        * beta
        * beta
        * (x_source_unlabeled.T @ h3)
        / len(x_source_unlabeled)
        - 2.0 * regularization * beta * x_target.mean(axis=0)
    )
    theta_gradient[1:] += 2.0 * RIDGE * theta[1:]
    beta_gradient = float(
        2.0 * regularization * beta * np.mean(h3 * h3)
        - 2.0 * regularization * np.mean(h2)
    )
    return value, theta_gradient, beta_gradient


def select_lambdas(problems: list[Problem]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    grid = lambda_grid()
    selected = np.zeros(len(problems), dtype=float)
    trace_rows: list[dict[str, Any]] = []
    for chunk_start in range(0, len(problems), CHUNK_SIZE):
        chunk = problems[chunk_start : chunk_start + CHUNK_SIZE]
        train_x1: list[np.ndarray] = []
        train_score: list[np.ndarray] = []
        train_x2: list[np.ndarray] = []
        train_x3: list[np.ndarray] = []
        candidate_lambdas: list[float] = []
        candidate_meta: list[tuple[int, int, int, np.ndarray]] = []
        for local_position, problem in enumerate(chunk):
            folds = folds_for(problem)
            for lambda_position, regularization in enumerate(grid):
                for fold_position, validation in enumerate(folds):
                    training = np.setdiff1d(np.arange(300), validation, assume_unique=True)
                    train_x1.append(problem.x_labeled[training])
                    train_score.append(problem.scores[training])
                    train_x2.append(problem.x_target[training])
                    train_x3.append(problem.x_source_unlabeled[training])
                    candidate_lambdas.append(float(regularization))
                    candidate_meta.append(
                        (local_position, lambda_position, fold_position, validation)
                    )
        theta, beta = batched_adam(
            np.stack(train_x1),
            np.stack(train_score),
            np.stack(train_x2),
            np.stack(train_x3),
            np.asarray(candidate_lambdas),
            np.ones(len(candidate_lambdas)),
            CV_STEPS,
        )
        fold_scores: dict[tuple[int, int], list[float]] = {}
        for candidate, meta in enumerate(candidate_meta):
            local_position, lambda_position, _, validation = meta
            problem = chunk[local_position]
            _, theta_gradient, beta_gradient = objective_and_gradient(
                theta[candidate],
                float(beta[candidate]),
                float(grid[lambda_position]),
                problem.x_labeled[validation],
                problem.scores[validation],
                problem.x_target[validation],
                problem.x_source_unlabeled[validation],
            )
            fold_scores.setdefault((local_position, lambda_position), []).append(
                float(np.linalg.norm(np.r_[theta_gradient, beta_gradient]))
            )
        for local_position, problem in enumerate(chunk):
            mean_scores = np.asarray(
                [np.mean(fold_scores[(local_position, position)]) for position in range(len(grid))]
            )
            best_position = int(np.argmin(mean_scores))
            global_position = chunk_start + local_position
            selected[global_position] = float(grid[best_position])
            for lambda_position, regularization in enumerate(grid):
                values = fold_scores[(local_position, lambda_position)]
                trace_rows.append(
                    {
                        "dataset": problem.dataset,
                        "shift": problem.shift,
                        "seed": problem.seed,
                        "replicate": problem.replicate,
                        "lambda": float(regularization),
                        "fold_0_gradient_norm": values[0],
                        "fold_1_gradient_norm": values[1],
                        "fold_2_gradient_norm": values[2],
                        "mean_validation_gradient_norm": float(np.mean(values)),
                        "selected": lambda_position == best_position,
                    }
                )
        print(
            f"selected lambda for {min(chunk_start + len(chunk), len(problems))}/{len(problems)} problems",
            flush=True,
        )
    return selected, trace_rows


def fit_final(
    problems: list[Problem], selected_lambdas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    coefficients = np.zeros((len(problems), MAX_COMPONENTS + 1), dtype=float)
    betas = np.zeros(len(problems), dtype=float)
    diagnostic_rows: list[dict[str, Any]] = []
    initial_betas = np.asarray((0.25, 1.0, 4.0), dtype=float)
    for chunk_start in range(0, len(problems), CHUNK_SIZE):
        chunk = problems[chunk_start : chunk_start + CHUNK_SIZE]
        x1 = np.repeat(np.stack([problem.x_labeled for problem in chunk]), 3, axis=0)
        score = np.repeat(np.stack([problem.scores for problem in chunk]), 3, axis=0)
        x2 = np.repeat(np.stack([problem.x_target for problem in chunk]), 3, axis=0)
        x3 = np.repeat(
            np.stack([problem.x_source_unlabeled for problem in chunk]), 3, axis=0
        )
        regularization = np.repeat(selected_lambdas[chunk_start : chunk_start + len(chunk)], 3)
        theta, beta = batched_adam(
            x1,
            score,
            x2,
            x3,
            regularization,
            np.tile(initial_betas, len(chunk)),
            FINAL_STEPS,
        )
        for local_position, problem in enumerate(chunk):
            candidates = slice(3 * local_position, 3 * local_position + 3)
            values = []
            gradients = []
            for candidate in range(candidates.start, candidates.stop):
                value, theta_gradient, beta_gradient = objective_and_gradient(
                    theta[candidate],
                    float(beta[candidate]),
                    float(regularization[candidate]),
                    problem.x_labeled,
                    problem.scores,
                    problem.x_target,
                    problem.x_source_unlabeled,
                )
                values.append(value)
                gradients.append(float(np.linalg.norm(np.r_[theta_gradient, beta_gradient])))
            best = int(np.argmin(values))
            candidate = candidates.start + best
            global_position = chunk_start + local_position
            coefficients[global_position] = theta[candidate]
            betas[global_position] = float(beta[candidate])
            diagnostic_rows.append(
                {
                    "dataset": problem.dataset,
                    "shift": problem.shift,
                    "seed": problem.seed,
                    "replicate": problem.replicate,
                    "selected_lambda": float(selected_lambdas[global_position]),
                    "selected_beta": float(beta[candidate]),
                    "objective": values[best],
                    "gradient_norm": gradients[best],
                    "selected_initial_beta": float(initial_betas[best]),
                    "raw_feature_count": problem.raw_feature_count,
                    "pca_components": problem.pca_components,
                    "source_unlabeled_seed": problem.source_unlabeled_seed,
                    "source_row_ids_sha256": problem.source_ids_hash,
                    "target_row_ids_sha256": problem.target_ids_hash,
                    "source_unlabeled_row_ids_sha256": problem.source_unlabeled_ids_hash,
                }
            )
        print(
            f"fit final LR-QR for {min(chunk_start + len(chunk), len(problems))}/{len(problems)} problems",
            flush=True,
        )
    return coefficients, betas, diagnostic_rows


def transform_test(problem: Problem, test: pd.DataFrame) -> np.ndarray:
    columns = list(problem.pair["covariate_columns"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories")
        raw = np.asarray(problem.preprocessor.transform(test[columns]), dtype=float)
    projected = problem.pca.transform(raw)
    if projected.shape[1] < MAX_COMPONENTS:
        projected = np.pad(projected, ((0, 0), (0, MAX_COMPONENTS - projected.shape[1])))
    return np.column_stack((np.ones(len(projected)), projected)).astype(np.float64)


def evaluate(test: pd.DataFrame, threshold: np.ndarray) -> dict[str, float]:
    probability = test["prob_1"].to_numpy(dtype=float)
    label = test["label"].to_numpy(dtype=np.int8)
    include0 = probability <= threshold
    include1 = 1.0 - probability <= threshold
    covered = np.where(label == 0, include0, include1)
    size = include0.astype(np.int8) + include1.astype(np.int8)
    class_coverage = [float(covered[label == candidate].mean()) for candidate in (0, 1)]
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


def evaluate_problems(
    problems: list[Problem], coefficients: np.ndarray, betas: np.ndarray, selected: np.ndarray
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    pair_groups: dict[tuple[str, str, int], list[int]] = {}
    for position, problem in enumerate(problems):
        pair_groups.setdefault((problem.dataset, problem.shift, problem.seed), []).append(position)
        for coefficient_position, value in enumerate(coefficients[position]):
            parameter_rows.append(
                {
                    "dataset": problem.dataset,
                    "shift": problem.shift,
                    "seed": problem.seed,
                    "replicate": problem.replicate,
                    "coefficient_position": coefficient_position,
                    "coefficient": float(value),
                }
            )
    for pair_position, positions in enumerate(pair_groups.values(), start=1):
        _, _, test = role_frames(problems[positions[0]].pair)
        for position in positions:
            problem = problems[position]
            x_test = transform_test(problem, test)
            threshold = x_test @ coefficients[position]
            metrics = evaluate(test, threshold)
            rows.append(
                {
                    "dataset": problem.dataset,
                    "shift": problem.shift,
                    "seed": problem.seed,
                    "replicate": problem.replicate,
                    "method": METHOD,
                    "alpha": ALPHA,
                    "n_source_sample": N_SOURCE_LABELED,
                    "n_source_unlabeled_sample": N_SOURCE_UNLABELED,
                    "n_target_sample": N_TARGET_UNLABELED,
                    "qhat_0": math.nan,
                    "qhat_1": math.nan,
                    "threshold_minimum": float(threshold.min()),
                    "threshold_median": float(np.median(threshold)),
                    "threshold_maximum": float(threshold.max()),
                    "selected_lambda": float(selected[position]),
                    "selected_beta": float(betas[position]),
                    **metrics,
                    "diagnostics_json": json.dumps(
                        {
                            "implementation": "independent_lrqr_reproduction",
                            "feature_map": "source_fitted_tabular_encoding_plus_source_unlabeled_whitened_pca",
                            "pca_components": problem.pca_components,
                        },
                        sort_keys=True,
                    ),
                }
            )
        print(f"evaluated target test for {pair_position}/{len(pair_groups)} pairs", flush=True)
    return pd.DataFrame(rows), parameter_rows


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


def extended_information_budget() -> pd.DataFrame:
    prior = pd.read_csv(PRIOR_INFORMATION_PATH)
    if "source_unlabeled_used" not in prior:
        prior.insert(2, "source_unlabeled_used", 0)
    row = {
        "method": METHOD,
        "source_labels_used": N_SOURCE_LABELED,
        "source_unlabeled_used": N_SOURCE_UNLABELED,
        "target_labels_used": 0,
        "target_unlabeled_used": N_TARGET_UNLABELED,
        "validity_scope": "asymptotic_covariate_shift_with_hypothesis_class_and_optimization_conditions",
        "implementation_note": "independent LR-QR reproduction with source-supported 50-component linear feature map and gradient-norm cross-validation",
    }
    return pd.concat((prior, pd.DataFrame([row])), ignore_index=True)


def main() -> int:
    started = time.time()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    provenance = json.loads(GENERATION_PROVENANCE_PATH.read_text())
    pairs = list(provenance["pairs"])
    if len(pairs) != 60:
        raise RuntimeError(f"Expected 60 shift-seed pairs, observed {len(pairs)}")
    prior_metrics = pd.read_parquet(PRIOR_METRICS_PATH)
    prior_draw = pd.read_csv(PRIOR_DRAW_PATH)
    if len(prior_metrics) != 2400 or len(prior_draw) != 300:
        raise RuntimeError("Prior matched evidence has unexpected dimensions")

    problems, input_hashes = prepare_problems(pairs)
    selected, cv_rows = select_lambdas(problems)
    coefficients, betas, diagnostic_rows = fit_final(problems, selected)
    lrqr_metrics, parameter_rows = evaluate_problems(problems, coefficients, betas, selected)

    draw_extension = pd.DataFrame(diagnostic_rows)[
        [
            "dataset",
            "shift",
            "seed",
            "replicate",
            "source_unlabeled_seed",
            "source_unlabeled_row_ids_sha256",
        ]
    ]
    draw_audit = prior_draw.merge(
        draw_extension, on=["dataset", "shift", "seed", "replicate"], validate="one_to_one"
    )
    if not (
        draw_audit["source_row_ids_sha256"]
        == pd.DataFrame(diagnostic_rows)["source_row_ids_sha256"]
    ).all():
        raise RuntimeError("LR-QR labeled-source draws differ from prior matched experiment")
    if not (
        draw_audit["target_row_ids_sha256"]
        == pd.DataFrame(diagnostic_rows)["target_row_ids_sha256"]
    ).all():
        raise RuntimeError("LR-QR target draws differ from prior matched experiment")

    extended_metrics = pd.concat((prior_metrics, lrqr_metrics), ignore_index=True, sort=False)
    _, shift_summary, dataset_summary, equal_dataset = aggregate_levels(extended_metrics)
    information = extended_information_budget()
    outputs: dict[str, pd.DataFrame] = {
        "lrqr_replicate_metrics.parquet": lrqr_metrics,
        "extended_replicate_metrics.parquet": extended_metrics,
        "extended_shift_summary.csv": shift_summary,
        "extended_dataset_summary.csv": dataset_summary,
        "extended_equal_dataset_summary.csv": equal_dataset,
        "lrqr_cv_trace.csv": pd.DataFrame(cv_rows),
        "lrqr_diagnostics.csv": pd.DataFrame(diagnostic_rows),
        "lrqr_parameters.parquet": pd.DataFrame(parameter_rows),
        "draw_audit.csv": draw_audit,
        "method_information_budget.csv": information,
    }
    for name, value in outputs.items():
        if name.endswith(".parquet"):
            atomic_parquet(RUN_DIR / name, value)
        else:
            atomic_csv(RUN_DIR / name, value.to_dict("records"))

    equal_lookup = equal_dataset.set_index("method")
    lrqr_shift = shift_summary.loc[shift_summary["method"] == METHOD]
    validation = {
        "input_hashes_match": True,
        "roles_disjoint": True,
        "source_labeled_and_unlabeled_disjoint": True,
        "matched_source_and_target_draws_reused": True,
        "no_target_pool_or_test_labels_used_for_fitting": True,
        "lrqr_row_count": len(lrqr_metrics) == 300,
        "extended_row_count": len(extended_metrics) == 2700,
        "method_count": extended_metrics["method"].nunique() == 9,
        "all_metrics_finite": bool(np.isfinite(lrqr_metrics[list(METRICS)].to_numpy()).all()),
        "one_selected_lambda_per_problem": bool(
            pd.DataFrame(cv_rows).groupby(["dataset", "shift", "seed", "replicate"])["selected"].sum().eq(1).all()
        ),
        "coefficient_rows": len(parameter_rows) == 300 * (MAX_COMPONENTS + 1),
    }
    artifact_records = {
        name: {"sha256": sha256_file(RUN_DIR / name), "size_bytes": (RUN_DIR / name).stat().st_size}
        for name in outputs
    }
    results = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if all(validation.values()) else "failed",
        "created_at_utc": utc_now(),
        "design": {
            "alpha": ALPHA,
            "source_labeled_per_problem": N_SOURCE_LABELED,
            "source_unlabeled_per_problem": N_SOURCE_UNLABELED,
            "target_unlabeled_per_problem": N_TARGET_UNLABELED,
            "replicates_per_shift_seed": N_REPLICATES,
            "shift_seed_pairs": 60,
            "dataset_shift_conditions": 12,
            "same_labeled_source_and_target_draws_as_prior_matched_experiment": True,
            "source_unlabeled_draw_disjoint_from_labeled_source": True,
            "untouched_target_test": True,
            "lrqr": {
                "reference": "Joshi et al., NeurIPS 2025, Algorithm 1 and Section 5.1",
                "implementation": "independent reproduction of the empirical LR-QR objective and three-fold gradient-norm lambda selection",
                "feature_map": "source-fitted numeric/one-hot encoding followed by a source-unlabeled fitted whitened PCA with at most 50 components",
                "lambda_grid": [float(value) for value in lambda_grid()],
                "beta_bounds": [BETA_LOWER, BETA_UPPER],
                "optimizer": "deterministic batched Adam with three final beta initializations",
                "cv_steps": CV_STEPS,
                "final_steps": FINAL_STEPS,
                "learning_rate": LEARNING_RATE,
                "ridge": RIDGE,
            },
            "inference": "descriptive equal-dataset aggregation; shifts and repeated draws are not independent deployment populations",
        },
        "headline": {
            method: {
                "coverage": float(equal_lookup.loc[method, "coverage"]),
                "mean_set_size": float(equal_lookup.loc[method, "mean_set_size"]),
                "class_coverage_gap": float(equal_lookup.loc[method, "class_coverage_gap"]),
            }
            for method in equal_lookup.index
        },
        "lrqr_heterogeneity": {
            "minimum_shift_coverage": float(lrqr_shift["coverage"].min()),
            "maximum_shift_coverage": float(lrqr_shift["coverage"].max()),
            "maximum_shift_class_coverage_gap": float(lrqr_shift["class_coverage_gap"].max()),
            "median_final_gradient_norm": float(pd.DataFrame(diagnostic_rows)["gradient_norm"].median()),
            "maximum_final_gradient_norm": float(pd.DataFrame(diagnostic_rows)["gradient_norm"].max()),
        },
        "validation": validation,
        "inputs": {
            "generation_provenance": {
                "path": str(GENERATION_PROVENANCE_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(GENERATION_PROVENANCE_PATH),
            },
            "prior_matched_metrics": {
                "path": str(PRIOR_METRICS_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(PRIOR_METRICS_PATH),
            },
            "prior_draw_audit": {
                "path": str(PRIOR_DRAW_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(PRIOR_DRAW_PATH),
            },
            "prediction_and_covariate_files": input_hashes,
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
    print(json.dumps(results["headline"][METHOD], indent=2, sort_keys=True))
    return 0 if results["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
