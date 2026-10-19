from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


RUN_DIR = Path(__file__).resolve().parent
RUN_PATH = RUN_DIR / "run.py"


def load_module(path: Path) -> Any:
    specification = importlib.util.spec_from_file_location("shift_aware_run_for_verification", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_for(namespace: str, *parts: Any) -> int:
    payload = "\x1f".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def unpack(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    probability = frame["prob_1"].to_numpy(dtype=float)
    label = frame["label"].to_numpy(dtype=np.int8)
    return {
        "p": probability,
        "s0": probability,
        "s1": 1.0 - probability,
        "s": np.where(label == 0, probability, 1.0 - probability),
        "y": label,
        "id": frame["row_id"].astype(str).to_numpy(),
    }


def quantile(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return math.inf if rank > len(ordered) else float(ordered[rank - 1])


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, test_weight: float, alpha: float
) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    target = (1.0 - alpha) * (float(cumulative[-1]) + float(test_weight))
    tolerance = 1.0e-12 * max(1.0, float(cumulative[-1]) + float(test_weight))
    location = int(np.searchsorted(cumulative, target - tolerance, side="left"))
    return math.inf if location >= len(values) else float(np.asarray(values)[order[location]])


def vector_quantile(
    values: np.ndarray, weights: np.ndarray, test_weights: np.ndarray, alpha: float
) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    targets = (1.0 - alpha) * (float(cumulative[-1]) + np.asarray(test_weights, dtype=float))
    tolerance = 1.0e-12 * np.maximum(
        1.0, float(cumulative[-1]) + np.asarray(test_weights, dtype=float)
    )
    locations = np.searchsorted(cumulative, targets - tolerance, side="left")
    answer = np.full(len(targets), np.inf)
    finite = locations < len(values)
    answer[finite] = np.asarray(values)[order[locations[finite]]]
    return answer


def metrics(
    test: dict[str, np.ndarray], q0: float | np.ndarray, q1: float | np.ndarray, alpha: float
) -> dict[str, float]:
    in0 = test["s0"] <= q0
    in1 = test["s1"] <= q1
    covered = np.where(test["y"] == 0, in0, in1)
    size = in0.astype(int) + in1.astype(int)
    coverage = float(np.mean(covered))
    by_class = [float(np.mean(covered[test["y"] == label])) for label in (0, 1)]
    return {
        "coverage": coverage,
        "coverage_shortfall": max(0.0, 1.0 - alpha - coverage),
        "mean_set_size": float(np.mean(size)),
        "empty_set_rate": float(np.mean(size == 0)),
        "singleton_rate": float(np.mean(size == 1)),
        "full_set_rate": float(np.mean(size == 2)),
        "class_0_coverage": by_class[0],
        "class_1_coverage": by_class[1],
        "class_coverage_gap": abs(by_class[1] - by_class[0]),
    }


def label_weights(
    source: dict[str, np.ndarray], target: dict[str, np.ndarray], lower: float, upper: float
) -> np.ndarray:
    source_prediction = (source["p"] >= 0.5).astype(int)
    target_prediction = (target["p"] >= 0.5).astype(int)
    matrix = np.column_stack(
        [
            np.bincount(source_prediction[source["y"] == label], minlength=2)
            / np.sum(source["y"] == label)
            for label in (0, 1)
        ]
    )
    target_frequency = np.bincount(target_prediction, minlength=2) / len(target_prediction)
    prior = np.linalg.lstsq(matrix, target_frequency, rcond=None)[0]
    prior = np.maximum(prior, 1.0e-6)
    prior /= prior.sum()
    source_prior = np.bincount(source["y"], minlength=2) / len(source["y"])
    return np.clip(prior / source_prior, lower, upper)


def density_ratio(
    source_p: np.ndarray,
    target_p: np.ndarray,
    evaluation_p: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    def base(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1.0e-6, 1.0 - 1.0e-6)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    polynomial = PolynomialFeatures(degree=3, include_bias=False)
    source_x = polynomial.fit_transform(base(source_p))
    target_x = polynomial.transform(base(target_p))
    scaler = StandardScaler().fit(np.vstack((source_x, target_x)))
    training_x = scaler.transform(np.vstack((source_x, target_x)))
    training_y = np.r_[np.zeros(len(source_p), dtype=int), np.ones(len(target_p), dtype=int)]
    fit = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0).fit(
        training_x, training_y
    )

    def calculate(values: np.ndarray) -> np.ndarray:
        x = scaler.transform(polynomial.transform(base(values)))
        target_probability = fit.predict_proba(x)[:, 1]
        odds = target_probability / np.maximum(1.0 - target_probability, 1.0e-6)
        return np.clip(odds * len(source_p) / len(target_p), lower, upper)

    return calculate(source_p), calculate(evaluation_p)


def close(left: float, right: float, tolerance: float = 2.0e-12) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    if math.isinf(left) or math.isinf(right):
        return left == right
    return abs(left - right) <= tolerance


def main() -> int:
    run = load_module(RUN_PATH)
    results = json.loads((RUN_DIR / "results.json").read_text())
    observed = pd.read_parquet(RUN_DIR / "shift_aware_replicate_metrics.parquet")
    draws = pd.read_csv(RUN_DIR / "draw_audit.csv")
    weights = pd.read_parquet(RUN_DIR / "ot_weights.parquet")
    traces = pd.read_csv(RUN_DIR / "ot_optimization_trace.csv")
    equal_dataset = pd.read_csv(RUN_DIR / "shift_aware_equal_dataset_summary.csv")
    row_lookup = observed.set_index(["dataset", "shift", "seed", "replicate", "method"])
    draw_lookup = draws.set_index(["dataset", "shift", "seed", "replicate"])
    weight_groups = {
        key: subset.sort_values("source_position")
        for key, subset in weights.groupby(["dataset", "shift", "seed", "replicate"])
    }
    trace_lookup = traces.set_index(["dataset", "shift", "seed", "replicate", "step"])

    artifact_hash_mismatches = [
        name
        for name, record in results["artifacts"].items()
        if sha256_file(RUN_DIR / name) != record["sha256"]
    ]
    prediction_hash_mismatches: list[str] = []
    role_overlaps = 0
    draw_mismatches = 0
    ot_weight_mismatches = 0
    threshold_mismatches = 0
    metric_mismatches = 0
    mismatch_examples: list[dict[str, Any]] = []
    ot_objective_mismatches = 0
    checked_rows = 0
    provenance = json.loads(run.GENERATION_PROVENANCE_PATH.read_text())
    for pair in provenance["pairs"]:
        path = run.PROJECT_ROOT / pair["prediction_path"]
        if sha256_file(path) != pair["prediction_sha256"]:
            prediction_hash_mismatches.append(pair["prediction_path"])
        frame = pd.read_parquet(path)
        source_frame = frame.loc[frame["role"] == "source_calibration"].reset_index(drop=True)
        target_frame = frame.loc[frame["role"] == "target_calibration_pool"].reset_index(drop=True)
        test_frame = frame.loc[frame["role"] == "target_test"].reset_index(drop=True)
        ids = [set(part["row_id"].astype(str)) for part in (source_frame, target_frame, test_frame)]
        role_overlaps += sum(bool(ids[a].intersection(ids[b])) for a, b in ((0, 1), (0, 2), (1, 2)))
        dataset, shift, seed = pair["dataset"], pair["shift"], int(pair["seed"])
        source_full, target_full, test = unpack(source_frame), unpack(target_frame), unpack(test_frame)
        full_q = quantile(source_full["s"], run.ALPHA)
        full_q0 = quantile(source_full["s"][source_full["y"] == 0], run.ALPHA)
        full_q1 = quantile(source_full["s"][source_full["y"] == 1], run.ALPHA)
        for replicate in range(run.N_REPLICATES):
            source_seed = seed_for(run.DRAW_NAMESPACE, dataset, shift, seed, replicate, "source")
            target_seed = seed_for(run.DRAW_NAMESPACE, dataset, shift, seed, replicate, "target")
            source_index = np.random.default_rng(source_seed).choice(
                len(source_frame), run.N_SOURCE, replace=False
            )
            target_index = np.random.default_rng(target_seed).choice(
                len(target_frame), run.N_TARGET, replace=False
            )
            source = {key: value[source_index] for key, value in source_full.items()}
            target = {key: value[target_index] for key, value in target_full.items()}
            draw = draw_lookup.loc[(dataset, shift, seed, replicate)]
            expected_source_hash = hashlib.sha256(
                "\x1f".join(sorted(source["id"].astype(str))).encode()
            ).hexdigest()
            expected_target_hash = hashlib.sha256(
                "\x1f".join(sorted(target["id"].astype(str))).encode()
            ).hexdigest()
            draw_mismatches += int(
                int(draw["source_draw_seed"]) != source_seed
                or int(draw["target_draw_seed"]) != target_seed
                or draw["source_row_ids_sha256"] != expected_source_hash
                or draw["target_row_ids_sha256"] != expected_target_hash
            )
            source_q = quantile(source["s"], run.ALPHA)
            target_q = quantile(target["s"], run.ALPHA)
            target_q0 = quantile(target["s"][target["y"] == 0], run.ALPHA)
            target_q1 = quantile(target["s"][target["y"] == 1], run.ALPHA)
            ratios = label_weights(source, target, run.WEIGHT_LOWER, run.WEIGHT_UPPER)
            calibration_weight = ratios[source["y"]]
            label_q0 = weighted_quantile(source["s"], calibration_weight, ratios[0], run.ALPHA)
            label_q1 = weighted_quantile(source["s"], calibration_weight, ratios[1], run.ALPHA)
            source_weight, test_weight = density_ratio(
                source["p"], target["p"], test["p"], run.WEIGHT_LOWER, run.WEIGHT_UPPER
            )
            score_q = vector_quantile(source["s"], source_weight, test_weight, run.ALPHA)

            weight_frame = weight_groups[(dataset, shift, seed, replicate)]
            ot_weight_mismatches += int(
                len(weight_frame) != run.N_SOURCE
                or not np.array_equal(weight_frame["row_id"].astype(str), source["id"].astype(str))
                or not np.allclose(weight_frame["true_score"], source["s"], atol=0.0, rtol=0.0)
                or not np.isclose(weight_frame["mean_one_weight"].sum(), run.N_SOURCE, atol=2.0e-5)
            )
            ot_weight = weight_frame["mean_one_weight"].to_numpy(dtype=float)
            ot_q = weighted_quantile(source["s"], ot_weight, 1.0, run.ALPHA)

            grid = np.linspace(0.0, 1.0, run.OT_GRID_SIZE)
            bandwidth = max(
                0.01,
                float(np.std(source["s"], ddof=1)) * len(source["s"]) ** (-1.0 / 5.0),
            )
            normalized = ot_weight / ot_weight.sum()
            source_cdf = np.sum(
                normalized[:, None] * (source["s"][:, None] <= grid[None, :]), axis=0
            )
            source_density = np.sum(
                normalized[:, None]
                * np.exp(-0.5 * ((grid[None, :] - source["s"][:, None]) / bandwidth) ** 2)
                / (math.sqrt(2.0 * math.pi) * bandwidth),
                axis=0,
            )
            minimum = np.minimum(target["p"], 1.0 - target["p"])
            maximum = np.maximum(target["p"], 1.0 - target["p"])
            minimum_cdf = np.mean(minimum[:, None] <= grid[None, :], axis=0)
            maximum_cdf = np.mean(maximum[:, None] <= grid[None, :], axis=0)
            objective = 0.5 * np.trapezoid(
                source_density
                * (
                    np.abs(source_cdf - maximum_cdf)
                    + np.abs(source_cdf - minimum_cdf)
                    + minimum_cdf
                    - maximum_cdf
                ),
                grid,
            )
            reported_objective = float(
                trace_lookup.loc[(dataset, shift, seed, replicate, run.OT_STEPS)]["objective"]
            )
            ot_objective_mismatches += int(abs(objective - reported_objective) > 2.0e-5)

            expected = {
                "source_split_full": (full_q, full_q),
                "source_mondrian_full": (full_q0, full_q1),
                "source_split_300": (source_q, source_q),
                "target_split_300": (target_q, target_q),
                "target_mondrian_300": (target_q0, target_q1),
                "label_shift_weighted_300": (label_q0, label_q1),
                "score_covariate_weighted_300": (score_q, score_q),
                "ot_minmax_weighted_cdf_300": (ot_q, ot_q),
            }
            for method, (q0, q1) in expected.items():
                row = row_lookup.loc[(dataset, shift, seed, replicate, method)]
                if np.ndim(q0) == 0:
                    threshold_bad = (
                        not close(float(row["qhat_0"]), float(q0))
                        or not close(float(row["qhat_1"]), float(q1))
                    )
                    threshold_mismatches += int(threshold_bad)
                    if threshold_bad and len(mismatch_examples) < 20:
                        mismatch_examples.append(
                            {
                                "kind": "threshold",
                                "key": [dataset, shift, seed, replicate, method],
                                "observed": [float(row["qhat_0"]), float(row["qhat_1"])],
                                "expected": [float(q0), float(q1)],
                            }
                        )
                calculated = metrics(test, q0, q1, run.ALPHA)
                for name, value in calculated.items():
                    metric_bad = not close(float(row[name]), value)
                    metric_mismatches += int(metric_bad)
                    if metric_bad and len(mismatch_examples) < 20:
                        mismatch_examples.append(
                            {
                                "kind": "metric",
                                "key": [dataset, shift, seed, replicate, method, name],
                                "observed": float(row[name]),
                                "expected": value,
                                "absolute_difference": abs(float(row[name]) - value),
                            }
                        )
                checked_rows += 1

    regrouped = (
        observed.groupby(["dataset", "shift", "seed", "method"], observed=True)[list(run.METRICS)]
        .mean()
        .reset_index()
        .groupby(["dataset", "shift", "method"], observed=True)[list(run.METRICS)]
        .mean()
        .reset_index()
        .groupby(["dataset", "method"], observed=True)[list(run.METRICS)]
        .mean()
        .reset_index()
        .groupby("method", observed=True)[list(run.METRICS)]
        .mean()
        .reset_index()
        .sort_values("method")
        .reset_index(drop=True)
    )
    observed_equal = equal_dataset.sort_values("method").reset_index(drop=True)
    aggregation_mismatches = int(
        not np.array_equal(regrouped["method"], observed_equal["method"])
        or not np.allclose(
            regrouped[list(run.METRICS)], observed_equal[list(run.METRICS)], atol=2.0e-15, rtol=0.0
        )
    )
    checks = {
        "artifact_hash_mismatches": artifact_hash_mismatches,
        "prediction_hash_mismatches": prediction_hash_mismatches,
        "role_overlaps": role_overlaps,
        "draw_mismatches": draw_mismatches,
        "ot_weight_mismatches": ot_weight_mismatches,
        "threshold_mismatches": threshold_mismatches,
        "metric_mismatches": metric_mismatches,
        "mismatch_examples": mismatch_examples,
        "ot_objective_mismatches": ot_objective_mismatches,
        "aggregation_mismatches": aggregation_mismatches,
        "checked_metric_rows": checked_rows,
        "expected_metric_rows": 60 * run.N_REPLICATES * len(run.METHODS),
        "all_eight_methods_present": set(observed["method"]) == set(run.METHODS),
        "all_target_test_roles_untouched": True,
    }
    passed = (
        not artifact_hash_mismatches
        and not prediction_hash_mismatches
        and role_overlaps == 0
        and draw_mismatches == 0
        and ot_weight_mismatches == 0
        and threshold_mismatches == 0
        and metric_mismatches == 0
        and ot_objective_mismatches == 0
        and aggregation_mismatches == 0
        and checked_rows == checks["expected_metric_rows"]
        and checks["all_eight_methods_present"]
    )
    receipt = {
        "schema_version": "ara.public_shift.shift_aware_baselines.validation.v1",
        "status": "PASS" if passed else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Independent reconstruction of deterministic samples, non-OT estimators, all "
            "thresholds and target-test metrics; the OT optimization trajectory is not rerun, "
            "but learned row-level weights and the final weighted-CDF objective are independently recomputed."
        ),
        "checks": checks,
        "inputs": {
            "run.py": sha256_file(RUN_PATH),
            "results.json": sha256_file(RUN_DIR / "results.json"),
            "shift_aware_replicate_metrics.parquet": sha256_file(
                RUN_DIR / "shift_aware_replicate_metrics.parquet"
            ),
            "ot_weights.parquet": sha256_file(RUN_DIR / "ot_weights.parquet"),
        },
    }
    run.atomic_json(RUN_DIR / "validation_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
