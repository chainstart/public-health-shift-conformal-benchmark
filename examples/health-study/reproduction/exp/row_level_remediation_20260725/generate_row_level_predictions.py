from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
RUN_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_LAB_ROOT = PROJECT_ROOT
OUTPUT_ROOT = RUN_ROOT / "artifacts"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
COVARIATE_ROOT = OUTPUT_ROOT / "covariates"
MANIFEST_PATH = RUN_ROOT / "remediation_inputs.json"
PROVENANCE_PATH = RUN_ROOT / "generation_provenance.json"

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
SUBGROUPS = {
    "brfss_multiyear_diabetes": ["sex_group", "age_group"],
    "nhanes_multicycle_diabetes": ["sex_group", "age_group"],
    "uci_diabetes_readmission": ["age_group", "race_group"],
}
BRFSS_CATEGORICAL = [
    "survey_year",
    "_STATE",
    "region",
    "GENHLTH",
    "HLTHPLN1",
    "PERSDOC2",
    "MEDCOST",
    "CHECKUP1",
    "BPHIGH4",
    "BLOODCHO",
    "SEX",
    "MARITAL",
    "EDUCA",
    "EMPLOY1",
    "INCOME2",
    "SMOKE100",
    "EXERANY2",
    "_RACE",
    "_AGEG5YR",
    "_AGE65YR",
    "_BMI5CAT",
]
UCI_CATEGORICAL = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "admission_mode",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "metformin",
    "repaglinide",
    "nateglinide",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_covariates(dataset: Any) -> tuple[list[str], list[str]]:
    if dataset.name == "brfss_multiyear_diabetes":
        categorical = BRFSS_CATEGORICAL
    elif dataset.name == "uci_diabetes_readmission":
        categorical = UCI_CATEGORICAL
    else:
        categorical = list(dataset.categorical_cols)
    numeric = [name for name in dataset.numeric_cols if name in dataset.X.columns]
    categorical = [name for name in categorical if name in dataset.X.columns]
    return numeric, categorical


def attach_stable_ids(dataset: Any) -> Any:
    groups = dataset.groups.copy()
    groups["__stable_row_id"] = [
        f"{dataset.name}:{index:09d}" for index in range(len(groups))
    ]
    return replace(dataset, groups=groups)


def probabilities(model: Any, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame)
    classes = list(model.named_steps["model"].classes_)
    if set(classes) != {0, 1}:
        raise RuntimeError(f"Expected binary classes 0 and 1, observed {classes}")
    return raw[:, [classes.index(0), classes.index(1)]]


def role_frames(
    dataset: Any,
    split: dict[str, Any],
    model: Any,
    shift: str,
    seed: int,
    numeric_covariates: list[str],
    categorical_covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    prediction_parts: list[pd.DataFrame] = []
    covariate_parts: list[pd.DataFrame] = []
    role_counts: dict[str, int] = {}
    positive_counts: dict[str, int] = {}
    specifications = [
        (
            "source_calibration",
            split["X_source_cal"],
            split["y_source_cal"],
            split["g_source_cal"],
        ),
        (
            "target_calibration_pool",
            split["X_target_pool"],
            split["y_target_pool"],
            split["g_target_pool"],
        ),
        ("target_test", split["X_test"], split["y_test"], split["g_test"]),
    ]
    for role, features, labels, groups in specifications:
        if features is None or groups is None:
            raise RuntimeError(f"Missing required role {role}")
        features = features.reset_index(drop=True)
        groups = groups.reset_index(drop=True)
        labels = np.asarray(labels, dtype=np.int8)
        probs = probabilities(model, features)
        row_ids = groups["__stable_row_id"].astype(str).reset_index(drop=True)
        common = pd.DataFrame(
            {
                "dataset": dataset.name,
                "shift": shift,
                "seed": np.int16(seed),
                "row_id": row_ids,
            }
        )
        prediction = common.copy()
        prediction["role"] = role
        prediction["setting"] = "covariate_shift"
        prediction["label"] = labels
        prediction["prob_1"] = probs[:, 1].astype(np.float32)
        for subgroup in SUBGROUPS[dataset.name]:
            prediction[f"group_{subgroup}"] = (
                groups[subgroup].astype("string").fillna("missing")
            )
        prediction_parts.append(prediction)

        covariates = common.copy()
        retained = numeric_covariates + categorical_covariates
        covariates = pd.concat(
            [covariates, features[retained].reset_index(drop=True)], axis=1
        )
        covariate_parts.append(covariates)
        role_counts[role] = int(len(labels))
        positive_counts[role] = int(labels.sum())
    predictions = pd.concat(prediction_parts, ignore_index=True)
    covariates = pd.concat(covariate_parts, ignore_index=True)
    if predictions["row_id"].duplicated().any():
        raise RuntimeError("Stable row IDs overlap across required split roles")
    if not predictions[["prob_1"]].notna().all().all():
        raise RuntimeError("Model produced missing candidate probabilities")
    if not predictions["prob_1"].between(0.0, 1.0).all():
        raise RuntimeError("Model produced an invalid class-1 probability")
    return predictions, covariates, role_counts, positive_counts


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(
        temporary,
        index=False,
        compression="zstd",
        engine="pyarrow",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate row-level predictions for the public-shift remediation."
    )
    parser.add_argument("--lab-root", type=Path, default=DEFAULT_LAB_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("brfss", "nhanes", "uci"),
        default=["brfss", "nhanes", "uci"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--shifts", nargs="*", default=None)
    parser.add_argument("--max-brfss-per-year", type=int, default=60_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lab_root = args.lab_root.resolve()
    sys.path.insert(0, str(lab_root))
    from scripts.run_large_scale_conformal import get_shift_split, make_model
    from scripts.run_public_shift_benchmark_v2 import (
        prepare_brfss_multiyear,
        prepare_nhanes_multicycle,
        prepare_uci_with_pseudo,
    )

    constructors = {
        "brfss": lambda: prepare_brfss_multiyear(
            max_rows_per_year=args.max_brfss_per_year
        ),
        "nhanes": prepare_nhanes_multicycle,
        "uci": prepare_uci_with_pseudo,
    }
    datasets = [attach_stable_ids(constructors[name]()) for name in args.datasets]
    requested_shifts = set(args.shifts or [])
    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    if args.resume and MANIFEST_PATH.is_file():
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for pair in payload.get("pairs", []):
            key = (str(pair["dataset"]), str(pair["shift"]), int(pair["seed"]))
            prediction = PROJECT_ROOT / pair["prediction_path"]
            covariate = PROJECT_ROOT / pair["covariate_path"]
            if prediction.is_file() and covariate.is_file():
                existing[key] = pair

    pair_records: list[dict[str, Any]] = list(existing.values())
    started = time.monotonic()
    for dataset in datasets:
        numeric_covariates, categorical_covariates = selected_covariates(dataset)
        for specification in dataset.shifts:
            if specification.name == "pseudo_random":
                continue
            if requested_shifts and specification.name not in requested_shifts:
                continue
            for seed in args.seeds:
                key = (dataset.name, specification.name, int(seed))
                if key in existing:
                    print(f"[resume] {dataset.name}/{specification.name}/seed={seed}")
                    continue
                print(
                    f"[fit] {dataset.name}/{specification.name}/seed={seed}",
                    flush=True,
                )
                split = get_shift_split(dataset, specification, seed)
                if split is None:
                    raise RuntimeError(f"Invalid shift domain for {key}")
                model = make_model(dataset, "sgd_logistic", seed)
                model.fit(split["X_train"], split["y_train"])
                predictions, covariates, role_counts, positive_counts = role_frames(
                    dataset,
                    split,
                    model,
                    specification.name,
                    seed,
                    numeric_covariates,
                    categorical_covariates,
                )
                stem = f"{dataset.name}__{specification.name}__seed_{seed}"
                prediction_path = PREDICTION_ROOT / f"{stem}__predictions.parquet"
                covariate_path = COVARIATE_ROOT / f"{stem}__covariates.parquet"
                write_parquet_atomic(predictions, prediction_path)
                write_parquet_atomic(covariates, covariate_path)
                pair_records.append(
                    {
                        "dataset": dataset.name,
                        "shift": specification.name,
                        "seed": int(seed),
                        "setting": "covariate_shift",
                        "prediction_path": str(
                            prediction_path.relative_to(PROJECT_ROOT)
                        ),
                        "covariate_path": str(
                            covariate_path.relative_to(PROJECT_ROOT)
                        ),
                        "prediction_sha256": sha256_file(prediction_path),
                        "covariate_sha256": sha256_file(covariate_path),
                        "row_count": int(len(predictions)),
                        "role_counts": role_counts,
                        "positive_counts": positive_counts,
                        "subgroup_columns": [
                            f"group_{name}" for name in SUBGROUPS[dataset.name]
                        ],
                        "covariate_columns": (
                            numeric_covariates + categorical_covariates
                        ),
                        "numeric_covariates": numeric_covariates,
                        "categorical_covariates": categorical_covariates,
                    }
                )
                del model, split, predictions, covariates
                gc.collect()
                ordered = sorted(
                    pair_records,
                    key=lambda row: (
                        str(row["dataset"]),
                        str(row["shift"]),
                        int(row["seed"]),
                    ),
                )
                artifacts: list[dict[str, Any]] = []
                for row in ordered:
                    common = {
                        "dataset": row["dataset"],
                        "shift": row["shift"],
                        "seed": row["seed"],
                        "setting": row["setting"],
                    }
                    artifacts.append(
                        {
                            **common,
                            "artifact_role": "row_level_predictions",
                            "prediction_path": row["prediction_path"],
                            "subgroup_columns": row["subgroup_columns"],
                            "probability_definition": (
                                "fixed health-outcome model P(Y=1|X); "
                                "P(Y=0|X) is canonically reconstructed as 1-p"
                            ),
                        }
                    )
                    artifacts.append(
                        {
                            **common,
                            "artifact_role": "domain_covariates",
                            "path": row["covariate_path"],
                            "covariate_columns": row["covariate_columns"],
                            "numeric_covariates": row["numeric_covariates"],
                            "categorical_covariates": row["categorical_covariates"],
                        }
                    )
                atomic_json(
                    MANIFEST_PATH,
                    {
                        "schema_version": "ara.public_shift.row_level_inputs.v1",
                        "created_at_utc": utc_now(),
                        "artifacts": artifacts,
                        "pairs": ordered,
                    },
                )

    ordered = sorted(
        pair_records,
        key=lambda row: (
            str(row["dataset"]),
            str(row["shift"]),
            int(row["seed"]),
        ),
    )
    dataset_shifts = sorted(
        {(row["dataset"], row["shift"]) for row in ordered}
    )
    atomic_json(
        PROVENANCE_PATH,
        {
            "schema_version": "ara.public_shift.row_level_generation.v1",
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "lab_root": str(lab_root),
            "health_outcome_model": "sgd_logistic",
            "model_refit_only_for_reproducing_archived_protocol": True,
            "outcome_model_changed": False,
            "binary_probability_canonicalization": (
                "Store P(Y=1|X) as float32 and reconstruct P(Y=0|X)=1-p "
                "to preserve exact complementarity in downstream set calculations."
            ),
            "pair_count": len(ordered),
            "dataset_shift_count": len(dataset_shifts),
            "datasets": sorted({row["dataset"] for row in ordered}),
            "seeds": sorted({int(row["seed"]) for row in ordered}),
            "pairs": ordered,
        },
    )
    print(
        f"[done] pairs={len(ordered)} dataset_shifts={len(dataset_shifts)} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
