from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
RUN_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
REFERENCE_GENERATOR_PATH = (
    PROJECT_ROOT
    / "exp"
    / "row_level_remediation_20260725"
    / "generate_row_level_predictions.py"
)
DEFAULT_LAB_ROOT = PROJECT_ROOT
ARTIFACT_ROOT = RUN_DIR / "artifacts"
PREDICTION_ROOT = ARTIFACT_ROOT / "predictions"
COVARIATE_ROOT = ARTIFACT_ROOT / "covariates"
MANIFEST_PATH = RUN_DIR / "remediation_inputs.json"
PROVENANCE_PATH = RUN_DIR / "generation_provenance.json"
ROLE_AUDIT_PATH = RUN_DIR / "role_assignment_audit.csv"
ROLE_NAMESPACE = "ara.source_target_role_randomization.v1"


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def assignment_seed(
    dataset: str,
    shift: str,
    model_seed: int,
    domain: str,
) -> int:
    payload = "\x1f".join(
        (ROLE_NAMESPACE, dataset, shift, str(model_seed), domain)
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def outcome_blind_partition(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: pd.DataFrame,
    heldout_fraction: float,
    *,
    dataset: str,
    shift: str,
    model_seed: int,
    domain: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0,1)")
    labels = np.asarray(labels, dtype=np.int8)
    row_count = len(labels)
    if not (len(features) == len(groups) == row_count):
        raise RuntimeError("Domain features, labels, and groups do not align")
    heldout_n = int(math.ceil(heldout_fraction * row_count))
    seed = assignment_seed(dataset, shift, model_seed, domain)
    permutation = np.random.default_rng(seed).permutation(row_count)
    retained_indices = permutation[:-heldout_n]
    heldout_indices = permutation[-heldout_n:]
    if set(retained_indices).intersection(heldout_indices):
        raise RuntimeError("Outcome-blind role partition overlaps")
    retained_ids = sorted(
        groups.iloc[retained_indices]["__stable_row_id"].astype(str)
    )
    heldout_ids = sorted(
        groups.iloc[heldout_indices]["__stable_row_id"].astype(str)
    )
    audit = {
        "dataset": dataset,
        "shift": shift,
        "seed": model_seed,
        "domain": domain,
        "namespace": ROLE_NAMESPACE,
        "assignment_seed": seed,
        "assignment_uses_outcome": False,
        "assignment_uses_features": False,
        "assignment_algorithm": (
            "PCG64 permutation of deterministic preprocessed row positions"
        ),
        "domain_n": row_count,
        "retained_n": len(retained_indices),
        "heldout_n": len(heldout_indices),
        "retained_positive_n": int(labels[retained_indices].sum()),
        "heldout_positive_n": int(labels[heldout_indices].sum()),
        "retained_prevalence": float(labels[retained_indices].mean()),
        "heldout_prevalence": float(labels[heldout_indices].mean()),
        "prevalence_gap": abs(
            float(labels[retained_indices].mean())
            - float(labels[heldout_indices].mean())
        ),
        "retained_ids_sha256": hashlib.sha256(
            "\x1f".join(retained_ids).encode("utf-8")
        ).hexdigest(),
        "heldout_ids_sha256": hashlib.sha256(
            "\x1f".join(heldout_ids).encode("utf-8")
        ).hexdigest(),
        "role_overlap": 0,
    }
    return (
        features.iloc[retained_indices].reset_index(drop=True),
        features.iloc[heldout_indices].reset_index(drop=True),
        labels[retained_indices],
        labels[heldout_indices],
        groups.iloc[retained_indices].reset_index(drop=True),
        groups.iloc[heldout_indices].reset_index(drop=True),
        audit,
    )


def outcome_blind_shift_split(
    dataset: Any,
    specification: Any,
    model_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_mask = dataset.groups[specification.group_col].isin(
        specification.source_values
    ).to_numpy()
    target_mask = dataset.groups[specification.group_col].isin(
        specification.target_values
    ).to_numpy()
    if np.any(source_mask & target_mask):
        raise RuntimeError("Source and target domains overlap")
    source_features = dataset.X.loc[source_mask].reset_index(drop=True)
    source_labels = np.asarray(dataset.y[source_mask], dtype=np.int8)
    source_groups = dataset.groups.loc[source_mask].reset_index(drop=True)
    target_features = dataset.X.loc[target_mask].reset_index(drop=True)
    target_labels = np.asarray(dataset.y[target_mask], dtype=np.int8)
    target_groups = dataset.groups.loc[target_mask].reset_index(drop=True)
    for domain, labels in (
        ("source", source_labels),
        ("target", target_labels),
    ):
        counts = np.bincount(labels, minlength=2)
        if len(labels) < 500 or np.any(counts < 25):
            raise RuntimeError(
                f"{dataset.name}/{specification.name}/{domain} is too small"
            )

    (
        x_train,
        x_source_cal,
        y_train,
        y_source_cal,
        g_train,
        g_source_cal,
        source_audit,
    ) = outcome_blind_partition(
        source_features,
        source_labels,
        source_groups,
        0.30,
        dataset=dataset.name,
        shift=specification.name,
        model_seed=model_seed,
        domain="source_train_calibration",
    )
    (
        x_target_pool,
        x_test,
        y_target_pool,
        y_test,
        g_target_pool,
        g_test,
        target_audit,
    ) = outcome_blind_partition(
        target_features,
        target_labels,
        target_groups,
        0.70,
        dataset=dataset.name,
        shift=specification.name,
        model_seed=model_seed,
        domain="target_pool_test",
    )
    split = {
        "setting": "covariate_shift",
        "shift_name": specification.name,
        "shift_description": specification.description,
        "X_train": x_train,
        "y_train": y_train,
        "X_source_cal": x_source_cal,
        "y_source_cal": y_source_cal,
        "g_source_cal": g_source_cal,
        "X_target_pool": x_target_pool,
        "y_target_pool": y_target_pool,
        "g_target_pool": g_target_pool,
        "X_test": x_test,
        "y_test": y_test,
        "g_test": g_test,
    }
    return split, [source_audit, target_audit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit all public-shift models with label-independent source and "
            "target role assignments."
        )
    )
    parser.add_argument("--lab-root", type=Path, default=DEFAULT_LAB_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("brfss", "nhanes", "uci"),
        default=["brfss", "nhanes", "uci"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--shifts", nargs="*", default=None)
    parser.add_argument("--max-brfss-per-year", type=int, default=60_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = load_module(
        "ara_public_shift_reference_generator",
        REFERENCE_GENERATOR_PATH,
    )
    lab_root = args.lab_root.resolve()
    sys.path.insert(0, str(lab_root))
    from scripts.run_large_scale_conformal import make_model
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
    datasets = [
        reference.attach_stable_ids(constructors[name]())
        for name in args.datasets
    ]
    requested_shifts = set(args.shifts or [])
    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    existing_audit: list[dict[str, Any]] = []
    if args.resume and PROVENANCE_PATH.is_file():
        payload = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        for pair in payload.get("pairs", []):
            key = (str(pair["dataset"]), str(pair["shift"]), int(pair["seed"]))
            prediction = PROJECT_ROOT / str(pair["prediction_path"])
            covariate = PROJECT_ROOT / str(pair["covariate_path"])
            if prediction.is_file() and covariate.is_file():
                existing[key] = pair
        if ROLE_AUDIT_PATH.is_file():
            existing_audit = pd.read_csv(ROLE_AUDIT_PATH).to_dict("records")

    pair_records: list[dict[str, Any]] = list(existing.values())
    role_audit_rows: list[dict[str, Any]] = list(existing_audit)
    started = time.monotonic()
    for dataset in datasets:
        numeric, categorical = reference.selected_covariates(dataset)
        for specification in dataset.shifts:
            if specification.name == "pseudo_random":
                continue
            if requested_shifts and specification.name not in requested_shifts:
                continue
            for model_seed in args.seeds:
                key = (dataset.name, specification.name, int(model_seed))
                if key in existing:
                    print(
                        f"[resume] {dataset.name}/{specification.name}/"
                        f"seed={model_seed}",
                        flush=True,
                    )
                    continue
                print(
                    f"[fit] {dataset.name}/{specification.name}/"
                    f"seed={model_seed}",
                    flush=True,
                )
                split, audits = outcome_blind_shift_split(
                    dataset,
                    specification,
                    model_seed,
                )
                model = make_model(dataset, "sgd_logistic", model_seed)
                model.fit(split["X_train"], split["y_train"])
                predictions, covariates, role_counts, positive_counts = (
                    reference.role_frames(
                        dataset,
                        split,
                        model,
                        specification.name,
                        model_seed,
                        numeric,
                        categorical,
                    )
                )
                stem = (
                    f"{dataset.name}__{specification.name}__seed_{model_seed}"
                )
                prediction_path = (
                    PREDICTION_ROOT / f"{stem}__predictions.parquet"
                )
                covariate_path = (
                    COVARIATE_ROOT / f"{stem}__covariates.parquet"
                )
                reference.write_parquet_atomic(predictions, prediction_path)
                reference.write_parquet_atomic(covariates, covariate_path)
                pair_records.append(
                    {
                        "dataset": dataset.name,
                        "shift": specification.name,
                        "seed": int(model_seed),
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
                            f"group_{name}"
                            for name in reference.SUBGROUPS[dataset.name]
                        ],
                        "covariate_columns": numeric + categorical,
                        "numeric_covariates": numeric,
                        "categorical_covariates": categorical,
                    }
                )
                role_audit_rows.extend(audits)
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
                atomic_json(
                    PROVENANCE_PATH,
                    {
                        "schema_version": (
                            "ara.public_shift.outcome_blind_source_roles.v1"
                        ),
                        "status": "running",
                        "role_namespace": ROLE_NAMESPACE,
                        "pairs": ordered,
                    },
                )
                atomic_csv(ROLE_AUDIT_PATH, pd.DataFrame(role_audit_rows))

    ordered = sorted(
        pair_records,
        key=lambda row: (
            str(row["dataset"]),
            str(row["shift"]),
            int(row["seed"]),
        ),
    )
    if len(ordered) != 60:
        raise RuntimeError(f"Expected 60 completed pairs, observed {len(ordered)}")
    if len(role_audit_rows) != 120:
        raise RuntimeError(
            f"Expected 120 source/target role audits, observed "
            f"{len(role_audit_rows)}"
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
            "schema_version": "ara.public_shift.row_level_inputs.v2",
            "role_namespace": ROLE_NAMESPACE,
            "artifacts": artifacts,
            "pairs": ordered,
        },
    )
    audit_frame = pd.DataFrame(role_audit_rows)
    atomic_csv(ROLE_AUDIT_PATH, audit_frame)
    validation = {
        "all_assignments_outcome_blind": bool(
            (~audit_frame["assignment_uses_outcome"].astype(bool)).all()
        ),
        "all_assignments_feature_blind": bool(
            (~audit_frame["assignment_uses_features"].astype(bool)).all()
        ),
        "all_roles_disjoint": bool((audit_frame["role_overlap"] == 0).all()),
        "all_partition_roles_have_both_classes": bool(
            (
                (audit_frame["retained_positive_n"] > 0)
                & (
                    audit_frame["retained_positive_n"]
                    < audit_frame["retained_n"]
                )
                & (audit_frame["heldout_positive_n"] > 0)
                & (
                    audit_frame["heldout_positive_n"]
                    < audit_frame["heldout_n"]
                )
            ).all()
        ),
    }
    payload = {
        "schema_version": "ara.public_shift.outcome_blind_source_roles.v1",
        "status": "completed" if all(validation.values()) else "failed",
        "elapsed_seconds": time.monotonic() - started,
        "role_namespace": ROLE_NAMESPACE,
        "stable_row_id_definition": (
            "dataset name plus zero-padded row position after deterministic "
            "dataset preprocessing; IDs are used only for auditing, while "
            "role assignment uses a namespaced PCG64 permutation."
        ),
        "assignment_protocol": (
            "Within each deterministic source or target domain, a seeded "
            "PCG64 permutation is generated without reading labels, scores, "
            "or features. Source uses 70/30 train/calibration and target uses "
            "30/70 calibration-pool/test counts, matching the archived sizes."
        ),
        "model": "sgd_logistic",
        "model_refit": True,
        "outcome_model_family_changed": False,
        "pair_count": len(ordered),
        "dataset_shift_count": len(
            {(row["dataset"], row["shift"]) for row in ordered}
        ),
        "maximum_role_prevalence_gap": float(
            audit_frame["prevalence_gap"].max()
        ),
        "mean_role_prevalence_gap": float(
            audit_frame["prevalence_gap"].mean()
        ),
        "validation": validation,
        "pairs": ordered,
        "outputs": [
            "remediation_inputs.json",
            "role_assignment_audit.csv",
            "artifacts/predictions/*.parquet",
            "artifacts/covariates/*.parquet",
        ],
        "provenance": {
            "script_sha256": sha256_file(SCRIPT_PATH),
            "reference_generator_path": str(
                REFERENCE_GENERATOR_PATH.relative_to(PROJECT_ROOT)
            ),
            "reference_generator_sha256": sha256_file(
                REFERENCE_GENERATOR_PATH
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "command": (
                "python exp/source_role_outcome_blind_20260725/generate.py"
            ),
        },
    }
    atomic_json(PROVENANCE_PATH, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
