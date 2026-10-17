"""Prediction-only binary calibration interface; no experimental results implied."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
from typing import Callable, Protocol


def probability(value: object) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("p1 must be finite and between zero and one")
    return value


@dataclass(frozen=True)
class CalibrationRow:
    row_id: str
    p1: float
    label: int

    def __post_init__(self) -> None:
        if not self.row_id or type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("A calibration row needs an ID and binary integer label")
        object.__setattr__(self, "p1", probability(self.p1))


class FittedMethod(Protocol):
    def predict_set(self, p1: float) -> frozenset[int]: ...


Fitter = Callable[[tuple[CalibrationRow, ...], float], FittedMethod]


def finite_quantile(scores: list[float], alpha: float) -> float:
    """Split-conformal order statistic with the appended +infinity convention."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return math.inf if rank > len(scores) else sorted(scores)[rank - 1]


@dataclass(frozen=True)
class ScoreThresholds:
    q0: float
    q1: float

    def predict_set(self, p1: float) -> frozenset[int]:
        p1 = probability(p1)
        return frozenset(label for label, score, q in
                         ((0, p1, self.q0), (1, 1 - p1, self.q1)) if score <= q)


def fit_split(rows: tuple[CalibrationRow, ...], alpha: float) -> ScoreThresholds:
    q = finite_quantile([r.p1 if r.label == 0 else 1 - r.p1 for r in rows], alpha)
    return ScoreThresholds(q, q)


def fit_mondrian(rows: tuple[CalibrationRow, ...], alpha: float) -> ScoreThresholds:
    scores = [[r.p1 if label == 0 else 1 - r.p1 for r in rows if r.label == label]
              for label in (0, 1)]
    return ScoreThresholds(*(finite_quantile(s, alpha) for s in scores))


BUILTINS: dict[str, Fitter] = {"split": fit_split, "mondrian": fit_mondrian}


def load_fitter(name: str) -> Fitter:
    if name in BUILTINS:
        return BUILTINS[name]
    module_name, separator, function_name = name.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Use split, mondrian, or an importable module:function")
    fitter = getattr(importlib.import_module(module_name), function_name)
    if not callable(fitter):
        raise ValueError("The plugin fitter must be callable")
    return fitter


def predict_sets(fitter: Fitter, calibration: tuple[CalibrationRow, ...],
                 test_predictions: tuple[tuple[str, float], ...], alpha: float
                 ) -> list[dict[str, object]]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    cal_ids = [r.row_id for r in calibration]
    test_ids = [r[0] for r in test_predictions]
    if (len(set(cal_ids)) != len(cal_ids) or len(set(test_ids)) != len(test_ids)
            or any(not x for x in test_ids) or set(cal_ids) & set(test_ids)):
        raise ValueError("Role IDs must be unique, nonempty, and disjoint")
    checked = [(row_id, probability(p1)) for row_id, p1 in test_predictions]
    # Only labeled calibration rows and alpha reach fit; test outcomes are absent.
    fitted = fitter(calibration, alpha)
    results = []
    for row_id, p1 in checked:
        labels = fitted.predict_set(p1)
        if not isinstance(labels, (set, frozenset)) or any(
                type(label) is not int or label not in (0, 1) for label in labels):
            raise ValueError("A prediction set must contain only binary integer labels")
        results.append({"row_id": row_id, "prediction_set": sorted(labels)})
    return results


def read_csv(path: Path, fields: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if (reader.fieldnames is None or len(reader.fieldnames) != len(fields)
                or set(reader.fieldnames) != fields):
            raise ValueError(f"{path.name} must have exactly columns {sorted(fields)}")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed CSV row")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--method", default="split")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = tuple(CalibrationRow(r["row_id"], probability(r["p1"]), int(r["label"]))
                        for r in read_csv(args.calibration, {"row_id", "p1", "label"}))
    predictions = tuple((r["row_id"], probability(r["p1"]))
                        for r in read_csv(args.predictions, {"row_id", "p1"}))
    if args.output.resolve() in (args.calibration.resolve(), args.predictions.resolve()):
        raise ValueError("Output must not overwrite an input")
    result = {"schema": "prediction-only-calibration.v1", "method": args.method,
              "alpha": args.alpha, "n_calibration": len(calibration),
              "predictions": predict_sets(load_fitter(args.method), calibration,
                                          predictions, args.alpha)}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
