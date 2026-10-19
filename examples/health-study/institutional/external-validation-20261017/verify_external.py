"""Independently rebuild calibration membership, sorted thresholds and metrics.

This verifier does not import the fitting runner or conformal implementation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def seed_for(*parts):
    payload = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def reference(cal, test, mondrian):
    py, yy = cal.prob_1.to_numpy(), cal.label.to_numpy(dtype=int)
    scores = np.array([p if y == 0 else 1 - p for p, y in zip(py, yy)])
    def quantile(values):
        rank = math.ceil((len(values) + 1) * .90)
        return float(sorted(values)[rank - 1]) if 1 <= rank <= len(values) else float("inf")
    thresholds = [quantile(scores[yy == c]) if mondrian else quantile(scores) for c in (0, 1)]
    p = test.prob_1.to_numpy()
    y = test.label.to_numpy(dtype=int)
    sets = np.stack([p <= thresholds[0], 1 - p <= thresholds[1]], axis=1)
    hits, size = sets[np.arange(len(y)), y], sets.sum(axis=1)
    classes = [float(hits[y == c].mean()) if np.any(y == c) else np.nan for c in (0, 1)]
    coverage = float(hits.mean())
    return {"threshold0": thresholds[0], "threshold1": thresholds[1], "coverage": coverage,
            "shortfall": max(0., .9 - coverage), "mean_set_size": float(size.mean()),
            "empty_set_rate": float((size == 0).mean()), "singleton_rate": float((size == 1).mean()),
            "full_set_rate": float((size == 2).mean()), "class_0_coverage": classes[0],
            "class_1_coverage": classes[1], "class_coverage_gap": abs(classes[1] - classes[0]),
            "positive_class_shortfall": max(0., .9 - classes[1]) if np.any(y == 1) else np.nan,
            "correct_singleton_rate": float(((size == 1) & hits).mean()),
            "incorrect_singleton_rate": float(((size == 1) & ~hits).mean()),
            "n_calibration": len(cal), "n_class0": int((yy == 0).sum()), "n_class1": int((yy == 1).sum()),
            "ids_sha256": hashlib.sha256("\n".join(cal.row_id).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.results / "run-receipt.json").read_text())
    assert not receipt["failures"], receipt["failures"]
    budget = pd.read_csv(args.results / "budget_metrics.csv")
    channels = pd.read_csv(args.results / "channel_metrics.csv")
    controls = pd.read_csv(args.results / "control_metrics.csv")
    verified, errors, prediction_count = 0, [], 0
    for inp in receipt["inputs"]:
        path = Path(inp["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == inp["sha256"]
        p = pd.read_parquet(path)
        prediction_count += 1
        assert p.row_id.is_unique
        fit = json.loads(Path(inp["fit_receipt"]).read_text())
        assert not set(fit["features"]) & {"row_id", "center", "label", "SepsisLabel", "ICULOS"}
        train = p[p.role == "source_train"]
        assert hashlib.sha256("\n".join(train.row_id).encode()).hexdigest() == fit["source_train_ids_sha256"]
        s = p[p.role == "source_calibration"].reset_index(drop=True)
        pool = p[p.role == "target_calibration_pool"].reset_index(drop=True)
        test = p[p.role == "target_test"].reset_index(drop=True)
        if inp["condition"] != "random_control":
            assert (test.center == inp["condition"]).all() and (pool.center == inp["condition"]).all()
            assert (train.center != inp["condition"]).all() and (s.center != inp["condition"]).all()
        for kind, f in [("budget", budget), ("channel", channels), ("control", controls)]:
            subset = f.copy()
            for key in ["dataset", "backbone", "condition", "seed"]:
                subset = subset[subset[key] == inp[key]]
            for row in subset.to_dict("records"):
                if kind == "control" or row["method"].startswith("source"):
                    cal = s
                elif kind == "budget":
                    n = max(1, round(row["fraction"] * (len(pool) + len(test))))
                    order = np.random.default_rng(seed_for("external-budget-v1", inp["dataset"], inp["condition"], inp["seed"], row["replicate"])).permutation(len(pool))
                    cal = pool.iloc[order[:n]]
                else:
                    n = max(1, round(.1 * (len(pool) + len(test))))
                    if row["method"].startswith("sentinel"):
                        order = np.random.default_rng(seed_for("external-sentinel-v1", inp["dataset"], inp["condition"], inp["seed"], row["replicate"], row["regime"])).permutation(len(pool))
                        cal = pool.iloc[order[:n]]
                    else:
                        weights = np.ones(len(pool)) if row["regime"] == "uniform" else np.where(pool.label == 1, .2, 1.) if row["regime"] == "outcome_depleted" else .1 + .9 * pool.prob_1.to_numpy()
                        scale = brentq(lambda c: np.minimum(1., c * weights).sum() - 2 * n, 0, 2 * n / weights.min(), xtol=1e-12)
                        rng = np.random.default_rng(seed_for("external-available-v1", inp["dataset"], inp["condition"], inp["seed"], row["replicate"], row["regime"]))
                        eligible = np.flatnonzero(rng.random(len(pool)) < np.minimum(1., scale * weights))
                        assert len(eligible) == row["eligible_n"]
                        cal = pool.iloc[rng.permutation(eligible)[:n]]
                actual = reference(cal, test, row["method"].endswith("mondrian"))
                for key, value in actual.items():
                    equal = value == row[key] if isinstance(value, str) else np.isclose(value, row[key], atol=1e-12, rtol=1e-12, equal_nan=True)
                    if not equal:
                        errors.append({k: row[k] for k in ["dataset", "backbone", "condition", "seed", "method"]} | {"metric": key, "expected": str(value), "reported": str(row[key])})
                verified += 1
    assert verified == len(budget) + len(channels) + len(controls)
    result = {"status": "pass" if not errors else "fail", "prediction_files": prediction_count,
              "reconstructed_rows": verified, "discrepancies": errors,
              "method": "independent sorted-score rank, explicit Boolean sets and metrics; brent root for selection weights; calibration IDs reconstructed from protocol seeds",
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.results / "independent-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "rows": verified, "discrepancies": len(errors)}))
    assert not errors, errors[:5]


if __name__ == "__main__":
    main()
