"""Frozen independent clinical-task validation; no target-guided model fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "jbi-r1-20261009/benchmark"))
from api import CalibrationInput, fit_split, set_metrics

PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
FRACTIONS = PROTOCOL["fractions_of_full_target"]
METRICS = ["coverage", "shortfall", "mean_set_size", "empty_set_rate", "singleton_rate", "full_set_rate",
           "class_0_coverage", "class_1_coverage", "class_coverage_gap", "correct_singleton_rate",
           "incorrect_singleton_rate", "positive_class_shortfall"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def split_roles(frame, dataset, condition, seed, control=False):
    if control:
        order = np.random.default_rng(seed_for("external-control-v1", dataset, seed)).permutation(len(frame))
        nt = int(np.ceil(.25 * len(frame)))
        test, development = frame.iloc[order[:nt]], frame.iloc[order[nt:]]
        order2 = np.random.default_rng(seed_for("external-control-cal-v1", dataset, seed)).permutation(len(development))
        nc = int(np.ceil(.3 * len(development)))
        return {"source_train": development.iloc[order2[nc:]], "source_calibration": development.iloc[order2[:nc]],
                "target_test": test}
    source = frame[frame.center != condition] if dataset == "heart" else frame[frame.center != condition]
    target = frame[frame.center == condition]
    sorder = np.random.default_rng(seed_for("external-source-v1", dataset, condition, seed)).permutation(len(source))
    torder = np.random.default_rng(seed_for("external-target-v1", dataset, condition, seed)).permutation(len(target))
    nc, npool = int(np.ceil(.3 * len(source))), int(np.ceil(.3 * len(target)))
    return {"source_train": source.iloc[sorder[nc:]], "source_calibration": source.iloc[sorder[:nc]],
            "target_calibration_pool": target.iloc[torder[:npool]], "target_test": target.iloc[torder[npool:]]}


def fit_probabilities(frame, roles, dataset, backbone, seed):
    features = [c for c in frame if c not in ["row_id", "center", "label"]]
    categorical = [c for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"] if c in features] if dataset == "heart" else []
    numeric = [c for c in features if c not in categorical]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("scale", StandardScaler())])
    transforms = [("numeric", numeric_pipe, numeric)]
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False))]), categorical))
    preprocess = ColumnTransformer(transforms, remainder="drop")
    if backbone == "elastic_net":
        clf = SGDClassifier(loss="log_loss", penalty="elasticnet", l1_ratio=.10, alpha=.0001,
                            max_iter=1000, tol=.001, class_weight="balanced", random_state=seed)
    else:
        clf = LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=.05,
                             class_weight="balanced", n_jobs=2, random_state=seed, verbosity=-1)
    model = Pipeline([("preprocess", preprocess), ("classifier", clf)])
    train = roles["source_train"]
    assert train.label.nunique() == 2
    with warnings.catch_warnings(record=True) as alerts:
        warnings.simplefilter("always")
        model.fit(train[features], train.label)
        result = []
        for role, f in roles.items():
            part = f[["row_id", "center", "label"]].copy()
            part["prob_1"] = model.predict_proba(f[features])[:, 1]
            part["role"] = role
            result.append(part)
    predictions = pd.concat(result, ignore_index=True)
    assert predictions.row_id.is_unique and predictions.prob_1.between(0, 1).all()
    return predictions, {"features": features, "source_train_n": len(train),
                         "source_train_ids_sha256": hashlib.sha256("\n".join(train.row_id).encode()).hexdigest(),
                         "warnings": sorted(set(str(w.message) for w in alerts))}


def metrics(sets, y):
    m = set_metrics(sets, y)
    sizes = sets.sum(axis=1)
    hits = sets[np.arange(len(y)), y]
    m.update(shortfall=max(0., .9 - m["coverage"]),
             positive_class_shortfall=max(0., .9 - m["class_1_coverage"]) if np.any(y == 1) else np.nan,
             correct_singleton_rate=float(((sizes == 1) & hits).mean()),
             incorrect_singleton_rate=float(((sizes == 1) & ~hits).mean()))
    return m


def fit_evaluate(cal, test, mondrian, base):
    data = CalibrationInput(np.array([]), np.array([], dtype=int), cal.prob_1.to_numpy(),
                            cal.label.to_numpy(dtype=int), .1, 0)
    fitted = fit_split(data, mondrian=mondrian)
    sets = fitted.predict_sets(test.prob_1.to_numpy())
    counts = np.bincount(cal.label.to_numpy(dtype=int), minlength=2)
    return dict(base, n_calibration=len(cal), n_class0=int(counts[0]), n_class1=int(counts[1]),
                threshold0=fitted.threshold_0, threshold1=fitted.threshold_1,
                ids_sha256=hashlib.sha256("\n".join(cal.row_id).encode()).hexdigest(),
                **metrics(sets, test.label.to_numpy(dtype=int)))


def availability(p, y, regime, n):
    weights = np.ones(len(p)) if regime == "uniform" else np.where(y == 1, .2, 1.) if regime == "outcome_depleted" else .1 + .9 * p
    assert 2 * n <= len(p)
    lo, hi = 0., 2 * n / weights.min()
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.minimum(1., mid * weights).sum() < 2 * n:
            lo = mid
        else:
            hi = mid
    return np.minimum(1., hi * weights)


def summarize(frame, keys):
    seed = frame.groupby(keys + ["seed"], dropna=False)[METRICS].mean().reset_index()
    return seed.groupby(keys, dropna=False)[METRICS].mean().reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["heart", "sepsis"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.artifacts / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    budget_rows, channel_rows, control_rows, diagnostics, input_records = [], [], [], [], []
    failures = []
    for dataset in args.datasets:
        cohort_path = args.artifacts / (dataset + ".parquet")
        frame = pd.read_parquet(cohort_path)
        assert frame.row_id.is_unique
        for backbone in ["elastic_net", "lightgbm"]:
            for condition in ["random_control", *sorted(frame.center.unique())]:
                for seed in range(5):
                    base = dict(dataset=dataset, backbone=backbone, condition=condition, seed=seed)
                    roles = split_roles(frame, dataset, condition, seed, condition == "random_control")
                    path = prediction_dir / ("-".join(map(str, base.values())) + ".parquet")
                    fit_receipt = path.with_suffix(".json")
                    if path.exists() and fit_receipt.exists():
                        predictions = pd.read_parquet(path)
                        receipt = json.loads(fit_receipt.read_text())
                        assert receipt["protocol_sha256"] == sha(ROOT / "protocol.json")
                        assert receipt["cohort_sha256"] == sha(cohort_path)
                        assert receipt["code_sha256"] == sha(Path(__file__))
                        assert receipt["prediction_sha256"] == sha(path)
                    else:
                        try:
                            predictions, record = fit_probabilities(frame, roles, dataset, backbone, seed)
                        except Exception as exc:
                            failures.append(dict(base, error=repr(exc)))
                            (args.out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
                            print("FAILED", base, repr(exc), flush=True)
                            continue
                        predictions.to_parquet(path, index=False)
                        receipt = dict(base, **record, cohort_sha256=sha(cohort_path),
                                       protocol_sha256=sha(ROOT / "protocol.json"), code_sha256=sha(Path(__file__)),
                                       prediction_sha256=sha(path))
                        fit_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
                    input_records.append(dict(base, path=str(path), sha256=sha(path), fit_receipt=str(fit_receipt)))
                    s = predictions[predictions.role == "source_calibration"].reset_index(drop=True)
                    test = predictions[predictions.role == "target_test"].reset_index(drop=True)
                    y, p = test.label.to_numpy(dtype=int), test.prob_1.to_numpy()
                    diagnostics.append(dict(base, n_test=len(test), class0_test=int((y == 0).sum()),
                                            class1_test=int((y == 1).sum()), prevalence=float(y.mean()),
                                            auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                                            balanced_accuracy=float(balanced_accuracy_score(y, p >= .5))))
                    if condition == "random_control":
                        for mondrian in [False, True]:
                            control_rows.append(fit_evaluate(s, test, mondrian,
                                dict(base, method="source_mondrian" if mondrian else "source_split")))
                        continue
                    pool = predictions[predictions.role == "target_calibration_pool"].reset_index(drop=True)
                    full_target_n = len(pool) + len(test)
                    for rep in range(5):
                        order = np.random.default_rng(seed_for("external-budget-v1", dataset, condition, seed, rep)).permutation(len(pool))
                        for fraction in FRACTIONS:
                            n = max(1, int(round(fraction * full_target_n)))
                            assert n <= len(pool)
                            cal = pool.iloc[order[:n]]
                            for name in ["source_split", "source_mondrian", "target_split", "target_mondrian"]:
                                used = s if name.startswith("source") else cal
                                budget_rows.append(fit_evaluate(used, test, name.endswith("mondrian"),
                                    dict(base, replicate=rep, fraction=fraction, method=name,
                                         target_budget=n, full_target_n=full_target_n)))
                        n = max(1, int(round(.10 * full_target_n)))
                        for regime in ["uniform", "outcome_depleted", "risk_priority"]:
                            probabilities = availability(pool.prob_1.to_numpy(), pool.label.to_numpy(dtype=int), regime, n)
                            rng = np.random.default_rng(seed_for("external-available-v1", dataset, condition, seed, rep, regime))
                            eligible = np.flatnonzero(rng.random(len(pool)) < probabilities)
                            chosen = rng.permutation(eligible)[:n]
                            sentinel = np.random.default_rng(seed_for("external-sentinel-v1", dataset, condition, seed, rep, regime)).permutation(len(pool))[:n]
                            for design, indices in [("available", chosen), ("sentinel", sentinel)]:
                                cal = pool.iloc[indices]
                                for mondrian in [False, True]:
                                    name = design + ("_mondrian" if mondrian else "_split")
                                    channel_rows.append(fit_evaluate(cal, test, mondrian,
                                        dict(base, replicate=rep, regime=regime, method=name, target_budget=n,
                                             eligible_n=len(eligible), quota_shortfall=max(0, n - len(chosen)))))
                    print("completed", dataset, backbone, condition, seed, flush=True)
    tables = {"budget_metrics": pd.DataFrame(budget_rows), "channel_metrics": pd.DataFrame(channel_rows),
              "control_metrics": pd.DataFrame(control_rows), "predictor_diagnostics": pd.DataFrame(diagnostics)}
    for name, f in tables.items():
        f.to_csv(args.out / (name + ".csv"), index=False)
    for name, keys in [("budget", ["dataset", "backbone", "condition", "fraction", "method"]),
                       ("channel", ["dataset", "backbone", "condition", "regime", "method"])]:
        condition = summarize(tables[name + "_metrics"], keys)
        condition.to_csv(args.out / (name + "_by_condition.csv"), index=False)
        condition.groupby([k for k in keys if k != "condition"])[METRICS].mean().reset_index().to_csv(
            args.out / (name + "_by_dataset.csv"), index=False)
    receipt = {"status": "completed" if not failures else "completed_with_failures",
               "protocol_sha256": sha(ROOT / "protocol.json"), "code_sha256": sha(Path(__file__)),
               "failures": failures, "inputs": input_records, "rows": {name: len(f) for name, f in tables.items()},
               "outputs": {p.name: sha(p) for p in args.out.glob("*.csv")}}
    (args.out / "run-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "rows": receipt["rows"]}), flush=True)


if __name__ == "__main__":
    main()
