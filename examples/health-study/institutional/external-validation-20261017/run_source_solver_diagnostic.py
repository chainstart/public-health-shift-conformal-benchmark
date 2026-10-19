"""Explicit posthoc source-only L2 optimization control; retain primary results."""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_external import split_roles, seed_for, fit_evaluate, availability, sha, summarize

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pred_dir = a.artifacts / "posthoc-l2-predictions"
    pred_dir.mkdir(exist_ok=True)
    cohort_path = a.artifacts / "sepsis.parquet"
    f = pd.read_parquet(cohort_path)
    features = [c for c in f if c not in ["label", "center", "row_id"]]
    budget, channel, control, diagnostics, inputs = [], [], [], [], []
    for condition in ["random_control", "A", "B"]:
        for seed in range(5):
            base = dict(dataset="sepsis", backbone="logistic_l2", condition=condition, seed=seed)
            roles = split_roles(f, "sepsis", condition, seed, condition == "random_control")
            train = roles["source_train"]
            model = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler()),
                              ("classifier", LogisticRegression(C=1, solver="lbfgs", max_iter=5000,
                                   class_weight="balanced", random_state=seed))])
            with warnings.catch_warnings(record=True) as alerts:
                warnings.simplefilter("always")
                model.fit(train[features], train.label)
            predictions = []
            for role, part in roles.items():
                p = part[["row_id", "center", "label"]].copy()
                p["prob_1"] = model.predict_proba(part[features])[:, 1]
                p["role"] = role
                predictions.append(p)
            p = pd.concat(predictions, ignore_index=True)
            assert p.row_id.is_unique
            path = pred_dir / (condition + "-" + str(seed) + ".parquet")
            p.to_parquet(path, index=False)
            record = dict(base, features=features, protocol_sha256=sha(ROOT / "protocol.json"),
                          diagnostic_protocol_sha256=sha(ROOT / "source_solver_diagnostic.json"),
                          code_sha256=sha(Path(__file__)), cohort_sha256=sha(cohort_path), prediction_sha256=sha(path),
                          source_train_ids_sha256=hashlib.sha256("\n".join(train.row_id).encode()).hexdigest(),
                          warnings=[str(w.message) for w in alerts], optimizer_iterations=model[-1].n_iter_.tolist())
            rp = path.with_suffix(".json")
            rp.write_text(json.dumps(record, indent=2) + "\n")
            inputs.append(dict(base, path=str(path), sha256=sha(path), fit_receipt=str(rp)))
            s = p[p.role == "source_calibration"].reset_index(drop=True)
            test = p[p.role == "target_test"].reset_index(drop=True)
            diagnostics.append(dict(base, source_calibration_auroc=roc_auc_score(s.label, s.prob_1),
                                    auroc=roc_auc_score(test.label, test.prob_1),
                                    balanced_accuracy=balanced_accuracy_score(test.label, test.prob_1 >= .5)))
            if condition == "random_control":
                for mondrian in [False, True]:
                    control.append(fit_evaluate(s, test, mondrian,
                        dict(base, method="source_mondrian" if mondrian else "source_split")))
                continue
            pool = p[p.role == "target_calibration_pool"].reset_index(drop=True)
            n = max(1, int(round(.1 * (len(pool) + len(test)))))
            for rep in range(5):
                order = np.random.default_rng(seed_for("external-budget-v1", "sepsis", condition, seed, rep)).permutation(len(pool))
                cal = pool.iloc[order[:n]]
                for method in ["source_split", "source_mondrian", "target_split", "target_mondrian"]:
                    used = s if method.startswith("source") else cal
                    budget.append(fit_evaluate(used, test, method.endswith("mondrian"),
                        dict(base, replicate=rep, fraction=.1, method=method, target_budget=n, full_target_n=len(pool) + len(test))))
                for regime in ["uniform", "outcome_depleted", "risk_priority"]:
                    prob = availability(pool.prob_1.to_numpy(), pool.label.to_numpy(dtype=int), regime, n)
                    rng = np.random.default_rng(seed_for("external-available-v1", "sepsis", condition, seed, rep, regime))
                    eligible = np.flatnonzero(rng.random(len(pool)) < prob)
                    chosen = rng.permutation(eligible)[:n]
                    sentinel = np.random.default_rng(seed_for("external-sentinel-v1", "sepsis", condition, seed, rep, regime)).permutation(len(pool))[:n]
                    for design, ids in [("available", chosen), ("sentinel", sentinel)]:
                        for mondrian in [False, True]:
                            channel.append(fit_evaluate(pool.iloc[ids], test, mondrian,
                                dict(base, replicate=rep, regime=regime,
                                     method=design + ("_mondrian" if mondrian else "_split"), target_budget=n,
                                     eligible_n=len(eligible), quota_shortfall=max(0, n - len(chosen)))))
            print("posthoc source L2", condition, seed, flush=True)
    for name, rows in [("budget_metrics", budget), ("channel_metrics", channel), ("control_metrics", control), ("predictor_diagnostics", diagnostics)]:
        pd.DataFrame(rows).to_csv(a.out / (name + ".csv"), index=False)
    for name, rows, keys in [("budget", budget, ["dataset", "backbone", "condition", "fraction", "method"]),
                             ("channel", channel, ["dataset", "backbone", "condition", "regime", "method"])]:
        c = summarize(pd.DataFrame(rows), keys)
        c.to_csv(a.out / (name + "_by_condition.csv"), index=False)
        c.groupby([k for k in keys if k != "condition"]).mean(numeric_only=True).reset_index().to_csv(a.out / (name + "_by_dataset.csv"), index=False)
    receipt = {"status": "completed", "evidence_type": "posthoc_source_optimizer_control", "failures": [],
               "protocol_sha256": sha(ROOT / "protocol.json"), "diagnostic_protocol_sha256": sha(ROOT / "source_solver_diagnostic.json"),
               "code_sha256": sha(Path(__file__)), "inputs": inputs,
               "rows": {"budget": len(budget), "channel": len(channel), "control": len(control)},
               "outputs": {p.name: sha(p) for p in a.out.glob("*.csv")}}
    (a.out / "run-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["rows"]), flush=True)


if __name__ == "__main__":
    main()
