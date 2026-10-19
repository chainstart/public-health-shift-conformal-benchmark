# Binary method interface for the additional outcome-blind protocol

See ../../PROTOCOL_MAP.md for protocol attribution.

# Binary probability-output method interface

The interface evaluates additional binary conformal methods on fixed source
calibration, target calibration pool, and target test roles. It accepts the
probabilities of an already source-fitted model; it does not train arbitrary
models, implement online adaptation, or provide a full-covariate plug-in API.
The separate LR-QR reproduction uses covariates through its own runner.

A Python extension defines `METHOD = MethodSpec(name, uses_target_labels, fit)`.
Its `fit(CalibrationInput)` returns an object with `predict_sets(probabilities)`.
The output is an `n_test x 2` Boolean matrix for membership of labels 0 and 1.
Methods declaring `uses_target_labels=False` receive no target outcomes.
Test outcomes are kept in the evaluation runner and are never passed to `fit`
or `predict_sets`. This is an interface contract, not a sandbox for malicious
Python code. Source and target sample counts and method information use must
be reported along with results; receiving target probabilities is additional
information even when their labels are withheld.

From this directory, use a local Python environment with the recorded
dependencies. `LAB` denotes a reproduction root containing `scripts/` and
`exp/source_role_outcome_blind_20260725/`, with regenerated predictions.

```
python check_contract.py
python run_method.py --lab-root "$LAB" --method example_method.py --out example.csv
python run_method.py --lab-root "$LAB" --method source_split --out builtin.csv
python run_budget.py --lab-root "$LAB" --out budget-results
python verify_budget.py --lab-root "$LAB" --out budget-results
python run_adult_sensitivity.py --lab-root "$LAB" --out adult-results
```

The default method comparison uses five matched draws of 300 source and 300
target-pool rows per condition and seed. Sampling is without replacement within
a draw; repeated draws reuse the same pools. Outcome-model training is fixed.
The extension example implements source split conformal and should match the
built-in method exactly after omitting the method-name column.

The budget experiment uses nested random prefixes at fractions 0.005, 0.01,
0.02, 0.05, 0.10 and 0.20 of the full target domain, five draws per model seed,
and all four basic methods. Its CSV outputs include absolute and class counts,
rank-overflow indicators, coverage, class coverage, mean set size, and empty,
singleton and two-label rates. The verifier reconstructs thresholds by sorted
order statistics instead of importing the implementation under test.

For all basic methods the finite rank is `ceil((n+1)*0.9)`. An absent class or
a rank greater than its sample size receives an infinite threshold. Ties are
included; no argmax label is inserted into an empty set. Results are averaged
within seed, then condition, then dataset; the three datasets have equal
weight. These summaries do not estimate effects over independent deployments.


### Additional outcome-blind eight-budget protocol

The article uses `../restore_eight_budgets.py`, with fractions 0.005, 0.01,
0.02, 0.03, 0.05, 0.10, 0.15 and 0.20: 2,400 draws and 9,600 method rows.
The standalone `run_budget.py` default is a separate six-fraction example.
Use the eight-fraction wrapper and the dynamic-count verifier to reconstruct
the article reference results; do not overwrite them with the default example.
