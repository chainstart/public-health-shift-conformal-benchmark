# Prediction-only binary calibration extensions

This additive interface accepts frozen class-1 probabilities from a chosen experiment. It does not construct cohorts, train a model, choose source/target splits, or replace the archived benchmark runner. Methods requiring covariates, feature-dependent weights, or unlabeled adaptation need an extended input contract; they are not represented by this score-only interface.

Use Python 3.10 or later; the interface and its tests use only the standard library. The actual verification environment and results are recorded separately. No clinical performance or coverage guarantee is inferred from the toy example.

Calibration CSV columns must be exactly `row_id,p1,label`, with binary labels 0/1 and finite probabilities in [0,1]. Test-prediction CSV columns must be exactly `row_id,p1`. IDs must be unique within each role and disjoint between roles. Use stable IDs exported from the actual frozen role assignment; this entrypoint does not create new roles. Test labels are not inputs to fitting or prediction. Evaluate the resulting sets separately against held-out outcomes.

```sh
python -m scripts.calibration_plugins --calibration cal.csv --predictions test-p1.csv --method split --alpha 0.1 --output sets.json
python -m scripts.calibration_plugins --calibration cal.csv --predictions test-p1.csv --method mondrian --alpha 0.1 --output class-sets.json
python -m unittest discover -s tests -v
```

Built-ins implement the order statistic ceil((n+1)(1-alpha)), appending positive infinity when the rank exceeds the available calibration count, including a missing Mondrian class. Inclusion uses score <= threshold. Empty sets remain empty; there is no forced argmax label. Validity still depends on the statistical assumptions of the method and sampling protocol.

To add a method without editing the runner, provide an importable `my_module:fit` function accepting `(tuple[CalibrationRow, ...], alpha)` and returning an object with `predict_set(p1) -> frozenset[int]`, then pass `--method my_module:fit`. Only labeled calibration rows and alpha reach the fitting function. The seven actual tests exercise exact ranks, rare/missing classes, ties, empty sets, ID overlap, plugin loading/fitting, invalid inputs, and a CSV-to-JSON subprocess run.

Historical cohort code and all archived result tables are unchanged. This is a development source extension for review, separate from the versioned manuscript supporting-material release. It has not been used to produce new manuscript results or asserted to have received new human code inspection.

## Difference from the historical public implementation

Direct execution of the original public functions with the pinned NumPy 2.3.4 confirmed differences on synthetic inputs. For 20 scores 0.05,0.10,...,1.00 at alpha=0.1, the required rank is 19: the historical higher-quantile call returns 1.00 and the new order-statistic implementation returns 0.95. With eight scores and required rank nine, the historical clipped implementation returns the maximum score whereas the candidate uses positive infinity. The historical Mondrian branch pools the threshold when a class has fewer than 20 rows; the candidate computes class-specific ranks, including infinity for a missing/too-small class. The historical benchmark calls prediction_sets with fallback=True, replacing empty sets by the argmax label; the candidate preserves empty sets.

These are concrete implementation differences, not evidence of an observed change in clinical performance. The historical runner and its archived results have not been rewritten. New benchmark results require a selected, frozen protocol and an actual rerun; they cannot be obtained by attaching new method names to old tables.
