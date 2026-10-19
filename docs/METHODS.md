# Add a method

Copy [examples/source_split.py](../examples/source_split.py) to a new Python file.
Import `MethodSpec` from `benchmark.api` and expose `METHOD`:

```python
from benchmark.api import MethodSpec, ThresholdSets, finite_quantile, true_scores

def fit(data):
    scores = true_scores(data.source_probabilities, data.source_labels)
    q = finite_quantile(scores, data.alpha)
    return ThresholdSets(q, q)

METHOD = MethodSpec("my_source_split", uses_target_labels=False, fit=fit)
```

`fit(CalibrationInput)` returns an object implementing
`predict_sets(probabilities) -> numpy.ndarray`. Return an `n_test × 2` Boolean
matrix whose columns specify label-0 and label-1 membership. You can implement
arbitrary set construction within this interface; `ThresholdSets` is a convenience.
`CalibrationInput` contains source probabilities/labels, target probabilities,
optional target labels, alpha and seed. A method declaring no target-label use
receives `target_labels=None`. Every method receives separate copies of inputs.
Use the provided seed for randomized methods; avoid global randomness.

```bash
python -m benchmark evaluate --workspace work/health \
  --methods source_split my_method.py --out work/custom.csv
python -m benchmark compare --input work/custom.csv --out work/custom-comparison
```

Report target-label access, label counts, unlabeled-target access, method settings
and predictor identity with results. Do not tune on target-test labels. The Python
interface is not a sandbox. Full-covariate methods require a separate runner and
must retain the same roles/budgets; example OT/LR-QR runners are in the paper case
study and are not silently represented by the four basic API baselines.

## Frozen prediction schema

A workspace contains `exp/source_role_outcome_blind_20260725/generation_provenance.json`.
Its `pairs` array records `dataset`, `shift`, `seed`, `prediction_path` (relative
to the workspace) and `prediction_sha256`. Each parquet has `role`, `prob_1` and
`label`; recognized evaluation roles are `source_calibration`,
`target_calibration_pool` and `target_test`. Training rows may also be present
and are not sampled for calibration/evaluation. Labels must be binary and
probabilities finite within [0,1]. Preserve row IDs and upstream role audits.

Receipts identify the supplied prediction files, not independently verified
absence of training leakage. New dataset/predictor providers must document and
validate training, role disjointness and preprocessing. The standard preparation
workflow supplies the archived role generator and its audits.

## Historical interfaces

`examples/legacy-experiments/scripts/calibration_plugins.py` is retained for compatibility with the
older CSV prediction-only example. New methods should use `benchmark.api` and
the unified runner. `examples/health-study/reproduction/benchmark/` preserves the exact
published API implementation for paper reproduction. The maintained API starts
from that implementation; versions and tests identify subsequent changes.
