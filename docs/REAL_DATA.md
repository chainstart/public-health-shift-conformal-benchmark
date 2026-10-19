# Real-data workflow

Run from the repository root with Python 3.12. Full cohort reconstruction needs
network access, disk space for annual survey files and approximately 32 GB RAM.
It is CPU based. Set `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2` and
`MKL_NUM_THREADS=2` if needed. Raw records and generated probabilities stay local.

## Prepare a new workspace

```bash
python -m pip install -r requirements.txt
python -m benchmark prepare --workspace work/health
```

`prepare` stages the maintained cohort/predictor preparation code (derived unchanged
from the published v1.4.15 implementation) to a new empty
workspace, obtains the 32 provider inputs, checks file signatures and records
hashes, builds the three cohorts, and fits 60 source predictors with fixed
outcome-blind source/target roles. It writes the prediction manifest at
`work/health/exp/source_role_outcome_blind_20260725/generation_provenance.json`.
Provider availability may change; inspect `work/health/data_receipts/` for acquisition
receipts. These record the downloaded bytes; they do not prove providers never
revise files. Retain these receipts with the evaluated configuration.

If acquisition is interrupted, repair/retry the staged steps directly, preserving
verified cache files:

```bash
python work/health/benchmark/prepare_data.py --lab-root work/health \
  --out work/health/data_receipts --download --probe-urls --rebuild-cohorts
python work/health/exp/source_role_outcome_blind_20260725/generate.py \
  --lab-root work/health --resume
```

An existing lab's raw provider cache can be reused without reusing its processed
cohorts or predictions:

```bash
python -m benchmark prepare --workspace work/rebuild --provider-cache /path/to/existing-lab
```

Cached inputs are linked, signature-checked and hashed again; processed cohorts
and all source models are regenerated in the new workspace. Keep the cache
available while the workspace is in use. Stronger OT/LR-QR case-study reproduction
additionally requires `examples/health-study/requirements.txt` (including PyTorch).

## Compare your method on real frozen predictions

```bash
python -m benchmark evaluate --workspace work/health \
  --methods source_split source_mondrian target_split target_mondrian examples/source_split.py \
  --out work/health/method-comparison.csv
python -m benchmark compare --input work/health/method-comparison.csv \
  --out work/health/comparison
```

The complete standard run produces 1,500 rows for five methods (60 model pairs ×
five draws × five methods). The external source-split example reproduces its
built-in counterpart; substitute your implementation for a scientific comparison.
A previously regenerated outcome-blind lab can be passed as `--workspace` without
rerunning preparation. Do not pass original-primary predictions as outcome-blind
inputs. Keep cohorts, roles and source predictors fixed across method comparisons.

## Extend other parts of the study

To introduce a new dataset, predictor or deployment condition, generate the same
prediction roles/schema and a hash manifest, publish your cohort/role-assignment
code and stable row-ID audits, and label the evaluation a new configuration.
The schema is documented in [METHODS.md](METHODS.md). A user-supplied manifest
alone cannot establish correct training or disjoint roles; audit the generator.
Covariate-dependent baselines, original-primary experiments, eight-budget studies
and independent external datasets have their own [reproduction commands](../examples/health-study/REPRODUCE.md).
