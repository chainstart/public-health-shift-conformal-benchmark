# Public-Health and Clinical Tabular Shift-Conformal Benchmark

[中文说明](README.zh-CN.md)

A reusable evaluation suite for binary conformal prediction under public-health
and clinical deployment shift. Use it to compare your calibration method with
fixed baselines on shared source/target roles, calibration draws and metrics.
**The associated paper is one case study using this benchmark.** Its tables,
figures and experimental records are examples of benchmark outputs.

## Supplementary article

[Read the complete supplementary article (Sections S1–S21)](docs/supplementary/20260706-ai-health-conformal-deployment-shift-v1.4.19-supplementary.pdf). The same PDF is separately downloadable from [Zenodo supporting release v1.4.19](https://doi.org/10.5281/zenodo.23619630).

## Start here

| Goal | Entry point |
| --- | --- |
| Understand what is being compared | [Benchmark specification](docs/BENCHMARK.md) |
| Run the interface without health data | Quick start below |
| Obtain public cohorts and run real evaluations | [Real-data workflow](docs/REAL_DATA.md) |
| Evaluate your own method | [Method development guide](docs/METHODS.md) |
| Reproduce the paper and stronger comparators | [Paper case study](examples/health-study/REPRODUCE.md) |
| Distinguish historical and additional protocols | [Protocol map](examples/health-study/PROTOCOL_MAP.md) |
| Locate reference results | [Reference-result guide](docs/REFERENCE_RESULTS.md) |

## Quick start: your method against four baselines

Python 3.12 is the recorded environment. Run these commands from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-benchmark.txt
python -m benchmark fixture --workspace work/smoke
python -m benchmark evaluate --workspace work/smoke \
  --methods source_split source_mondrian target_split target_mondrian examples/source_split.py \
  --out work/smoke/results.csv
python -m benchmark compare --input work/smoke/results.csv --out work/smoke/comparison
```

This produces 50 method measurements (two synthetic model seeds, five shared
calibration draws, five methods), a provenance receipt, per-condition and
per-dataset summaries, and an equal-dataset summary. `examples/source_split.py`
is an external method file using the public API: its results must match the
built-in `source_split`. Copy and change this file to implement your method.
**Synthetic smoke data test software integration; they are not health datasets
or scientific evidence.** For real evaluations follow [REAL_DATA.md](docs/REAL_DATA.md).

## What is fixed, and what you change

The benchmark fixes cohort recipes, deployment conditions, train/calibration/test
roles, prediction inputs, matched calibration draws, and metric definitions.
You supply a binary probability-output calibration method. The runner withholds
test labels from the method, applies all methods to the same draws, and computes
coverage, set size, empty/singleton/two-label rates and class coverage.

The current public method API covers calibration of fixed binary predictors.
Full-covariate methods such as LR-QR have separate reference runners in the paper
case study. This is not a universal API for every predictor or adaptation method.
See [METHODS.md](docs/METHODS.md) for information access and fair comparisons.

## Included real-data tasks

The paper case study uses BRFSS diabetes surveys (2011–2023), NHANES diabetes
surveys (2011–2018), and UCI diabetes readmission, with 12 deployment conditions
and five model seeds per condition. Exact recipes, variables, exclusions and
source–target definitions are in [DATA_SOURCES.md](examples/health-study/DATA_SOURCES.md)
and its 32-file [provider inventory](examples/health-study/data-source-files.csv).
The original analytic cohorts contain 528,458 records. Random controls are
15 whole-dataset seed evaluations; they are not condition-size/prevalence-matched
controls. Independent cardiac and sepsis extensions are separate case studies.

## Repository layout

- `benchmark/`: maintained method API, cohort/predictor preparation, task
  configuration, evaluation runner and comparison commands.
- `examples/source_split.py`: minimal user-method implementation.
- `docs/`: benchmark specification, real-data workflow and method-development guide.
- `examples/health-study/`: v1.4.15 case-study experiment code, protocols and reference summaries.
- `tests/`: interface, label access, metric, pairing and end-to-end checks.
- `examples/legacy-experiments/`: previous experimental-code snapshot, including its
  scripts, results, figures and tables. The old fixed tag/commit URLs remain valid.

## Versions and citation

The maintained interface is version **2.0.0**. The immutable historical
[`paper-release`](https://github.com/chainstart/public-health-shift-conformal-benchmark/releases/tag/paper-release)
remains available. The latest published code-and-results support archive is **v1.4.16**,
[DOI 10.5281/zenodo.23476545](https://doi.org/10.5281/zenodo.23476545).
The interface and the paper-support archive have separate version numbers;
support v1.4.16 excludes article documents; scientific code and original results are retained.
See [verified scope](docs/VERIFICATION.md) and the [fixed-300 reference run](reference/fixed300/README.md).
Use [CITATION.cff](CITATION.cff), and report the exact commit/tag, protocol,
cohort inputs, model seeds, calibration budget and method information access.

## Tests and license

```bash
python -m unittest discover -s tests -v
```

Original code and derived outputs are MIT licensed. Raw health records are not
redistributed; provider acquisition and terms remain applicable. See [NOTICE.md](NOTICE.md).
