# Benchmark specification: outcome-blind-binary-v1

## Evaluation unit

A task is a named dataset and source-to-target condition. Each task has five
source-model seeds in the paper case study. Within each source domain, 70% of
rows train a fixed predictor and 30% form source calibration; within each target
domain, 30% form the target calibration pool and 70% form target test. Role
assignment is deterministic and outcome blind; stable row IDs and role audits
are produced by the cohort/prediction generators. Feature construction and
source-to-target definitions are fixed by the named cohort recipes.

The maintained API operates on the resulting frozen binary probabilities.
For each dataset/condition/model seed it samples 300 source calibration and
300 target-pool rows without replacement, repeating five draws. A SHA-256-derived
PCG64 seed (namespace `jbi-extension-v1`) makes the draws identical for all methods.
The standard real-data run has 60 prediction pairs and 300 draws per method.
Budgets, alpha and replicate count are explicit options and recorded in receipts.
Alternative settings are separate configurations, not interchangeable reference results.

The machine-readable default task list and evaluation settings are in
[`benchmark/tasks.json`](../benchmark/tasks.json). It lists all 12 conditions,
group definitions and excluded shift-defining predictors. Historical group
labels are retained for compatibility; the NHANES younger-than-65 source includes
children, and the obesity complement includes unavailable BMI records.

## Information access

Every method sees source calibration probabilities and labels, target-pool
probabilities, alpha and a draw seed. Only methods declaring
`uses_target_labels=True` see sampled target labels. Test probabilities enter
`predict_sets`; test outcomes remain inside the evaluation runner. The predictor
is fitted before calibration. Test outcomes must not choose methods, parameters
or optimizer budgets. The API is an access contract, not a security sandbox for
untrusted Python; users remain responsible for external files their code reads.

## Baseline rules

Built-ins are source split, source classwise (`source_mondrian`), target split
and target classwise (`target_mondrian`). Scores are `p` for label 0 and `1-p`
for label 1. The threshold is order statistic `ceil((n+1)*(1-alpha))`; rank
overflow or an absent class yields infinity. Membership includes ties. Empty
sets remain empty. No label is inserted by argmax. Default alpha is 0.1.

The historical paper's primary protocol differs: stratified role allocation,
100-label floor, NumPy higher quantile, small-class pooling and argmax completion.
Its results remain valid records of that named protocol; do not pool them with
this API's results. See [the protocol map](../examples/health-study/PROTOCOL_MAP.md).

## Metrics and aggregation

Coverage is the fraction containing the true label. Set size is membership count
in `{0,1}`. Empty, singleton and full-set rates partition all test cases. Class
coverage is calculated separately for observed outcomes 0 and 1; class gap is
the absolute difference. An absent test class yields NaN for that class/gap.
The comparison tool averages draws within model seed, seeds within condition,
conditions within dataset, then gives each dataset equal weight. It also writes
condition and dataset summaries so heterogeneous performance remains visible.
These are descriptive comparisons, not independent-deployment confidence intervals.
NaNs are excluded by pandas means: inspect per-condition files and report missing
classes; do not interpret a partially observed class metric as complete coverage.

## Reproducibility contract

The recorded CPU preparation environment is pinned in `requirements-lock.txt`.
Prediction parquet hashes are checked before evaluation. Receipts record protocol,
input manifest and prediction hashes, method-file hashes, budgets and output hash.
The comparison tool rejects duplicate measurements, unmatched condition/seed/draw
keys and inconsistent budgets/alpha. Use the same frozen prediction manifest for
all methods; the multi-method evaluation command guarantees shared draws.
Report information access alongside numerical performance. A method using target
labels has a different information budget from an unlabeled-target method.
