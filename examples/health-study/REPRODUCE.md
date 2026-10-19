# Additional outcome-blind experiment

Read PROTOCOL_MAP.md for the separate original primary and institutional protocols. The commands below run from this archive root and reproduce the additional experiment.

# Reproduction

Use Python 3.12 and `requirements.txt`. The experiments run on CPU;
GPU memory is not required. A full source-cohort rebuild benefits from
32 GB RAM because the public annual survey tables are large. Create an
isolated environment, install the recorded versions, and run from this
package's root. BLAS and OpenMP thread counts can be bounded by setting
`OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2` and `MKL_NUM_THREADS=2`.

```
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Provider inputs and outcome-blind predictions

```
.venv/bin/python reproduction/benchmark/prepare_data.py --lab-root reproduction --out results/data_preparation --download --probe-urls --rebuild-cohorts
.venv/bin/python reproduction/exp/source_role_outcome_blind_20260725/generate.py --lab-root reproduction
```

The first command downloads 32 missing provider inputs, rejects incorrect
file types, records their hashes and prepares all three frozen cohorts. A
cache-assisted repeat can omit `--rebuild-cohorts` and reuse verified provider
inputs. The generator fits 60 source models and creates row-level probability
and covariate intermediates locally. It does not read target-test labels to
fit a predictor or select a calibration method.

## Budgets, efficiency, subgroups and implementation controls

```
.venv/bin/python reproduction/benchmark/check_contract.py
.venv/bin/python reproduction/restore_eight_budgets.py --lab-root reproduction --out results/budgets
.venv/bin/python reproduction/benchmark/verify_budget.py --lab-root reproduction --out results/budgets
.venv/bin/python reproduction/benchmark/run_controls.py --lab-root reproduction --out results/controls
.venv/bin/python reproduction/benchmark/run_adult_sensitivity.py --lab-root reproduction --out results/adult_sensitivity
.venv/bin/python reproduction/benchmark/run_backbone.py --lab-root reproduction --out results/nonlinear
```

The four-method budget experiment produces 9,600 method rows and 2,400
target calibration draws. The within-source control produces 600 rows.
The adult-only NHANES sensitivity fits 20 models and produces 400 rows.
The nonlinear sensitivity fits 60 models and produces 1,200 rows using
exactly the primary 10% target draws. The rank checker independently sorts
scores and reconstructs draws, thresholds, sets and efficiency metrics.

## Stronger unlabeled-target methods

```
.venv/bin/python reproduction/exp/shift_aware_baselines_20260813/run.py
.venv/bin/python reproduction/exp/shift_aware_baselines_20260813/verify.py
.venv/bin/python reproduction/exp/lrqr_baseline_20260824/run.py
.venv/bin/python reproduction/exp/lrqr_baseline_20260824/verify.py
```

The second runner depends on the first runner's regenerated matched draws
and metrics. Their row-level artifacts remain under `reproduction/exp/`.
Both numerical optimizers are explicitly configured for CPU. The first
runner compares eight methods using five draws per condition/seed; LR-QR
adds a ninth method with 300 additional unlabeled source observations.
The checkers reconstruct saved optimizer outputs, not independent optimizer
trajectories or global optima. Small floating-point differences across
hardware and math libraries should be reported; strict file hashes identify
inputs, whereas metric comparisons should use an explicit numerical tolerance.

## Add a probability-output method

```
.venv/bin/python reproduction/benchmark/run_method.py --lab-root reproduction --method reproduction/benchmark/example_method.py --out example.csv
.venv/bin/python reproduction/benchmark/run_method.py --lab-root reproduction --method source_split --out builtin.csv
```

After omitting `method`, the two outputs should agree in all 300 paired
evaluations. Define a `MethodSpec` and implement `fit`/`predict_sets` as
documented in `reproduction/benchmark/README.md`. Test outcomes are held by
the runner. This interface is limited to binary probability outputs; LR-QR's
covariate-dependent implementation is supplied separately.

## Presentation assets

After reconstructing or comparing the reference CSVs, run:

```
.venv/bin/python make_paper_assets.py --paper-root reconstructed-paper
```

This generates the evidence plots and most TeX tables from `results/` into
`reconstructed-paper/manuscript/`, plus EPS/TIFF exports. The complete editable
article source is supplied separately with the journal revision.

## Exploratory class coverage and singleton diagnostics

```
.venv/bin/python analyze_decision_tradeoffs.py --results results --out results/decision_tradeoffs
.venv/bin/python verify_decision_tradeoffs.py --results results
```

This extension summarizes every existing fixed-sample method and condition,
without fitting or choosing methods from test outcomes. Correct singleton
mass equals coverage minus the two-label mass. All three class floors
(85%, 88%, 90%) are retained as retrospective diagnostics, not clinical
tolerances. Condition minima and correct/incorrect singleton rates are
reconstructed independently using the Python standard library.

## Numerical-budget sensitivity and analytic controls

Main budgets are OT4000 and LR-QR1800CV/8000final. The fixed pre-run
protocols, lower-budget results, all80problem sensitivity settings,
objective/gradient summaries and independently solved controls are retained.
The lower-budget OT age result is an optimization-budget sensitivity.
The reference_low_budget files preserve the exact implementations imported
by the sensitivity/control drivers; they are distinct from the increased-budget
main reproduction runners. No target-test outcomes select a setting.

```
.venv/bin/python validate_optimizer_stability.py --lab-root reproduction --reference-code reproduction/reference_low_budget/exp/shift_aware_baselines_20260813/run.py --out local-optimizer/ot --method ot
.venv/bin/python validate_optimizer_stability.py --lab-root reproduction --reference-code reproduction/reference_low_budget/exp/lrqr_baseline_20260824/run.py --out local-optimizer/lrqr --method lrqr
.venv/bin/python validate_ot_long_budget.py --lab-root reproduction --reference-code reproduction/reference_low_budget/exp/shift_aware_baselines_20260813/run.py --out local-optimizer/ot-long
.venv/bin/python validate_optimizer_stability.py --lab-root reproduction --reference-code reproduction/reference_low_budget/exp/shift_aware_baselines_20260813/run.py --out local-optimizer/ot-full --method ot --full
.venv/bin/python verify_optimized_ot.py --lab-root reproduction --out local-optimizer/ot-full
```

The full OT driver generates exactly the mean-one float64 weights used by
the frozen high-budget reference summaries. The standard comparator runner
multiplies normalized float32 weights before casting, so weight rounding may
differ slightly; its threshold construction and objective are the same.
Full LR-QR uses the original full batching order because gradient clipping
can couple optimization trajectories within a batch. The80problem LR-QR
sensitivity is therefore reported separately from the300problem full run.

```
.venv/bin/python optimizer_analytic_controls.py --reference-code reproduction/reference_low_budget --out local-optimizer/controls
.venv/bin/python predictor_discrimination.py --lab-root reproduction --out local-optimizer/discrimination
```

The controls do not require health records. Discrimination requires the
local reconstructed prediction lab. Small CSV outputs correspond to
results/optimizer_controls and results/predictor_discrimination.

Scientific figures/tables are regenerated with make_paper_assets.py and
make_optimizer_assets.py using --paper-root reconstructed-paper. Numerical
selectors are in numeric_evidence.json. Local probability arrays, calibration
weights, raw records and model parameters are generated rather than uploaded.

Files under results/optimizer_full_lrqr retain the original full LR-QR
runner output and its seven/eight-method contextual inputs, including the
older1000-step OT context. The nine-method estimates combine independently verified OT and LR-QR
rows in results/matched/. Use that directory for the fixed-300 comparison
and singleton/class-coverage diagnostics. The raw full-run outputs remain unchanged to
preserve their independently verified provenance.

## Computational measurement and reconstruction inventory

Primary budgets: 2,400 calibration draws and 9,600 method measurements. Fixed
comparison:2700method rows, including300LR-QR fits; adult sensitivity400
method rows; nonlinear sensitivity1200. Full OT NumPy reconstruction checks
all300weight vectors, draws, thresholds, memberships, metrics and fitted
objectives (maximum objective discrepancy1.55e-7). Full LR-QR verification
checks300feature maps, final objectives/gradients, CV selections, thresholds
and memberships. Those reconstructions evaluate saved fits; the separate
budget/init reruns and independently solved controls supply optimization
evidence. Objective agreement is not zero optimization gap or clinical validity.

## Random controls and retained sensitivity questions

```
.venv/bin/python reproduction/restore_random_controls.py --lab-root reproduction --out results/restored_random_controls
.venv/bin/python reproduction/restore_submitted_sensitivities.py --lab-root reproduction --out results/restored_submitted_sensitivities
.venv/bin/python reproduction/restore_clipped_weighting.py --lab-root reproduction --out results/restored_clipped_weighting
.venv/bin/python reproduction/rebuild_original_diagnostics.py --lab-root reproduction --out results/restored_threshold_diagnostic
```

These commands fit 15 random-control models, 120 feature-ablation/L2 models,
and 60 clipped-domain-weighting models, respectively; the diagnostic reuses
the 60 primary prediction files. The sensitivity run checks role-ID and label
identity against primary predictions and independently sorts calibration
scores to reconstruct all 1,200 prediction-set measurements. The clipped
weighting estimator receives one unlabeled 10% target-pool draw per seed and
uses its stated empirical weighted-CDF heuristic without test-point infinity
mass. Neither target-test features nor target outcomes fit those weights.
No distribution-free weighted-validity guarantee is assigned to that heuristic.

The random controls are 15 dataset–seed evaluations, not 60 condition-size-
matched controls. Their test fraction is 25%, followed by 30% calibration
within the remaining development rows; both allocations are outcome blind.

To generate the retained result figures, color schematic and supplemental
tables after the common `make_paper_assets.py` command, run:

```
.venv/bin/python reproduction/make_original_revision_assets.py --results results --paper-root reconstructed-paper
.venv/bin/python reproduction/rebuild_color_graphical_abstract.py --results results --paper-root reconstructed-paper
.venv/bin/python reproduction/make_restored_supplement_tables.py --results results --paper-root reconstructed-paper
```

Empirical bar labels and scatter points come from the reference CSVs. People,
dataset icons and score-distribution curves in the graphical abstract are
schematic. The journal source supplies the original color workflow artwork
and the complete article text separately. Regenerated figure bytes may vary
with plotting-library metadata; numeric values and displayed labels, rather
than PDF byte identity, establish computational agreement.

## Controlled label observation

```
.venv/bin/python reproduction/run_label_acquisition.py --lab-root reproduction --out results/label_acquisition
.venv/bin/python reproduction/verify_label_acquisition.py --lab-root reproduction --out results/label_acquisition
.venv/bin/python reproduction/make_label_acquisition_assets.py --results results --paper-root reconstructed-paper
```

The generator fixes the protocol before computation and reuses all 60 frozen
predictors and their original tests. It produces 9,000 metric rows and 3,600
draw records. The verifier independently reconstructs eligibility probabilities,
row identities, finite-sample ranks, prediction memberships and metrics.
Outcome-dependent observation is a retrospective controlled stress test; no
unobserved outcome is assumed available to a deployment procedure. The known-
design rule is prespecified from observation metadata and never uses test
outcomes for selection. Representative sentinel and available-label designs
are alternative collection protocols; their budgets must not be added or
interpreted as equal monetary costs. The plotted 12 condition means are
descriptive and reuse the primary test cohorts.
