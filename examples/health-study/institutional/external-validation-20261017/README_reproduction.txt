Independent cardiac and hospital sepsis validation

This archive extends the primary benchmark; it does not replace its three
datasets, twelve conditions, prior public archive or submitted Figures 1-3.
The main extension has two independent dataset collections, six related
institutional conditions, five seeds, two fixed predictor families and eight
target-label fractions. See protocol.json for the prospectively fixed local
analysis plan (not a claim of public trial registration).

Data and licenses
Heart Disease: UCI dataset 45, DOI 10.24432/C52P4X. Four processed files,
920 diagnostic records. Credit Janosi, Steinbrunn, Pfisterer and Detrano.
PhysioNet/CinC 2019: version 1.0.0, DOI 10.13026/v64v-d857. All 40,336
public hospital A/B patient files. Cite the provider dataset and Reyna et al.,
Critical Care Medicine 48 (2020) 210-217; DOI 10.1097/CCM.0000000000004145.
Follow the providers' CC BY 4.0 attribution and access terms. The official
public S3 mirror is an alternative transport for the same patient objects.
Raw records are acquired from the providers, not redistributed in this archive.
data-source-summary.json gives exact URLs, file/cohort hashes and exclusions.

Endpoint and leakage boundary
The cardiac outcome is contemporaneous diagnosis num>0, not future incidence.
The sepsis evaluation has one observation per patient: features use ICU hours
1-12; the target is any supplied positive Challenge label in hours 13-24.
Require an observed, sequential ICU clock 1-24 and observation through hour24.
These are retrospective selection rules. Challenge labels begin six hours
before the clinical onset; this does not reproduce the original hourly
Challenge utility metric or prove prospective clinical benefit.

Reproduction (Python 3.12; dependencies pinned in requirements-external.txt)
Create an isolated environment and install the pinned dependencies.
Use a local ARTIFACTS directory outside the source tree. All example shell
commands below assume this archive's top-level directory as working directory.
When working in OpenLabs, prepend openlabs/bin/openlabs-resource-guard
--task-type cpu -- to EVERY interpreter, package manager, archive or build job.

export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python external-validation-20261017/prepare_external.py --artifacts ARTIFACTS --workers 8
python external-validation-20261017/run_external.py --artifacts ARTIFACTS --out external-validation-20261017/results
python external-validation-20261017/audit_external_cohorts.py --artifacts ARTIFACTS --output external-validation-20261017/results/cohort-audit.json
python external-validation-20261017/verify_external.py --results external-validation-20261017/results
python external-validation-20261017/verify_role_counts.py --results external-validation-20261017/results

The original frozen predictors are never replaced after inspecting target
results. A separately declared POSTHOC source-model diagnostic follows
source_solver_diagnostic.json. It uses source-only L2 logistic regression
(C=1, lbfgs, balanced weights, max_iter=5000), unchanged roles and no target
tuning. This is exploratory, not a third prespecified predictor family.

python external-validation-20261017/run_source_solver_diagnostic.py --artifacts ARTIFACTS --out external-validation-20261017/source-solver-results
python external-validation-20261017/verify_external.py --results external-validation-20261017/source-solver-results
python external-validation-20261017/make_external_assets.py --artifacts ARTIFACTS --results external-validation-20261017/results --output manuscript-assets

Verification scope and interpretation
The raw-cohort verifier independently parses every source file and rebuilds
all eligible first-twelve-hour features, endpoint labels and inclusion rules.
The metric verifier imports neither the fitting runner nor its conformal API:
it reconstructs sampled IDs, uses sorted-score ranks and explicit set metrics,
and uses a separate root solver for label-availability probabilities.
Primary external measurements: 9,600 budget + 3,600 channel + 40 control rows.
Posthoc diagnostic measurements: 200 budget + 600 channel + 10 control rows.
These are repeated computational measurements, not independent deployments.

reference/ contains small condition/collection summaries, not patient arrays,
fitted models or selected favorable runs. Sepsis does not uniformly reproduce
source undercoverage; rare-class protection can require nearly two labels,
and interhospital discrimination can remain weak. Keep those findings.

The raw provider snapshot, package sources and numeric summaries are hash-bound
in the local paper evidence records. Different PDF creation dates can change
PDF byte hashes; plotted values and source tables are the comparison target.

Exact role allocation
Source calibration uses ceil(0.30*Nsource), training its complement. External
target pool uses ceil(0.30*Ntarget), test its complement; primary target test
instead uses ceil(0.70*Ntarget). Thus target rounding can differ by one record.
Random controls use ceil(0.25*N) test, then ceil(0.30*Ndevelopment) calibration.
reference/results/role_counts.csv reports every model/seed role count.


Computational reconstruction records retained from Supplement S14-S15
===================================================================
The independent parser audits every original file, raw byte hash, inclusion decision, endpoint and eligible patient's 107 features. Its maximum feature discrepancy is 1.14e-13 from floating-point summation. This verifies computational timing and reconstruction, not clinical endpoint adjudication.

Independent sorted-score reconstruction verifies all 9,600 fractional-budget rows, 3,600 observation-channel rows and 40 control rows with zero discrepancies. The separate posthoc source-model reconstruction verifies all 810 measurements with zero discrepancies. Cohort timing, feature construction, patient-disjoint roles, sampling design, predictor and sensitivity definitions, and scientific results remain in Supplement S14-S16.
