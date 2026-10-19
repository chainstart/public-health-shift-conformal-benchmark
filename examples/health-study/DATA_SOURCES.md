# Data acquisition and cohort construction

The accompanying `data-source-files.csv` enumerates 32 provider inputs: seven
BRFSS annual ZIPs, six NHANES components for each of four cycles, and one UCI
ZIP. It gives exact file identifiers, release years and actionable URLs.
`reproduction/benchmark/prepare_data.py` checks file signatures, records
SHA-256 hashes, optionally downloads missing files atomically, and calls the
included cohort-construction code. A successful HTTP response containing HTML
is rejected. Provider files are not redistributed in this package.

BRFSS annual documentation and codebooks:
https://www.cdc.gov/brfss/annual_data/annual_data.htm

NHANES public component documentation:
https://wwwn.cdc.gov/nchs/nhanes/Default.aspx

UCI dataset 296 (provider DOI `10.24432/C5230J`):
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999+2008

## BRFSS

Years are 2011, 2013, 2015, 2017, 2019, 2021 and 2023. SAS transport data
are read in 50,000-row chunks. `DIABETE4` is used when present with nonmissing
values, otherwise `DIABETE3`; only codes 1 (diagnosed) and 3 (no diabetes) are
retained. Each year is capped at 60,000 records by outcome-proportional quotas
and deterministic sampling within each outcome stratum, followed by a shuffle.
This is a computational cap, not survey weighting. Seeds and quota rounding
are implemented in `stable_stratified_sample`.

State FIPS codes are mapped by the explicit sets in `state_region`.
Territories/unmapped states are excluded from the South-versus-other-US-region
condition. Age groups use `_AGE65YR` codes 1 and 2; unknown categories are
excluded from the age condition. The BMI condition compares `_BMI5CAT==4`
against every other record, including unavailable BMI. It is therefore a
recorded BMI-category stress test, not a clinical obesity diagnosis. Numeric
BMI is `_BMI5/100`, with values outside [10,80] treated as unavailable.
Days coded 88 are zero; 77/99 and out-of-range values become unavailable.

The race proxy uses the first available column in the explicit priority list
`_RACE`, `_PRACE1`, `_MRACE1`, `_RACE_G1`, `_RACEGR3`, `_RACEG21`, `SEXVAR`;
codes 1/2 map to White/Black and other codes to other/unknown. The current
provider files contain race columns before `SEXVAR`; the last fallback is
retained in the frozen construction source and is not used for these files.
The model's frozen `SEX` feature and the derived sex-group fallback rules are
preserved exactly; this package does not claim semantic equivalence of every
questionnaire item across annual releases. Source code exposes the variable
selection and coding rules for inspection or alternative harmonization.

## NHANES

Cycles 2011–2012/G, 2013–2014/H, 2015–2016/I and 2017–2018/J use
`DEMO`, `DIQ`, `BMX`, `BPQ`, `SMQ` and `MCQ`. Selected component variables are
listed in `NHANES_FILES`. Demographics form the left table; components are
joined on `SEQN`. Only `DIQ010` codes 1/2 are retained, with 1 as the positive
outcome. No adult-age restriction is imposed in the primary frozen cohort.

The frozen group code `18_64` actually means age below 65 and includes
children. All published cohort definitions describe the actual population.
`BMXBMI>=30` is compared against every other record, including missing BMI;
in children this is a numeric threshold rather than an age-adjusted clinical
obesity definition. `RIDRETH3` codes 3/4 supply White/Black proxies; other
categories are outside this two-group condition. Sex groups use `RIAGENDR`.
The separate sensitivity restricts age to at least 18 and, only for its BMI
condition, requires observed BMI. It does not overwrite primary results.

## UCI diabetes readmission

The input is `diabetic_data.csv` in the provider ZIP. Question marks become
missing values; `Unknown/Invalid` gender rows are excluded. Records are sorted
by encounter identifier and only the first encounter for each patient is
retained. The positive outcome is `readmitted=="<30"`. Age midpoints define
the <70/≥70 condition. The other conditions compare recorded Caucasian versus
AfricanAmerican categories and `admission_source_id==7` (emergency room)
versus all other admission sources. These are retrospective patient-table
stress tests, not measured changes between hospitals. Features include the
complete index hospitalization, including discharge disposition, so this is
not an admission-time clinical prediction model.

## Features, roles and estimand

`results/data_preparation/outcome-model-features.json` enumerates every numeric
and categorical outcome-model feature. LR-QR's auxiliary covariate subset is
separately enumerated by the probability generator. These lists are different
and should not be conflated. Numeric features are median-imputed and
standardized. Categoricals use mode imputation and one-hot encoding with
`min_frequency=20` and unseen-level ignoring. All preprocessing and model
fitting use source-training observations only.

The role generator permutes row positions without inspecting labels, scores
or features: 70/30 source training/calibration and 30/70 target pool/test.
Stable row identifiers and saved role records check separation. Budget draws
are simple random samples without replacement, not outcome-stratified.
The estimand is behavior on capped, unweighted analytic rows. BRFSS/NHANES
survey weights, strata and primary sampling units are not used; reported
prevalences are not national population estimates. Repeated seeds, draws and
overlapping conditions do not supply independent deployments.

## Terms and reproducibility

Consult each provider's current data-use conditions. The package license
applies to supplied research code/documentation, not to the providers' data.
Provider hashes and retrieved file signatures identify the exact local inputs;
if a provider replaces a file, report the new hash rather than silently
claiming byte-identical reproduction. Full regeneration of cohorts from the
provider files uses `--rebuild-cohorts`. No credentials or institutional
datasets are required.
