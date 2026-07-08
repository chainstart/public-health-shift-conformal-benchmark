# Public-Health and Clinical Tabular Shift-Conformal Benchmark v3

This repository is the standalone benchmark support package associated with the manuscript:

**A public-health and clinical tabular benchmark for target-domain recalibration of conformal prediction under deployment shift**

The benchmark evaluates conformal prediction under deployment-style source-to-target shifts in public-health and clinical tabular data. The target-domain recalibration method is general and model-agnostic, but the empirical claims supported by this package are limited to the datasets, source-target partitions, model pipeline, and baselines included here.

## What "benchmark" means here

This is not a single static test dataset. It is a reusable evaluation suite containing:

- public data-source recipes and audit metadata;
- harmonized source-target split definitions;
- run-level conformal outputs for all evaluated seeds and target-calibration budgets;
- summary metrics, statistical tests, subgroup coverage outputs, and sensitivity analyses;
- manuscript-ready tables and figures;
- runner scripts and release notes for reproducing or auditing the analyses.

## Included tasks

- **BRFSS diabetes survey prediction**, 2011-2023, with temporal, geographic, demographic, and risk-profile shifts.
- **NHANES multicycle diabetes survey prediction**, 2011-2018, with temporal and demographic shifts.
- **UCI diabetes readmission prediction**, patient-first encounters, with demographic and clinical-setting shifts.

The benchmark contains 528,458 analytic records in the completed experiment, 12 non-pseudo deployment shifts, five random seeds per shift, and matched pseudo-random exchangeable controls.

## Directory layout

- `code/`: v3 benchmark runner and shared conformal utilities copied from the ARA experiment directory.
- `results/`: run-level and analysis-level CSV outputs from the completed v3 experiments.
- `summaries/`: compact evidence packet used during manuscript synthesis.
- `tables/`: manuscript tables in CSV and TeX form.
- `figures/`: manuscript-ready diagnostic figures from the completed benchmark analyses.
- `artifact_index.ara-local.json`: original ARA-local artifact index.
- `MANIFEST.md`: file inventory and release-validation summary.
- `CHECKSUMS.sha256`: SHA-256 checksums for release files.
- `RELEASE_NOTES.md`: packaging status and known limitations.
- `CITATION.cff`: citation metadata for GitHub.
- `requirements.txt`: Python package requirements used by the analysis scripts.

## Key files

- `results/public_benchmark_v3_conformal_results.csv`: run-level conformal metrics.
- `results/public_benchmark_v3_subgroup_coverage.csv`: subgroup and class-coverage outputs.
- `results/public_benchmark_v3_shift_audit.csv`: source-target shift definitions and counts.
- `results/public_benchmark_v3_dataset_audit.csv`: dataset-level audit metadata.
- `results/public_benchmark_v3_statistical_tests.csv`: paired statistical comparisons.
- `results/public_benchmark_v3_ablation_summary.csv`: shift-defining feature ablation summary.
- `results/public_benchmark_v3_mechanism_score_diagnostics.csv`: score-threshold diagnostic outputs.
- `results/public_benchmark_v3_l2_backbone_sensitivity.csv`: L2-backbone sensitivity outputs.

## Reproducibility status and scope

The reported manuscript analyses can be audited from the run-level outputs, split identifiers, audit CSV files, generated tables, and figures included here. The release is designed to support independent inspection of the benchmark evidence used in the manuscript.

Raw public datasets are not redistributed. The included code documents the analysis pipeline and model/calibration logic used for the v3 experiments, but one historical public-data preparation helper imported by `code/run_public_shift_benchmark_v3.py` was not found in the local project tree at release-packaging time. Therefore, this release should be cited as an auditable benchmark-results and paper-support package, not as a complete one-command raw-data regeneration archive.

Run the lightweight release check with:

```bash
python code/validate_release.py
```

## Citation

Use `CITATION.cff` to cite this repository. The manuscript should cite the public repository URL:

`https://github.com/chainstart/public-health-shift-conformal-benchmark-v3`

## License

Original code, metadata, and derived benchmark outputs in this repository are released under the MIT License. Raw BRFSS, NHANES, and UCI source datasets are not redistributed here and remain subject to their original terms of use.
