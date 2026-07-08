# Release Notes

## Packaging status

Created on 2026-07-08 from the local manuscript experiment workspace.

This release folder consolidates benchmark outputs that were previously spread across local experiment, manuscript-table, figure, and code directories.

- experiment result CSVs
- manuscript tables
- manuscript figures
- benchmark runner scripts

## What is complete

- Completed benchmark run-level outputs are included.
- Dataset and shift audit files are included.
- Summary metrics and statistical tests are included.
- Ablation, mechanism-diagnostic, subgroup, and L2-backbone sensitivity outputs are included.
- Manuscript figures and tables are included.
- The benchmark runner, public-data preparation helpers, and shared conformal utility script are included.
- Citation metadata and package-level README are included.

## Reproducibility scope

The release now includes the public-data preparation helpers needed by
`scripts/run_public_shift_benchmark.py`; it no longer depends on a separate
historical preprocessing script. Raw BRFSS, NHANES, and UCI source files are not
redistributed. Full reruns download or read those public files from the original
providers and may depend on provider availability, network access, and local
compute/storage.

## GitHub release checklist

- Replace placeholder repository URL in `CITATION.cff`. Done.
- Select and add a repository license. Done.
- Run release validation over included CSV, table, figure, and metadata files.
- Confirm that public source URLs for BRFSS, NHANES, and UCI are current.
- Confirm that manuscript text cites the final GitHub URL.
- Tag the release as the manuscript-linked paper release.
