# Manifest

Release: `public-health-shift-conformal-benchmark-v3`

Repository URL: `https://github.com/chainstart/public-health-shift-conformal-benchmark-v3`

Release date: 2026-07-08

## Contents

- `code/`: benchmark runner and conformal-analysis utilities used in the local v3 experiment.
- `results/`: run-level and analysis-level CSV outputs from the completed v3 experiments.
- `summaries/`: compact evidence packet used during manuscript synthesis.
- `tables/`: manuscript tables in CSV and TeX form.
- `figures/`: benchmark figures generated from completed analyses.
- `DATA_SOURCES.md`: public data-source description and audit file pointers.
- `CITATION.cff`: citation metadata.
- `LICENSE` and `NOTICE.md`: repository licensing and source-data notice.
- `CHECKSUMS.sha256`: SHA-256 checksums generated for this release.

## Validation Summary

The packaged release contains 80 files, including metadata, validation, and
checksum files. It includes:

- 18 CSV result files under `results/`.
- 25 CSV table or summary files under `tables/`.
- 3 compact summary CSV files under `summaries/`.
- 5 PNG figures under `figures/`.

The included `code/validate_release.py` script checks that required files exist,
that key CSV files can be parsed, that expected row counts for headline outputs
match the packaged release, and that PNG files have valid PNG signatures.

## Scope

This is an auditable benchmark-results and paper-support release. It is not a
complete one-command raw-data regeneration archive because one historical
public-data preparation helper referenced by the v3 runner was not present in
the local project tree at packaging time.
