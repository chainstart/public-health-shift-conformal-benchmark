# Manifest

Release: `public-health-shift-conformal-benchmark`

Repository URL: `https://github.com/chainstart/public-health-shift-conformal-benchmark`

Release date: 2026-07-08

## Contents

- `scripts/`: benchmark runner, public-data preparation helpers, conformal-analysis utilities, and release validator.
- `results/`: run-level and analysis-level CSV outputs from the completed benchmark experiments.
- `summaries/`: compact evidence packet used during manuscript synthesis.
- `tables/`: manuscript tables in CSV and TeX form.
- `figures/`: benchmark figures generated from completed analyses.
- `scripts/create_manuscript_figures.py`: regenerates manuscript empirical result figures from archived CSV outputs.
- `DATA_SOURCES.md`: public data-source description and audit file pointers.
- `CITATION.cff`: citation metadata.
- `LICENSE` and `NOTICE.md`: repository licensing and source-data notice.
- `CHECKSUMS.sha256`: SHA-256 checksums generated for this release.

## Validation Summary

The packaged release contains the manuscript-support metadata, validation
script, checksum file, executable benchmark scripts, run-level results, tables,
and figures. It includes:

- 18 CSV result files under `results/`.
- 25 CSV table or summary files under `tables/`.
- 3 compact summary CSV files under `summaries/`.
- 5 PNG figures under `figures/`.

The included `scripts/validate_release.py` script checks that required files exist,
that key CSV files can be parsed, that expected row counts for headline outputs
match the packaged release, and that PNG files have valid PNG signatures.

## Scope

This is the manuscript-linked benchmark release. Raw public data are not
redistributed; the included runner downloads or reads public-source files and
recreates the benchmark outputs subject to provider availability and local
compute/storage constraints.
