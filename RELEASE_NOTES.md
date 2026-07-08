# Release Notes

## Packaging status

Created on 2026-07-08 from the local ARA project:

`/home/biostar/work/projects/ara/projects/public-shift-conformal-benchmark-v3`

This release folder consolidates the benchmark outputs that were previously spread across:

- `exp/tasks/v3_benchmark/results/`
- `exp/v3_benchmark/`
- `writing/paper/tables/`
- `writing/paper/figures/`
- `exp/tasks/v3_benchmark/code/`

## What is complete

- Completed v3 run-level outputs are included.
- Dataset and shift audit files are included.
- Summary metrics and statistical tests are included.
- Ablation, mechanism-diagnostic, subgroup, and L2-backbone sensitivity outputs are included.
- Manuscript figures and tables are included.
- The v3 runner and shared conformal utility script are included.
- Citation metadata and package-level README are included.

## Known limitation

`code/run_public_shift_benchmark_v3.py` imports:

```python
from scripts.run_public_shift_benchmark_v2 import (
    evaluate_confidence_threshold,
    estimate_domain_auc,
    prepare_brfss_multiyear,
    prepare_nhanes_multicycle,
    prepare_uci_with_pseudo,
)
```

The file `run_public_shift_benchmark_v2.py` was not found under `/home/biostar` during packaging. This means the current release is sufficient for auditing the reported outputs, tables, figures, split definitions, and statistical summaries, but it is not a fully self-contained raw-data rerun package.

The GitHub release is therefore intentionally described as an auditable benchmark-results and paper-support package. A future release can restore or refactor the missing public-data preparation helper if one-command regeneration from raw public downloads is required.

## GitHub release checklist

- Replace placeholder repository URL in `CITATION.cff`. Done.
- Select and add a repository license. Done.
- Run release validation over included CSV, table, figure, and metadata files.
- Confirm that public source URLs for BRFSS, NHANES, and UCI are current.
- Confirm that manuscript text cites the final GitHub URL.
- Tag the release version, for example `v3.0.0`.
