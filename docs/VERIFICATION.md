# Verification performed for interface v2.0.0

Checked 2026-10-19 in a new Python 3.12 virtual environment. The full CPU
dependency inventory is `requirements-lock.txt`.

- 15 unit/integration checks passed, covering finite ranks/ties, test-label access,
  malformed outputs, metrics, end-to-end external-method use, paired budgets,
  input tampering, role-ID overlap and preparation staging/overwrite protection.
- All 32 public source URLs returned valid expected file signatures.
- Reusing the existing raw-provider cache, all inputs were checked and hashed,
  all three cohorts were rebuilt (528,458 rows), and all 60 source predictors
  across the 12 conditions were fitted again. No processed cohorts or old
  predictions were used in this reconstruction.
- All role-disjointness and outcome/feature-blind allocation checks passed.
- The new predictions produced 1,500 multi-method measurements. The external
  method matched the built-in source split in all 300 paired evaluations.
- The maintained runner agreed exactly with the published v1.4.15 method runner
  on the same 300 source-split evaluations. The full fresh 1,500-row result also
  agreed exactly with the earlier frozen-prediction evaluation in this environment.
- All 88 historical Git blobs were retained. The experiment package excludes
  supplementary article PDF/TeX and its article bibliography; experimental
  code, numerical results, figure assets and table fragments remain available. The historical release validator passed.

[Reference results, input receipts and verification record](../reference/fixed300/README.md)
are public. Probability arrays and raw health records remain local.

The rebuild used cached raw downloads; it was not a new transfer of every provider
file. URL probes check headers, not complete downloads. The separate OT/LR-QR
optimizer experiments and independent cardiac/sepsis extensions were not all rerun
for this interface release; their published implementations and results are retained
as named case studies. GitHub CI runs the lightweight interface checks, not the
full health-data reconstruction.
