# Standard fixed-300 reference run

Generated with interface 2.0.0 on 2026-10-19 from newly reconstructed cohorts
and 60 newly fitted source models. See `verification.json` for actual checks.
The provider cache was reused, but processed cohorts and prediction arrays were
not. Raw inputs were signature-checked and hashed; provider metadata are included.

`run-level.csv` contains 1,500 rows: four baselines and one external source-split
example, each on 300 paired evaluations. The external example duplicates a
baseline to validate method integration; it is not an additional scientific method.
`run-level.receipt.json` identifies code, inputs, settings and exact output bytes.
The CSV summaries average draws within seed, seeds within condition, conditions
within dataset, then weight datasets equally. All baseline measurements match
our previously frozen-prediction run exactly; the external example matches the
built-in source split exactly in all 300 pairs.

This is a named benchmark configuration: 300 source and 300 target-pool samples,
five draws, alpha 0.1, strict finite rank and unmodified prediction sets. Its
means are not the primary paper's retrospective medians. It does not replace any
original paper number. Compare methods on the same predictions and settings.

Input manifest hashes identify a particular preparation run, including its
timing/provenance. They need not be identical across fresh runs. Prediction-file
hashes identify exact bytes; use explicit numerical tolerances when comparing
metric values across software/hardware environments. This tested environment
matched exactly; that observation is not a universal bitwise-reproducibility claim.
