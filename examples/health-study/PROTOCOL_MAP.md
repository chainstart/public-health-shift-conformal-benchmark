# Protocol map

The article's primary analysis is the retrospective stratified protocol in
`primary-protocol/`. Its control, shifted-source, ordinary-target10 and
classwise-target10 coverage medians are 0.9009, 0.8620, 0.9011 and 0.9012.
It uses a 100-label floor, NumPy's higher quantile, pooled thresholds for
classes with fewer than twenty labels, argmax completion and one draw.

The additional outcome-blind experiment uses `reproduction/` and `results/`
at this archive's root. It fits separate fixed predictors and averages five
calibration draws within seed before taking sixty-pair medians. Exact ranks,
infinite thresholds on rank overflow, strict class thresholds and unmodified
empty sets apply. Its control and shifted-source medians are 0.9000/0.8616.
Historical labels such as primary predictions or primary roles in the
additional experiment's code refer to that experiment, not the article's
stratified primary analysis. Do not pool or relabel their measurements.

`institutional/` contains the independent cardiac and sepsis studies,
prespecified settings, provider acquisition, source-only posthoc sensitivity,
all condition/collection summaries and independent verifiers. The two
collections and six related conditions are not additional independent primary
benchmark datasets or pooled into either preceding median.

`supplement/figures/` and `supplement/tables/` contain experimental figure assets
and numerical table fragments. The supplementary article itself is not distributed.
Current implementations, acquisition instructions and results remain associated
with their named protocol.
For infinite thresholds the ordinary real-valued threshold SD is undefined:
the analytical n=8 example is N/A, while expected coverage remains one.
