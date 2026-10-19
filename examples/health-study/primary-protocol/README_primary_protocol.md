# Original stratified primary protocol

Read ../PROTOCOL_MAP.md first. Run the commands below from this directory.

The immutable `paper-release` commit is 2ef735d198037e5badd2cbe1fedf47a0f8f4dcfa of https://github.com/chainstart/public-health-shift-conformal-benchmark. `archived-release/` preserves its code, aggregate per-run metrics, tables, citation and license. No raw patient or survey records are included. This directory supplies the original-protocol source and aggregate reference evidence.

The primary protocol retains stratified allocation and calibration draws, the 100-label floor, higher quantile, pooled thresholds for classes with fewer than twenty labels, argmax completion and one draw per budget/seed. Original clipped weighting uses unlabeled target-test features; the additional outcome-blind experiment has different information access and roles.

Run in an environment with the recorded original requirements, under the OpenLabs CPU guard when working in this repository:

```sh
python audit_primary.py --release archived-release --out audit-regenerated
python render_primary_figures.py --output-dir regenerated-figures
python render_primary_graphical_abstract.py --results plot-inputs --paper-root reconstructed-paper
```

The first command independently aggregates archived observations and reconstructs retrospective stratified calibration class counts. It does not rerun historical source-model fitting or independently recover historical row-level predictions. Original fitting can be run in a separate working copy with `python archived-release/scripts/run_public_shift_benchmark.py`, after acquiring provider data. Input hashes, the archived protocol and recorded dependency versions identify the reference; a changed provider file or environment must not be represented as an identical historical execution.

The main audit checks 58 printed statistics/cells. 57 agree. The one confirmed rounding discrepancy is the 0.5% ordinary-target class gap: 0.0920481182628204 rounds to 0.0920 rather than 0.0921. The archived original table remains unchanged. The primary medians 0.9009/0.8620/0.9011/0.9012 and correlations 0.946/0.961 are reconstructed. Condition-bootstrap intervals and nominal p values describe related benchmark conditions, not independent deployment populations.

All 480 condition-seed-budget allocations yield 960 matching workflow label counts. The 120 archived class counts at 10% also agree. Counts include indicators for the primary pooled-small-class fallback. The actual primary label range is 100–48,000; the additional one-label-floor range is 26–48,000.

The original ablation figures 0.0609 and 0.0006 average twelve within-condition medians. Means over all sixty run-level shortfalls are separately supplied as additional estimands (0.0614 and 0.0031). Neither calculation changes the observations.

`primary-subgroups-with-counts-and-size.csv` preserves the original subgroup coverage, count, prevalence and singleton fraction. Its inherited global metrics, including size, AUC and class coverage, have explicit `parent_run_` names; the additional `subgroup_mean_set_size = 2 - singleton_rate` uses the primary method's nonempty binary sets. The original subgroup reporting threshold was fifty observations.

The method interface and its 300 paired example evaluations belong to the additional outcome-blind protocol in ../reproduction/benchmark/. Independent institutional code is in ../institutional/.



## Full original subgroup aggregates

The complete original subgroup CSV is available at a fixed public Git commit;
subgroup-source.json supplies its URL, exact byte count and SHA-256. It and the
derived full CSV are not bundled because they exceed the source upload limit.
The original remains independently downloadable and byte-verifiable. Rebuild:

```sh
python fetch_and_rebuild_subgroups.py
```

This downloads original aggregate measurements (not raw participant records),
then preserves the primary observations, labels inherited parent-run fields
and computes subgroup set size as two minus singleton fraction. Current S18
and its table source provide the manuscript summaries. All local original
and previously supplied complete evidence remains preserved.
