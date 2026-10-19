# Public data acquisition and preparation

This recipe describes the public implementation at commit `2ef735d198037e5badd2cbe1fedf47a0f8f4dcfa`. It does not certify a newly reproduced cohort or change the archived experimental results. Raw data are obtained from the providers, not redistributed here.

## Exact provider files

All 32 files, local cache paths, NHANES selected columns, and merge keys are enumerated in `data_sources.json`. BRFSS uses seven selected annual files, not every year from 2011 through 2023.

| Dataset | Period | File | Official URL |
|---|---|---|---|
| BRFSS | 2011 | LLCP2011XPT.zip | https://www.cdc.gov/brfss/annual_data/2011/files/LLCP2011XPT.zip |
| BRFSS | 2013 | LLCP2013XPT.zip | https://www.cdc.gov/brfss/annual_data/2013/files/LLCP2013XPT.zip |
| BRFSS | 2015 | LLCP2015XPT.zip | https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip |
| BRFSS | 2017 | LLCP2017XPT.zip | https://www.cdc.gov/brfss/annual_data/2017/files/LLCP2017XPT.zip |
| BRFSS | 2019 | LLCP2019XPT.zip | https://www.cdc.gov/brfss/annual_data/2019/files/LLCP2019XPT.zip |
| BRFSS | 2021 | LLCP2021XPT.zip | https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip |
| BRFSS | 2023 | LLCP2023XPT.zip | https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip |
| NHANES | 2011-2012 | DEMO_G.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.xpt |
| NHANES | 2011-2012 | DIQ_G.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DIQ_G.xpt |
| NHANES | 2011-2012 | BMX_G.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BMX_G.xpt |
| NHANES | 2011-2012 | BPQ_G.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BPQ_G.xpt |
| NHANES | 2011-2012 | SMQ_G.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/SMQ_G.xpt |
| NHANES | 2011-2012 | MCQ_G.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/MCQ_G.xpt |
| NHANES | 2013-2014 | DEMO_H.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.xpt |
| NHANES | 2013-2014 | DIQ_H.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DIQ_H.xpt |
| NHANES | 2013-2014 | BMX_H.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BMX_H.xpt |
| NHANES | 2013-2014 | BPQ_H.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BPQ_H.xpt |
| NHANES | 2013-2014 | SMQ_H.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SMQ_H.xpt |
| NHANES | 2013-2014 | MCQ_H.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/MCQ_H.xpt |
| NHANES | 2015-2016 | DEMO_I.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt |
| NHANES | 2015-2016 | DIQ_I.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DIQ_I.xpt |
| NHANES | 2015-2016 | BMX_I.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.xpt |
| NHANES | 2015-2016 | BPQ_I.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPQ_I.xpt |
| NHANES | 2015-2016 | SMQ_I.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.xpt |
| NHANES | 2015-2016 | MCQ_I.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/MCQ_I.xpt |
| NHANES | 2017-2018 | DEMO_J.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt |
| NHANES | 2017-2018 | DIQ_J.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.xpt |
| NHANES | 2017-2018 | BMX_J.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.xpt |
| NHANES | 2017-2018 | BPQ_J.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPQ_J.xpt |
| NHANES | 2017-2018 | SMQ_J.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.xpt |
| NHANES | 2017-2018 | MCQ_J.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/MCQ_J.xpt |
| UCI | 1999–2008 | 296 diabetes 130-US hospitals | https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip |

## Preparation implemented by the existing loaders

BRFSS: `scripts/public_data_preparation.py::read_brfss_year` reads the XPT member in each annual ZIP, retains the configured questionnaire columns, uses DIABETE4 when available with nonmissing values and DIABETE3 otherwise, retains responses 1 and 3, and defines diabetes as response 1. BMI is `_BMI5 / 100` with values outside [10,80] set missing. The loader builds survey-year, region, age, sex, obesity, race, and deterministic pseudo-domain groups. Its default per-year cap is 60,000 with label-stratified sampling. Cached files must not be mixed across different caps or force-preprocessing choices.

NHANES: for each of 2011–2012 (G), 2013–2014 (H), 2015–2016 (I), and 2017–2018 (J), `read_nhanes_cycle` starts from DEMO and left-merges DIQ, BMX, BPQ, SMQ, and MCQ on SEQN. Cycles are concatenated; DIQ010 responses 1 and 2 are retained, with response 1 positive. See the JSON catalog for component-level variables.

**Cohort issue requiring resolution:** the current NHANES loader does not impose RIDAGEYR >= 18, although it names the below-65 group `18_64`. The group label alone does not establish adult eligibility or a count of minors. An adult-only analysis requires a predeclared filter and regeneration of the corresponding results. A retained all-age cohort instead requires accurate group labels and manuscript descriptions. This candidate does not silently change that filter or historical results.

UCI: `scripts/large_scale_conformal.py::prepare_uci_readmission` reads `diabetic_data.csv`, converts `?` to missing, excludes `Unknown/Invalid` gender, then sorts by encounter_id and keeps the first retained encounter per patient_nbr. The positive outcome is `readmitted == "<30"`. This is the first retained encounter after the stated exclusion, not an unqualified first encounter before exclusions.

## Commands

Install the original benchmark dependencies in a separate environment, then run from the repository root:

```sh
python -m scripts.run_public_shift_benchmark --seeds 0 --target-fractions 0.1 --max-brfss-per-year 60000
```

This example is a single-seed rerun, not reproduction of all archived tables. It downloads and prepares the public cohorts, trains the registered model, and writes results. The original default/full seed and fraction settings must be matched to the experiment being reproduced; JBI and later ESWA protocols are not interchangeable. `--force-preprocess` rebuilds caches but does not erase or redownload nonempty raw files. The original requirements pin NumPy, pandas, SciPy and scikit-learn but do not pin matplotlib/seaborn; a full clean-environment execution remains to be recorded.

For a bounded network check without complete cohort downloads:

```sh
python -m scripts.check_data_sources --output provider-probes.json
```

The checker reads at most 512 bytes per response and verifies ZIP/XPT signatures. It does not calculate whole-file checksums or validate cohort row counts. In the actual 16 October 2026 check, 24 NHANES files and the UCI archive passed; all seven BRFSS requests returned HTTP 403 in that client. These failures are preserved, not treated as successful downloads. If automated access is unavailable, obtain the named file through its official CDC annual-data page and put it at the exact cache path in the JSON. Record the actual downloaded bytes and hashes before claiming a complete rerun. Do not place HTML error pages or zero-byte files in the cache.

## Independent calibration extensions

See `METHOD_PLUGINS.md` for an additive prediction-only interface. Its synthetic tests do not validate the old complete experiment, and its introduction does not relabel historical results as plugin-generated.

## Label budgets and statistical roles

The public historical loader stratifies source/target roles and target-calibration sampling using labels. sample_target_pool uses max(100, round(fraction * (pool_size + test_size))), capped by pool size. Therefore the nominal percentage is not an exact realized budget for every dataset. Record absolute and class-specific counts for the selected protocol. The later manuscript's label-blind role construction is a different protocol, not a description of this historical runner.
