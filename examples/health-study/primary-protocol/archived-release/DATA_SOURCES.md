# Public Data Sources

This benchmark was built from public tabular data sources. Raw public data files are not redistributed in this release package.

## BRFSS

Behavioral Risk Factor Surveillance System annual public-use survey files, 2011-2023.

The completed benchmark used annual CDC BRFSS XPT downloads and harmonized common questionnaire variables across years.

## NHANES

National Health and Nutrition Examination Survey public-use files, 2011-2018.

The completed benchmark used multicycle demographic and examination variables available across the included cycles.

## UCI Diabetes Readmission

UCI Machine Learning Repository diabetes 130-US hospitals readmission dataset:

`https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999+2008`

The completed benchmark sorted encounters by identifier and retained the first encounter per patient before defining the 30-day readmission outcome.

## Audit files

Use these files to inspect dataset sizes, features, source-target partitions, and empirical prevalence differences:

- `results/public_benchmark_dataset_audit.csv`
- `results/public_benchmark_shift_audit.csv`
- `tables/dataset_protocol.csv`
- `tables/shift_protocol.csv`

