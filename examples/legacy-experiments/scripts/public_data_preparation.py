from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.large_scale_conformal import (  # noqa: E402
    BRFSS_COLS,
    LargeDataset,
    PROCESSED_DIR,
    RAW_DIR,
    ShiftSpec,
    TARGET_COVERAGE,
    clean_days,
    download_if_missing,
    ensure_dirs as ensure_large_scale_dirs,
    make_preprocessor,
    prepare_uci_readmission,
    safe_auc,
    state_region,
)


RAW_PUBLIC_DIR = RAW_DIR / "public_benchmark"
PROCESSED_PUBLIC_DIR = PROCESSED_DIR / "public_benchmark"
BRFSS_YEARS = [2011, 2013, 2015, 2017, 2019, 2021, 2023]
NHANES_CYCLES = [(2011, "G"), (2013, "H"), (2015, "I"), (2017, "J")]
BRFSS_EXTRA_COLS = ["_RACE", "_PRACE1", "_MRACE1", "_RACE_G1", "_RACEGR3", "_RACEG21", "SEXVAR"]


def ensure_public_data_dirs() -> None:
    ensure_large_scale_dirs()
    for path in [RAW_PUBLIC_DIR, PROCESSED_PUBLIC_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def brfss_url(year: int) -> str:
    return f"https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"


def nhanes_url(cycle_start: int, file_stem: str, suffix: str) -> str:
    return f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle_start}/DataFiles/{file_stem}_{suffix}.xpt"


def stable_stratified_sample(df: pd.DataFrame, target_col: str, max_n: int | None, seed: int) -> pd.DataFrame:
    if max_n is None or len(df) <= max_n:
        return df.reset_index(drop=True)
    parts = []
    rng = np.random.default_rng(seed)
    counts = df[target_col].value_counts(normalize=True)
    for label, frac in counts.items():
        subset = df[df[target_col] == label]
        n = max(1, int(round(max_n * frac)))
        n = min(n, len(subset))
        parts.append(subset.sample(n=n, random_state=int(rng.integers(0, 1_000_000))))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_n:
        out = out.sample(n=max_n, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def brfss_race_group(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if code == 1:
        return "white"
    if code == 2:
        return "black"
    if code in {3, 4, 5, 6, 7, 8}:
        return "other"
    return "unknown"


def read_brfss_year(year: int, force: bool = False, max_rows: int | None = None) -> pd.DataFrame:
    ensure_public_data_dirs()
    zip_path = RAW_PUBLIC_DIR / f"brfss_{year}_xpt.zip"
    selected_path = PROCESSED_PUBLIC_DIR / f"brfss_{year}_selected.csv"
    processed_path = PROCESSED_PUBLIC_DIR / f"brfss_{year}_diabetes_processed_max{max_rows or 'all'}.csv"
    download_if_missing(brfss_url(year), zip_path)

    if force or not selected_path.exists():
        cols = sorted(set(BRFSS_COLS + ["DIABETE4"] + BRFSS_EXTRA_COLS))
        parts = []
        raw_n = 0
        with ZipFile(zip_path) as archive:
            xpt_name = archive.namelist()[0]
            with archive.open(xpt_name) as handle:
                reader = pd.read_sas(handle, format="xport", chunksize=50_000)
                for chunk in reader:
                    raw_n += len(chunk)
                    keep = [col for col in cols if col in chunk.columns]
                    parts.append(chunk[keep].copy())
        selected = pd.concat(parts, ignore_index=True)
        selected.attrs["raw_n"] = raw_n
        selected.to_csv(selected_path, index=False)

    if not force and processed_path.exists():
        return pd.read_csv(processed_path)

    df = pd.read_csv(selected_path)
    target_col = "DIABETE4" if "DIABETE4" in df.columns and df["DIABETE4"].notna().sum() > 0 else "DIABETE3"
    df = df[df[target_col].isin([1.0, 3.0])].copy()
    df["target"] = (df[target_col] == 1.0).astype(int)
    df["survey_year"] = str(year)
    df["region"] = df["_STATE"].map(state_region)
    df["age_group"] = np.select(
        [df["_AGE65YR"] == 1.0, df["_AGE65YR"] == 2.0],
        ["18_64", "65_plus"],
        default="unknown",
    )
    sex_col = "SEX" if "SEX" in df.columns else ("SEXVAR" if "SEXVAR" in df.columns else None)
    if sex_col:
        df["sex_group"] = np.select([df[sex_col] == 1.0, df[sex_col] == 2.0], ["male", "female"], default="unknown")
    else:
        df["sex_group"] = "unknown"
    df["obesity_group"] = np.where(df["_BMI5CAT"] == 4.0, "obese", "not_obese")
    race_col = next((col for col in BRFSS_EXTRA_COLS if col in df.columns), None)
    df["race_group"] = df[race_col].map(brfss_race_group) if race_col else "unknown"
    df["physhlth_days"] = clean_days(df["PHYSHLTH"])
    df["menthlth_days"] = clean_days(df["MENTHLTH"])
    df["bmi"] = pd.to_numeric(df["_BMI5"], errors="coerce") / 100.0
    df.loc[(df["bmi"] < 10) | (df["bmi"] > 80), "bmi"] = np.nan
    if "_AGE80" in df.columns:
        df["age80"] = pd.to_numeric(df["_AGE80"], errors="coerce")
        df.loc[(df["age80"] < 18) | (df["age80"] > 100), "age80"] = np.nan
    else:
        df["age80"] = np.nan
    df = stable_stratified_sample(df, "target", max_rows, seed=year)
    rng = np.random.default_rng(year + 999)
    df["pseudo_group"] = rng.choice(["pseudo_a", "pseudo_b"], size=len(df), replace=True)
    df.to_csv(processed_path, index=False)
    return df


def prepare_brfss_multiyear(force: bool = False, max_rows_per_year: int | None = 60_000) -> LargeDataset:
    ensure_public_data_dirs()
    cache_path = PROCESSED_PUBLIC_DIR / f"brfss_multiyear_max{max_rows_per_year or 'all'}.csv"
    if force or not cache_path.exists():
        dfs = [read_brfss_year(year, force=force, max_rows=max_rows_per_year) for year in BRFSS_YEARS]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(cache_path, index=False)
    else:
        df = pd.read_csv(cache_path)

    numeric_cols = ["physhlth_days", "menthlth_days", "bmi", "age80"]
    categorical_cols = [
        "survey_year",
        "_STATE",
        "region",
        "GENHLTH",
        "HLTHPLN1",
        "PERSDOC2",
        "MEDCOST",
        "CHECKUP1",
        "BPHIGH4",
        "BLOODCHO",
        "CHOLCHK",
        "TOLDHI2",
        "CVDINFR4",
        "CVDCRHD4",
        "CVDSTRK3",
        "ASTHMA3",
        "CHCSCNCR",
        "CHCOCNCR",
        "CHCCOPD1",
        "HAVARTH3",
        "ADDEPEV2",
        "CHCKIDNY",
        "SEX",
        "MARITAL",
        "EDUCA",
        "EMPLOY1",
        "INCOME2",
        "QLACTLM2",
        "BLIND",
        "DECIDE",
        "DIFFWALK",
        "SMOKE100",
        "SMOKDAY2",
        "ALCDAY5",
        "EXERANY2",
        "_RACE",
        "_AGEG5YR",
        "_AGE65YR",
        "_BMI5CAT",
        "_RFHYPE5",
        "_RFCHOL",
        "_MICHD",
        "_SMOKER3",
        "_RFSMOK3",
        "_TOTINDA",
        "_FRTLT1",
        "_VEGLT1",
        "_PA150R2",
    ]
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    X = df[numeric_cols + categorical_cols].copy()
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for col in categorical_cols:
        X[col] = X[col].astype("string").fillna("missing")
    groups = df[["survey_year", "region", "age_group", "sex_group", "obesity_group", "race_group", "pseudo_group"]].copy()
    for col in groups.columns:
        groups[col] = groups[col].astype(str)
    shifts = [
        ShiftSpec("time_future", "survey_year", ("2011", "2013", "2015"), ("2017", "2019", "2021", "2023"), "train/calibrate on 2011-2015 BRFSS, test on 2017-2023"),
        ShiftSpec("region_south", "region", ("northeast", "midwest", "west"), ("south",), "train/calibrate outside US South, test in Southern states"),
        ShiftSpec("age_65plus", "age_group", ("18_64",), ("65_plus",), "train/calibrate adults 18-64, test adults 65+"),
        ShiftSpec("obesity", "obesity_group", ("not_obese",), ("obese",), "train/calibrate non-obese respondents, test obese respondents"),
        ShiftSpec("race_black_vs_white", "race_group", ("white",), ("black",), "train/calibrate white respondents, test black respondents"),
        ShiftSpec("pseudo_random", "pseudo_group", ("pseudo_a",), ("pseudo_b",), "negative-control random pseudo-domain split"),
    ]
    return LargeDataset(
        name="brfss_multiyear_diabetes",
        task="self-reported diagnosed diabetes across BRFSS 2011-2023",
        source_url=";".join(brfss_url(y) for y in BRFSS_YEARS),
        X=X.reset_index(drop=True),
        y=df["target"].to_numpy(dtype=int),
        groups=groups.reset_index(drop=True),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        shifts=shifts,
        raw_n=int(len(df)),
        final_n=int(len(df)),
    )


NHANES_FILES = {
    "DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", "INDFMPIR"],
    "DIQ": ["SEQN", "DIQ010"],
    "BMX": ["SEQN", "BMXBMI", "BMXWT", "BMXHT", "BMXWAIST"],
    "BPQ": ["SEQN", "BPQ020", "BPQ080"],
    "SMQ": ["SEQN", "SMQ020"],
    "MCQ": ["SEQN", "MCQ160B", "MCQ160C", "MCQ160F", "MCQ160M"],
}


def nhanes_race_group(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if code == 3:
        return "white"
    if code == 4:
        return "black"
    if code in {1, 2, 6, 7}:
        return "other"
    return "unknown"


def read_nhanes_file(cycle_start: int, suffix: str, stem: str, force: bool = False) -> pd.DataFrame:
    ensure_public_data_dirs()
    raw_path = RAW_PUBLIC_DIR / f"nhanes_{cycle_start}_{stem}_{suffix}.xpt"
    out_path = PROCESSED_PUBLIC_DIR / f"nhanes_{cycle_start}_{stem}_{suffix}.csv"
    download_if_missing(nhanes_url(cycle_start, stem, suffix), raw_path)
    if force or not out_path.exists():
        df = pd.read_sas(raw_path, format="xport")
        keep = [col for col in NHANES_FILES[stem] if col in df.columns]
        df = df[keep].copy()
        df.to_csv(out_path, index=False)
    return pd.read_csv(out_path)


def read_nhanes_cycle(cycle_start: int, suffix: str, force: bool = False) -> pd.DataFrame:
    cache_path = PROCESSED_PUBLIC_DIR / f"nhanes_{cycle_start}_{suffix}_merged.csv"
    if force or not cache_path.exists():
        merged = None
        for stem in NHANES_FILES:
            part = read_nhanes_file(cycle_start, suffix, stem, force=force)
            merged = part if merged is None else merged.merge(part, on="SEQN", how="left")
        merged["cycle_start"] = str(cycle_start)
        merged["cycle"] = f"{cycle_start}_{cycle_start + 1}"
        merged.to_csv(cache_path, index=False)
    return pd.read_csv(cache_path)


def prepare_nhanes_multicycle(force: bool = False) -> LargeDataset:
    ensure_public_data_dirs()
    cache_path = PROCESSED_PUBLIC_DIR / "nhanes_multicycle_diabetes.csv"
    if force or not cache_path.exists():
        df = pd.concat([read_nhanes_cycle(start, suffix, force=force) for start, suffix in NHANES_CYCLES], ignore_index=True)
        df = df[df["DIQ010"].isin([1.0, 2.0])].copy()
        df["target"] = (df["DIQ010"] == 1.0).astype(int)
        df["age_group"] = np.where(pd.to_numeric(df["RIDAGEYR"], errors="coerce") >= 65, "65_plus", "18_64")
        df["sex_group"] = np.select([df["RIAGENDR"] == 1.0, df["RIAGENDR"] == 2.0], ["male", "female"], default="unknown")
        df["obesity_group"] = np.where(pd.to_numeric(df["BMXBMI"], errors="coerce") >= 30.0, "obese", "not_obese")
        df["race_group"] = df["RIDRETH3"].map(nhanes_race_group)
        rng = np.random.default_rng(20260705)
        df["pseudo_group"] = rng.choice(["pseudo_a", "pseudo_b"], size=len(df), replace=True)
        df.to_csv(cache_path, index=False)
    else:
        df = pd.read_csv(cache_path)

    numeric_cols = ["RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWT", "BMXHT", "BMXWAIST"]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    categorical_cols = [
        "cycle_start",
        "RIAGENDR",
        "RIDRETH3",
        "DMDEDUC2",
        "DMDMARTL",
        "BPQ020",
        "BPQ080",
        "SMQ020",
        "MCQ160B",
        "MCQ160C",
        "MCQ160F",
        "MCQ160M",
    ]
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    X = df[numeric_cols + categorical_cols].copy()
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for col in categorical_cols:
        X[col] = X[col].astype("string").fillna("missing")
    groups = df[["cycle_start", "age_group", "sex_group", "obesity_group", "race_group", "pseudo_group"]].copy()
    for col in groups.columns:
        groups[col] = groups[col].astype(str)
    shifts = [
        ShiftSpec("time_future", "cycle_start", ("2011", "2013"), ("2015", "2017"), "train/calibrate NHANES 2011-2014, test 2015-2018"),
        ShiftSpec("age_65plus", "age_group", ("18_64",), ("65_plus",), "train/calibrate adults 18-64, test adults 65+"),
        ShiftSpec("obesity", "obesity_group", ("not_obese",), ("obese",), "train/calibrate non-obese respondents, test obese respondents"),
        ShiftSpec("race_black_vs_white", "race_group", ("white",), ("black",), "train/calibrate white respondents, test black respondents"),
        ShiftSpec("pseudo_random", "pseudo_group", ("pseudo_a",), ("pseudo_b",), "negative-control random pseudo-domain split"),
    ]
    return LargeDataset(
        name="nhanes_multicycle_diabetes",
        task="self-reported diabetes across NHANES 2011-2018",
        source_url=";".join(nhanes_url(start, "DEMO", suffix) for start, suffix in NHANES_CYCLES),
        X=X.reset_index(drop=True),
        y=df["target"].to_numpy(dtype=int),
        groups=groups.reset_index(drop=True),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        shifts=shifts,
        raw_n=int(len(df)),
        final_n=int(len(df)),
    )


def prepare_uci_with_pseudo(force: bool = False) -> LargeDataset:
    ensure_public_data_dirs()
    ds = prepare_uci_readmission(force=force)
    groups = ds.groups.copy()
    rng = np.random.default_rng(130)
    groups["pseudo_group"] = rng.choice(["pseudo_a", "pseudo_b"], size=len(groups), replace=True)
    ds.groups = groups
    ds.shifts = [
        *ds.shifts,
        ShiftSpec("pseudo_random", "pseudo_group", ("pseudo_a",), ("pseudo_b",), "negative-control random pseudo-domain split"),
    ]
    return ds


def evaluate_confidence_threshold(cal_probs: np.ndarray, y_cal: np.ndarray, test_probs: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    cal_conf = cal_probs.max(axis=1)
    cal_pred = cal_probs.argmax(axis=1)
    thresholds = np.unique(np.r_[np.linspace(0.50, 0.99, 80), cal_conf])
    best_threshold = 1.01
    best_auto_rate = 0.0
    for threshold in thresholds:
        auto = cal_conf >= threshold
        if auto.sum() < 25:
            continue
        acc = float((cal_pred[auto] == y_cal[auto]).mean())
        if acc >= TARGET_COVERAGE and float(auto.mean()) > best_auto_rate:
            best_auto_rate = float(auto.mean())
            best_threshold = float(threshold)
    test_conf = test_probs.max(axis=1)
    test_pred = test_probs.argmax(axis=1)
    auto = test_conf >= best_threshold
    return {
        "threshold": best_threshold,
        "calibration_auto_rate": best_auto_rate,
        "auto_rate": float(auto.mean()),
        "auto_accuracy": float((test_pred[auto] == y_test[auto]).mean()) if auto.any() else np.nan,
        "manual_review_rate": float(1 - auto.mean()),
    }


def estimate_domain_auc(ds: LargeDataset, X_source: pd.DataFrame, X_target: pd.DataFrame, seed: int) -> float:
    max_n = 20_000
    if len(X_source) > max_n:
        X_source = X_source.sample(n=max_n, random_state=seed)
    if len(X_target) > max_n:
        X_target = X_target.sample(n=max_n, random_state=seed + 1)
    X_domain = pd.concat([X_source, X_target], ignore_index=True)
    y_domain = np.r_[np.zeros(len(X_source), dtype=int), np.ones(len(X_target), dtype=int)]
    if len(np.unique(y_domain)) < 2:
        return np.nan
    X_tr, X_te, y_tr, y_te = train_test_split(X_domain, y_domain, test_size=0.35, stratify=y_domain, random_state=seed)
    model = Pipeline(
        [
            ("preprocess", make_preprocessor(ds)),
            ("model", SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=500, tol=1e-3, random_state=seed)),
        ]
    )
    model.fit(X_tr, y_tr)
    return safe_auc(y_te, model.predict_proba(X_te)[:, 1])
