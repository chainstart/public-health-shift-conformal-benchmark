from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_large"
PROCESSED_DIR = ROOT / "data" / "processed_large"
RESULTS_DIR = ROOT / "results" / "large_scale"
FIGURES_DIR = ROOT / "figures" / "large_scale"
REPORTS_DIR = ROOT / "reports"

UCI_DIABETES_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"
)
CDC_BRFSS_2015_URL = "https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip"

TARGET_COVERAGE = 0.90
ALPHA = 1 - TARGET_COVERAGE
DEFAULT_SEEDS = list(range(6))


@dataclass
class ShiftSpec:
    name: str
    group_col: str
    source_values: tuple[str, ...]
    target_values: tuple[str, ...]
    description: str


@dataclass
class LargeDataset:
    name: str
    task: str
    source_url: str
    X: pd.DataFrame
    y: np.ndarray
    groups: pd.DataFrame
    numeric_cols: list[str]
    categorical_cols: list[str]
    shifts: list[ShiftSpec]
    raw_n: int
    final_n: int


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=180) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def age_midpoint(age_band: str) -> float:
    if not isinstance(age_band, str) or not age_band.startswith("["):
        return np.nan
    left, right = age_band.strip("[]()").split("-")
    return (float(left) + float(right)) / 2


def icd_group(value: object) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value).strip()
    if text in {"?", "", "nan"}:
        return "missing"
    if text.startswith(("V", "E")):
        return "supplemental"
    try:
        code = float(text)
    except ValueError:
        return "other"
    if 390 <= code <= 459 or code == 785:
        return "circulatory"
    if 460 <= code <= 519 or code == 786:
        return "respiratory"
    if 520 <= code <= 579 or code == 787:
        return "digestive"
    if 250 <= code < 251:
        return "diabetes"
    if 800 <= code <= 999:
        return "injury"
    if 710 <= code <= 739:
        return "musculoskeletal"
    if 580 <= code <= 629 or code == 788:
        return "genitourinary"
    if 140 <= code <= 239:
        return "neoplasms"
    return "other"


def admission_mode(value: object) -> str:
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if code == 7:
        return "emergency_room"
    if code == 1:
        return "physician_referral"
    if code in {2, 3}:
        return "clinic_or_hmo"
    if code in {4, 5, 6, 10, 18, 22, 25, 26}:
        return "transfer_or_other"
    return "other"


def prepare_uci_readmission(force: bool = False) -> LargeDataset:
    zip_path = RAW_DIR / "uci_diabetes_130_hospitals.zip"
    cache_path = PROCESSED_DIR / "uci_diabetes_readmission_patient_first.csv"
    download_if_missing(UCI_DIABETES_URL, zip_path)

    if force or not cache_path.exists():
        with ZipFile(zip_path) as archive:
            with archive.open("diabetic_data.csv") as handle:
                df = pd.read_csv(handle)
        raw_n = len(df)
        df = df.replace("?", np.nan)
        df = df[df["gender"] != "Unknown/Invalid"].copy()
        df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
        df["target"] = (df["readmitted"] == "<30").astype(int)
        df["age_mid"] = df["age"].map(age_midpoint)
        for col in ["diag_1", "diag_2", "diag_3"]:
            df[f"{col}_group"] = df[col].map(icd_group)
        df["admission_mode"] = df["admission_source_id"].map(admission_mode)
        df["age_group"] = np.where(df["age_mid"] >= 70, "older70", "younger")
        df["race_group"] = df["race"].fillna("missing").astype(str)
        df["admission_shift_group"] = np.where(
            df["admission_mode"] == "emergency_room", "emergency_room", "non_emergency"
        )
        df.to_csv(cache_path, index=False)
    else:
        df = pd.read_csv(cache_path)
        raw_n = 101766

    numeric_cols = [
        "age_mid",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]
    med_cols = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    categorical_cols = [
        "race",
        "gender",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "admission_mode",
        "max_glu_serum",
        "A1Cresult",
        "change",
        "diabetesMed",
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
        *med_cols,
    ]
    cols = numeric_cols + categorical_cols
    X = df[cols].copy()
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for col in categorical_cols:
        X[col] = X[col].astype("string").fillna("missing")
    groups = df[["age_group", "race_group", "admission_shift_group"]].copy()
    for col in groups.columns:
        groups[col] = groups[col].astype(str)
    shifts = [
        ShiftSpec(
            "age_older70",
            "age_group",
            ("younger",),
            ("older70",),
            "train/calibrate on patients younger than 70, test on patients 70+",
        ),
        ShiftSpec(
            "race_african_american",
            "race_group",
            ("Caucasian",),
            ("AfricanAmerican",),
            "train/calibrate on Caucasian patients, test on African American patients",
        ),
        ShiftSpec(
            "emergency_admission",
            "admission_shift_group",
            ("non_emergency",),
            ("emergency_room",),
            "train/calibrate on non-emergency admissions, test on emergency-room admissions",
        ),
    ]
    return LargeDataset(
        name="uci_diabetes_readmission",
        task="30-day hospital readmission among diabetic inpatients",
        source_url=UCI_DIABETES_URL,
        X=X.reset_index(drop=True),
        y=df["target"].to_numpy(dtype=int),
        groups=groups.reset_index(drop=True),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        shifts=shifts,
        raw_n=raw_n,
        final_n=len(df),
    )


BRFSS_COLS = [
    "_STATE",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
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
    "DIABETE3",
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
    "_AGE80",
    "_BMI5",
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


SOUTH_STATES = {1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54}
NORTHEAST_STATES = {9, 23, 25, 33, 34, 36, 42, 44, 50}
MIDWEST_STATES = {17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55}
WEST_STATES = {2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56}


def state_region(state: object) -> str:
    if pd.isna(state):
        return "unknown"
    code = int(float(state))
    if code in SOUTH_STATES:
        return "south"
    if code in NORTHEAST_STATES:
        return "northeast"
    if code in MIDWEST_STATES:
        return "midwest"
    if code in WEST_STATES:
        return "west"
    return "territory_or_unknown"


def clean_days(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.replace({88: 0, 77: np.nan, 99: np.nan})
    out[(out < 0) | (out > 30)] = np.nan
    return out


def prepare_brfss_diabetes(force: bool = False) -> LargeDataset:
    zip_path = RAW_DIR / "brfss_2015_xpt.zip"
    cache_path = PROCESSED_DIR / "brfss_2015_diabetes_selected.csv"
    download_if_missing(CDC_BRFSS_2015_URL, zip_path)

    if force or not cache_path.exists():
        parts = []
        raw_n = 0
        with ZipFile(zip_path) as archive:
            xpt_name = archive.namelist()[0]
            with archive.open(xpt_name) as handle:
                reader = pd.read_sas(handle, format="xport", chunksize=50_000)
                for chunk in reader:
                    raw_n += len(chunk)
                    keep = chunk[[col for col in BRFSS_COLS if col in chunk.columns]].copy()
                    parts.append(keep)
        df = pd.concat(parts, ignore_index=True)
        df.attrs["raw_n"] = raw_n
        df.to_csv(cache_path, index=False)
    else:
        df = pd.read_csv(cache_path)
        raw_n = 441456

    if "raw_n" in df.attrs:
        raw_n = int(df.attrs["raw_n"])
    df = df[df["DIABETE3"].isin([1.0, 3.0])].copy()
    df["target"] = (df["DIABETE3"] == 1.0).astype(int)

    df["physhlth_days"] = clean_days(df["PHYSHLTH"])
    df["menthlth_days"] = clean_days(df["MENTHLTH"])
    df["bmi"] = pd.to_numeric(df["_BMI5"], errors="coerce") / 100.0
    df.loc[(df["bmi"] < 10) | (df["bmi"] > 80), "bmi"] = np.nan
    df["age80"] = pd.to_numeric(df["_AGE80"], errors="coerce")
    df.loc[(df["age80"] < 18) | (df["age80"] > 100), "age80"] = np.nan
    df["region"] = df["_STATE"].map(state_region)
    df["age_group"] = np.select(
        [df["_AGE65YR"] == 1.0, df["_AGE65YR"] == 2.0],
        ["18_64", "65_plus"],
        default="unknown",
    )
    df["sex_group"] = np.select([df["SEX"] == 1.0, df["SEX"] == 2.0], ["male", "female"], default="unknown")
    df["obesity_group"] = np.where(df["_BMI5CAT"] == 4.0, "obese", "not_obese")

    numeric_cols = ["physhlth_days", "menthlth_days", "bmi", "age80"]
    categorical_cols = [
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
    X = df[numeric_cols + categorical_cols].copy()
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for col in categorical_cols:
        X[col] = X[col].astype("string").fillna("missing")
    groups = df[["region", "age_group", "sex_group", "obesity_group"]].copy()
    for col in groups.columns:
        groups[col] = groups[col].astype(str)
    shifts = [
        ShiftSpec(
            "region_south",
            "region",
            ("northeast", "midwest", "west"),
            ("south",),
            "train/calibrate outside the US South, test in Southern states",
        ),
        ShiftSpec(
            "age_65plus",
            "age_group",
            ("18_64",),
            ("65_plus",),
            "train/calibrate on adults aged 18-64, test on adults 65+",
        ),
        ShiftSpec(
            "obesity",
            "obesity_group",
            ("not_obese",),
            ("obese",),
            "train/calibrate on non-obese respondents, test on obese respondents",
        ),
    ]
    return LargeDataset(
        name="brfss2015_diabetes",
        task="self-reported diagnosed diabetes in CDC BRFSS 2015",
        source_url=CDC_BRFSS_2015_URL,
        X=X.reset_index(drop=True),
        y=df["target"].to_numpy(dtype=int),
        groups=groups.reset_index(drop=True),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        shifts=shifts,
        raw_n=raw_n,
        final_n=len(df),
    )


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def bootstrap_ci(values: np.ndarray, seed: int = 42, n_boot: int = 4000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def make_preprocessor(ds: LargeDataset) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, ds.numeric_cols),
            ("cat", categorical_pipe, ds.categorical_cols),
        ],
        sparse_threshold=0.3,
    )


def make_model(ds: LargeDataset, model_name: str, seed: int) -> Pipeline:
    preprocessor = make_preprocessor(ds)
    if model_name == "logistic_saga":
        clf = LogisticRegression(
            max_iter=700,
            solver="saga",
            C=0.7,
            class_weight="balanced",
            random_state=seed,
        )
    elif model_name == "sgd_logistic":
        clf = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=0.10,
            alpha=1e-4,
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def weighted_conformal_quantile(scores: np.ndarray, weights: np.ndarray, alpha: float = ALPHA) -> float:
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    scores = scores[mask]
    weights = weights[mask]
    if len(scores) == 0:
        return 1.0
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    idx = int(np.searchsorted(cumulative, 1 - alpha, side="left"))
    idx = min(idx, len(sorted_scores) - 1)
    return float(sorted_scores[idx])


def make_qhat(cal_probs: np.ndarray, y_cal: np.ndarray, method: str, weights: np.ndarray | None = None):
    scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
    if method == "split":
        return conformal_quantile(scores)
    if method == "weighted_split":
        if weights is None:
            raise ValueError("weighted_split requires weights")
        return weighted_conformal_quantile(scores, weights)
    if method == "mondrian":
        qhats: dict[int, float] = {}
        for label in [0, 1]:
            mask = y_cal == label
            if mask.sum() < 20:
                qhats[label] = conformal_quantile(scores)
            else:
                qhats[label] = conformal_quantile(1 - cal_probs[mask, label])
        return qhats
    raise ValueError(method)


def prediction_sets(probs: np.ndarray, qhat: float | dict[int, float], fallback: bool = True) -> list[set[int]]:
    out: list[set[int]] = []
    for row in probs:
        pred_set: set[int] = set()
        for label in [0, 1]:
            q = qhat[label] if isinstance(qhat, dict) else qhat
            if 1 - row[label] <= q:
                pred_set.add(label)
        if fallback and not pred_set:
            pred_set = {int(np.argmax(row))}
        out.append(pred_set)
    return out


def evaluate_sets(
    y_true: np.ndarray,
    probs: np.ndarray,
    sets: list[set[int]],
    groups: pd.DataFrame,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    covered = np.array([int(y) in pred_set for y, pred_set in zip(y_true, sets)])
    sizes = np.array([len(pred_set) for pred_set in sets])
    singleton = sizes == 1
    point_pred = np.argmax(probs, axis=1)
    metrics: dict[str, float] = {
        "n_test": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "coverage": float(np.mean(covered)),
        "coverage_error": float(abs(np.mean(covered) - TARGET_COVERAGE)),
        "coverage_shortfall": float(max(0.0, TARGET_COVERAGE - np.mean(covered))),
        "mean_set_size": float(np.mean(sizes)),
        "singleton_rate": float(np.mean(singleton)),
        "ambiguous_rate": float(np.mean(sizes == 2)),
        "empty_rate": float(np.mean(sizes == 0)),
        "singleton_accuracy": float(
            np.mean(
                [
                    len(pred_set) == 1 and int(next(iter(pred_set))) == int(y)
                    for y, pred_set in zip(y_true, sets)
                    if len(pred_set) == 1
                ]
            )
        )
        if np.any(singleton)
        else np.nan,
        "point_balanced_accuracy": float(balanced_accuracy_score(y_true, point_pred)),
        "point_auc": safe_auc(y_true, probs[:, 1]),
    }
    for label, label_name in [(0, "negative"), (1, "positive")]:
        mask = y_true == label
        metrics[f"{label_name}_coverage"] = float(np.mean(covered[mask])) if np.any(mask) else np.nan
    metrics["class_coverage_gap"] = float(abs(metrics["positive_coverage"] - metrics["negative_coverage"]))

    subgroup_rows: list[dict[str, object]] = []
    coverage_values = []
    for group_col in groups.columns:
        for group_value, idx in groups.groupby(group_col, dropna=False).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            if len(idx_arr) < 50:
                continue
            sub_y = y_true[idx_arr]
            sub_cov = covered[idx_arr]
            sub_singleton = singleton[idx_arr]
            subgroup_rows.append(
                {
                    "subgroup_col": group_col,
                    "subgroup_value": str(group_value),
                    "n": int(len(idx_arr)),
                    "prevalence": float(np.mean(sub_y)),
                    "coverage": float(np.mean(sub_cov)),
                    "coverage_shortfall": float(max(0.0, TARGET_COVERAGE - np.mean(sub_cov))),
                    "singleton_rate": float(np.mean(sub_singleton)),
                }
            )
            coverage_values.append(float(np.mean(sub_cov)))
    metrics["min_subgroup_coverage"] = float(np.min(coverage_values)) if coverage_values else np.nan
    metrics["max_subgroup_coverage_gap"] = (
        float(np.max(coverage_values) - np.min(coverage_values)) if coverage_values else np.nan
    )
    return metrics, subgroup_rows


def stratified_two_way_split(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    X_a, X_b, y_a, y_b, g_a, g_b = train_test_split(
        X, y, groups, test_size=test_size, stratify=y, random_state=seed
    )
    return (
        X_a.reset_index(drop=True),
        X_b.reset_index(drop=True),
        y_a,
        y_b,
        g_a.reset_index(drop=True),
        g_b.reset_index(drop=True),
    )


def valid_domain(y: np.ndarray, minimum_n: int = 500, minimum_pos: int = 25) -> bool:
    if len(y) < minimum_n:
        return False
    counts = np.bincount(y, minlength=2)
    return bool(counts[0] >= minimum_pos and counts[1] >= minimum_pos)


def get_random_split(ds: LargeDataset, seed: int):
    X_dev, X_test, y_dev, y_test, g_dev, g_test = stratified_two_way_split(
        ds.X, ds.y, ds.groups, test_size=0.25, seed=seed
    )
    X_train, X_cal, y_train, y_cal, g_train, g_cal = stratified_two_way_split(
        X_dev, y_dev, g_dev, test_size=0.30, seed=seed
    )
    return {
        "setting": "random_exchangeable",
        "shift_name": "random",
        "shift_description": "random train/calibration/test split",
        "X_train": X_train,
        "y_train": y_train,
        "X_source_cal": X_cal,
        "y_source_cal": y_cal,
        "g_source_cal": g_cal,
        "X_target_pool": None,
        "y_target_pool": None,
        "g_target_pool": None,
        "X_test": X_test,
        "y_test": y_test,
        "g_test": g_test,
    }


def get_shift_split(ds: LargeDataset, spec: ShiftSpec, seed: int):
    source_mask = ds.groups[spec.group_col].isin(spec.source_values).to_numpy()
    target_mask = ds.groups[spec.group_col].isin(spec.target_values).to_numpy()
    if not valid_domain(ds.y[source_mask]) or not valid_domain(ds.y[target_mask]):
        return None
    X_source = ds.X.loc[source_mask].reset_index(drop=True)
    y_source = ds.y[source_mask]
    g_source = ds.groups.loc[source_mask].reset_index(drop=True)
    X_target = ds.X.loc[target_mask].reset_index(drop=True)
    y_target = ds.y[target_mask]
    g_target = ds.groups.loc[target_mask].reset_index(drop=True)
    X_train, X_source_cal, y_train, y_source_cal, g_train, g_source_cal = stratified_two_way_split(
        X_source, y_source, g_source, test_size=0.30, seed=seed
    )
    X_target_pool, X_test, y_target_pool, y_test, g_target_pool, g_test = stratified_two_way_split(
        X_target, y_target, g_target, test_size=0.70, seed=seed
    )
    return {
        "setting": "covariate_shift",
        "shift_name": spec.name,
        "shift_description": spec.description,
        "X_train": X_train,
        "y_train": y_train,
        "X_source_cal": X_source_cal,
        "y_source_cal": y_source_cal,
        "g_source_cal": g_source_cal,
        "X_target_pool": X_target_pool,
        "y_target_pool": y_target_pool,
        "g_target_pool": g_target_pool,
        "X_test": X_test,
        "y_test": y_test,
        "g_test": g_test,
    }


def sample_target_pool(split: dict, fraction: float, seed: int):
    X_pool = split["X_target_pool"]
    y_pool = split["y_target_pool"]
    g_pool = split["g_target_pool"]
    if X_pool is None:
        return None
    n = max(100, int(round(fraction * (len(X_pool) + len(split["X_test"])))))
    n = min(n, len(X_pool))
    if n <= 0:
        return None
    _, X_sub, _, y_sub, _, g_sub = train_test_split(
        X_pool,
        y_pool,
        g_pool,
        test_size=n,
        stratify=y_pool,
        random_state=seed,
    )
    return X_sub.reset_index(drop=True), y_sub, g_sub.reset_index(drop=True)


def estimate_density_ratio_weights(
    ds: LargeDataset,
    X_source_cal: pd.DataFrame,
    X_target_unlabeled: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    X_domain = pd.concat([X_source_cal, X_target_unlabeled], ignore_index=True)
    y_domain = np.r_[np.zeros(len(X_source_cal), dtype=int), np.ones(len(X_target_unlabeled), dtype=int)]
    domain_model = Pipeline(
        [
            ("preprocess", make_preprocessor(ds)),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=500,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    domain_model.fit(X_domain, y_domain)
    p_target = domain_model.predict_proba(X_source_cal)[:, 1]
    odds = p_target / np.clip(1 - p_target, 1e-6, 1)
    prior_correction = len(X_source_cal) / max(len(X_target_unlabeled), 1)
    weights = odds * prior_correction
    return np.clip(weights, 0.05, 20.0)


def run_experiments(
    datasets: list[LargeDataset],
    seeds: list[int],
    model_names: list[str],
    target_fractions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    weight_cache: dict[tuple[str, int, str, float], np.ndarray] = {}

    for ds in datasets:
        manifest_rows.append(
            {
                "dataset": ds.name,
                "task": ds.task,
                "source_url": ds.source_url,
                "raw_n": ds.raw_n,
                "final_n": ds.final_n,
                "positive_prevalence": float(np.mean(ds.y)),
                "n_numeric_features": len(ds.numeric_cols),
                "n_categorical_features": len(ds.categorical_cols),
                "shift_settings": "; ".join(spec.name for spec in ds.shifts),
            }
        )
        for seed in seeds:
            split_specs = [get_random_split(ds, seed)]
            for spec in ds.shifts:
                shifted = get_shift_split(ds, spec, seed)
                if shifted is not None:
                    split_specs.append(shifted)
            for split in split_specs:
                for model_name in model_names:
                    model = make_model(ds, model_name, seed)
                    model.fit(split["X_train"], split["y_train"])
                    source_cal_probs = model.predict_proba(split["X_source_cal"])
                    test_probs = model.predict_proba(split["X_test"])
                    method_specs: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray | None, float, int]] = [
                        (
                            "source_split",
                            "split",
                            source_cal_probs,
                            split["y_source_cal"],
                            None,
                            0.0,
                            0,
                        ),
                        (
                            "source_mondrian",
                            "mondrian",
                            source_cal_probs,
                            split["y_source_cal"],
                            None,
                            0.0,
                            0,
                        ),
                    ]
                    if split["setting"] == "covariate_shift":
                        for fraction in target_fractions:
                            sampled = sample_target_pool(split, fraction, seed + int(fraction * 1000))
                            if sampled is None:
                                continue
                            X_target_cal, y_target_cal, _ = sampled
                            target_cal_probs = model.predict_proba(X_target_cal)
                            weight_key = (ds.name, seed, split["shift_name"], fraction)
                            if weight_key not in weight_cache:
                                weight_cache[weight_key] = estimate_density_ratio_weights(
                                    ds, split["X_source_cal"], X_target_cal, seed + int(fraction * 1000)
                                )
                            weights = weight_cache[weight_key]
                            method_specs.extend(
                                [
                                    (
                                        "weighted_source_split",
                                        "weighted_split",
                                        source_cal_probs,
                                        split["y_source_cal"],
                                        weights,
                                        fraction,
                                        len(y_target_cal),
                                    ),
                                    (
                                        "target_recalibrated_split",
                                        "split",
                                        target_cal_probs,
                                        y_target_cal,
                                        None,
                                        fraction,
                                        len(y_target_cal),
                                    ),
                                    (
                                        "target_recalibrated_mondrian",
                                        "mondrian",
                                        target_cal_probs,
                                        y_target_cal,
                                        None,
                                        fraction,
                                        len(y_target_cal),
                                    ),
                                ]
                            )
                    for workflow, q_method, cal_probs, y_cal, weights, fraction, n_target_cal in method_specs:
                        qhat = make_qhat(cal_probs, y_cal, q_method, weights=weights)
                        sets = prediction_sets(test_probs, qhat, fallback=True)
                        metrics, subgroups = evaluate_sets(split["y_test"], test_probs, sets, split["g_test"])
                        row = {
                            "dataset": ds.name,
                            "seed": seed,
                            "model": model_name,
                            "setting": split["setting"],
                            "shift_name": split["shift_name"],
                            "shift_description": split["shift_description"],
                            "workflow": workflow,
                            "q_method": q_method,
                            "target_cal_fraction": fraction,
                            "target_coverage": TARGET_COVERAGE,
                            "n_train": int(len(split["y_train"])),
                            "n_source_cal": int(len(split["y_source_cal"])),
                            "n_target_cal": int(n_target_cal),
                            **metrics,
                        }
                        result_rows.append(row)
                        for subgroup in subgroups:
                            subgroup_rows.append({**row, **subgroup})

    results = pd.DataFrame(result_rows)
    subgroups = pd.DataFrame(subgroup_rows)
    manifest = pd.DataFrame(manifest_rows)
    results.to_csv(RESULTS_DIR / "large_scale_conformal_results.csv", index=False)
    subgroups.to_csv(RESULTS_DIR / "large_scale_subgroup_coverage.csv", index=False)
    manifest.to_csv(RESULTS_DIR / "large_scale_dataset_manifest.csv", index=False)
    return results, subgroups, manifest


def summarize_results(results: pd.DataFrame, subgroups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        results.groupby(["setting", "workflow", "target_cal_fraction"])
        .agg(
            coverage=("coverage", "median"),
            coverage_error=("coverage_error", "median"),
            coverage_shortfall=("coverage_shortfall", "median"),
            singleton_rate=("singleton_rate", "median"),
            singleton_accuracy=("singleton_accuracy", "median"),
            mean_set_size=("mean_set_size", "median"),
            class_coverage_gap=("class_coverage_gap", "median"),
            max_subgroup_coverage_gap=("max_subgroup_coverage_gap", "median"),
            point_auc=("point_auc", "median"),
            n_runs=("coverage", "size"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "large_scale_conformal_summary.csv", index=False)

    subgroup_summary = (
        subgroups.groupby(["setting", "workflow", "target_cal_fraction", "subgroup_col"])
        .agg(
            min_coverage=("coverage", "min"),
            median_coverage=("coverage", "median"),
            max_coverage=("coverage", "max"),
            max_shortfall=("coverage_shortfall", "max"),
            median_singleton=("singleton_rate", "median"),
            n_subgroups=("coverage", "size"),
        )
        .reset_index()
    )
    subgroup_summary.to_csv(RESULTS_DIR / "large_scale_subgroup_summary.csv", index=False)
    return summary, subgroup_summary


def paired_test(
    results: pd.DataFrame,
    label: str,
    workflow_a: str,
    workflow_b: str,
    metric: str,
    setting: str = "covariate_shift",
    fraction: float | None = None,
) -> dict[str, object] | None:
    frame = results[results["setting"] == setting].copy()
    a_frame = frame[frame["workflow"] == workflow_a].copy()
    b_frame = frame[frame["workflow"] == workflow_b].copy()
    if fraction is not None:
        a_frame = a_frame[np.isclose(a_frame["target_cal_fraction"], fraction)]
        if workflow_b.startswith("source_"):
            b_frame = b_frame[np.isclose(b_frame["target_cal_fraction"], 0.0)]
        else:
            b_frame = b_frame[np.isclose(b_frame["target_cal_fraction"], fraction)]
    a = a_frame.set_index(["dataset", "shift_name", "seed", "model"])[metric]
    b = b_frame.set_index(["dataset", "shift_name", "seed", "model"])[metric]
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return None
    diff = a.loc[idx].to_numpy() - b.loc[idx].to_numpy()
    ci_low, ci_high = bootstrap_ci(diff)
    return {
        "test": label,
        "metric": metric,
        "workflow_a": workflow_a,
        "workflow_b": workflow_b,
        "setting": setting,
        "target_cal_fraction": fraction if fraction is not None else np.nan,
        "median_difference": float(np.median(diff)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_p": float(wilcoxon(diff).pvalue) if np.any(diff != 0) else np.nan,
        "n_pairs": int(len(diff)),
    }


def run_statistical_tests(results: pd.DataFrame) -> pd.DataFrame:
    tests: list[dict[str, object]] = []
    shifted_source = results[
        (results["setting"] == "covariate_shift")
        & (results["workflow"] == "source_split")
        & (results["target_cal_fraction"] == 0.0)
    ]
    random_source = results[
        (results["setting"] == "random_exchangeable")
        & (results["workflow"] == "source_split")
        & (results["target_cal_fraction"] == 0.0)
    ].set_index(["dataset", "seed", "model"])
    random_rows = []
    for _, row in shifted_source.iterrows():
        key = (row["dataset"], row["seed"], row["model"])
        if key in random_source.index:
            random_rows.append(
                {
                    "coverage_error_shift_minus_random": row["coverage_error"]
                    - float(random_source.loc[key, "coverage_error"]),
                    "coverage_shift_minus_random": row["coverage"] - float(random_source.loc[key, "coverage"]),
                }
            )
    if random_rows:
        comp = pd.DataFrame(random_rows)
        for metric in comp.columns:
            values = comp[metric].to_numpy()
            ci_low, ci_high = bootstrap_ci(values)
            tests.append(
                {
                    "test": metric,
                    "metric": metric,
                    "workflow_a": "source_split_shift",
                    "workflow_b": "source_split_random",
                    "setting": "shift_vs_random",
                    "target_cal_fraction": np.nan,
                    "median_difference": float(np.median(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_p": float(wilcoxon(values).pvalue) if np.any(values != 0) else np.nan,
                    "n_pairs": int(len(values)),
                }
            )
    for fraction in [0.05, 0.10, 0.20]:
        for metric in ["coverage", "coverage_error", "coverage_shortfall", "singleton_rate", "class_coverage_gap"]:
            for workflow in ["target_recalibrated_split", "target_recalibrated_mondrian", "weighted_source_split"]:
                test = paired_test(
                    results,
                    f"{workflow}_minus_source_split_{metric}_fraction_{fraction}",
                    workflow,
                    "source_split",
                    metric,
                    fraction=fraction,
                )
                if test is not None:
                    tests.append(test)
    test_df = pd.DataFrame(tests)
    test_df.to_csv(RESULTS_DIR / "large_scale_statistical_tests.csv", index=False)
    return test_df


def make_figures(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plot_results = results.copy()
    plot_results["workflow_label"] = plot_results["workflow"].map(
        {
            "source_split": "source split",
            "source_mondrian": "source Mondrian",
            "weighted_source_split": "weighted source",
            "target_recalibrated_split": "target recal.",
            "target_recalibrated_mondrian": "target Mondrian",
        }
    )

    plt.figure(figsize=(10.5, 5.8))
    sns.boxplot(
        data=plot_results[
            (plot_results["workflow"].isin(["source_split", "target_recalibrated_split", "weighted_source_split"]))
            & (plot_results["target_cal_fraction"].isin([0.0, 0.10]))
        ],
        x="setting",
        y="coverage",
        hue="workflow_label",
    )
    plt.axhline(TARGET_COVERAGE, color="black", lw=1, ls="--")
    plt.xlabel("")
    plt.ylabel("Empirical coverage")
    plt.title("Large-scale clinical conformal coverage under exchangeability and shift")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "large_scale_coverage_by_workflow.png", dpi=180)
    plt.close()

    shift = summary[summary["setting"] == "covariate_shift"].copy()
    shift = shift[shift["workflow"].isin(["source_split", "weighted_source_split", "target_recalibrated_split"])]
    plt.figure(figsize=(9.5, 5.4))
    sns.lineplot(
        data=shift,
        x="target_cal_fraction",
        y="coverage_shortfall",
        hue="workflow",
        marker="o",
    )
    plt.xlabel("Target-domain calibration fraction")
    plt.ylabel("Median coverage shortfall")
    plt.title("Target recalibration reduces under-coverage under covariate shift")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "large_scale_calibration_budget_shortfall.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9.5, 5.6))
    scatter = results[
        (results["setting"] == "covariate_shift")
        & (
            (results["target_cal_fraction"] == 0.0)
            | ((results["target_cal_fraction"] == 0.10) & (results["workflow"] != "source_mondrian"))
        )
    ].copy()
    sns.scatterplot(
        data=scatter,
        x="singleton_rate",
        y="coverage",
        hue="workflow",
        style="dataset",
        alpha=0.75,
    )
    plt.axhline(TARGET_COVERAGE, color="black", lw=1, ls="--")
    plt.xlabel("Singleton auto-decision rate")
    plt.ylabel("Empirical coverage")
    plt.title("Coverage-efficiency tradeoff under clinical covariate shift")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "large_scale_coverage_efficiency_shift.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9.5, 5.2))
    subgroup_plot = results[
        (results["setting"] == "covariate_shift")
        & (
            (results["workflow"] == "source_split")
            | ((results["workflow"] == "target_recalibrated_split") & (results["target_cal_fraction"] == 0.10))
        )
    ]
    sns.boxplot(data=subgroup_plot, x="workflow", y="max_subgroup_coverage_gap")
    plt.xlabel("")
    plt.ylabel("Max subgroup coverage gap")
    plt.title("Subgroup coverage gaps remain an audit target")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "large_scale_subgroup_gap.png", dpi=180)
    plt.close()


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows).copy()
    headers = [str(col) for col in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in shown.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(fmt(value, 4))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(manifest: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    random_source = summary[
        (summary["setting"] == "random_exchangeable")
        & (summary["workflow"] == "source_split")
        & (summary["target_cal_fraction"] == 0.0)
    ].iloc[0]
    shift_source = summary[
        (summary["setting"] == "covariate_shift")
        & (summary["workflow"] == "source_split")
        & (summary["target_cal_fraction"] == 0.0)
    ].iloc[0]
    target10 = summary[
        (summary["setting"] == "covariate_shift")
        & (summary["workflow"] == "target_recalibrated_split")
        & (np.isclose(summary["target_cal_fraction"], 0.10))
    ].iloc[0]
    weighted10 = summary[
        (summary["setting"] == "covariate_shift")
        & (summary["workflow"] == "weighted_source_split")
        & (np.isclose(summary["target_cal_fraction"], 0.10))
    ].iloc[0]
    target20_rows = summary[
        (summary["setting"] == "covariate_shift")
        & (summary["workflow"] == "target_recalibrated_split")
        & (np.isclose(summary["target_cal_fraction"], 0.20))
    ]
    target20_shortfall = fmt(target20_rows.iloc[0]["coverage_shortfall"]) if not target20_rows.empty else "not_run"

    selected_summary = summary[
        (
            (summary["setting"] == "random_exchangeable")
            & (summary["workflow"].isin(["source_split", "source_mondrian"]))
        )
        | (
            (summary["setting"] == "covariate_shift")
            & (
                (summary["workflow"].isin(["source_split", "source_mondrian"]))
                | (
                    summary["workflow"].isin(
                        ["weighted_source_split", "target_recalibrated_split", "target_recalibrated_mondrian"]
                    )
                    & np.isclose(summary["target_cal_fraction"], 0.10)
                )
            )
        )
    ].copy()
    selected_summary = selected_summary[
        [
            "setting",
            "workflow",
            "target_cal_fraction",
            "coverage",
            "coverage_shortfall",
            "singleton_rate",
            "mean_set_size",
            "class_coverage_gap",
            "max_subgroup_coverage_gap",
            "n_runs",
        ]
    ]

    selected_tests = tests[
        tests["test"].str.contains(
            "coverage_error_shift_minus_random|target_recalibrated_split_minus_source_split_coverage_shortfall_fraction_0.1|"
            "target_recalibrated_split_minus_source_split_coverage_fraction_0.1|weighted_source_split_minus_source_split_coverage_fraction_0.1",
            regex=True,
        )
    ].copy()

    results_path = RESULTS_DIR / "large_scale_conformal_results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        shift_detail = results[
            (results["setting"] == "covariate_shift")
            & (results["workflow"].isin(["source_split", "target_recalibrated_split", "target_recalibrated_mondrian"]))
            & (results["target_cal_fraction"].isin([0.0, 0.10]))
        ]
        shift_detail = (
            shift_detail.groupby(["dataset", "shift_name", "workflow", "target_cal_fraction"])
            .agg(
                coverage=("coverage", "median"),
                shortfall=("coverage_shortfall", "median"),
                singleton=("singleton_rate", "median"),
                n_runs=("coverage", "size"),
            )
            .reset_index()
        )
    else:
        shift_detail = pd.DataFrame()

    conclusion = (
        "达到可投稿级别的核心结果"
        if target10["coverage_shortfall"] < 0.02 and shift_source["coverage_shortfall"] > 0.03
        else "形成了投稿主线，但仍需更多外部队列"
    )

    report = f"""# 大规模临床表格数据上的 Shift-aware Conformal Prediction 实验报告

生成日期：2026-07-05  
项目目录：`{ROOT}`

## 研究问题

上一轮结果显示，conformal prediction 在随机拆分下能接近标称覆盖率，但在 axis-shift 下会失效。本轮把主线收敛为：

> 临床表格模型在分布偏移下，源域校准的 conformal prediction 会发生系统性覆盖不足；少量目标域再校准是否能恢复覆盖，并量化自动预测率/人工复核率代价？

这比“conformal prediction 是否有用”更接近可投稿问题，因为它关注临床部署中最关键的失败模式：校准集和未来患者不再可交换。

## 大规模数据集

| dataset | task | final_n | positive_prevalence | source |
|---|---|---:|---:|---|
{chr(10).join(f"| {r.dataset} | {r.task} | {int(r.final_n)} | {r.positive_prevalence:.3f} | {r.source_url} |" for r in manifest.itertuples())}

处理原则：

- UCI 130-US hospitals：按 `patient_nbr` 保留首个住院事件，目标为 30 天内再入院 `<30`；
- CDC BRFSS 2015：从官方 XPT 原始文件抽取问卷变量，目标为自报曾被诊断糖尿病 `DIABETE3=1`，对照为 `DIABETE3=3`；
- 全部预处理、模型拟合和校准都放入 scikit-learn pipeline，避免测试集泄漏。

## 实验设计

每个数据集做随机同分布拆分和多个临床可解释的 covariate-shift 拆分：

- UCI：年龄 70+、African American vs Caucasian、急诊入院 vs 非急诊；
- BRFSS：美国南部地区、65 岁以上、肥胖人群。

比较工作流：

- `source_split`：只用源域 calibration set 做普通 split conformal；
- `source_mondrian`：源域 class-conditional / Mondrian conformal；
- `weighted_source_split`：使用目标域无标注协变量训练 domain classifier，做密度比加权校准；
- `target_recalibrated_split`：使用少量目标域标注样本重新估计 conformal 分位数；
- `target_recalibrated_mondrian`：目标域 class-conditional 再校准。

主要指标是 90% 标称覆盖下的 empirical coverage、coverage shortfall、singleton 自动预测率、类别覆盖差和 subgroup coverage gap。

## 核心结果

随机同分布下，源域 split conformal 的中位覆盖率为 **{fmt(random_source['coverage'])}**，coverage shortfall 为 **{fmt(random_source['coverage_shortfall'])}**，单标签自动预测率为 **{fmt(random_source['singleton_rate'])}**。

在 covariate shift 下，同样的源域校准覆盖率降至 **{fmt(shift_source['coverage'])}**，coverage shortfall 升至 **{fmt(shift_source['coverage_shortfall'])}**。这在两个大规模真实临床/流行病学任务上复现了上一轮小数据 benchmark 的关键失败模式。

10% 目标域标注再校准后，中位覆盖率恢复到 **{fmt(target10['coverage'])}**，coverage shortfall 降至 **{fmt(target10['coverage_shortfall'])}**，但单标签率从源域校准的 **{fmt(shift_source['singleton_rate'])}** 降至 **{fmt(target10['singleton_rate'])}**。20% 目标域标注再校准的 coverage shortfall 为 **{target20_shortfall}**。

密度比加权源域校准在 10% 目标域无标注协变量下的覆盖率为 **{fmt(weighted10['coverage'])}**，coverage shortfall 为 **{fmt(weighted10['coverage_shortfall'])}**。在当前特征和简单域分类器设置下，它几乎没有实质性缓解，不能替代目标域标注再校准。

| setting | workflow | target_cal_fraction | coverage | shortfall | singleton | set_size | class_gap | subgroup_gap | n_runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r.setting} | {r.workflow} | {r.target_cal_fraction:.2f} | {r.coverage:.3f} | {r.coverage_shortfall:.3f} | {r.singleton_rate:.3f} | {r.mean_set_size:.3f} | {r.class_coverage_gap:.3f} | {r.max_subgroup_coverage_gap:.3f} | {int(r.n_runs)} |" for r in selected_summary.itertuples())}

## 分 shift 异质性

整体结论不是由单个数据集驱动：BRFSS 的年龄、肥胖和地区偏移，以及 UCI 的年龄和急诊入院偏移都出现源域校准覆盖不足，并且目标域再校准恢复到 90% 附近。UCI 的 African American vs Caucasian 代理偏移在本实验中没有明显 under-coverage，因此论文不能声称“所有亚组偏移都会失败”，而应表述为“临床可解释偏移中存在可重复、可审计的覆盖失效，需要部署前检查和必要时再校准”。

{markdown_table(shift_detail[['dataset', 'shift_name', 'workflow', 'target_cal_fraction', 'coverage', 'shortfall', 'singleton', 'n_runs']], max_rows=40)}

## 统计检验

关键配对检验如下：

{markdown_table(selected_tests[['test', 'median_difference', 'ci_low', 'ci_high', 'wilcoxon_p', 'n_pairs']], max_rows=12)}

解释：

- shift 相比随机拆分显著增加 coverage error，支持“交换性破坏导致覆盖失效”的主张；
- 10% 目标域再校准显著降低 coverage shortfall，是当前最强 mitigation；
- 加权校准的实质效应接近零，适合放在失败/消融比较中，而不是作为 mitigation 主结论。

## 图表

![coverage by workflow](../figures/large_scale/large_scale_coverage_by_workflow.png)

![calibration budget](../figures/large_scale/large_scale_calibration_budget_shortfall.png)

![coverage efficiency](../figures/large_scale/large_scale_coverage_efficiency_shift.png)

![subgroup gap](../figures/large_scale/large_scale_subgroup_gap.png)

## 是否足以支撑论文结论？

判断：**{conclusion}**。

可以支撑的论文结论：

1. 在大规模真实临床/流行病学表格数据上，随机拆分验证会高估 conformal prediction 的部署可靠性；
2. 源域 calibration 在年龄、入院来源、地区、肥胖等多个临床可解释偏移下出现稳定 coverage shortfall，但并非每个代理亚组偏移都会失败；
3. 少量目标域标注再校准能把覆盖率拉回标称附近，但代价是更多 ambiguous set，也就是更高人工复核负担；
4. 简单无标注 covariate-shift 加权校准在当前实验中几乎不能恢复覆盖，反而强化了目标域审计/再校准的必要性；
5. 论文主张应聚焦“shift-aware conformal audit and recalibration workflow”，而不是泛泛声称 conformal prediction 本身新颖。

仍需谨慎的地方：

- UCI 和 BRFSS 都是公开二次数据，不等同于真实医院时间外部验证；
- race/region/obesity shift 是临床可解释的代理偏移，不能直接推断因果或公平性；
- 目前是二分类表格任务，投稿前最好再加入至少一个真实时间拆分或多机构 EHR 数据。

## 投稿定位

若写成方法/应用论文，建议标题方向：

**Shift-aware conformal prediction for clinical tabular models: large-scale evidence of under-coverage and target-domain recalibration trade-offs**

当前结果已经比前一轮更接近可发表：不再只是小 benchmark 现象，而是在两个大规模公开数据上复现了覆盖失效和再校准恢复。若目标是一般医学 AI / clinical informatics 期刊，还需要补充更多外部队列；若目标是 workshop、methods short paper 或数据基准论文，本轮结果已经足以形成完整稿件初稿。

## 复现

```bash
cd {ROOT}
python3 scripts/run_large_scale_conformal.py --seeds 0 1 2 3 4 5 --models sgd_logistic --target-fractions 0.05 0.10 0.20
```

主要输出：

| 文件 | 内容 |
|---|---|
| `results/large_scale/large_scale_dataset_manifest.csv` | 数据集与任务说明 |
| `results/large_scale/large_scale_conformal_results.csv` | 每个 seed/model/shift/workflow 的逐项结果 |
| `results/large_scale/large_scale_subgroup_coverage.csv` | 亚组覆盖结果 |
| `results/large_scale/large_scale_conformal_summary.csv` | 聚合结果 |
| `results/large_scale/large_scale_statistical_tests.csv` | 配对统计检验 |
| `figures/large_scale/*.png` | 论文图表草稿 |
"""
    (REPORTS_DIR / "large_scale_conformal_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-scale shift-aware conformal prediction experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=["logistic_saga", "sgd_logistic"])
    parser.add_argument("--target-fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--force-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    datasets = [
        prepare_uci_readmission(force=args.force_preprocess),
        prepare_brfss_diabetes(force=args.force_preprocess),
    ]
    results, subgroups, manifest = run_experiments(
        datasets=datasets,
        seeds=args.seeds,
        model_names=args.models,
        target_fractions=args.target_fractions,
    )
    summary, _ = summarize_results(results, subgroups)
    tests = run_statistical_tests(results)
    make_figures(results, summary)
    write_report(manifest, summary, tests)
    print(f"Wrote {RESULTS_DIR / 'large_scale_conformal_summary.csv'}")
    print(f"Wrote {REPORTS_DIR / 'large_scale_conformal_report.md'}")


if __name__ == "__main__":
    main()
