"""Audit every source patient independently of the preparation implementation."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import zipfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = args.artifacts
    heart = pd.read_parquet(a / "heart.parquet").set_index("row_id")
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
               "exang", "oldpeak", "slope", "ca", "thal", "num"]
    n_heart = 0
    with zipfile.ZipFile(a / "raw/heart-disease.zip") as archive:
        for center, suffix in [("cleveland", "cleveland"), ("hungarian", "hungarian"),
                               ("switzerland", "switzerland"), ("long_beach", "va")]:
            for i, fields in enumerate(csv.reader(io.StringIO(archive.read("processed." + suffix + ".data").decode()))):
                values = [float(v) if v.strip() != "?" else np.nan for v in fields]
                values = [np.nan if v == -9 else v for v in values]
                if not math.isfinite(values[-1]) or values[-1] not in [0, 1, 2, 3, 4]:
                    continue
                row = heart.loc["heart:" + center + ":" + str(i)]
                assert np.allclose(row[columns[:-1]].to_numpy(dtype=float), values[:-1], equal_nan=True)
                assert row.label == int(values[-1] > 0) and row.center == center
                n_heart += 1
    assert n_heart == len(heart)
    cohort = pd.read_parquet(a / "sepsis.parquet").set_index("row_id")
    manifest = pd.read_csv(a / "raw-source-manifest.csv")
    checked, exclusions = set(), {}
    max_error = 0.
    for i, entry in enumerate(manifest.to_dict("records"), 1):
        raw = gzip.decompress((a / "raw" / ("hospital_" + entry["hospital"]) / (entry["file"] + ".gz")).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"] and len(raw) == entry["bytes"]
        reader = csv.DictReader(io.StringIO(raw.decode("ascii")), delimiter="|")
        records = list(itertools.islice(reader, 24))
        if len(records) < 24:
            reason = "fewer_than_24_observed_hours"
        elif [float(r["ICULOS"]) for r in records] != list(range(1, 25)):
            reason = "nonsequential_ICULOS_1_to_24"
        elif not all(float(r["SepsisLabel"]) in [0., 1.] for r in records):
            reason = "invalid_observed_label"
        elif any(float(r["SepsisLabel"]) == 1 for r in records[:12]):
            reason = "positive_Challenge_label_by_hour_12"
        else:
            reason = "eligible"
        assert reason == entry["exclusion"], (entry["file"], reason, entry["exclusion"])
        exclusions[(entry["hospital"], reason)] = exclusions.get((entry["hospital"], reason), 0) + 1
        if reason != "eligible":
            continue
        key = "sepsis:" + entry["hospital"] + ":" + entry["file"]
        assert key not in checked
        checked.add(key)
        actual = cohort.loc[key]
        assert actual.label == int(any(float(r["SepsisLabel"]) == 1 for r in records[12:24]))
        assert actual.center == entry["hospital"]
        expected = {}
        for name in list(reader.fieldnames)[:34]:
            values = [float(r[name]) for r in records[:12]]
            observed = [v for v in values if not math.isnan(v)]
            expected[name + "_mean"] = sum(observed) / len(observed) if observed else np.nan
            expected[name + "_last"] = observed[-1] if observed else np.nan
            expected[name + "_missing_fraction"] = sum(math.isnan(v) for v in values) / 12
        for name in list(reader.fieldnames)[34:39]:
            observed = [float(r[name]) for r in records[:12] if not math.isnan(float(r[name]))]
            expected[name] = observed[-1] if observed else np.nan
        assert set(actual.index) == set(expected) | {"center", "label"}
        for name, value in expected.items():
            assert np.isclose(actual[name], value, atol=1e-10, rtol=1e-12, equal_nan=True), (key, name)
            if math.isfinite(value):
                max_error = max(max_error, abs(actual[name] - value))
        if i % 5000 == 0:
            print("audited source patients", i, flush=True)
    assert checked == set(cohort.index)
    result = {"status": "pass", "heart_source_records": n_heart,
              "sepsis_raw_patient_files": len(manifest), "sepsis_eligible_patients": len(checked),
              "all_raw_byte_hashes_checked": True, "every_eligible_feature_rebuilt_from_hours_1_to_12": True,
              "future_label_window_independently_rebuilt": [13, 24], "max_absolute_feature_discrepancy": max_error,
              "exclusions": [{"hospital": h, "reason": r, "n": n} for (h, r), n in exclusions.items()],
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
