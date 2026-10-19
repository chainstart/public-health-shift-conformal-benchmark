"""Acquire official open clinical cohorts and enforce a fixed landmark endpoint.

Run under the OpenLabs CPU resource guard. Raw records and patient-level data
belong in the local artifact directory, not Git or the public code package.
"""
from pathlib import Path
import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import re
import threading
import time
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
LOCAL = threading.local()
BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"
TRANSPORT_BASE = "https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get(url):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers["User-Agent"] = "OpenLabs-academic-reproducibility/1.0"
    for attempt in range(5):
        try:
            response = LOCAL.session.get(url, timeout=(15, 90))
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(min(20, 2 ** attempt))


def prepare_heart(artifact):
    url = json.loads((ROOT / "protocol.json").read_text())["datasets"]["heart_multicenter"]["url"]
    archive = artifact / "raw/heart-disease.zip"
    if not archive.exists():
        archive.write_bytes(get(url))
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
               "exang", "oldpeak", "slope", "ca", "thal", "num"]
    frames, files = [], []
    with zipfile.ZipFile(archive) as z:
        for center, suffix in [("cleveland", "cleveland"), ("hungarian", "hungarian"),
                               ("switzerland", "switzerland"), ("long_beach", "va")]:
            name = "processed." + suffix + ".data"
            raw = z.read(name)
            f = pd.read_csv(io.BytesIO(raw), header=None, names=columns, na_values=["?", -9])
            assert f.shape[1] == 14
            for column in columns:
                f[column] = pd.to_numeric(f[column], errors="raise")
            valid = f.num.notna() & f.num.isin([0, 1, 2, 3, 4])
            ids = ["heart:" + center + ":" + str(i) for i in range(len(f))]
            f["row_id"] = ids
            f = f.loc[valid].copy()
            f["label"] = (f.num > 0).astype(int)
            f = f.drop(columns="num")
            f["center"] = center
            frames.append(f)
            files.append({"file": name, "sha256": sha(raw), "input_rows": len(valid),
                          "eligible_rows": int(valid.sum()), "excluded_invalid_label": int((~valid).sum())})
    cohort = pd.concat(frames, ignore_index=True)
    assert len(cohort) == 920 and cohort.row_id.is_unique
    cohort.to_parquet(artifact / "heart.parquet", index=False)
    return {"url": url, "archive_sha256": sha(archive.read_bytes()), "files": files,
            "centers": cohort.groupby("center").label.agg(["size", "sum", "mean"]).reset_index().to_dict("records")}


def prepare_patient(item):
    artifact, hospital, name = item
    cache = artifact / "raw" / ("hospital_" + hospital) / (name + ".gz")
    raw = None
    if cache.exists():
        try:
            raw = gzip.decompress(cache.read_bytes())
        except (OSError, EOFError):
            cache.rename(cache.with_suffix(".interrupted.gz"))
    if raw is None:
        raw = get(TRANSPORT_BASE + "training_set" + hospital + "/" + name)
        # Cache complete original bytes, never only a favorable or eligible case.
        temporary = cache.with_suffix(".gz.tmp")
        temporary.write_bytes(gzip.compress(raw, mtime=0))
        temporary.replace(cache)
    receipt = {"hospital": hospital, "file": name, "sha256": sha(raw), "bytes": len(raw)}
    text = raw.decode("ascii")
    columns = text.splitlines()[0].strip().split("|")
    assert len(columns) == 41 and columns[-1] == "SepsisLabel" and columns[39] == "ICULOS"
    a = np.genfromtxt(io.StringIO(text), delimiter="|", skip_header=1, ndmin=2)
    assert a.shape[1] == 41
    if len(a) < 24:
        return None, dict(receipt, exclusion="fewer_than_24_observed_hours")
    first = a[:24]
    if not np.array_equal(first[:, 39], np.arange(1, 25)):
        return None, dict(receipt, exclusion="nonsequential_ICULOS_1_to_24")
    if not np.isin(first[:, 40], [0, 1]).all():
        return None, dict(receipt, exclusion="invalid_observed_label")
    if np.any(first[:12, 40] == 1):
        return None, dict(receipt, exclusion="positive_Challenge_label_by_hour_12")
    x = first[:12, :34]
    row = {"row_id": "sepsis:" + hospital + ":" + name, "center": hospital,
           "label": int(np.any(first[12:24, 40] == 1))}
    for j, name in enumerate(columns[:34]):
        observed = x[:, j][~np.isnan(x[:, j])]
        row[name + "_mean"] = float(observed.mean()) if len(observed) else np.nan
        row[name + "_last"] = float(observed[-1]) if len(observed) else np.nan
        row[name + "_missing_fraction"] = float(np.isnan(x[:, j]).mean())
    for j in range(34, 39):
        observed = first[:12, j][~np.isnan(first[:12, j])]
        row[columns[j]] = float(observed[-1]) if len(observed) else np.nan
    assert "SepsisLabel" not in row and "ICULOS" not in row
    return row, dict(receipt, exclusion="eligible")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    artifact = args.artifacts
    (artifact / "raw").mkdir(parents=True, exist_ok=True)
    receipt = {"protocol_sha256": sha((ROOT / "protocol.json").read_bytes()),
               "code_sha256": sha(Path(__file__).read_bytes()), "heart": prepare_heart(artifact)}
    tasks = []
    for hospital, expected in [("A", 20336), ("B", 20000)]:
        (artifact / "raw" / ("hospital_" + hospital)).mkdir(exist_ok=True)
        url = BASE + "training_set" + hospital + "/"
        index = get(url)
        (artifact / ("hospital_" + hospital + "_provider_index.html")).write_bytes(index)
        names = sorted(re.findall(r'href="([^"/]+\.psv)"', index.decode()))
        assert len(names) == expected, (hospital, len(names), expected)
        tasks.extend((artifact, hospital, name) for name in names)
    rows, manifest = [], []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (row, entry) in enumerate(pool.map(prepare_patient, tasks), 1):
            manifest.append(entry)
            if row is not None:
                rows.append(row)
            if i % 500 == 0:
                print(json.dumps({"downloaded": i, "total": len(tasks), "eligible": len(rows),
                                  "seconds": round(time.time() - start)}), flush=True)
    cohort = pd.DataFrame(rows)
    assert cohort.row_id.is_unique
    cohort.to_parquet(artifact / "sepsis.parquet", index=False)
    f = pd.DataFrame(manifest)
    f.to_csv(artifact / "raw-source-manifest.csv", index=False)
    receipt["sepsis"] = {"provider_url": BASE, "alternative_publisher_transport": TRANSPORT_BASE,
                         "raw_files": len(f), "eligible_patients": len(cohort),
                         "exclusions": f.groupby(["hospital", "exclusion"]).size().reset_index(name="n").to_dict("records"),
                         "hospitals": cohort.groupby("center").label.agg(["size", "sum", "mean"]).reset_index().to_dict("records"),
                         "manifest_sha256": sha((artifact / "raw-source-manifest.csv").read_bytes())}
    receipt["cohort_hashes"] = {name: sha((artifact / name).read_bytes()) for name in ["heart.parquet", "sepsis.parquet"]}
    (artifact / "preparation-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
