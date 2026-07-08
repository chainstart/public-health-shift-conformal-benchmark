from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "DATA_SOURCES.md",
    "CITATION.cff",
    "LICENSE",
    "NOTICE.md",
    "MANIFEST.md",
    "RELEASE_NOTES.md",
    "CHECKSUMS.sha256",
    "requirements.txt",
    "results/public_benchmark_v3_conformal_results.csv",
    "results/public_benchmark_v3_subgroup_coverage.csv",
    "results/public_benchmark_v3_shift_audit.csv",
    "results/public_benchmark_v3_dataset_audit.csv",
    "results/public_benchmark_v3_statistical_tests.csv",
    "tables/primary_shift_level_inference.csv",
    "tables/target_calibration_budget_table.csv",
    "summaries/headline_metrics.csv",
    "figures/quality_by_task.png",
    "figures/score_threshold_mismatch.png",
]

EXPECTED_ROWS = {
    "results/public_benchmark_v3_conformal_results.csv": 1575,
    "results/public_benchmark_v3_subgroup_coverage.csv": 26175,
    "results/public_benchmark_v3_shift_audit.csv": 15,
    "results/public_benchmark_v3_dataset_audit.csv": 3,
    "results/public_benchmark_v3_statistical_tests.csv": 41,
    "tables/primary_shift_level_inference.csv": 3,
    "tables/target_calibration_budget_table.csv": 16,
    "summaries/headline_metrics.csv": 19,
}


def validate_png(path: Path) -> None:
    with path.open("rb") as handle:
        signature = handle.read(8)
    if signature != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.relative_to(ROOT)} is not a valid PNG file")


def main() -> None:
    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    for rel, expected_rows in EXPECTED_ROWS.items():
        df = pd.read_csv(ROOT / rel)
        if len(df) != expected_rows:
            raise ValueError(f"{rel}: expected {expected_rows} rows, found {len(df)}")
        if not len(df.columns):
            raise ValueError(f"{rel}: no columns found")

    for figure in sorted((ROOT / "figures").glob("*.png")):
        validate_png(figure)

    print("Release validation passed.")


if __name__ == "__main__":
    main()
