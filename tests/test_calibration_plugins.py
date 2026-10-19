import csv
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HISTORICAL = Path(__file__).resolve().parents[1] / 'examples/legacy-experiments'
sys.path.insert(0, str(HISTORICAL))

from scripts.calibration_plugins import (
    CalibrationRow, finite_quantile, fit_mondrian, fit_split,
    load_fitter, predict_sets, read_csv,
)


class CalibrationContractTests(unittest.TestCase):
    def test_order_statistic_and_rare_sample_infinity(self):
        self.assertEqual(finite_quantile([0.1, 0.2, 0.3, 0.4], 0.4), 0.3)
        self.assertTrue(math.isinf(finite_quantile([0.2] * 8, 0.1)))
        self.assertTrue(math.isinf(finite_quantile([], 0.1)))

    def test_ties_and_empty_sets_are_preserved(self):
        rows = tuple(CalibrationRow(str(i), 0.25, 0) for i in range(4))
        fitted = fit_split(rows, 0.4)
        self.assertEqual(fitted.predict_set(0.25), {0})
        self.assertEqual(fitted.predict_set(0.5), set())

    def test_missing_class_uses_infinite_threshold(self):
        rows = tuple(CalibrationRow(str(i), 0.2, 0) for i in range(9))
        fitted = fit_mondrian(rows, 0.1)
        self.assertTrue(math.isinf(fitted.q1))
        self.assertEqual(fitted.predict_set(0.1), {0, 1})
        self.assertEqual(fitted.predict_set(0.9), {1})

    def test_role_overlap_fails_before_fitter_runs(self):
        calls = []
        def spy(rows, alpha):
            calls.append(rows)
            return fit_split(rows, alpha)
        with self.assertRaises(ValueError):
            predict_sets(spy, (CalibrationRow("same", 0.2, 0),), (("same", 0.3),), 0.1)
        self.assertEqual(calls, [])

    def test_plugin_import_and_fit_receive_no_test_outcomes(self):
        rows = tuple(CalibrationRow(str(i), 0.25, 0) for i in range(4))
        self.assertIs(load_fitter("scripts.calibration_plugins:fit_split"), fit_split)
        seen = []
        def plugin(calibration, alpha):
            seen.append((calibration, alpha))
            return fit_split(calibration, alpha)
        out = predict_sets(plugin, rows, (("test", 0.25),), 0.4)
        self.assertEqual(seen, [(rows, 0.4)])
        self.assertEqual(out[0]["prediction_set"], [0])

    def test_invalid_probabilities_and_prediction_labels(self):
        for value in (math.nan, math.inf, -0.1, 1.1):
            with self.assertRaises(ValueError):
                CalibrationRow("cal", value, 0)
        with self.assertRaises(ValueError):
            CalibrationRow("cal", 0.5, True)
        class BadOutput:
            def predict_set(self, p1):
                return {2}
        with self.assertRaises(ValueError):
            predict_sets(lambda rows, alpha: BadOutput(), (), (("test", 0.3),), 0.1)

    def test_csv_contract_and_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cal, pred, out = (root / n for n in ("cal.csv", "pred.csv", "out.json"))
            cal.write_text("row_id,p1,label\nc0,0.1,0\nc1,0.2,0\nc2,0.3,0\nc3,0.4,0\n")
            pred.write_text("row_id,p1\nt0,0.2\nt1,0.5\n")
            subprocess.run([sys.executable, "-m", "scripts.calibration_plugins",
                            "--calibration", str(cal), "--predictions", str(pred),
                            "--alpha", "0.4", "--output", str(out)], check=True, cwd=HISTORICAL)
            self.assertEqual([r["prediction_set"] for r in json.loads(out.read_text())["predictions"]],
                             [[0], []])
            pred.write_text("row_id,p1,label\nt0,0.2,1\n")
            with self.assertRaises(ValueError):
                read_csv(pred, {"row_id", "p1"})


if __name__ == "__main__":
    unittest.main()
