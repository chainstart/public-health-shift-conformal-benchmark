"""Independent stdlib reconstruction of every class-floor/yield summary."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    out = args.results / 'decision_tradeoffs'
    source = args.results / 'matched/extended_replicate_metrics.csv'
    expected = defaultdict(list)
    computed = {}
    for r in read(source):
        k = tuple(r[x] for x in ['dataset', 'shift', 'seed', 'replicate', 'method'])
        assert k not in computed
        cover, full, single = [float(r[x]) for x in ['coverage', 'full_set_rate', 'singleton_rate']]
        # A binary full set always covers; an empty set never covers.
        correct = cover - full
        error = 1 - float(r['empty_set_rate']) - full - correct
        assert math.isclose(error, single - correct, abs_tol=1e-12)
        class_min = min(float(r['class_0_coverage']), float(r['class_1_coverage']))
        computed[k] = {'correct_singleton_rate': correct, 'singleton_error_rate': error,
                       'minimum_class_coverage': class_min}
        expected[(r['dataset'], r['shift'], r['method'])].append(r)
    for r in read(out / 'replicate_diagnostics.csv'):
        k = tuple(r[x] for x in ['dataset', 'shift', 'seed', 'replicate', 'method'])
        for metric, value in computed[k].items():
            assert math.isclose(float(r[metric]), value, abs_tol=1e-12)
    condition_values = {}
    for k, group in expected.items():
        assert len(group) == 25
        c0 = mean(float(r['class_0_coverage']) for r in group)
        c1 = mean(float(r['class_1_coverage']) for r in group)
        condition_values[k] = {'minimum_of_class_means': min(c0, c1),
            'correct_singleton_rate': mean(float(r['coverage']) - float(r['full_set_rate']) for r in group),
            'singleton_error_rate': mean(float(r['singleton_rate']) - float(r['coverage']) + float(r['full_set_rate']) for r in group)}
    for r in read(out / 'condition_diagnostics.csv'):
        for metric, value in condition_values[(r['dataset'], r['shift'], r['method'])].items():
            assert math.isclose(float(r[metric]), value, abs_tol=1e-12)
    for r in read(out / 'floor_sensitivity.csv'):
        values = [v['minimum_of_class_means'] for (d, s, m), v in condition_values.items() if m == r['method']]
        assert len(values) == 12
        assert int(r['conditions_passing']) == sum(v >= float(r['floor']) for v in values)
        assert math.isclose(float(r['worst_condition_class_mean']), min(values), abs_tol=1e-12)
    for r in read(out / 'equal_dataset_diagnostics.csv'):
        for metric in ['correct_singleton_rate', 'singleton_error_rate']:
            datasets = defaultdict(list)
            for (d, s, m), value in condition_values.items():
                if m == r['method']:
                    datasets[d].append(value[metric])
            assert len(datasets) == 3
            value = mean(mean(v) for v in datasets.values())
            assert math.isclose(float(r[metric]), value, abs_tol=1e-12)
    receipt = {'status': 'passed', 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'independent_implementation': 'Python standard library; no imports from analysis or threshold code',
        'replicate_rows': len(computed), 'conditions_methods': len(expected), 'floor_checks': 27,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
