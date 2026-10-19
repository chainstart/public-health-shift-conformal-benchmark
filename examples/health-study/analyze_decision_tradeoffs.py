"""Exploratory, exhaustive class-floor and singleton summaries of frozen results.

No fitting or method selection uses these retrospective test-set summaries.
The yield identity holds for binary prediction sets, including empty sets.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.results / 'matched/extended_replicate_metrics.csv'
    rows = pd.read_csv(source)
    assert len(rows) == 2700 and rows.method.nunique() == 9
    keys = ['dataset', 'shift', 'seed', 'replicate', 'method']
    assert not rows.duplicated(keys).any()
    rows['correct_singleton_rate'] = rows.coverage - rows.full_set_rate
    rows['singleton_error_rate'] = rows.singleton_rate - rows.correct_singleton_rate
    rows['minimum_class_coverage'] = rows[['class_0_coverage', 'class_1_coverage']].min(axis=1)
    assert np.allclose(rows.empty_set_rate + rows.singleton_rate + rows.full_set_rate, 1)
    assert np.allclose(rows.singleton_rate + 2 * rows.full_set_rate, rows.mean_set_size)
    for c in ('correct_singleton_rate', 'singleton_error_rate'):
        assert (rows[c] >= -1e-12).all() and (rows[c] <= rows.singleton_rate + 1e-12).all()
    rows['singleton_accuracy'] = np.divide(rows.correct_singleton_rate, rows.singleton_rate,
        out=np.full(len(rows), np.nan), where=rows.singleton_rate > 0)
    floors = (0.85, 0.88, 0.90)
    for floor in floors:
        rows[f'both_classes_at_{floor:.2f}'] = (rows.minimum_class_coverage >= floor).astype(float)
    measures = ['coverage', 'mean_set_size', 'empty_set_rate', 'singleton_rate', 'full_set_rate',
                'class_0_coverage', 'class_1_coverage', 'class_coverage_gap',
                'correct_singleton_rate', 'singleton_error_rate', 'minimum_class_coverage']
    measures += [f'both_classes_at_{floor:.2f}' for floor in floors]
    # Equal draws and seeds here; keep the hierarchy explicit.
    seeds = rows.groupby(['dataset', 'shift', 'method', 'seed'])[measures].mean()
    conditions = seeds.groupby(['dataset', 'shift', 'method']).mean().reset_index()
    conditions['minimum_of_class_means'] = conditions[['class_0_coverage', 'class_1_coverage']].min(axis=1)
    datasets = conditions.groupby(['dataset', 'method'])[measures].mean().reset_index()
    overall = datasets.groupby('method')[measures].mean().reset_index()
    # Ratio of equally weighted masses; not an average of per-draw ratios.
    overall['singleton_accuracy_from_weighted_mass'] = overall.correct_singleton_rate / overall.singleton_rate
    floor_rows = []
    for method, group in conditions.groupby('method'):
        assert len(group) == 12
        for floor in floors:
            passes = group.minimum_of_class_means >= floor
            floor_rows.append({'method': method, 'floor': floor, 'conditions_passing': int(passes.sum()),
                'conditions_total': len(group), 'worst_condition_class_mean': group.minimum_of_class_means.min(),
                'worst_condition_marginal_mean': group.coverage.min(),
                'condition_weighted_draw_pass_rate': group[f'both_classes_at_{floor:.2f}'].mean()})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in [('replicate_diagnostics.csv', rows[keys + measures + ['singleton_accuracy']]),
                        ('condition_diagnostics.csv', conditions), ('dataset_diagnostics.csv', datasets),
                        ('equal_dataset_diagnostics.csv', overall), ('floor_sensitivity.csv', pd.DataFrame(floor_rows))]:
        frame.to_csv(args.out / name, index=False)
    receipt = {'input': str(source), 'input_sha256': sha(source), 'script_sha256': sha(Path(__file__)),
               'method_rows': len(rows), 'conditions': 12, 'methods': 9, 'floors': list(floors),
               'analysis': 'Exploratory descriptive; thresholds are retrospective evaluation criteria, not clinical tolerances or tuning objectives.',
               'checks': 'Binary set rate and size identities, nonnegative correct/error singleton masses, unique keys, complete methods/conditions.',
               'files': {p.name: sha(p) for p in args.out.glob('*.csv')}}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(overall[['method', 'coverage', 'singleton_rate', 'correct_singleton_rate', 'singleton_error_rate']].to_string(index=False))
    print(pd.DataFrame(floor_rows).to_string(index=False))


if __name__ == '__main__':
    main()
