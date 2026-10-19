"""Generate numerical-validation and discrimination tables from frozen results."""
from pathlib import Path
import argparse
import pandas as pd
from make_paper_assets import table, DS, SH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-root', type=Path, required=True)
    paper = ap.parse_args().paper_root
    root = Path(__file__).parent
    results = root / 'results'
    tables = paper / 'manuscript/tables'
    rows = []
    for method, folder in [('OT', 'optimizer_sensitivity_ot'),
                           ('OT', 'optimizer_sensitivity_ot_long'),
                           ('LR-QR', 'optimizer_sensitivity_lrqr')]:
        cond = pd.read_csv(results / folder / 'condition_summary.csv')
        age = cond[(cond.dataset == 'nhanes_multicycle_diabetes') & (cond['shift'] == 'age_65plus')]
        trace = pd.read_csv(results / folder / ('optimization_trace.csv' if method == 'OT' else 'optimization_diagnostics.csv'))
        for _, row in age.iterrows():
            diag = trace[trace.variant == row.variant]
            if method == 'OT':
                diag = diag[diag.step == diag.step.max()]
                gradient = 'logit_gradient_norm'
            else:
                diag = diag.loc[diag.groupby(['dataset', 'shift', 'seed', 'replicate']).objective.idxmin()]
                gradient = 'projected_gradient_norm'
            rows.append(dict(method=method, variant=row.variant,
                             age_worst_class=row.minimum_of_class_means,
                             age_coverage=row.coverage, age_two_label=row.full_set_rate,
                             objective_mean80=diag.objective.mean(), objective_max80=diag.objective.max(),
                             gradient_mean80=diag[gradient].mean(), gradient_max80=diag[gradient].max()))
    summary = pd.DataFrame(rows)
    summary.to_csv(results / 'optimizer_comparison.csv', index=False)
    labels = {'baseline': '1,000 steps', 'steps4000': '4,000 steps', 'steps8000': '8,000 steps',
              'jitter731': '4,000; jitter 731', 'jitter991': '4,000; jitter 991', 'starts_extra': '8,000; extra starts'}
    display = []
    for _, row in summary.iterrows():
        label = labels[row.variant]
        if row.method == 'LR-QR':
            label = {'baseline': '600 CV / 2,000 final', 'steps8000': '1,800 CV / 8,000 final',
                     'starts_extra': '1,800 / 8,000; extra starts'}[row.variant]
        display.append([row.method, label, *[f'{row[c]:.4f}' for c in
                         ['age_worst_class', 'age_coverage', 'age_two_label', 'objective_mean80', 'gradient_mean80', 'gradient_max80']]])
    table(tables / 'optimizer_sensitivity.tex', ['Method', 'Budget/initialization', 'Age class min',
          'Age coverage', 'Age two-label', 'Objective', 'Gradient mean', 'Gradient max'], display, 'llrrrrrr')
    diag = pd.read_csv(results / 'predictor_discrimination/condition_diagnostics.csv')
    table(tables / 'discrimination.tex', ['Condition', '$N$ test', 'Prevalence', 'AUROC', 'AP', 'Brier'],
          [[DS[row.dataset] + ' ' + SH[row['shift']], f'{int(row.test_n):,}',
            *[f'{row[c]:.4f}' for c in ['prevalence', 'auroc', 'average_precision', 'brier_score']]]
           for _, row in diag.iterrows()], 'lrrrrr')
    controls = []
    for method, filename, column in [('OT', 'ot_global_controls.csv', 'objective_implementation_error'),
                                     ('LR-QR', 'lrqr_global_controls.csv', None)]:
        frame = pd.read_csv(results / 'optimizer_controls' / filename)
        for steps, group in frame.groupby('steps'):
            controls.append([method, int(steps), len(group), f'{group.global_objective_gap.max():.2e}',
                             f'{group[column].max():.2e}' if column else r'$\le 10^{-12}$'])
    table(tables / 'optimizer_controls.tex', ['Method', 'Steps', 'Controls', 'Max global gap', 'Objective agreement'], controls, 'lrrrr')
    print('Generated numerical-validation and discrimination summaries and three tables.')


if __name__ == '__main__':
    main()
