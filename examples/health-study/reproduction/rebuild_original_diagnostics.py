"""Restore the submitted diagnostic using verified, corrected-protocol predictions.

Run under the OpenLabs CPU resource guard. No model fitting, tuning, or test-label
selection is performed. The target quantile is retrospective and nondeployable.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lab-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    interface = Path(__file__).parent / 'benchmark'
    if not interface.exists():
        interface = Path(__file__).parent / 'reproduction/benchmark'
    sys.path.insert(0, str(interface))
    from api import finite_quantile, true_scores, ThresholdSets, set_metrics
    manifest = a.lab_root / 'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    pairs = json.loads(manifest.read_text())['pairs']
    assert len(pairs) == 60
    rows = []
    for pair in pairs:
        path = a.lab_root / pair['prediction_path']
        assert sha(path) == pair['prediction_sha256']
        f = pd.read_parquet(path)
        s = f.loc[f.role == 'source_calibration']
        t = f.loc[f.role == 'target_test']
        pool = f.loc[f.role == 'target_calibration_pool'].reset_index(drop=True)
        assert not set(s.row_id) & set(t.row_id)
        source_scores = true_scores(s.prob_1.to_numpy(), s.label.to_numpy(int))
        target_scores = true_scores(t.prob_1.to_numpy(), t.label.to_numpy(int))
        q = finite_quantile(source_scores)
        # The submitted diagnostic used a descriptive empirical 90th percentile,
        # not a fitted threshold. Keep that diagnostic definition explicit.
        target_q = float(np.quantile(target_scores, .9, method='linear'))
        target_thresholds = []
        for replicate in range(5):
            payload = '|'.join(map(str, ('jbi-budget-v1', pair['dataset'], pair['shift'], pair['seed'], replicate)))
            draw_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big')
            n = max(1, int(round(.1 * (len(pool) + len(t)))))
            cal = pool.iloc[np.random.default_rng(draw_seed).permutation(len(pool))[:n]]
            target_thresholds.append(finite_quantile(true_scores(cal.prob_1.to_numpy(), cal.label.to_numpy(int))))
        metric = set_metrics(ThresholdSets(q, q).predict_sets(t.prob_1.to_numpy()), t.label.to_numpy(int))
        rows.append({**{k: pair[k] for k in ('dataset', 'shift', 'seed')},
                     'source_n': len(s), 'test_n': len(t), 'source_threshold': q,
                     'diagnostic_target_q90': target_q, 'threshold_gap': target_q - q,
                     'target10_threshold_mean': float(np.mean(target_thresholds)),
                     'coverage': metric['coverage'], 'shortfall': max(0., .9 - metric['coverage']),
                     'prediction_sha256': pair['prediction_sha256']})
    f = pd.DataFrame(rows)
    assert len(f) == 60 and not f.isna().any().any()
    f.to_csv(a.out / 'threshold_diagnostic_by_seed.csv', index=False)
    g = f.groupby(['dataset', 'shift']).mean(numeric_only=True).reset_index()
    g.to_csv(a.out / 'threshold_diagnostic_by_condition.csv', index=False)
    r60 = float(spearmanr(f.threshold_gap, f.shortfall).statistic)
    r12 = float(spearmanr(g.threshold_gap, g.shortfall).statistic)
    colors = ['#0072B2', '#D55E00', '#009E73']
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for (name, frame), color in zip(f.groupby('dataset', sort=True), colors):
        ax.scatter(frame.threshold_gap, frame.shortfall, c=color, alpha=.72,
                   edgecolors='white', linewidths=.4, label=name.split('_')[0].upper())
    ax.axvline(0, color='.6', lw=.8, ls='--')
    ax.set(xlabel='Diagnostic target-test 90th percentile − source threshold',
           ylabel='Source-calibrated coverage shortfall',
           title=f'Retrospective threshold diagnostic (60 seed runs; Spearman r = {r60:.3f})')
    ax.legend(frameon=False)
    ax.spines[['right', 'top']].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out / 'score_threshold_mismatch.pdf')
    plt.close(fig)
    receipt = {'status': 'completed', 'pairs': 60, 'conditions': 12,
               'spearman_seed_descriptive': r60, 'spearman_condition_descriptive': r12,
               'inferential_pvalues_reported': False,
               'interpretation': 'Shared target-score diagnostic; not independent causal evidence. Test labels never fit a deployable threshold.',
               'target_diagnostic_quantile': 'numpy linear empirical percentile 0.9; descriptive only',
               'source_threshold': 'ceil((n+1)*0.9) order statistic, infinity on overflow',
               'input_manifest_sha256': sha(manifest), 'code_sha256': sha(Path(__file__)),
               'outputs': {p.name: sha(p) for p in a.out.glob('*') if p.suffix in ('.csv', '.pdf')}}
    (a.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('outputs',)}, indent=2))


if __name__ == '__main__':
    main()
