"""Fixed-budget optimizer sensitivity; selection never reads target test labels.

Imports the archived independent implementations, reuses exactly their draws,
and records all tested budgets and initializations, including unfavorable ones.
Run under the CPU resource guard; no CUDA devices or arrays are used.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load(name, path, lab):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = lab
    module.REFERENCE_DIR = lab / 'exp/source_role_outcome_blind_20260725'
    module.GENERATION_PROVENANCE_PATH = module.REFERENCE_DIR / 'generation_provenance.json'
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ot_fit(problems, steps, initial_seed):
    grid = torch.tensor(np.stack([p['grid'] for p in problems]), dtype=torch.float32)
    indicators = torch.tensor(np.stack([p['indicator'] for p in problems]), dtype=torch.float32)
    kernels = torch.tensor(np.stack([p['kernel'] for p in problems]), dtype=torch.float32)
    lower = torch.tensor(np.stack([p['target_max_cdf'] for p in problems]), dtype=torch.float32)
    upper = torch.tensor(np.stack([p['target_min_cdf'] for p in problems]), dtype=torch.float32)
    initial = np.zeros((len(problems), 300), dtype=np.float32)
    if initial_seed is not None:
        initial = np.random.default_rng(initial_seed).normal(0, 0.2, initial.shape).astype(np.float32)
    logits = torch.tensor(initial, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.001)
    def objective():
        weights = torch.softmax(logits, dim=1)
        cdf = torch.einsum('bn,bng->bg', weights, indicators)
        density = torch.einsum('bn,bng->bg', weights, kernels)
        return 0.5 * torch.trapezoid(density * (abs(cdf-lower) + abs(cdf-upper) + upper-lower), grid, dim=1)
    traces = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.sum().backward()
        if step in {0, 1000, 2000, 4000, steps}:
            values = loss.detach().numpy()
            gradients = torch.linalg.vector_norm(logits.grad, dim=1).detach().numpy()
            traces.extend({'case': i, 'step': step, 'objective': float(v), 'logit_gradient_norm': float(gradients[i])}
                          for i, v in enumerate(values))
        if step < steps:
            optimizer.step()
    return torch.softmax(logits, dim=1).detach().numpy().astype(float) * 300, traces


def summarize(frame, out):
    frame['correct_singleton_rate'] = frame.coverage - frame.full_set_rate
    frame['incorrect_singleton_rate'] = frame.singleton_rate - frame.correct_singleton_rate
    metrics = ['coverage', 'class_0_coverage', 'class_1_coverage', 'mean_set_size',
               'empty_set_rate', 'singleton_rate', 'full_set_rate', 'correct_singleton_rate', 'incorrect_singleton_rate']
    seed = frame.groupby(['dataset', 'shift', 'seed', 'variant'])[metrics].mean()
    cond = seed.groupby(['dataset', 'shift', 'variant']).mean().reset_index()
    cond['minimum_of_class_means'] = cond[['class_0_coverage', 'class_1_coverage']].min(axis=1)
    # All-condition seed-0 subset is separate from the enriched age condition.
    base = frame[frame.seed == 0].groupby(['dataset', 'shift', 'variant'])[metrics].mean()
    overall = base.groupby(['dataset', 'variant']).mean().groupby('variant').mean().reset_index()
    frame.to_csv(out / 'replicate_metrics.csv', index=False)
    cond.to_csv(out / 'condition_summary.csv', index=False)
    overall.to_csv(out / 'seed0_equal_dataset_summary.csv', index=False)


def run_ot(module, pairs, lab, out):
    problems, metadata, tests = [], [], []
    for pair in pairs:
        path = lab / pair['prediction_path']
        assert sha(path) == pair['prediction_sha256']
        frame = pd.read_parquet(path)
        source = module.arrays(frame[frame.role == 'source_calibration'].reset_index(drop=True))
        target = module.arrays(frame[frame.role == 'target_calibration_pool'].reset_index(drop=True))
        test = module.arrays(frame[frame.role == 'target_test'].reset_index(drop=True))
        for replicate in range(5):
            parts = (pair['dataset'], pair['shift'], pair['seed'], replicate)
            si = np.random.default_rng(module.deterministic_seed(module.DRAW_NAMESPACE, *parts, 'source')).choice(len(source['p']), 300, replace=False)
            ti = np.random.default_rng(module.deterministic_seed(module.DRAW_NAMESPACE, *parts, 'target')).choice(len(target['p']), 300, replace=False)
            problems.append(module.prepare_ot_problem(source['true_score'][si], target['p'][ti]))
            metadata.append({'dataset': pair['dataset'], 'shift': pair['shift'], 'seed': pair['seed'], 'replicate': replicate,
                             'scores': source['true_score'][si]})
            tests.append(test)
    assert len(problems) == 80
    rows, diagnostics = [], []
    variants = [('baseline', 1000, None), ('steps4000', 4000, None), ('jitter731', 4000, 731), ('jitter991', 4000, 991)]
    for name, steps, seed in variants:
        weights, trace = ot_fit(problems, steps, seed)
        for i, item in enumerate(trace):
            meta = metadata[item['case']]
            diagnostics.append({k: v for k, v in meta.items() if k != 'scores'} | {'variant': name} | item)
        for i, meta in enumerate(metadata):
            q = module.weighted_quantile(meta['scores'], weights[i], test_weight=1.0)
            metrics = module.evaluate(tests[i], q, q)
            rows.append({k: v for k, v in meta.items() if k != 'scores'} | {'variant': name, 'steps': steps, 'threshold': q} | metrics)
        print('Completed OT', name, 'cases', len(problems), flush=True)
        pd.DataFrame(rows).to_csv(out / 'partial_metrics.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out / 'optimization_trace.csv', index=False)
    summarize(pd.DataFrame(rows), out)


def lr_fit_starts(module, problems, selected, starts):
    coefficients, betas, diagnostics = [], [], []
    for offset in range(0, len(problems), module.CHUNK_SIZE):
        chunk = problems[offset:offset+module.CHUNK_SIZE]
        count = len(starts)
        x1 = np.repeat(np.stack([p.x_labeled for p in chunk]), count, axis=0)
        score = np.repeat(np.stack([p.scores for p in chunk]), count, axis=0)
        x2 = np.repeat(np.stack([p.x_target for p in chunk]), count, axis=0)
        x3 = np.repeat(np.stack([p.x_source_unlabeled for p in chunk]), count, axis=0)
        lam = np.repeat(selected[offset:offset+len(chunk)], count)
        theta, beta = module.batched_adam(x1, score, x2, x3, lam, np.tile(starts, len(chunk)), module.FINAL_STEPS)
        for j, problem in enumerate(chunk):
            values = []
            for k in range(count):
                index = count*j+k
                value, gt, gb = module.objective_and_gradient(theta[index], beta[index], lam[index],
                    problem.x_labeled, problem.scores, problem.x_target, problem.x_source_unlabeled)
                values.append(value)
                projected_gb = gb
                if beta[index] <= module.BETA_LOWER+1e-6:
                    projected_gb = min(gb, 0)
                if beta[index] >= module.BETA_UPPER-1e-6:
                    projected_gb = max(gb, 0)
                diagnostics.append({'dataset': problem.dataset, 'shift': problem.shift, 'seed': problem.seed,
                    'replicate': problem.replicate, 'initial_beta': starts[k], 'lambda': lam[index],
                    'objective': value, 'gradient_norm': float(np.linalg.norm(np.r_[gt,gb])),
                    'projected_gradient_norm': float(np.linalg.norm(np.r_[gt,projected_gb])), 'fitted_beta': beta[index]})
            best = count*j + int(np.argmin(values))
            coefficients.append(theta[best]);betas.append(beta[best])
        print('LR-QR final fits',offset+len(chunk),'/',len(problems),flush=True)
    return np.array(coefficients), np.array(betas), diagnostics


def run_lr(module, pairs, out):
    problems, _ = module.prepare_problems(pairs)
    assert len(problems) == 80
    rows, diagnostics = [], []
    selections = {}
    for name, cv_steps, final_steps, starts in [('baseline',600,2000,[.25,1,4]),
                    ('steps8000',1800,8000,[.25,1,4]), ('starts_extra',1800,8000,[.1,2,10])]:
        module.CV_STEPS = cv_steps;module.FINAL_STEPS = final_steps
        if cv_steps not in selections:
            selected, cvtrace = module.select_lambdas(problems)
            selections[cv_steps] = selected
            pd.DataFrame(cvtrace).to_csv(out / f'cv_trace_{cv_steps}.csv',index=False)
        selected = selections[cv_steps]
        theta, beta, diag = lr_fit_starts(module, problems, selected, starts)
        frame, _ = module.evaluate_problems(problems, theta, beta, selected)
        frame['variant']=name;frame['cv_steps']=cv_steps;frame['final_steps']=final_steps
        rows.append(frame);diagnostics.extend(dict(row,variant=name) for row in diag)
        pd.concat(rows).to_csv(out/'partial_metrics.csv',index=False)
        print('Completed LR-QR',name,flush=True)
    pd.DataFrame(diagnostics).to_csv(out/'optimization_diagnostics.csv',index=False)
    summarize(pd.concat(rows,ignore_index=True),out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lab-root',type=Path,required=True)
    parser.add_argument('--reference-code',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--method',choices=['ot','lrqr'],required=True)
    args=parser.parse_args();started=time.monotonic()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    lab=args.lab_root.resolve();out=args.out/args.method;out.mkdir(parents=True,exist_ok=True)
    folder='shift_aware_baselines_20260813' if args.method=='ot' else 'lrqr_baseline_20260824'
    code=args.reference_code/'exp'/folder/'run.py'
    module=load('stability_'+args.method,code,lab)
    provenance=lab/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    pairs=[p for p in json.loads(provenance.read_text())['pairs'] if p['seed']==0 or
           (p['dataset']=='nhanes_multicycle_diabetes' and p['shift']=='age_65plus')]
    assert len(pairs)==16
    if args.method=='ot':run_ot(module,pairs,lab,out)
    else:run_lr(module,pairs,out)
    receipt={'status':'completed','method':args.method,'cases':80,'pair_count':16,'device':'cpu',
             'reference_code_sha256':sha(code),'script_sha256':sha(Path(__file__)),
             'generation_provenance_sha256':sha(provenance),'elapsed_seconds':time.monotonic()-started,
             'files':{p.name:sha(p) for p in out.glob('*.csv')}}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
