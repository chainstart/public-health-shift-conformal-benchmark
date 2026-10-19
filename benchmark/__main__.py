"""Unified commands for preparing roles, evaluating methods and comparing results."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from .api import BUILTINS, MethodSpec, evaluate_method

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_method(name):
    if name in BUILTINS:
        return BUILTINS[name]
    path = Path(name).resolve()
    loader = importlib.util.spec_from_file_location('benchmark_user_method', path)
    if loader is None or loader.loader is None:
        raise ValueError('Expected a Python file defining METHOD')
    module = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = module
    loader.loader.exec_module(module)
    if not isinstance(module.METHOD, MethodSpec):
        raise TypeError('METHOD must use benchmark.api.MethodSpec')
    return module.METHOD


def prepare(a):
    workspace = a.workspace.resolve()
    template = ROOT / 'benchmark/preparation'
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError('Use a new empty workspace; existing scientific records are never overwritten')
    shutil.copytree(template, workspace, dirs_exist_ok=True)
    if a.provider_cache:
        source = a.provider_cache.resolve()
        loader = importlib.util.spec_from_file_location('provider_input_recipe', workspace/'benchmark/prepare_data.py')
        recipe = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(recipe)
        for item in recipe.inputs(workspace):
            destination = item[4]
            cached = source / destination.relative_to(workspace)
            if cached.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(cached)
    subprocess.run([sys.executable, str(workspace/'benchmark/prepare_data.py'),
                    '--lab-root', str(workspace), '--out', str(workspace/'data_receipts'),
                    '--download', '--probe-urls', '--rebuild-cohorts'], check=True)
    subprocess.run([sys.executable, str(workspace/'exp/source_role_outcome_blind_20260725/generate.py'),
                    '--lab-root', str(workspace)], check=True)


def fixture(a):
    workspace = a.workspace.resolve()
    manifest = workspace / 'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    if manifest.exists():
        raise ValueError('Fixture manifest already exists')
    pairs = []
    for seed in range(2):
        rng = np.random.default_rng(seed)
        frames = []
        for role in ('source_calibration', 'target_calibration_pool', 'target_test'):
            p = rng.uniform(.01, .99, 400)
            y = rng.binomial(1, p if role == 'source_calibration' else np.clip(p*.8+.15, 0, 1))
            frames.append(pd.DataFrame({'role':role, 'prob_1':p, 'label':y,
                'row_id':[f'{seed}:{role}:{i}' for i in range(len(p))]}))
        path = workspace / f'fixture/seed-{seed}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames).to_parquet(path, index=False)
        pairs.append(dict(dataset='SYNTHETIC_SMOKE_ONLY', shift='illustrative_shift', seed=seed,
                          prediction_path=str(path.relative_to(workspace)), prediction_sha256=sha(path)))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'synthetic':True, 'pairs':pairs}, indent=2)+'\n')
    print('Synthetic interface fixture created; not empirical benchmark evidence.')


def evaluate(a):
    if min(a.source_n, a.target_n, a.replicates) < 1 or not 0 < a.alpha < 1:
        raise ValueError('Positive budgets/replicates and 0 < alpha < 1 required')
    specs = [load_method(name) for name in a.methods]
    if len({s.name for s in specs}) != len(specs):
        raise ValueError('Method names must be unique')
    workspace = a.workspace.resolve()
    provenance = workspace/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    manifest = json.loads(provenance.read_text())
    rows, inputs, seen = [], [], set()
    for pair in manifest['pairs']:
        key = tuple(pair[k] for k in ('dataset', 'shift', 'seed'))
        if key in seen:
            raise ValueError('Duplicate dataset/shift/seed')
        seen.add(key)
        path = (workspace/pair['prediction_path']).resolve()
        if not path.is_relative_to(workspace):
            raise ValueError('Prediction path escapes workspace')
        actual = sha(path)
        if actual != pair['prediction_sha256']:
            raise ValueError('Prediction hash mismatch')
        frame = pd.read_parquet(path)
        if not np.isfinite(frame.prob_1).all() or not frame.prob_1.between(0,1).all():
            raise ValueError('Invalid binary probabilities')
        if not frame.label.isin([0,1]).all():
            raise ValueError('Invalid binary outcomes')
        if 'row_id' in frame and frame.row_id.duplicated().any():
            raise ValueError('Row IDs overlap or repeat across roles')
        s, t, test = [frame[frame.role == role].reset_index(drop=True) for role in
                     ('source_calibration','target_calibration_pool','target_test')]
        if len(s) < a.source_n or len(t) < a.target_n or test.empty:
            raise ValueError('Insufficient role samples')
        inputs.append({**{k:pair[k] for k in ('dataset','shift','seed')}, 'sha256':actual})
        for rep in range(a.replicates):
            payload = f"jbi-extension-v1|{pair['dataset']}|{pair['shift']}|{pair['seed']}|{rep}"
            draw_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big')
            rng = np.random.default_rng(draw_seed)
            sc = s.iloc[rng.choice(len(s), a.source_n, replace=False)]
            tc = t.iloc[rng.choice(len(t), a.target_n, replace=False)]
            for spec in specs:
                metrics = evaluate_method(spec, sc.prob_1.to_numpy(), sc.label.to_numpy(int),
                    tc.prob_1.to_numpy(), tc.label.to_numpy(int), test.prob_1.to_numpy(),
                    test.label.to_numpy(int), alpha=a.alpha, seed=draw_seed)
                rows.append({**{k:pair[k] for k in ('dataset','shift','seed')},
                    'replicate':rep, 'method':spec.name, 'uses_target_labels':spec.uses_target_labels,
                    'source_n':a.source_n, 'target_n':a.target_n, 'test_n':len(test),
                    'alpha':a.alpha, **metrics})
    if not rows:
        raise ValueError('No evaluation pairs')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out,index=False)
    config = json.loads(Path(__file__).with_name('tasks.json').read_text())
    expected = {(task['dataset'],task['shift_name'],seed) for task in config['tasks'] for seed in config['model_seeds']}
    standard = (seen == expected and a.source_n == config['source_n'] and a.target_n == config['target_n']
                and a.replicates == config['replicates'] and a.alpha == config['alpha'])
    receipt = {'standard_configuration':standard, 'protocol':'outcome-blind-binary-v1' , 'benchmark_version':'2.0.0',
        'synthetic':bool(manifest.get('synthetic',False)), 'source_n':a.source_n,
        'target_n':a.target_n, 'replicates':a.replicates, 'alpha':a.alpha,
        'prediction_inputs':inputs, 'manifest_sha256':sha(provenance),
        'implementation_sha256':{name:sha(Path(__file__).with_name(name)) for name in ('api.py','__main__.py','tasks.json')},
        'environment':{'python':sys.version.split()[0], **{name:importlib.metadata.version(name) for name in ('numpy','pandas','pyarrow')}},
        'method_inputs':[{'argument':name, 'name':spec.name,
            'sha256':sha(Path(name)) if Path(name).is_file() else sha(Path(__file__).with_name('api.py'))}
            for name,spec in zip(a.methods,specs)], 'output_sha256':sha(a.out)}
    a.out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'{len(rows)} method evaluations written to {a.out}')


def compare(a):
    frame = pd.read_csv(a.input)
    keys = ['dataset','shift','seed','replicate']
    if frame.duplicated(keys+['method']).any():
        raise ValueError('Duplicate evaluation rows')
    reference = None
    for method, group in frame.groupby('method'):
        index = set(map(tuple,group[keys].to_numpy()))
        settings = group[['alpha','source_n','target_n']].drop_duplicates()
        if len(settings) != 1:
            raise ValueError('Mixed evaluation budgets or alpha')
        counts = dict(zip(map(tuple,group[keys].to_numpy()),group.test_n))
        signature = (index, tuple(settings.iloc[0]), counts)
        if reference is not None and signature != reference:
            raise ValueError('Methods must share all evaluation keys and settings')
        reference = signature
    metrics = ['coverage','mean_set_size','empty_set_rate','singleton_rate','full_set_rate',
               'class_0_coverage','class_1_coverage','class_coverage_gap']
    seed = frame.groupby(['method','dataset','shift','seed'])[metrics].mean()
    condition = seed.groupby(['method','dataset','shift']).mean()
    dataset = condition.groupby(['method','dataset']).mean()
    summary = dataset.groupby('method').mean()
    a.out.mkdir(parents=True,exist_ok=True)
    dataset.to_csv(a.out/'by_dataset.csv')
    condition.to_csv(a.out/'by_condition.csv')
    summary.to_csv(a.out/'equal_dataset_summary.csv')
    frame[['method','uses_target_labels','source_n','target_n','alpha']].drop_duplicates().to_csv(a.out/'information_budget.csv',index=False)
    print(summary.to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', action='version', version='2.0.0')
    commands = p.add_subparsers(dest='command',required=True)
    for name, handler in [('prepare',prepare),('fixture',fixture)]:
        sub=commands.add_parser(name);sub.add_argument('--workspace',type=Path,required=True)
        if name == 'prepare':
            sub.add_argument('--provider-cache',type=Path,help='Existing lab root with raw provider files; linked read-only by convention, revalidated and hashed')
        sub.set_defaults(handler=handler)
    sub=commands.add_parser('evaluate');sub.set_defaults(handler=evaluate)
    sub.add_argument('--workspace',type=Path,required=True)
    sub.add_argument('--methods',nargs='+',default=list(BUILTINS))
    sub.add_argument('--source-n',type=int,default=300);sub.add_argument('--target-n',type=int,default=300)
    sub.add_argument('--replicates',type=int,default=5);sub.add_argument('--alpha',type=float,default=.1)
    sub.add_argument('--out',type=Path,required=True)
    sub=commands.add_parser('compare');sub.set_defaults(handler=compare)
    sub.add_argument('--input',type=Path,required=True);sub.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.handler(args)


if __name__ == '__main__':
    main()
