"""Evaluate a user plug-in on fixed roles, shared calibration draws and test sets."""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
from api import BUILTINS, MethodSpec, evaluate_method


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--method',required=True,help='built-in name or Python file defining METHOD')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--source-n',type=int,default=300)
    ap.add_argument('--target-n',type=int,default=300);ap.add_argument('--replicates',type=int,default=5)
    args=ap.parse_args()
    if args.method in BUILTINS:spec=BUILTINS[args.method]
    else:
        path=Path(args.method).resolve();loader=importlib.util.spec_from_file_location('user_method',path)
        module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module);spec=module.METHOD
    if not isinstance(spec,MethodSpec):raise TypeError('METHOD must be a MethodSpec')
    provenance=args.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    pairs=json.loads(provenance.read_text())['pairs'];rows=[]
    for pair in pairs:
        path=args.lab_root/pair['prediction_path'];actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=pair['prediction_sha256']:raise ValueError('Input hash mismatch')
        frame=pd.read_parquet(path);s=frame[frame.role=='source_calibration'].reset_index(drop=True)
        t=frame[frame.role=='target_calibration_pool'].reset_index(drop=True)
        test=frame[frame.role=='target_test'].reset_index(drop=True)
        if len(s)<args.source_n or len(t)<args.target_n:raise ValueError('Budget exceeds available pool')
        for rep in range(args.replicates):
            payload=f"jbi-extension-v1|{pair['dataset']}|{pair['shift']}|{pair['seed']}|{rep}"
            seed=int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big');rng=np.random.default_rng(seed)
            sc=s.iloc[rng.choice(len(s),args.source_n,replace=False)]
            tc=t.iloc[rng.choice(len(t),args.target_n,replace=False)]
            m=evaluate_method(spec,sc.prob_1.to_numpy(),sc.label.to_numpy(int),tc.prob_1.to_numpy(),
                              tc.label.to_numpy(int),test.prob_1.to_numpy(),test.label.to_numpy(int),seed=seed)
            rows.append({**{k:pair[k] for k in ('dataset','shift','seed')},'replicate':rep,'method':spec.name,**m})
    args.out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.out,index=False)
    print(f'{len(rows)} paired evaluations written to {args.out}')


if __name__=='__main__':main()
