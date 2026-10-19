"""Descriptive untouched-target diagnostics; no fitting or tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lab-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    provenance=args.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    rows=[]
    for pair in json.loads(provenance.read_text())['pairs']:
        path=args.lab_root/pair['prediction_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pair['prediction_sha256']
        f=pd.read_parquet(path);f=f[f.role=='target_test']
        y=f.label;p=f.prob_1
        rows.append({'dataset':pair['dataset'],'shift':pair['shift'],'seed':pair['seed'],
                     'test_n':len(f),'prevalence':y.mean(),'auroc':roc_auc_score(y,p),
                     'average_precision':average_precision_score(y,p),'brier_score':brier_score_loss(y,p)})
    f=pd.DataFrame(rows);assert len(f)==60
    args.out.mkdir(parents=True,exist_ok=True)
    f.to_csv(args.out/'seed_diagnostics.csv',index=False)
    f.groupby(['dataset','shift']).mean(numeric_only=True).drop(columns='seed').to_csv(args.out/'condition_diagnostics.csv')
    (args.out/'receipt.json').write_text(json.dumps({'status':'completed','fits':60,'model_selection':'none; frozen predictions only',
        'provenance_sha256':hashlib.sha256(provenance.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(f.groupby(['dataset','shift'])[['auroc','average_precision','brier_score']].mean().to_string())


if __name__=='__main__':main()
