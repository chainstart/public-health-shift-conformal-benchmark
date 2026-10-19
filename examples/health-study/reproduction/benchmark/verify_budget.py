"""Independent sorted-rank reconstruction of budget outputs (does not import api)."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

def threshold(p,y,mondrian):
    scores=np.where(y==0,p,1-p)
    def rank(a):
        k=int(np.ceil(.9*(len(a)+1)))
        return np.inf if len(a)==0 or k>len(a) else np.sort(a)[k-1]
    return [rank(scores[y==c]) for c in (0,1)] if mondrian else [rank(scores)]*2

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    receipt=json.loads((a.out/'receipt.json').read_text())
    for name,wanted in receipt['outputs'].items():
        assert hashlib.sha256((a.out/name).read_bytes()).hexdigest()==wanted,name
    metrics=pd.read_csv(a.out/'budget_metrics.csv');draws=pd.read_csv(a.out/'budget_draws.csv')
    pairs=json.loads((a.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json').read_text())['pairs']
    mismatches=[];checked=0;set_columns=['coverage','mean_set_size','empty_set_rate','singleton_rate','full_set_rate','class_0_coverage','class_1_coverage','class_coverage_gap']
    for pair in pairs:
        path=a.lab_root/pair['prediction_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==pair['prediction_sha256']
        f=pd.read_parquet(path);s=f[f.role=='source_calibration'].reset_index(drop=True)
        pool=f[f.role=='target_calibration_pool'].reset_index(drop=True);t=f[f.role=='target_test'].reset_index(drop=True)
        wanted=metrics[(metrics.dataset==pair['dataset'])&(metrics['shift']==pair['shift'])&(metrics.seed==pair['seed'])]
        for rep in range(receipt['replicates']):
            payload='|'.join(map(str,('jbi-budget-v1',pair['dataset'],pair['shift'],pair['seed'],rep)))
            seed=int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big')
            order=np.random.default_rng(seed).permutation(len(pool))
            for fraction in receipt['fractions']:
                n=max(1,int(round(fraction*(len(pool)+len(t)))));cal=pool.iloc[order[:n]]
                d=draws[(draws.dataset==pair['dataset'])&(draws['shift']==pair['shift'])&(draws.seed==pair['seed'])&(draws.replicate==rep)&(draws.fraction==fraction)].iloc[0]
                assert d.n_class1==cal.label.sum() and d.n_class0==n-cal.label.sum()
                assert d.ids_sha256==hashlib.sha256('\x1f'.join(cal.row_id.astype(str)).encode()).hexdigest()
                for method in ('source_split','source_mondrian','target_split','target_mondrian'):
                    c=s if method.startswith('source') else cal
                    q=threshold(c.prob_1.to_numpy(float),c.label.to_numpy(int),method.endswith('mondrian'))
                    p=t.prob_1.to_numpy(float);y=t.label.to_numpy(int)
                    sets=np.column_stack((p<=q[0],1-p<=q[1]));hit=sets[np.arange(len(y)),y];size=sets.sum(axis=1)
                    c0=hit[y==0].mean();c1=hit[y==1].mean()
                    observed=np.array([hit.mean(),size.mean(),(size==0).mean(),(size==1).mean(),(size==2).mean(),c0,c1,abs(c0-c1)])
                    row=wanted[(wanted.replicate==rep)&(wanted.fraction==fraction)&(wanted.method==method)].iloc[0]
                    if not np.allclose(observed,row[set_columns].to_numpy(float),rtol=0,atol=1e-12):mismatches.append((pair['dataset'],pair['shift'],pair['seed'],rep,fraction,method))
                    checked+=1
    expected_rows=len(pairs)*receipt['replicates']*len(receipt['fractions'])*4
    assert checked==len(metrics)==expected_rows
    result={'status':'PASS' if not mismatches else 'FAIL','checked_metric_rows':checked,
            'mismatches':mismatches,'scope':'Independent sorted finite-rank thresholds, paired draws, class counts, prediction sets and all efficiency metrics. Does not assert optimizer convergence or clinical validity.',
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert not mismatches

if __name__=='__main__':main()
