"""Within-source checks at the same 300-label budget as the main comparison."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from api import CalibrationInput,fit_split,set_metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    manifest=a.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    pairs=json.loads(manifest.read_text())['pairs'];rows=[];assert len(pairs)==60
    for pair in pairs:
        path=a.lab_root/pair['prediction_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==pair['prediction_sha256']
        frame=pd.read_parquet(path);s=frame[frame.role=='source_calibration'].reset_index(drop=True)
        for rep in range(5):
            payload=f"jbi-source-control-v1|{pair['dataset']}|{pair['shift']}|{pair['seed']}|{rep}"
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big'))
            order=rng.permutation(len(s));cal=s.iloc[order[:300]];test=s.iloc[order[300:]]
            assert not set(cal.row_id)&set(test.row_id)
            d=CalibrationInput(cal.prob_1.to_numpy(),cal.label.to_numpy(int),np.array([]),None,.1,rep)
            for mondrian in (False,True):
                fitted=fit_split(d,'source',mondrian)
                rows.append({**{k:pair[k] for k in ('dataset','shift','seed')},'replicate':rep,
                             'method':'source_mondrian' if mondrian else 'source_split',
                             'n_calibration':300,'n_test':len(test),
                             **set_metrics(fitted.predict_sets(test.prob_1.to_numpy()),test.label.to_numpy(int))})
    f=pd.DataFrame(rows);f.to_csv(a.out/'control_metrics.csv',index=False)
    metrics=['coverage','mean_set_size','class_coverage_gap','empty_set_rate','singleton_rate','full_set_rate']
    shift=f.groupby(['dataset','shift','seed','method'])[metrics].mean().groupby(['dataset','shift','method']).mean()
    ds=shift.groupby(['dataset','method']).mean().reset_index();ds.to_csv(a.out/'control_by_dataset.csv',index=False)
    eq=ds.groupby('method')[metrics].mean().reset_index();eq.to_csv(a.out/'control_summary.csv',index=False)
    ranks=[]
    for n in [8,9,26,44,91,300]:
        k=int(np.ceil((n+1)*.9));expected=1 if k>n else k/(n+1)
        ranks.append({'n':n,'rank':k,'threshold_infinite':k>n,'expected_coverage_continuous_iid':expected,
                      'threshold_sd_uniform_scores':float('nan') if k>n else np.sqrt(k*(n+1-k)/((n+1)**2*(n+2)))})
    pd.DataFrame(ranks).to_csv(a.out/'finite_rank_examples.csv',index=False)
    assert len(f)==600
    receipt={'status':'completed','metric_rows':600,'pairs':60,'calibration_n':300,'replicates':5,
             'scope':'Source-domain diagnostic with the already source-fitted model; not part of the transported-method contrasts.',
             'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.csv')}}
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(eq.to_string(index=False))

if __name__=='__main__':main()
