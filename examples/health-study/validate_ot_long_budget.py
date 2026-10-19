"""Additional fixed 8000-step OT sensitivity on the prespecified 80 problems."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from validate_optimizer_stability import load,sha,ot_fit,summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--lab-root',type=Path,required=True);p.add_argument('--reference-code',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 m=load('ot_reference_long',a.reference_code,a.lab_root)
 provenance=a.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
 pairs=[x for x in json.loads(provenance.read_text())['pairs'] if x['seed']==0 or (x['dataset']=='nhanes_multicycle_diabetes' and x['shift']=='age_65plus')]
 problems=[];meta=[];tests=[]
 for pair in pairs:
  path=a.lab_root/pair['prediction_path'];assert sha(path)==pair['prediction_sha256'];df=pd.read_parquet(path)
  source=m.arrays(df[df.role=='source_calibration'].reset_index(drop=True));target=m.arrays(df[df.role=='target_calibration_pool'].reset_index(drop=True));test=m.arrays(df[df.role=='target_test'].reset_index(drop=True))
  for rep in range(5):
   parts=(pair['dataset'],pair['shift'],pair['seed'],rep)
   si=np.random.default_rng(m.deterministic_seed(m.DRAW_NAMESPACE,*parts,'source')).choice(len(source['p']),300,replace=False);ti=np.random.default_rng(m.deterministic_seed(m.DRAW_NAMESPACE,*parts,'target')).choice(len(target['p']),300,replace=False)
   problems.append(m.prepare_ot_problem(source['true_score'][si],target['p'][ti]));meta.append(dict(dataset=pair['dataset'],shift=pair['shift'],seed=pair['seed'],replicate=rep,scores=source['true_score'][si]));tests.append(test)
 assert len(problems)==80
 weights,trace=ot_fit(problems,8000,None);np.save(a.out/'weights-steps8000.npy',weights)
 rows=[]
 for i,x in enumerate(meta):
  q=m.weighted_quantile(x['scores'],weights[i],1.0)
  rows.append({k:v for k,v in x.items() if k!='scores'}|dict(variant='steps8000',steps=8000,threshold=q)|m.evaluate(tests[i],q,q))
 summarize(pd.DataFrame(rows),a.out)
 pd.DataFrame([{k:v for k,v in meta[x['case']].items() if k!='scores'}|dict(variant='steps8000')|x for x in trace]).to_csv(a.out/'optimization_trace.csv',index=False)
 (a.out/'receipt.json').write_text(json.dumps(dict(status='completed',cases=80,steps=8000,script_sha256=sha(Path(__file__)),driver_sha256=sha(Path(__file__).with_name('validate_optimizer_stability.py')),reference_sha256=sha(a.reference_code)),indent=2)+'\n')
if __name__=='__main__':main()
