"""Download a SHA-bound original public aggregate and reconstruct subgroup fields."""
from pathlib import Path
import hashlib,json,os,urllib.request
import pandas as pd

ROOT=Path(__file__).resolve().parent
def main():
    source=json.loads((ROOT/'subgroup-source.json').read_text())
    target=ROOT/source['local_path'];target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary=target.with_suffix('.download')
        try:
            with urllib.request.urlopen(source['url'],timeout=90) as response,temporary.open('wb') as out:
                while chunk:=response.read(1024*1024):out.write(chunk)
            assert temporary.stat().st_size==source['bytes']
            assert hashlib.sha256(temporary.read_bytes()).hexdigest()==source['sha256']
            os.replace(temporary,target)
        finally:
            if temporary.exists():temporary.unlink()
    assert target.stat().st_size==source['bytes']
    assert hashlib.sha256(target.read_bytes()).hexdigest()==source['sha256']
    frame=pd.read_csv(target)
    frame=frame[(frame.experiment=='primary')&(frame.ablation=='none')&(frame.shift_name!='pseudo_random')].copy()
    assert frame.empty_rate.eq(0).all()
    frame['subgroup_mean_set_size']=2-frame.singleton_rate
    inherited=['n_test','coverage_error','mean_set_size','ambiguous_rate','empty_rate',
      'singleton_accuracy','point_balanced_accuracy','point_auc','negative_coverage',
      'positive_coverage','class_coverage_gap','min_subgroup_coverage','max_subgroup_coverage_gap',
      'weight_mean','weight_median','weight_p95','weight_max','weight_ess']
    frame=frame.rename(columns={name:'parent_run_'+name for name in inherited})
    out=ROOT/source['derived_path'];out.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(out,index=False)
    print(json.dumps({'original_sha256':source['sha256'],'subgroup_rows':len(frame),
      'output':str(out),'scope':'Reconstruction from original aggregate observations; no model refitting.'}))
if __name__=='__main__':main()
