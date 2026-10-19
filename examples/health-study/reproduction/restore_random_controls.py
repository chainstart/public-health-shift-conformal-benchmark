"""Retain submitted whole-dataset random controls with outcome-blind roles."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd


def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    codehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol={'question':'Retain the original whole-dataset random train/calibration/test control.',
              'test_fraction':.25,'calibration_fraction_of_development':.3,'seeds':list(range(5)),
              'allocation':'permutation independent of labels/features; original fractions retained',
              'interpretation':'15 dataset-seed controls, not 60 independent controls and not condition-size matched',
              'code_sha256':codehash}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    interface=Path(__file__).parent/'benchmark'
    if not interface.exists():interface=Path(__file__).parent/'reproduction/benchmark'
    sys.path.insert(0,str(interface));from api import finite_quantile,set_metrics
    gen=load('random_roles',a.lab_root/'exp/source_role_outcome_blind_20260725/generate.py')
    ref=load('random_reference',a.lab_root/'exp/row_level_remediation_20260725/generate_row_level_predictions.py')
    sys.path.insert(0,str(a.lab_root))
    from scripts.run_public_shift_benchmark_v2 import prepare_brfss_multiyear,prepare_nhanes_multicycle,prepare_uci_with_pseudo
    from scripts.run_large_scale_conformal import make_model
    rows=[];audits=[]
    for construct in (prepare_brfss_multiyear,prepare_nhanes_multicycle,prepare_uci_with_pseudo):
        ds=ref.attach_stable_ids(construct())
        for seed in range(5):
            xd,xt,yd,yt,gd,gt,record=gen.outcome_blind_partition(ds.X,ds.y,ds.groups,.25,dataset=ds.name,shift='random_control',model_seed=seed,domain='development_test')
            audits.append(record)
            xtr,xc,ytr,yc,gtr,gc,record=gen.outcome_blind_partition(xd,yd,gd,.30,dataset=ds.name,shift='random_control',model_seed=seed,domain='training_calibration')
            audits.append(record)
            assert not set(gc['__stable_row_id'])&set(gt['__stable_row_id'])
            model=make_model(ds,'sgd_logistic',seed);model.fit(xtr,ytr)
            assert model.classes_.tolist()==[0,1]
            cp=model.predict_proba(xc)[:,1];tp=model.predict_proba(xt)[:,1]
            scores=np.where(yc==0,cp,1-cp);q=finite_quantile(scores)
            k=int(np.ceil((len(scores)+1)*.9));assert q==np.sort(scores)[k-1]
            sets=np.column_stack([tp<=q,1-tp<=q]);metric=set_metrics(sets,yt)
            rows.append(dict(dataset=ds.name,seed=seed,method='source_split',n_train=len(ytr),
                             n_calibration=len(yc),n_test=len(yt),threshold=q,**metric,
                             shortfall=max(0.,.9-metric['coverage'])))
            pd.DataFrame(rows).to_csv(a.out/'control_metrics.partial.csv',index=False)
            print('whole-dataset random control',ds.name,seed,flush=True)
    f=pd.DataFrame(rows);assert len(f)==15;f.to_csv(a.out/'control_metrics.csv',index=False)
    pd.DataFrame(audits).to_csv(a.out/'role_assignment.csv',index=False)
    cols=['coverage','mean_set_size','class_coverage_gap','empty_set_rate','singleton_rate','full_set_rate','shortfall']
    f.groupby('dataset')[cols].mean().to_csv(a.out/'dataset_summary.csv')
    (a.out/'control_metrics.partial.csv').unlink()
    receipt={'status':'completed','fits':15,'seed_runs':15,'calibration_thresholds_independently_reconstructed':True,
             'median_coverage':float(f.coverage.median()),'equal_dataset_mean_coverage':float(f.groupby('dataset').coverage.mean().mean()),
             'code_sha256':codehash,'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.csv')}}
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
