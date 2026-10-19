"""Adult-only NHANES cohort sensitivity; no target labels enter model fitting."""
import argparse, dataclasses, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from api import CalibrationInput,fit_split,set_metrics

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    gen=load('adult_role_generator',a.lab_root/'exp/source_role_outcome_blind_20260725/generate.py')
    ref=load('adult_reference',a.lab_root/'exp/row_level_remediation_20260725/generate_row_level_predictions.py')
    sys.path.insert(0,str(a.lab_root))
    from scripts.run_public_shift_benchmark_v2 import prepare_nhanes_multicycle
    from scripts.run_large_scale_conformal import make_model
    original=prepare_nhanes_multicycle();rows=[];cohorts=[];audits=[]
    for shift in (s for s in original.shifts if s.name!='pseudo_random'):
        mask=(original.X.RIDAGEYR>=18).to_numpy()
        if shift.name=='obesity':mask&=original.X.BMXBMI.notna().to_numpy()
        ds=dataclasses.replace(original,X=original.X.loc[mask].reset_index(drop=True),y=original.y[mask],
                               groups=original.groups.loc[mask].reset_index(drop=True))
        ds=ref.attach_stable_ids(ds)
        for seed in range(5):
            split,record=gen.outcome_blind_shift_split(ds,shift,seed);audits.extend(record)
            model=make_model(ds,'sgd_logistic',seed);model.fit(split['X_train'],split['y_train'])
            numeric,categorical=ref.selected_covariates(ds)
            pred,_,_,_=ref.role_frames(ds,split,model,shift.name,seed,numeric,categorical)
            source=pred[pred.role=='source_calibration'].reset_index(drop=True)
            pool=pred[pred.role=='target_calibration_pool'].reset_index(drop=True)
            test=pred[pred.role=='target_test'].reset_index(drop=True)
            domains={r['domain']:r for r in record}
            cohorts.append({'shift':shift.name,'seed':seed,'source_n':domains['source_train_calibration']['domain_n'],
                            'target_n':len(pool)+len(test),'test_n':len(test),'test_prevalence':float(test.label.mean())})
            for rep in range(5):
                payload=f'jbi-adult-v1|{shift.name}|{seed}|{rep}'
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big'))
                s=source.iloc[rng.choice(len(source),300,replace=False)]
                t=pool.iloc[rng.choice(len(pool),300,replace=False)]
                data=CalibrationInput(s.prob_1.to_numpy(),s.label.to_numpy(int),t.prob_1.to_numpy(),t.label.to_numpy(int),.1,rep)
                for domain in ('source','target'):
                    for mondrian in (False,True):
                        name=domain+('_mondrian' if mondrian else '_split')
                        fitted=fit_split(data,domain,mondrian)
                        rows.append({'shift':shift.name,'seed':seed,'replicate':rep,'method':name,
                                     **set_metrics(fitted.predict_sets(test.prob_1.to_numpy()),test.label.to_numpy(int))})
            print('adult sensitivity',shift.name,seed,flush=True)
    f=pd.DataFrame(rows);f.to_csv(a.out/'adult_metrics.csv',index=False)
    pd.DataFrame(cohorts).to_csv(a.out/'adult_cohort_counts.csv',index=False)
    pd.DataFrame(audits).to_csv(a.out/'adult_role_assignment.csv',index=False)
    metrics=['coverage','mean_set_size','class_coverage_gap','empty_set_rate','singleton_rate','full_set_rate']
    byseed=f.groupby(['shift','seed','method'])[metrics].mean()
    byshift=byseed.groupby(['shift','method']).mean().reset_index()
    byshift.to_csv(a.out/'adult_by_shift.csv',index=False)
    eq=byshift.groupby('method')[metrics].mean().reset_index();eq.to_csv(a.out/'adult_summary.csv',index=False)
    assert len(f)==400
    receipt={'status':'completed','metric_rows':400,'model_fits':20,'adult_only':True,
             'observed_bmi_required_for_bmi_shift':True,'source_labels':300,'target_labels':300,
             'inference':'Separate NHANES cohort sensitivity; excluded from primary all-age aggregation.',
             'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.csv')}}
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(eq.to_string(index=False))

if __name__=='__main__':main()
