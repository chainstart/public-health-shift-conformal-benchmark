"""Source-fitted nonlinear sensitivity with identical primary roles and target draws."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from api import CalibrationInput,fit_split,set_metrics

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    gen=load('backbone_role_generator',a.lab_root/'exp/source_role_outcome_blind_20260725/generate.py')
    ref=load('backbone_reference',a.lab_root/'exp/row_level_remediation_20260725/generate_row_level_predictions.py')
    sys.path.insert(0,str(a.lab_root))
    from scripts.run_public_shift_benchmark_v2 import prepare_brfss_multiyear,prepare_nhanes_multicycle,prepare_uci_with_pseudo
    from scripts.run_large_scale_conformal import make_preprocessor
    rows=[];audits=[]
    for construct in (prepare_brfss_multiyear,prepare_nhanes_multicycle,prepare_uci_with_pseudo):
        ds=ref.attach_stable_ids(construct())
        for shift in (s for s in ds.shifts if s.name!='pseudo_random'):
            for seed in range(5):
                split,record=gen.outcome_blind_shift_split(ds,shift,seed);audits.extend(record)
                clf=LGBMClassifier(n_estimators=120,num_leaves=31,learning_rate=.05,class_weight='balanced',random_state=seed,n_jobs=2,verbosity=-1)
                model=Pipeline([('preprocess',make_preprocessor(ds)),('model',clf)])
                model.fit(split['X_train'],split['y_train']);numeric,categorical=ref.selected_covariates(ds)
                f,_,_,_=ref.role_frames(ds,split,model,shift.name,seed,numeric,categorical)
                source=f[f.role=='source_calibration'].reset_index(drop=True)
                pool=f[f.role=='target_calibration_pool'].reset_index(drop=True);test=f[f.role=='target_test'].reset_index(drop=True)
                for rep in range(5):
                    payload='|'.join(map(str,('jbi-budget-v1',ds.name,shift.name,seed,rep)))
                    draw_seed=int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big')
                    n=max(1,int(round(.1*(len(pool)+len(test)))))
                    cal=pool.iloc[np.random.default_rng(draw_seed).permutation(len(pool))[:n]]
                    data=CalibrationInput(source.prob_1.to_numpy(),source.label.to_numpy(int),cal.prob_1.to_numpy(),cal.label.to_numpy(int),.1,rep)
                    for domain in ('source','target'):
                        for mondrian in (False,True):
                            fitted=fit_split(data,domain,mondrian)
                            rows.append({'dataset':ds.name,'shift':shift.name,'seed':seed,'replicate':rep,
                                         'method':domain+('_mondrian' if mondrian else '_split'),
                                         **set_metrics(fitted.predict_sets(test.prob_1.to_numpy()),test.label.to_numpy(int))})
                print('LightGBM',ds.name,shift.name,seed,flush=True)
    f=pd.DataFrame(rows);f.to_csv(a.out/'nonlinear_metrics.csv',index=False);pd.DataFrame(audits).to_csv(a.out/'role_assignment.csv',index=False)
    metrics=['coverage','mean_set_size','class_coverage_gap','empty_set_rate','singleton_rate','full_set_rate']
    shifts=f.groupby(['dataset','shift','seed','method'])[metrics].mean().groupby(['dataset','shift','method']).mean().reset_index()
    shifts.to_csv(a.out/'nonlinear_by_shift.csv',index=False)
    ds=shifts.groupby(['dataset','method'])[metrics].mean().reset_index();ds.to_csv(a.out/'nonlinear_by_dataset.csv',index=False)
    eq=ds.groupby('method')[metrics].mean().reset_index();eq.to_csv(a.out/'nonlinear_summary.csv',index=False)
    assert len(f)==1200
    receipt={'status':'completed','model_fits':60,'metric_rows':1200,'backbone':'LightGBM','target_fraction':.1,'replicates':5,
             'settings':{'n_estimators':120,'num_leaves':31,'learning_rate':.05,'class_weight':'balanced','n_jobs':2},
             'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.csv')}}
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(eq.to_string(index=False))

if __name__=='__main__':main()
