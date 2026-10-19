"""Retain submitted ablation/L2 questions under the corrected role/quantile rules."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

DROP = {
 'brfss_multiyear_diabetes': {
  'time_future':['survey_year'], 'region_south':['region','_STATE'],
  'age_65plus':['age80','_AGE65YR','_AGEG5YR'], 'obesity':['bmi','_BMI5CAT'],
  'race_black_vs_white':['_RACE']},
 'nhanes_multicycle_diabetes': {
  'time_future':['cycle_start'], 'age_65plus':['RIDAGEYR'],
  'obesity':['BMXBMI','BMXWT','BMXHT','BMXWAIST'], 'race_black_vs_white':['RIDRETH3']},
 'uci_diabetes_readmission': {
  'age_older70':['age_mid'], 'race_african_american':['race'],
  'emergency_admission':['admission_source_id','admission_mode']}}


def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    protocol={'question':'Retain the submitted feature-ablation and L2 sensitivity, correcting the role/quantile implementation only.',
              'variants':['shift_feature_ablation','l2_logistic'],'drop_mapping':DROP,
              'model_seeds':list(range(5)),'calibration_draws':5,'target_fraction':.1,
              'alpha':.1,'selection':'All twelve conditions; no test-label tuning or favorable subset selection.',
              'summary':'Average draws, then seeds, then conditions within dataset; equal weight to datasets.',
              'code_sha256':sha(Path(__file__))}
    pp=a.out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
    sys.path.insert(0,str(Path(__file__).parent/'benchmark'))
    from api import CalibrationInput,fit_split,set_metrics
    gen=load('restore_role',a.lab_root/'exp/source_role_outcome_blind_20260725/generate.py')
    ref=load('restore_reference',a.lab_root/'exp/row_level_remediation_20260725/generate_row_level_predictions.py')
    sys.path.insert(0,str(a.lab_root))
    from scripts.run_public_shift_benchmark_v2 import prepare_brfss_multiyear,prepare_nhanes_multicycle,prepare_uci_with_pseudo
    from scripts.run_large_scale_conformal import make_model
    pairs=json.loads((a.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json').read_text())['pairs']
    pairmap={(p['dataset'],p['shift'],p['seed']):p for p in pairs}
    rows=[];fits=[]
    for construct in (prepare_brfss_multiyear,prepare_nhanes_multicycle,prepare_uci_with_pseudo):
        ds=ref.attach_stable_ids(construct())
        for shift in (s for s in ds.shifts if s.name!='pseudo_random'):
            drop=DROP[ds.name][shift.name]
            for seed in range(5):
                split,audits=gen.outcome_blind_shift_split(ds,shift,seed)
                pair=pairmap[(ds.name,shift.name,seed)];prediction=a.lab_root/pair['prediction_path']
                assert sha(prediction)==pair['prediction_sha256']
                original=pd.read_parquet(prediction)
                for variant in protocol['variants']:
                    fitds=replace(ds,numeric_cols=[c for c in ds.numeric_cols if c not in drop],
                                  categorical_cols=[c for c in ds.categorical_cols if c not in drop]) if variant=='shift_feature_ablation' else ds
                    model=make_model(fitds,'sgd_logistic',seed)
                    if variant=='l2_logistic':model.set_params(model__penalty='l2')
                    model.fit(split['X_train'],split['y_train'])
                    numeric,categorical=ref.selected_covariates(ds)
                    f,_,_,_=ref.role_frames(ds,split,model,shift.name,seed,numeric,categorical)
                    assert f.row_id.tolist()==original.row_id.tolist()
                    assert f.role.tolist()==original.role.tolist()
                    assert f.label.tolist()==original.label.tolist()
                    source=f[f.role=='source_calibration'].reset_index(drop=True)
                    pool=f[f.role=='target_calibration_pool'].reset_index(drop=True)
                    test=f[f.role=='target_test'].reset_index(drop=True)
                    for rep in range(5):
                        payload='|'.join(map(str,('jbi-budget-v1',ds.name,shift.name,seed,rep)))
                        drawseed=int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big')
                        n=max(1,int(round(.1*(len(pool)+len(test)))))
                        cal=pool.iloc[np.random.default_rng(drawseed).permutation(len(pool))[:n]]
                        data=CalibrationInput(source.prob_1.to_numpy(),source.label.to_numpy(int),cal.prob_1.to_numpy(),cal.label.to_numpy(int),.1,rep)
                        for domain in ('source','target'):
                            fitted=fit_split(data,domain)
                            sets=fitted.predict_sets(test.prob_1.to_numpy());labels=test.label.to_numpy(int)
                            # Independent sorted-rank reconstruction, no implementation import.
                            c=source if domain=='source' else cal
                            scores=np.where(c.label.to_numpy()==0,c.prob_1.to_numpy(),1-c.prob_1.to_numpy())
                            k=int(np.ceil((len(scores)+1)*.9));q=np.sort(scores)[k-1] if k<=len(scores) else np.inf
                            tp=test.prob_1.to_numpy();independent=np.column_stack([tp<=q,1-tp<=q])
                            assert np.array_equal(sets,independent)
                            metric=set_metrics(sets,labels)
                            rows.append(dict(dataset=ds.name,shift=shift.name,seed=seed,replicate=rep,
                                             variant=variant,method=domain+'_split',**metric,
                                             shortfall=max(0.,.9-metric['coverage'])))
                    fits.append(dict(dataset=ds.name,shift=shift.name,seed=seed,variant=variant,
                                     dropped='|'.join(drop) if variant=='shift_feature_ablation' else '',
                                     n_train=len(split['y_train']),n_test=len(test),
                                     frozen_roles_equal=True,reference_prediction_sha256=pair['prediction_sha256']))
                    pd.DataFrame(rows).to_csv(a.out/'replicate_metrics.partial.csv',index=False)
                    print(variant,ds.name,shift.name,seed,flush=True)
    f=pd.DataFrame(rows);assert len(f)==1200 and len(fits)==120
    f.to_csv(a.out/'replicate_metrics.csv',index=False)
    pd.DataFrame(fits).to_csv(a.out/'fit_provenance.csv',index=False)
    cols=['coverage','shortfall','mean_set_size','class_coverage_gap','empty_set_rate','singleton_rate','full_set_rate']
    condition=f.groupby(['dataset','shift','seed','variant','method'])[cols].mean().groupby(['dataset','shift','variant','method']).mean().reset_index()
    condition.to_csv(a.out/'condition_summary.csv',index=False)
    datasets=condition.groupby(['dataset','variant','method'])[cols].mean().reset_index()
    summary=datasets.groupby(['variant','method'])[cols].mean().reset_index()
    summary.to_csv(a.out/'equal_dataset_summary.csv',index=False)
    receipt={'status':'completed','model_fits':120,'measurements':1200,'all_original_conditions':True,
             'roles_match_verified_primary':True,'sorted_rank_membership_reconstruction':True,
             'protocol_sha256':sha(pp),'outputs':{p.name:sha(p) for p in a.out.glob('*.csv') if '.partial.' not in p.name}}
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (a.out/'replicate_metrics.partial.csv').unlink()
    print(summary.to_string(index=False))


if __name__=='__main__':main()
