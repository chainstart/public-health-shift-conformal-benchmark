"""Reconstruct submitted-protocol summaries from immutable archived run metrics.

No model fitting, calibration changes, or edits to the archived release.
Run inside the OpenLabs CPU resource guard.
"""
from pathlib import Path
import argparse, hashlib, json, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import train_test_split

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.release/'results/public_benchmark_conformal_results.csv')
    primary=raw[(raw.experiment=='primary')&(raw.ablation=='none')&(raw.shift_name!='pseudo_random')].copy()
    controls=primary[(primary.setting=='random_exchangeable')&(primary.workflow=='source_split')]
    shifted=primary[primary.setting=='covariate_shift'].copy()
    keys=['dataset','shift_name','seed']
    source=shifted[shifted.workflow=='source_split'];target=shifted[(shifted.workflow=='target_recalibrated_split')&(shifted.target_cal_fraction==.1)]
    mondrian=shifted[(shifted.workflow=='target_recalibrated_mondrian')&(shifted.target_cal_fraction==.1)]
    weighted=shifted[shifted.workflow=='source_weighted_split']
    assert len(controls)==15 and all(len(x)==60 for x in (source,target,mondrian,weighted))
    checks=[]
    def check(name,value,expected,digits=4):
        actual=f'{value:.{digits}f}'; ok=actual==expected
        checks.append({'claim':name,'computed':float(value),'display':actual,'submitted_display':expected,'matches':ok})
    check('Random-control coverage median',controls.coverage.median(),'0.9009')
    check('Shifted source coverage median',source.coverage.median(),'0.8620')
    check('Shifted source shortfall median',source.coverage_shortfall.median(),'0.0380')
    check('Target10 ordinary coverage median',target.coverage.median(),'0.9011')
    check('Target10 ordinary shortfall median',target.coverage_shortfall.median(),'0.0000')
    check('Target10 ordinary shortfall mean',target.coverage_shortfall.mean(),'0.0027')
    check('Target10 Mondrian coverage median',mondrian.coverage.median(),'0.9012')
    check('Target10 Mondrian class-gap median',mondrian.class_coverage_gap.median(),'0.0100')
    check('Weighted-source coverage median',weighted.coverage.median(),'0.8620')
    for dataset,shift,expected in [('nhanes_multicycle_diabetes','age_65plus','0.3645'),('brfss_multiyear_diabetes','obesity','0.6902')]:
        check(dataset+' '+shift+' source mean',source[(source.dataset==dataset)&(source.shift_name==shift)].coverage.mean(),expected)
    budget=[]
    for (method,fraction),g in shifted[shifted.workflow.str.startswith('target_recalibrated')].groupby(['workflow','target_cal_fraction']):
        assert len(g)==60
        budget.append({'workflow':method,'fraction':fraction,'coverage':g.coverage.median(),'mean_shortfall':g.coverage_shortfall.mean(),
                       'median_shortfall':g.coverage_shortfall.median(),'below_nominal':int((g.coverage<.9).sum()),
                       'class_gap':g.class_coverage_gap.median(),'mean_set_size':g.mean_set_size.mean(),
                       'minimum_labels':int(g.n_target_cal.min()),'maximum_labels':int(g.n_target_cal.max())})
    budget=pd.DataFrame(budget);budget.to_csv(a.out/'primary-budget-reconstructed.csv',index=False)
    # Every printed numerical cell of submitted Table 3, independently aggregated.
    submitted=[('target_recalibrated_split',.005,['0.9124','0.0032','19','0.0921']),
      ('target_recalibrated_split',.01,['0.9024','0.0058','25','0.0968']),
      ('target_recalibrated_split',.05,['0.9012','0.0032','27','0.1058']),
      ('target_recalibrated_split',.1,['0.9011','0.0027','28','0.1044']),
      ('target_recalibrated_mondrian',.005,['0.9112','0.0043','19','0.0468']),
      ('target_recalibrated_mondrian',.01,['0.9047','0.0050','23','0.0366']),
      ('target_recalibrated_mondrian',.05,['0.9018','0.0028','24','0.0162']),
      ('target_recalibrated_mondrian',.1,['0.9012','0.0023','26','0.0100'])]
    for method,fraction,expected in submitted:
        row=budget[(budget.workflow==method)&(budget.fraction==fraction)].iloc[0]
        for col,old in zip(['coverage','mean_shortfall','below_nominal','class_gap'],expected):
            check(f'Table 3 {method} {fraction} {col}',row[col],old,0 if col=='below_nominal' else 4)
    joined=source[keys+['coverage_error']].merge(controls[['dataset','seed','coverage_error']],on=['dataset','seed'],suffixes=('_source','_control'))
    delta=(joined.coverage_error_source-joined.coverage_error_control).groupby([joined.dataset,joined.shift_name]).mean()
    check('Shift-control condition mean error difference',delta.mean(),'0.1129');check('Shift-control condition median error difference',delta.median(),'0.0493')
    check('Shift-control nominal Wilcoxon p',wilcoxon(delta).pvalue,'0.000488',6)
    paired={}
    for name,other,expected in [('target10',target,('-0.1013','-0.0340','0.0210')),('weighted',weighted,('-0.0344','0.0000104','0.7422'))]:
        j=source[keys+['coverage_shortfall']].merge(other[keys+['coverage_shortfall']],on=keys,suffixes=('_source','_other'))
        d=(j.coverage_shortfall_other-j.coverage_shortfall_source).groupby([j.dataset,j.shift_name]).mean()
        paired[name]=d
        check(name+' condition mean shortfall difference',d.mean(),expected[0]);check(name+' condition median shortfall difference',d.median(),expected[1],7 if name=='weighted' else 4)
        check(name+' nominal Wilcoxon p',wilcoxon(d).pvalue,expected[2])
    diag=pd.read_csv(a.release/'tables/mechanism_score_diagnostics_run_level.csv');assert len(diag)==60
    dg=diag.groupby(['dataset','shift_name']).mean(numeric_only=True)
    check('Threshold correlation, sixty pairs',spearmanr(diag.oracle_minus_source_qhat,diag.source_shortfall).statistic,'0.946',3)
    check('Threshold correlation, twelve means',spearmanr(dg.oracle_minus_source_qhat,dg.source_shortfall).statistic,'0.961',3)
    hetero=pd.read_csv(a.release/'tables/shift_heterogeneity_diagnostics.csv')
    check('Prevalence gap versus source shortfall',spearmanr(hetero.abs_prev_diff,hetero.source_short_mean).statistic,'0.826',3)
    check('Prevalence gap versus shortfall reduction',spearmanr(hetero.abs_prev_diff,hetero.shortfall_reduction).statistic,'0.741',3)
    ab=raw[raw.experiment=='drop_defining_features']
    ab_cond=ab.groupby(['dataset','shift_name','workflow']).coverage_shortfall.median().reset_index()
    check('Ablation source: mean of twelve condition medians',ab_cond[ab_cond.workflow=='source_split'].coverage_shortfall.mean(),'0.0609')
    check('Ablation target: mean of twelve condition medians',ab_cond[ab_cond.workflow=='target_recalibrated_split'].coverage_shortfall.mean(),'0.0006')
    ab.groupby('workflow').coverage_shortfall.mean().to_csv(a.out/'ablation-mean-run-shortfall-additional-estimand.csv')
    # Reconstruct stratified calibration class counts from exact domain counts.
    inventory=pd.read_csv(a.release/'results/public_benchmark_shift_audit.csv');counts=[]
    for _,row in inventory[inventory.shift_name!='pseudo_random'].iterrows():
        n=int(row.target_n);n1=round(n*row.target_positive_prevalence);y=np.r_[np.zeros(n-n1,dtype=np.int8),np.ones(n1,dtype=np.int8)]
        for seed in range(5):
            pool,test=train_test_split(y,test_size=.7,stratify=y,random_state=seed)
            for f in (.005,.01,.02,.03,.05,.1,.15,.2):
                k=min(max(100,int(round(f*n))),len(pool))
                _,cal=train_test_split(pool,test_size=k,stratify=pool,random_state=seed+int(f*100000))
                matched=shifted[(shifted.dataset==row.dataset)&(shifted.shift_name==row.shift_name)&(shifted.seed==seed)&(shifted.target_cal_fraction==f)]
                assert matched.n_target_cal.eq(k).all() and len(matched)==2
                rec={'dataset':row.dataset,'shift_name':row.shift_name,'seed':seed,'fraction':f,'labels':k,'class_0':int((cal==0).sum()),'class_1':int((cal==1).sum())}
                counts.append(rec)
                if f==.1:
                    q=diag[(diag.dataset==row.dataset)&(diag.shift_name==row.shift_name)&(diag.seed==seed)].iloc[0]
                    assert rec['class_1']==q.target_cal_positive and rec['class_0']==q.target_cal_negative
    counts=pd.DataFrame(counts)
    counts['class0_uses_pooled_threshold']=counts.class_0<20
    counts['class1_uses_pooled_threshold']=counts.class_1<20
    counts.to_csv(a.out/'primary-calibration-counts-reconstructed.csv',index=False)
    # Empirical summaries use the same source-model family and information access as the primary release.
    eff=shifted[(shifted.workflow.isin(['source_split','source_mondrian','source_weighted_split']))|((shifted.target_cal_fraction==.1)&shifted.workflow.str.startswith('target_recalibrated'))]
    cols=['coverage','coverage_shortfall','mean_set_size','empty_rate','singleton_rate','ambiguous_rate','negative_coverage','positive_coverage','class_coverage_gap']
    eff.groupby(['dataset','shift_name','workflow'])[cols].mean().groupby('workflow').mean().to_csv(a.out/'primary-efficiency-condition-means.csv')
    eff.groupby(['dataset','shift_name','workflow'])[cols].mean().groupby(['dataset','workflow']).mean().groupby('workflow').mean().to_csv(a.out/'primary-efficiency-equal-dataset-means.csv')
    pd.DataFrame(checks).to_csv(a.out/'submitted-numeric-claim-audit.csv',index=False)
    receipt={'basis':'Archived original per-run outputs; no new fitting, no replacement of methods or observations',
       'primary_rows':len(shifted),'random_control_rows':len(controls),'unique_shift_seed_pairs':len(source),
       'checks':len(checks),'matching_checks':sum(x['matches'] for x in checks),'mismatches':[x for x in checks if not x['matches']],
       'all_original_primary_medians_reconstructed':all(x['matches'] for x in checks[:11]),
       'calibration_count_rows':len(counts),'count_matches_to_archived_target_runs':len(counts)*2,'independent_10pct_class_count_checks':120,
       'primary_target_label_range':[int(counts.labels.min()),int(counts.labels.max())],
       'minimum_source_calibration_count':int(source.n_source_cal.min()),
       'overflow_counterexample_n8_applies_to_reported_ordinary_primary_runs':False,
       'archived_execution_independently_rerun':False,'row_level_predictions_reconstructed':False,
       'bootstrap_intervals':'Retained archived condition-resampling intervals; not population inference over independent deployments',
       'inputs':{str(p.relative_to(a.release)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.release/'results/public_benchmark_conformal_results.csv',a.release/'tables/mechanism_score_diagnostics_run_level.csv',a.release/'results/public_benchmark_shift_audit.csv']}}
    (a.out/'audit-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
