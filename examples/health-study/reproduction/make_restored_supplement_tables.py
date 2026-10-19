"""Generate retained ablation/L2/control/threshold tables from reference CSVs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DS={'brfss_multiyear_diabetes':'BRFSS','nhanes_multicycle_diabetes':'NHANES','uci_diabetes_readmission':'UCI'}
SH={'time_future':'Later years','region_south':'South region','age_65plus':'Age 65+', 'obesity':'BMI threshold','race_black_vs_white':'Race proxy','age_older70':'Age 70+','race_african_american':'Race proxy','emergency_admission':'Admission source'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--paper-root',type=Path,required=True);a=ap.parse_args()
    r=a.results;out=a.paper_root/'manuscript/tables';out.mkdir(parents=True,exist_ok=True)
    def table(name,head,rows,align):
        s=['\\begin{tabular}{'+align+'}',r'\toprule',' & '.join(head)+r' \\',r'\midrule']
        s+=[' & '.join(map(str,row))+r' \\' for row in rows];s += [r'\bottomrule',r'\end{tabular}']
        (out/name).write_text('\n'.join(s)+'\n')
    g=pd.read_csv(r/'restored_submitted_sensitivities/condition_summary.csv')
    for variant,name in [('shift_feature_ablation','restored_ablation.tex'),('l2_logistic','restored_l2.tex')]:
        rows=[]
        for (ds,shift),f in g[g.variant==variant].groupby(['dataset','shift']):
            f=f.set_index('method');s=f.loc['source_split'];t=f.loc['target_split']
            rows.append([DS[ds]+' '+SH[shift],f'{s.coverage:.4f}',f'{s.shortfall:.4f}',f'{t.coverage:.4f}',f'{t.shortfall:.4f}',f'{t.mean_set_size:.4f}'])
        table(name,['Condition','Source cov.','Source shortfall','Target cov.','Target shortfall','Target size'],rows,'lrrrrr')
    controls=pd.read_csv(r/'restored_random_controls/control_metrics.csv')
    table('restored_random_controls.tex',['Data','Seed','Train $N$','Cal. $N$','Test $N$','Coverage','Size'],[[DS[x.dataset],x.seed,int(x.n_train),int(x.n_calibration),int(x.n_test),f'{x.coverage:.4f}',f'{x.mean_set_size:.4f}'] for _,x in controls.iterrows()],'lrrrrrr')
    table('restored_clipped_weighting.tex',['Condition','Coverage','Size','Empty','Singleton','Two-label','Class gap'],[[DS[x.dataset]+' '+SH[x['shift']],*[f'{x[k]:.4f}' for k in ['coverage','mean_set_size','empty_set_rate','singleton_rate','full_set_rate','class_coverage_gap']]] for _,x in pd.read_csv(r/'restored_clipped_weighting/condition_summary.csv').iterrows()],'lrrrrrr')
    dg=pd.read_csv(r/'restored_threshold_diagnostic/threshold_diagnostic_by_condition.csv')
    table('restored_thresholds.tex',['Condition','Source $q$','Test $q_{.9}$','Target10 $q$','Source shortfall'],[[DS[x.dataset]+' '+SH[x['shift']],*[f'{x[k]:.4f}' for k in ['source_threshold','diagnostic_target_q90','target10_threshold_mean','shortfall']]] for _,x in dg.iterrows()],'lrrrr')
    budget=r/'restored_original_budgets'
    if not budget.exists():budget=r/'budgets'
    b=pd.read_csv(budget/'budget_metrics.csv');b=b[b.fraction==.1]
    b['shortfall']=np.maximum(0.,.9-b.coverage);b['coverage_error']=abs(b.coverage-.9)
    seed=b.groupby(['dataset','shift','seed','method']).mean(numeric_only=True).reset_index()
    source=seed[seed.method=='source_split'].merge(controls[['dataset','seed','coverage']],on=['dataset','seed'],suffixes=('','_control'),validate='many_to_one')
    source['effect']=source.coverage_error-abs(source.coverage_control-.9);er=source.groupby(['dataset','shift']).effect.mean()
    pair=seed.pivot(index=['dataset','shift','seed'],columns='method',values='shortfall');target=(pair.target_split-pair.source_split).groupby(['dataset','shift']).mean()
    clip=pd.read_csv(r/'restored_clipped_weighting/metrics.csv').merge(seed[seed.method=='source_split'][['dataset','shift','seed','shortfall']],on=['dataset','shift','seed'],suffixes=('','_source'),validate='one_to_one')
    clip['effect']=clip.shortfall-clip.shortfall_source;weighted=clip.groupby(['dataset','shift']).effect.mean()
    rows=[]
    for title,v in [('Shift minus random-control coverage error',er),('Target10 minus source shortfall',target),('Clipped weighting minus source shortfall',weighted)]:
        rows.append([title,f'{v.mean():.4f}',f'{v.median():.4f}',f'[{v.min():.4f}, {v.max():.4f}]'])
    table('original_paired_descriptive.tex',['Comparison','Mean','Median','Condition range'],rows,'lrrr')
    print('Generated retained submitted-analysis tables and descriptive paired effects.')


if __name__=='__main__':main()
