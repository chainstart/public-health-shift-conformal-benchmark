"""Generate the label-channel tables and condition plot from complete reference CSVs."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REG={'uniform':'Uniform','outcome_depleted':'Outcome-dependent','risk_priority':'Risk-prioritized'}
NAMES={'available_split':'Available split','available_mondrian':'Available Mondrian','sentinel_split':'Sentinel split',
       'sentinel_mondrian':'Sentinel Mondrian','known_design_rule':'Protocol rule'}
DS={'brfss_multiyear_diabetes':'BRFSS','nhanes_multicycle_diabetes':'NHANES','uci_diabetes_readmission':'UCI'}
SH={'time_future':'time','region_south':'region','age_65plus':'age','obesity':'BMI','race_black_vs_white':'race',
    'age_older70':'age','race_african_american':'race','emergency_admission':'admission'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--paper-root',type=Path,required=True);a=ap.parse_args()
    r=a.results/'label_acquisition';m=a.paper_root/'manuscript';tabs=m/'tables';figs=m/'figures'
    tabs.mkdir(parents=True,exist_ok=True);figs.mkdir(exist_ok=True)
    eq=pd.read_csv(r/'equal_dataset_summary.csv')
    def table(name,head,rows,align):
        s=['\\begin{tabular}{'+align+'}',r'\toprule',' & '.join(head)+r' \\',r'\midrule']
        s+=[' & '.join(map(str,row))+r' \\' for row in rows];s += [r'\bottomrule',r'\end{tabular}']
        (tabs/name).write_text('\n'.join(s)+'\n')
    rows=[]
    for (n,regime),g in eq.groupby(['planned_labels','regime']):
        g=g.set_index('method')
        selected='sentinel_mondrian' if regime=='risk_priority' else 'available_mondrian'
        for method in ['available_split','available_mondrian','sentinel_mondrian']:
            x=g.loc[method]
            rows.append([n,REG[regime],NAMES[method],*[f'{x[k]:.4f}' for k in
                         ['class_0_coverage','class_1_coverage','mean_set_size']],
                         'Yes' if method==selected else ''])
    table('label_acquisition.tex',['Labels','Observation channel','Method','$C_0$','$C_1$','Size','Rule selects'],rows,'rllrrrl')
    rows=[]
    for _,x in eq.iterrows():
        rows.append([int(x.planned_labels),REG[x.regime],NAMES[x.method],*[f'{x[k]:.4f}' for k in
                     ['coverage','class_0_coverage','class_1_coverage','mean_set_size','correct_singleton_rate','incorrect_singleton_rate']]])
    table('label_acquisition_full.tex',['Labels','Channel','Method','Coverage','$C_0$','$C_1$','Size','Correct single','Incorrect single'],rows,'rllrrrrrr')
    draw=pd.read_csv(r/'calibration_draws.csv');rows=[]
    for (n,regime,design),g in draw.groupby(['planned_labels','regime','design']):
        rows.append([n,REG[regime],design,f'{g.n_calibration.min()}--{g.n_calibration.max()}',
                     f'{g.n_class0.min()}--{g.n_class0.max()}',f'{g.n_class1.min()}--{g.n_class1.max()}',int((g.n_class1<9).sum())])
    table('label_acquisition_counts.tex',['Labels','Channel','Design','$N$ used','Class 0','Class 1','Class-1 overflow'],rows,'rllrrrr')
    condition=pd.read_csv(r/'condition_summary.csv');g=condition[(condition.planned_labels==300)&(condition.regime=='risk_priority')]
    pivot=g.pivot(index=['dataset','shift'],columns='method',values='class_1_coverage').sort_index()
    fig,ax=plt.subplots(figsize=(8,5.8));y=np.arange(len(pivot))
    for i,(_,x) in enumerate(pivot.iterrows()):ax.plot([x.available_mondrian,x.sentinel_mondrian],[i,i],c='.72',lw=1.3)
    ax.scatter(pivot.available_mondrian,y,c='#D55E00',s=35,label='Risk-prioritized labels, Mondrian',zorder=3)
    ax.scatter(pivot.sentinel_mondrian,y,c='#009E73',s=35,label='Uniform sentinel labels, Mondrian',zorder=3)
    ax.axvline(.9,c='.4',ls='--',lw=1)
    ax.set_yticks(y,[DS[d]+' '+SH[s] for d,s in pivot.index]);ax.set_xlim(.75,1.0)
    ax.set_xlabel('Positive-outcome prediction-set coverage');ax.invert_yaxis()
    ax.legend(loc='lower right',fontsize=8,frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(figs/'label_acquisition_class_coverage.pdf');plt.close(fig)
    # A small numerical entrypoint for every headline in the added analysis.
    summary={'risk300':eq[(eq.planned_labels==300)&(eq.regime=='risk_priority')].set_index('method').to_dict(orient='index'),
             'condition_minima':{k:float(pivot[k].min()) for k in ['available_mondrian','sentinel_mondrian']},
             'conditions_positive_below90':{k:int((pivot[k]<.9).sum()) for k in ['available_mondrian','sentinel_mondrian']},
             'figures_are_descriptive':True,'all_twelve_conditions_in_plot':len(pivot)==12}
    (r/'headline_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (r/'asset-receipt.json').write_text(json.dumps({'source':'Complete reference CSVs, not selected conditions',
        'outputs':{str(p.relative_to(m)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [tabs/'label_acquisition.tex',tabs/'label_acquisition_full.tex',tabs/'label_acquisition_counts.tex',figs/'label_acquisition_class_coverage.pdf']}},indent=2)+'\n')
    print('Generated label-observation tables and all-condition positive-coverage plot.')


if __name__=='__main__':main()
