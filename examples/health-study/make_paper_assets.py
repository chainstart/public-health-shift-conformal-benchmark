"""Generate manuscript tables, metrics and figures from reference CSVs only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DS={'brfss_multiyear_diabetes':'BRFSS','nhanes_multicycle_diabetes':'NHANES','uci_diabetes_readmission':'UCI'}
SH={'time_future':'Later years','region_south':'South region','age_65plus':'Age 65+','obesity':'BMI threshold',
    'race_black_vs_white':'Race proxy','age_older70':'Age 70+','race_african_american':'Race proxy','emergency_admission':'Admission source'}
NAMES={'source_split':'Source split','source_mondrian':'Source Mondrian','target_split':'Target split','target_mondrian':'Target Mondrian'}
MATCHED={'source_split_full':'Source split, full','source_mondrian_full':'Source Mondrian, full','source_split_300':'Source split, 300',
 'target_split_300':'Target split, 300','target_mondrian_300':'Target Mondrian, 300',
 'label_shift_weighted_300':'Label-shift weighting','score_covariate_weighted_300':'Score-space weighting',
 'ot_minmax_weighted_cdf_300':'Optimal-transport weighting','lrqr_300_300_300':'LR-QR (extended budget)'}

def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(path,headers,rows,fmt):
    text=[r'\begin{tabular}{'+fmt+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    text+=[' & '.join(map(str,row))+r' \\' for row in rows];text+=[r'\bottomrule',r'\end{tabular}']
    path.write_text('\n'.join(text)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paper-root',type=Path,required=True);a=ap.parse_args()
    paper=a.paper_root;root=Path(__file__).parent;results=root/'results';man=paper/'manuscript';tabs=man/'tables';figs=man/'figures'
    upload=paper/'production/jbi-26-3391-r1/figures'
    for d in [tabs,figs,upload]:d.mkdir(parents=True,exist_ok=True)
    eq=pd.read_csv(results/'budgets/budget_equal_dataset.csv');primary=eq[eq.fraction==.1].set_index('method')
    matched=pd.read_csv(results/'matched/extended_equal_dataset_summary.csv').set_index('method')
    adult=pd.read_csv(results/'adult_sensitivity/adult_summary.csv').set_index('method')
    control=pd.read_csv(results/'controls/control_summary.csv').set_index('method')
    nonlinear=pd.read_csv(results/'nonlinear/nonlinear_summary.csv').set_index('method')
    domains=pd.read_csv(results/'data_preparation/cohort_shift_definitions.csv')
    budget=pd.read_csv(results/'budgets/absolute_budget_summary.csv')
    shifts=pd.read_csv(results/'budgets/budget_by_shift.csv');shifts=shifts[shifts.fraction==.1]
    strong=pd.read_csv(results/'matched/extended_shift_summary.csv')
    decision=pd.read_csv(results/'decision_tradeoffs/equal_dataset_diagnostics.csv').set_index('method')
    floors=pd.read_csv(results/'decision_tradeoffs/floor_sensitivity.csv')
    class_conditions=pd.read_csv(results/'decision_tradeoffs/condition_diagnostics.csv')
    rows=[]
    definitions={('BRFSS','time_future'):('2011, 2013, 2015','2017, 2019, 2021, 2023','Temporal'),('BRFSS','region_south'):('Other US regions','South','Geographic'),
                 ('BRFSS','age_65plus'):('18--64','65+','Stress test'),('NHANES','time_future'):('2011--2014','2015--2018','Temporal'),
                 ('NHANES','age_65plus'):('Below 65 (all ages)','65+','Stress test'),('UCI','age_older70'):('Below 70','70+','Stress test'),
                 ('UCI','race_african_american'):('Caucasian','AfricanAmerican','Stress test'),('UCI','emergency_admission'):('Other admission sources','Emergency room','Stress test'),
                 ('BRFSS','obesity'):('Other/unknown BMI category','BMI category 4 ($\\geq30$ kg/m$^2$)','Stress test'),
                 ('NHANES','obesity'):('BMI below 30 or missing','BMI $\\geq30$ kg/m$^2$','Stress test')}
    for _,r in domains.iterrows():
        d=DS[r.dataset];s,t,kind=definitions.get((d,r.shift_name),('Other/unknown BMI','High BMI','Stress test') if r.shift_name=='obesity' else ('White proxy','Black proxy','Stress test'))
        rows.append([d,s+' $\\rightarrow$ '+t,f'{r.source_n:,}',f'{r.target_n:,}',f'{100*r.source_prevalence:.1f}',f'{100*r.target_prevalence:.1f}',kind])
    table(tabs/'conditions.tex',['Data','Source $\\rightarrow$ target','$N_S$','$N_T$','$p_S$ (\\%)','$p_T$ (\\%)','Interpretation'],rows,'lp{4.7cm}rrrrl')
    rows=[]
    for _,r in budget[budget.fraction==.1].iterrows():
        group=budget[(budget.dataset==r.dataset)&(budget['shift']==r['shift'])]
        rows.append([DS[r.dataset]+' '+SH[r['shift']],f'{int(group.n_target_labels.min()):,}--{int(group.n_target_labels.max()):,}',
                     f'{r.n_target_labels:,}',f'{r.class0_min:,}--{r.class0_max:,}',f'{r.class1_min:,}--{r.class1_max:,}'])
    table(tabs/'absolute_budgets.tex',['Condition','0.5--20\\% labels','10\\% labels','10\\% class 0','10\\% class 1'],rows,'lrrrr')
    rows=[]
    for m in NAMES:
        r=primary.loc[m];rows.append([NAMES[m],*[f'{r[c]:.4f}' for c in ['coverage','mean_set_size','empty_set_rate','singleton_rate','full_set_rate','class_coverage_gap']]])
    table(tabs/'primary_efficiency.tex',['Method','Coverage','Size','Empty','Singleton','Two-label','Class gap'],rows,'lrrrrrr')
    rows=[]
    order=['source_split_full','source_mondrian_full','source_split_300','label_shift_weighted_300','score_covariate_weighted_300','ot_minmax_weighted_cdf_300','lrqr_300_300_300','target_split_300','target_mondrian_300']
    for m in order:
        r=matched.loc[m];rows.append([MATCHED[m],*[f'{r[c]:.4f}' for c in ['coverage','mean_set_size','empty_set_rate','singleton_rate','full_set_rate','class_coverage_gap']]])
    table(tabs/'matched_efficiency.tex',['Method','Coverage','Size','Empty','Singleton','Two-label','Class gap'],rows,'lrrrrrr')
    rows=[]
    for _,r in domains.iterrows():
        small=shifts[(shifts.dataset==r.dataset)&(shifts['shift']==r.shift_name)].set_index('method')
        st=strong[(strong.dataset==r.dataset)&(strong['shift']==r.shift_name)].set_index('method')
        rows.append([DS[r.dataset]+' '+SH[r.shift_name],f'{small.loc["source_split","coverage"]:.4f}',f'{small.loc["target_split","coverage"]:.4f}',f'{small.loc["target_mondrian","coverage"]:.4f}',f'{st.loc["ot_minmax_weighted_cdf_300","coverage"]:.4f}',f'{st.loc["lrqr_300_300_300","coverage"]:.4f}'])
    table(tabs/'condition_coverage.tex',['Condition','Source split, full','Target split 10\\%','Target M. 10\\%','OT 300','LR-QR'],rows,'lrrrrr')
    rows=[]
    for name,frame in [('Elastic-net',primary),('LightGBM',nonlinear)]:
        for m in NAMES:rows.append([name,NAMES[m],*[f'{frame.loc[m,c]:.4f}' for c in ['coverage','mean_set_size','class_coverage_gap']]])
    table(tabs/'backbone.tex',['Predictor','Method','Coverage','Size','Class gap'],rows,'llrrr')
    rows=[[NAMES[m],*[f'{adult.loc[m,c]:.4f}' for c in ['coverage','mean_set_size','class_coverage_gap']]] for m in NAMES]
    table(tabs/'adult.tex',['Method','Coverage','Size','Class gap'],rows,'lrrr')
    # Full absolute/class budget table and subgroup tables are supplementary.
    btext=[r'\begin{longtable}{llrrrrr}',r'\caption{Absolute target labels and per-class ranges across five seeds and five draws.}\\',r'\toprule Data & Condition & Fraction & Labels & Class 0 & Class 1 & Overflow 1 \\ \midrule\endfirsthead']
    btext.append(r'\toprule Data & Condition & Fraction & Labels & Class 0 & Class 1 & Overflow 1 \\ \midrule\endhead')
    for _,r in budget.iterrows():
        btext.append(' & '.join([DS[r.dataset],SH[r['shift']],f'{100*r.fraction:g}\\%',str(int(r.n_target_labels)),f'{r.class0_min}--{r.class0_max}',f'{r.class1_min}--{r.class1_max}',str(int(r.overflow1_draws))])+r' \\')
    btext +=[r'\bottomrule\end{longtable}'];(tabs/'supplement_budgets.tex').write_text('\n'.join(btext)+'\n')
    subgroup=pd.read_csv(results/'budgets/subgroup_summary.csv');subgroup=subgroup[subgroup.method.isin(['source_split','target_split','target_mondrian'])]
    text=[r'\begin{longtable}{llllp{3.0cm}rrrr}',r'\caption{Target-test subgroups at the 10\% target-label setting. Counts are seed ranges; prevalences and metrics are means.}\\',r'\toprule Data & Condition & Method & Variable & Subgroup & $N$ & Prevalence & Coverage & Size \\ \midrule\endfirsthead']
    text.append(r'\toprule Data & Condition & Method & Variable & Subgroup & $N$ & Prevalence & Coverage & Size \\ \midrule\endhead')
    for _,r in subgroup.iterrows():
        value=r.value
        if r.dataset=='nhanes_multicycle_diabetes' and value=='18_64':value='below 65 (all ages)'
        dimension={'group_age_group':'Age','group_sex_group':'Sex','group_race_group':'Race'}.get(r['group'],tex(r['group']))
        text.append(' & '.join([DS[r.dataset],SH[r['shift']],NAMES[r.method].replace('Source ','S. ').replace('Target ','T. '),dimension,tex(value),f'{r.n_test_min}--{r.n_test_max}',f'{r.prevalence:.3f}',f'{r.coverage:.3f}',f'{r.mean_set_size:.3f}'])+r' \\')
    text+=[r'\bottomrule\end{longtable}'];(tabs/'supplement_subgroups.tex').write_text('\n'.join(text)+'\n')
    rank=pd.read_csv(results/'controls/finite_rank_examples.csv')
    table(tabs/'rank_examples.tex',['$n$','$k_n$','Expected coverage','Threshold SD'],[[int(r.n),int(r['rank']),f'{r.expected_coverage_continuous_iid:.4f}',('N/A' if pd.isna(r.threshold_sd_uniform_scores) else f'{r.threshold_sd_uniform_scores:.4f}')] for _,r in rank.iterrows()],'rrrr')
    rows=[]
    for m in order:
        f=floors[floors.method==m].set_index('floor')
        rows.append([MATCHED[m],f'{f.loc[.88,"worst_condition_class_mean"]:.4f}',
            *[str(int(f.loc[x,'conditions_passing'])) for x in [.85,.88,.90]],
            f'{decision.loc[m,"correct_singleton_rate"]:.4f}',f'{decision.loc[m,"singleton_error_rate"]:.4f}'])
    table(tabs/'decision_tradeoffs.tex',['Method','Worst class','85\\%','88\\%','90\\%','Correct single','Incorrect single'],rows,'lrrrrrr')
    # Macros are the only manually referenced headline quantities.
    macros={}
    for prefix,frame in [('Primary',primary),('Adult',adult),('Nonlinear',nonlinear),('Control',control)]:
        for m in frame.index:
            for c in ['coverage','mean_set_size','class_coverage_gap']:
                key=prefix+''.join(x.title() for x in m.split('_'))+''.join(x.title() for x in c.split('_'))
                macros[key]=f'{frame.loc[m,c]:.4f}'
    for m in ['ot_minmax_weighted_cdf_300','lrqr_300_300_300','score_covariate_weighted_300']:
        prefix={'ot_minmax_weighted_cdf_300':'OT','lrqr_300_300_300':'LRQR','score_covariate_weighted_300':'Weighted'}[m]
        for c in ['coverage','mean_set_size','class_coverage_gap']:macros[prefix+''.join(x.title() for x in c.split('_'))]=f'{matched.loc[m,c]:.4f}'
        v=strong[strong.method==m].coverage;macros[prefix+'MinCoverage']=f'{v.min():.4f}';macros[prefix+'MaxCoverage']=f'{v.max():.4f}'
    b10=budget[budget.fraction==.1];macros['BudgetMin']=f'{int(b10.n_target_labels.min()):,}';macros['BudgetMax']=f'{int(b10.n_target_labels.max()):,}'
    for m,prefix in [('target_mondrian_300','DecisionMondrian'),('target_split_300','DecisionSplit'),
                     ('score_covariate_weighted_300','DecisionWeighted'),('ot_minmax_weighted_cdf_300','DecisionOT'),('lrqr_300_300_300','DecisionLRQR')]:
        f=floors[(floors.method==m)&(floors.floor==.88)].iloc[0]
        macros[prefix+'WorstClass']=f'{f.worst_condition_class_mean:.4f}'
        macros[prefix+'PassCount']=str(int(f.conditions_passing))
        macros[prefix+'CorrectSingleton']=f'{decision.loc[m,"correct_singleton_rate"]:.4f}'
        macros[prefix+'Singleton']=f'{decision.loc[m,"singleton_rate"]:.4f}'
        macros[prefix+'FullSet']=f'{decision.loc[m,"full_set_rate"]:.4f}'
    pv=shifts.pivot(index=['dataset','shift'],columns='method',values='coverage');effects=pv.target_mondrian-pv.source_mondrian
    macros['NegativeConditions']=str(int((effects<0).sum()));macros['MinEffect']=f'{effects.min():.4f}';macros['MaxEffect']=f'{effects.max():.4f}'
    (tabs/'metrics.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    def save(fig,name):
        fig.savefig(figs/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(upload/(name+'.eps'),bbox_inches='tight')
        fig.savefig(upload/(name+'.tif'),dpi=400,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.2,3.5),layout='constrained')
    colors=['#6b7280','#92400e','#2563eb','#047857']
    for ax,metric,label in zip(axes,['coverage','mean_set_size','class_coverage_gap'],['Marginal coverage','Mean prediction-set size','Class-coverage gap']):
        for m,col in zip(NAMES,colors):
            f=eq[eq.method==m].sort_values('fraction');ax.plot(f.fraction*100,f[metric],marker='o',color=col,label=NAMES[m])
        ax.set_xscale('log');ax.set_xticks([.5,1,2,5,10,20],['0.5','1','2','5','10','20']);ax.set_xlabel('Labels (% of target domain)');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].axhline(.9,color='black',ls='--',lw=.8);axes[0].legend(fontsize=8,loc='lower right');save(fig,'budget_tradeoffs')
    fig,ax=plt.subplots(figsize=(10.5,5.7),layout='constrained')
    pivot=shifts.pivot(index=['dataset','shift'],columns='method',values='coverage').sort_index();positions=np.arange(len(pivot))
    ax.hlines(positions,pivot.source_mondrian,pivot.target_mondrian,color='#cbd5e1',lw=2)
    ax.scatter(pivot.source_mondrian,positions,color=colors[1],label='Source Mondrian (full)')
    ax.scatter(pivot.target_mondrian,positions,color=colors[3],label='Target Mondrian (10%)')
    ax.axvline(.9,color='black',ls='--',lw=.8);ax.set_yticks(positions,[DS[d]+': '+SH[s] for d,s in pivot.index]);ax.invert_yaxis();ax.set_xlabel('Coverage on the same target test role');ax.set_xlim(0,1.02);ax.legend(loc='lower left');ax.grid(axis='x',alpha=.2);save(fig,'condition_effects')
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.2),sharey=True,layout='constrained')
    pos=np.arange(len(order));palette=[plt.cm.tab10(i%10) for i in pos]
    for ax,column,label,limits in zip(axes,['coverage','mean_set_size'],['Marginal coverage','Mean prediction-set size'],[(.7,1.0),(0,2.0)]):
        ax.barh(pos,matched.loc[order,column],color=palette,height=.65)
        ax.set_xlabel(label);ax.set_xlim(*limits);ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(pos,[MATCHED[m] for m in order]);axes[0].invert_yaxis()
    axes[0].axvline(.9,color='black',ls='--',lw=.8);save(fig,'matched_tradeoffs')
    class_pivot=class_conditions.pivot(index=['dataset','shift'],columns='method',values='minimum_of_class_means').sort_index()[order]
    fig,ax=plt.subplots(figsize=(12.4,7.8),layout='constrained')
    im=ax.imshow(class_pivot.to_numpy(),vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_xticks(np.arange(len(order)),[MATCHED[m].replace(' (extended budget)','') for m in order],rotation=45,ha='right',fontsize=9)
    ax.set_yticks(np.arange(len(class_pivot)),[DS[d]+': '+SH[s] for d,s in class_pivot.index])
    for i in range(len(class_pivot)):
        for j in range(len(order)):
            value=class_pivot.iloc[i,j]
            ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white' if value<.65 else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='Minimum of the two condition-mean class coverages')
    save(fig,'class_coverage_conditions')
    fig,ax=plt.subplots(figsize=(12.4,5.3));ax.set_xlim(0,12.4);ax.set_ylim(0,5.3);ax.axis('off')
    boxes=[(.2,1.1,2.6,3.4,'PUBLIC DATA\n\nBRFSS: 420,000\nNHANES: 36,943\nUCI: 71,515\n\n12 source–target\nconditions'),(3.2,1.1,2.6,3.4,'FIXED PREDICTOR\n\nFit on source only\n\nKeep target test\nlabels separate\n\nCompare calibration\ninformation'),(6.2,1.1,2.6,3.4,'TARGET CALIBRATION\n\n0.5%–20% labels\n\nReport absolute and\nper-class counts\n\nCoverage + set size\n+ class differences'),(9.2,1.1,3.0,3.4,'BENCHMARK RESULT\n\nSource coverage: '+macros['PrimarySourceSplitCoverage']+'\nTarget Mondrian: '+macros['PrimaryTargetMondrianCoverage']+'\n\nMean set size: '+macros['PrimaryTargetMondrianMeanSetSize']+'\n\n10% labeled target;\ndescriptive averages')]
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor='#eff6ff',edgecolor='#1e3a8a',lw=1.3));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12)
    for x in [2.82,5.82,8.82]:ax.annotate('',xy=(x+.31,2.8),xytext=(x,2.8),arrowprops={'arrowstyle':'->','lw':1.7})
    ax.text(6.2,.45,'Near-nominal coverage has a labeling and prediction-set-size cost.',ha='center',fontsize=13);save(fig,'graphical_abstract')
    inputs={str(p.relative_to(results)):hashlib.sha256(p.read_bytes()).hexdigest() for p in results.rglob('*.csv')}
    outputs={str(p.relative_to(man)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(tabs.glob('*.tex'))+list(figs.glob('*.pdf'))}
    (root/'paper-assets-receipt.json').write_text(json.dumps({'status':'completed','inputs':inputs,'outputs':outputs,'macros':macros,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('Generated tables, numerical macros, five PDF/EPS/TIFF figures and hash receipt.')

if __name__=='__main__':main()
