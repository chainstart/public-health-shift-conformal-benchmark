"""Generate the retained submitted result figures and budget table from CSVs."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path(__file__).parent/'results')
    ap.add_argument('--paper-root',type=Path,required=True);a=ap.parse_args()
    m=a.paper_root/'manuscript';(m/'figures').mkdir(parents=True,exist_ok=True);(m/'tables').mkdir(exist_ok=True)
    budget_dir=a.results/'restored_original_budgets'
    if not (budget_dir/'receipt.json').exists():budget_dir=a.results/'budgets'
    f=pd.read_csv(budget_dir/'budget_metrics.csv')
    f['shortfall']=np.maximum(0.,.9-f.coverage)
    metrics=['coverage','shortfall','class_coverage_gap','mean_set_size']
    seed=f.groupby(['dataset','shift','seed','fraction','method'])[metrics].mean().reset_index()
    primary=seed[seed.fraction==.1]
    random_controls=a.results/'restored_random_controls/control_metrics.csv'
    if random_controls.exists():control=pd.read_csv(random_controls)
    else:
        c=pd.read_csv(a.results/'controls/control_metrics.csv')
        control=c[c.method=='source_split'].groupby(['dataset','shift','seed']).coverage.mean().reset_index()
    fig,axes=plt.subplots(1,3,figsize=(11,4),sharey=True)
    colors=['#0072B2','#D55E00','#009E73']
    for ax,dataset in zip(axes,sorted(primary.dataset.unique())):
        groups=[control[control.dataset==dataset].coverage.to_numpy(),
                primary[(primary.dataset==dataset)&(primary.method=='source_split')].coverage.to_numpy(),
                primary[(primary.dataset==dataset)&(primary.method=='target_split')].coverage.to_numpy()]
        bp=ax.boxplot(groups,positions=[1,2,3],widths=.45,patch_artist=True,showfliers=False)
        for box,color in zip(bp['boxes'],colors):box.set_facecolor(color);box.set_alpha(.22)
        for i,(values,color) in enumerate(zip(groups,colors),1):
            ax.scatter(i+np.linspace(-.12,.12,len(values)),values,s=12,c=color,alpha=.65)
        ax.axhline(.9,c='.4',ls='--',lw=1);ax.set_ylim(0,1.03)
        ax.set_xticks([1,2,3],['Random\ncontrol','Source\nshift','Target\n10%'])
        ax.set_title(dataset.split('_')[0].upper());ax.spines[['top','right']].set_visible(False)
        inset=ax.inset_axes([.50,.14,.46,.22])
        for i,(values,color) in enumerate(zip(groups,colors),1):inset.scatter(np.full(len(values),i),values,s=5,c=color,alpha=.5)
        inset.axhline(.9,c='.4',ls='--',lw=.6);inset.set_ylim(.89,.91);inset.set_xlim(.5,3.5);inset.set_xticks([]);inset.tick_params(labelsize=6)
    axes[0].set_ylabel('Marginal coverage');fig.tight_layout()
    fig.savefig(m/'figures/quality_by_task.pdf');plt.close(fig)
    diag=pd.read_csv(a.results/'restored_threshold_diagnostic/threshold_diagnostic_by_seed.csv')
    assert len(diag)==60 and diag[['dataset','shift','seed']].duplicated().sum()==0
    r=float(spearmanr(diag.threshold_gap,diag.shortfall).statistic)
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for (name,g),color in zip(diag.groupby('dataset',sort=True),colors):
        ax.scatter(g.threshold_gap,g.shortfall,c=color,alpha=.72,edgecolors='white',linewidths=.4,label=name.split('_')[0].upper())
    ax.axvline(0,color='.6',lw=.8,ls='--')
    ax.set(xlabel='Diagnostic target-test 90th percentile − source threshold',ylabel='Source-calibrated coverage shortfall',
           title=f'Retrospective threshold diagnostic (60 seed runs; Spearman r = {r:.3f})')
    ax.legend(frameon=False);ax.spines[['right','top']].set_visible(False);fig.tight_layout()
    fig.savefig(m/'figures/score_threshold_mismatch.pdf');plt.close(fig)
    lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Workflow & Fraction & Coverage & Mean shortfall & Runs $<0.90$ & Class gap & Size \\',r'\midrule']
    for (method,frac),g in seed[seed.method.str.startswith('target')].groupby(['method','fraction']):
        label={'target_split':'Target split','target_mondrian':'Target Mondrian'}[method]
        lines.append(f'{label} & {frac:.3f} & {g.coverage.median():.4f} & {g.shortfall.mean():.4f} & {(g.coverage<.9).sum()}/60 & {g.class_coverage_gap.median():.4f} & {g.mean_set_size.mean():.4f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (m/'tables/original_budget_corrected.tex').write_text('\n'.join(lines)+'\n')
    out=a.results/'restored_threshold_diagnostic/presentation-receipt.json'
    receipt={'source':'reference CSVs; all score/label reconstruction remains separately reproducible',
             'seed_summary':'draw means; medians across 60 condition-seed pairs',
             'shortfall':'max(0,.9-coverage) per draw before aggregation',
             'source_control':'Whole-dataset random controls when supplied; not a condition-size-matched reference',
             'outputs':{str(p.relative_to(m)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [m/'figures/quality_by_task.pdf',m/'figures/score_threshold_mismatch.pdf',m/'tables/original_budget_corrected.tex']}}
    out.write_text(json.dumps(receipt,indent=2)+'\n');print('Retained result figures and budget table generated.')


if __name__=='__main__':main()
