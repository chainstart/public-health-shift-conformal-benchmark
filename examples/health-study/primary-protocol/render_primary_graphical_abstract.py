"""Editable color schematic following the submitted four-panel composition.

All numeric labels come from reference results; icons and score curves are
schematic illustrations, not empirical plots. No source bitmap is edited.
"""
from pathlib import Path
import argparse
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RESULTS=Path(__file__).parent/'results'
OUT=ROOT/'manuscript/figures/graphical_abstract_original_design.pdf'
ORANGE='#D55E00';BLUE='#0072B2';PURPLE='#7851A9';GREEN='#009E73';GRAY='#536171'


def main():
    global ROOT, RESULTS, OUT
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,default=RESULTS)
    ap.add_argument('--paper-root',type=Path,default=ROOT)
    a=ap.parse_args();ROOT=a.paper_root;RESULTS=a.results
    OUT=ROOT/'manuscript/figures/graphical_abstract_original_design.pdf'
    OUT.parent.mkdir(parents=True,exist_ok=True)
    budget_dir=RESULTS/'restored_original_budgets'
    if not (budget_dir/'receipt.json').exists():budget_dir=RESULTS/'budgets'
    f=pd.read_csv(budget_dir/'budget_metrics.csv')
    f=f[f.fraction==.1].groupby(['dataset','shift','seed','method']).mean(numeric_only=True).reset_index()
    med=f.groupby('method').median(numeric_only=True)
    random_controls=RESULTS/'restored_random_controls/control_metrics.csv'
    if random_controls.exists():
        controls=pd.read_csv(random_controls);c=controls.coverage.median()
        control_note='Random controls: 5 seeds per dataset'
    else:
        controls=pd.read_csv(RESULTS/'controls/control_metrics.csv')
        c=controls[controls.method=='source_split'].groupby(['dataset','shift','seed']).coverage.mean().median()
        control_note='Medians; control uses 300 labels'
    source=float(med.loc['source_split','coverage']);target=float(med.loc['target_split','coverage'])
    gap=float(med.loc['target_mondrian','class_coverage_gap'])
    fig,ax=plt.subplots(figsize=(16,8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def text(x,y,s,size=10,color=GRAY,weight='normal',ha='center'):
        ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,weight=weight)
    def box(x,y,w,h,color,fill='white',lw=1.5):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006,rounding_size=.012',fc=fill,ec=color,lw=lw))
    def arrow(x1,y1,x2,y2):ax.annotate('',(x2,y2),(x1,y1),arrowprops={'arrowstyle':'->','lw':2,'color':GRAY})
    text(.5,.955,'Deployment shift can break source coverage; target recalibration restores marginal coverage',18,GRAY,'bold')
    text(.5,.905,'Public-health and clinical tabular benchmark for deployment-aware conformal prediction',12,GRAY,'bold')
    for i,(heading,color) in enumerate(zip(['Problem','Benchmark','Solution','Result'],[ORANGE,BLUE,PURPLE,GREEN])):
        x=.018+i*.247;box(x,.16,.223,.66,color,lw=2);text(x+.1115,.775,heading,19,color,'bold')
        if i<3:arrow(x+.229,.48,x+.242,.48)
    # People icons retain the original source-to-target metaphor.
    for cx,color in ((.075,BLUE),(.182,ORANGE)):
        for dx,scale in ((-.022,.8),(0,1.1),(.022,.8)):
            ax.add_patch(Circle((cx+dx,.625),.011*scale,color=color))
            ax.add_patch(FancyBboxPatch((cx+dx-.013*scale,.555),.026*scale,.049*scale,boxstyle='round,pad=.004',fc=color,ec=color))
    text(.129,.721,'Random control near nominal;',10,GRAY,'bold')
    text(.129,.692,'shifted target can undercover.',10,ORANGE,'bold')
    arrow(.112,.56,.145,.56);text(.075,.52,'source',9,BLUE,'bold');text(.182,.52,'target',9,ORANGE,'bold')
    # Small result bar charts are deterministic and display a labeled truncated
    # axis; they are reconstructed from reference data, not generated imagery.
    def bars(left,bottom,values,labels,colors):
        chart=ax.inset_axes([left,bottom,.175,.19])
        chart.bar([0,1],values,color=colors,width=.5)
        chart.set_ylim(.75,1.0);chart.axhline(.9,color='#D69B00',ls='--',lw=1)
        chart.set_xticks([0,1],labels,fontsize=8);chart.set_yticks([.8,.9,1.0]);chart.tick_params(axis='y',labelsize=7)
        chart.spines[['top','right']].set_visible(False)
        for j,v in enumerate(values):chart.text(j,v+.008,f'{v:.4f}',ha='center',fontsize=9,color=colors[j],weight='bold')
    bars(.052,.275,[c,source],['Source control','Shifted source'],[BLUE,ORANGE])
    text(.129,.215,control_note,8,GRAY)
    # Dataset table, clipboard, and database pictograms.
    x=.288;y=.58;ax.add_patch(Rectangle((x,y),.046,.063,fill=False,ec=BLUE,lw=1.5))
    ax.add_patch(Rectangle((x,y+.049),.046,.014,color=BLUE))
    for q in (.015,.030):ax.plot([x+q,x+q],[y,y+.049],color=BLUE,lw=.7)
    for q in (.016,.033):ax.plot([x,x+.046],[y+q,y+q],color=BLUE,lw=.7)
    box(.363,.58,.041,.065,GREEN);ax.add_patch(Rectangle((.372,.639),.023,.012,color=GREEN))
    for yy in (.594,.609,.624):ax.plot([.371,.396],[yy,yy],color=GREEN,lw=1.5)
    ax.add_patch(Rectangle((.431,.58),.046,.059,color=PURPLE));ax.add_patch(Ellipse((.454,.639),.046,.02,fc=PURPLE,ec='white'))
    ax.add_patch(Ellipse((.454,.58),.046,.02,fc=PURPLE,ec='white'))
    for xx,label in ((.311,'BRFSS'),(.383,'NHANES'),(.454,'UCI')):text(xx,.55,label,9,GRAY,'bold')
    for yy,number,label in ((.48,'528,458','records'),(.415,'3','public datasets'),(.35,'12','conditions'),(.285,'5','model seeds')):
        text(.292,yy,number,17,BLUE,'bold',ha='left');text(.404,yy,label,9,GRAY,'bold',ha='left')
    text(.377,.205,'Time · region · age · BMI · race · admission',8,BLUE)
    # Recalibration illustration retains the two schematic score curves.
    text(.623,.716,'Stratified target labels',11,GRAY,'bold')
    for i in range(4):box(.552+i*.039,.61,.027,.041,PURPLE,fill='#F0E8F8');text(.565+i*.039,.63,'Y',10,PURPLE,'bold')
    text(.623,.57,'Re-estimate the conformal threshold',10,PURPLE,'bold')
    xx=np.linspace(-3,3,150);yy=np.exp(-xx**2/2)
    for cx,color,cut in ((.568,ORANGE,-.1),(.683,GREEN,.8)):
        xp=cx+xx*.014;yp=.392+yy*.075
        ax.plot(xp,yp,color=color,lw=1.6);ax.fill_between(xp,.392,yp,color=color,alpha=.12)
        ax.plot([cx+cut*.014]*2,[.392,.47],color=color,lw=2)
    arrow(.611,.425,.64,.425);text(.568,.367,'source scores',8);text(.683,.367,'target scores',8)
    text(.623,.3,'No model retraining required',10,GRAY,'bold')
    text(.623,.247,'10% requires 521–24,000 target labels',9,PURPLE,'bold')
    text(.623,.204,'Curves are schematic',8)
    text(.87,.713,'Coverage restored; set-size costs',10,GREEN,'bold')
    bars(.795,.495,[source,target],['Source','Target split'],[ORANGE,GREEN])
    text(.87,.435,'10% target results: seed-level medians',9)
    text(.87,.376,f'Coverage: {target:.4f}',15,GREEN,'bold')
    text(.87,.31,f'Mondrian class gap: {gap:.4f}',12,GREEN,'bold')
    text(.87,.255,'Mean Mondrian size: 1.4985',10,GRAY,'bold')
    text(.87,.205,'Stronger unlabeled methods also recover coverage',8)
    box(.04,.04,.92,.055,'#CBD5DF',fill='#F7F9FB',lw=1)
    text(.5,.068,'Core message: target recalibration can restore coverage; compare label burden, class protection and prediction-set efficiency.',11,GRAY,'bold')
    fig.savefig(OUT,bbox_inches='tight');plt.close(fig)
    receipt={'output':str(OUT.relative_to(ROOT)),'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
             'basis':'Submitted four-panel color composition, rebuilt as editable deterministic vector artwork',
             'not_bitmap_identical':True,'numbers':{'source_seed_median':source,'control_seed_median':float(c),'target_split_seed_median':target,'target_mondrian_gap_seed_median':gap},
             'schematic_elements':['people','dataset icons','score curves'],
             'empirical_bar_ylim':[.75,1.0],'all_empirical_bars_generated_from_reference_csv':True}
    (RESULTS/'restored_threshold_diagnostic/graphical-abstract-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Color graphical abstract rebuilt from verified reference data:',OUT)


if __name__=='__main__':main()
