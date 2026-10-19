"""Deterministic budget/efficiency experiment on source-fitted probability files."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from api import CalibrationInput, fit_split, set_metrics

FRACTIONS=(.005,.01,.02,.05,.10,.20)
METRICS=('coverage','mean_set_size','empty_set_rate','singleton_rate','full_set_rate',
         'class_0_coverage','class_1_coverage','class_coverage_gap')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--replicates',type=int,default=5)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    provenance=args.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    pairs=json.loads(provenance.read_text())['pairs'];assert len(pairs)==60
    audits=pd.read_csv(args.lab_root/'exp/source_role_outcome_blind_20260725/role_assignment_audit.csv')
    rows=[];budgets=[];subgroups=[];domainrows=[];inputs=[]
    for pair in pairs:
        path=args.lab_root/pair['prediction_path'];assert sha(path)==pair['prediction_sha256']
        frame=pd.read_parquet(path)
        source=frame.loc[frame.role=='source_calibration'].reset_index(drop=True)
        pool=frame.loc[frame.role=='target_calibration_pool'].reset_index(drop=True)
        test=frame.loc[frame.role=='target_test'].reset_index(drop=True)
        role_ids=[set(x.row_id) for x in (source,pool,test)]
        assert all(not(role_ids[i]&role_ids[j]) for i,j in ((0,1),(0,2),(1,2)))
        sp=source.prob_1.to_numpy();sy=source.label.to_numpy(int)
        tp=test.prob_1.to_numpy();ty=test.label.to_numpy(int)
        target_n=len(pool)+len(test)
        common={k:pair[k] for k in ('dataset','shift','seed')}
        audit=audits.loc[(audits.dataset==pair['dataset'])&(audits['shift']==pair['shift'])&
                        (audits.seed==pair['seed'])&(audits.domain=='source_train_calibration')].iloc[0]
        source_n=int(audit.domain_n)
        source_pos=int(audit.retained_positive_n)+int(audit.heldout_positive_n)
        domainrows.append(dict(**common,source_n=source_n,target_n=target_n,
                               source_prevalence=source_pos/source_n,
                               target_prevalence=(int(pool.label.sum())+int(ty.sum()))/target_n,
                               source_calibration_n=len(source),target_pool_n=len(pool),test_n=len(test)))
        empty=CalibrationInput(sp,sy,np.array([]),np.array([],int),.1,0)
        fixed={name:fit_split(empty,'source',mondrian=name.endswith('mondrian'))
               for name in ('source_split','source_mondrian')}
        for replicate in range(args.replicates):
            order=np.random.default_rng(seed_for('jbi-budget-v1',pair['dataset'],pair['shift'],pair['seed'],replicate)).permutation(len(pool))
            for frac in FRACTIONS:
                n=max(1,int(round(frac*target_n)));assert n<=len(pool)
                cal=pool.iloc[order[:n]];cp=cal.prob_1.to_numpy();cy=cal.label.to_numpy(int)
                data=CalibrationInput(sp,sy,cp,cy,.1,replicate)
                fitted=dict(fixed)
                fitted['target_split']=fit_split(data)
                fitted['target_mondrian']=fit_split(data,mondrian=True)
                counts=np.bincount(cy,minlength=2)
                ranks=[int(np.ceil((int(c)+1)*.9)) for c in counts]
                budgets.append(dict(**common,replicate=replicate,fraction=frac,n_target_labels=n,
                                    n_class0=int(counts[0]),n_class1=int(counts[1]),
                                    rank0=ranks[0],rank1=ranks[1],
                                    overflow0=ranks[0]>int(counts[0]),overflow1=ranks[1]>int(counts[1]),
                                    ids_sha256=hashlib.sha256('\x1f'.join(cal.row_id.astype(str)).encode()).hexdigest()))
                for name,method in fitted.items():
                    sets=method.predict_sets(tp);metric=set_metrics(sets,ty)
                    rows.append(dict(**common,replicate=replicate,fraction=frac,method=name,**metric))
                    if frac==.10:
                        covered=sets[np.arange(len(ty)),ty]
                        for column in (c for c in test.columns if c.startswith('group_')):
                            values=test[column].fillna('missing').astype(str)
                            for value in sorted(values.unique()):
                                mask=(values==value).to_numpy()
                                subgroups.append(dict(**common,replicate=replicate,method=name,group=column,
                                                      value=value,n_test=int(mask.sum()),
                                                      prevalence=float(ty[mask].mean()),
                                                      coverage=float(covered[mask].mean()),
                                                      mean_set_size=float(sets[mask].sum(axis=1).mean())))
        inputs.append({'path':str(path),'sha256':pair['prediction_sha256']})
        print('completed',pair['dataset'],pair['shift'],pair['seed'],flush=True)
    frame=pd.DataFrame(rows);b=pd.DataFrame(budgets);domains=pd.DataFrame(domainrows)
    frame.to_csv(args.out/'budget_metrics.csv',index=False)
    b.to_csv(args.out/'budget_draws.csv',index=False)
    domains.to_csv(args.out/'domain_counts_by_seed.csv',index=False)
    pd.DataFrame(subgroups).to_csv(args.out/'subgroup_metrics.csv',index=False)
    seedmeans=frame.groupby(['dataset','shift','seed','fraction','method'])[list(METRICS)].mean().reset_index()
    shifts=seedmeans.groupby(['dataset','shift','fraction','method'])[list(METRICS)].mean().reset_index()
    ds=shifts.groupby(['dataset','fraction','method'])[list(METRICS)].mean().reset_index()
    eq=ds.groupby(['fraction','method'])[list(METRICS)].mean().reset_index()
    for name,table in [('budget_by_shift',shifts),('budget_by_dataset',ds),('budget_equal_dataset',eq)]:
        table.to_csv(args.out/(name+'.csv'),index=False)
    summary=b.groupby(['dataset','shift','fraction']).agg(n_target_labels=('n_target_labels','first'),
                class0_min=('n_class0','min'),class0_max=('n_class0','max'),class1_min=('n_class1','min'),
                class1_max=('n_class1','max'),missing_class0_draws=('n_class0',lambda x:int((x==0).sum())),
                missing_class1_draws=('n_class1',lambda x:int((x==0).sum())),
                overflow0_draws=('overflow0','sum'),overflow1_draws=('overflow1','sum')).reset_index()
    summary.to_csv(args.out/'absolute_budget_summary.csv',index=False)
    sg=pd.DataFrame(subgroups).groupby(['dataset','shift','method','group','value']).agg(
        n_test_min=('n_test','min'),n_test_max=('n_test','max'),prevalence=('prevalence','mean'),
        coverage=('coverage','mean'),mean_set_size=('mean_set_size','mean')).reset_index()
    sg.to_csv(args.out/'subgroup_summary.csv',index=False)
    assert len(frame)==60*args.replicates*len(FRACTIONS)*4
    assert np.isfinite(frame[list(METRICS)].to_numpy()).all()
    assert np.allclose(frame.empty_set_rate+frame.singleton_rate+frame.full_set_rate,1)
    assert np.allclose(frame.mean_set_size,frame.singleton_rate+2*frame.full_set_rate)
    receipt={'status':'completed','pairs':len(pairs),'metric_rows':len(frame),
             'fractions':FRACTIONS,'replicates':args.replicates,'inputs':inputs,
             'code_sha256':sha(Path(__file__)),'provenance_sha256':sha(provenance),
             'aggregation':'draw->seed->shift->dataset; equal weight to datasets',
             'validation':{'roles_disjoint':True,'row_count':True,'efficiency_identities':True},
             'outputs':{p.name:sha(p) for p in args.out.glob('*.csv')}}
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(eq.to_string(index=False))


if __name__=='__main__':main()
