"""Independent normalization, draws and sorted-rank reconstruction of label channels."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def seed(namespace,pair,n,regime,rep):
    message=f"{namespace}|{pair['dataset']}|{pair['shift']}|{pair['seed']}|{n}|{regime}|{rep}"
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8],'big')


def waterfill(weights,total):
    # Active-set solution, independent of the generator's bisection routine.
    weights=np.asarray(weights,float);fixed=np.zeros(len(weights),bool)
    for _ in range(len(weights)+1):
        c=(total-fixed.sum())/weights[~fixed].sum() if (~fixed).any() else 0.
        next_fixed=fixed|(c*weights>=1.)
        if np.array_equal(next_fixed,fixed):return np.where(fixed,1.,c*weights)
        fixed=next_fixed
    raise AssertionError('Water filling did not terminate')


def metrics(cal,test,mondrian):
    cp=cal.prob_1.to_numpy(float);cy=cal.label.to_numpy(int)
    scores=np.where(cy==0,cp,1-cp)
    def quantile(s):
        k=int(np.ceil((len(s)+1)*.9))
        return np.inf if not len(s) or k>len(s) else float(np.sort(s)[k-1])
    q=[quantile(scores[cy==k]) for k in (0,1)] if mondrian else [quantile(scores)]*2
    p=test.prob_1.to_numpy(float);y=test.label.to_numpy(int)
    sets=np.column_stack((p<=q[0],1-p<=q[1]));hit=sets[np.arange(len(y)),y];size=sets.sum(axis=1)
    c0=float(hit[y==0].mean());c1=float(hit[y==1].mean())
    return {'coverage':float(hit.mean()),'class_0_coverage':c0,'class_1_coverage':c1,
            'class_coverage_gap':abs(c1-c0),'mean_set_size':float(size.mean()),
            'empty_set_rate':float((size==0).mean()),'singleton_rate':float((size==1).mean()),
            'full_set_rate':float((size==2).mean()),'positive_class_shortfall':max(0.,.9-c1),
            'correct_singleton_rate':float(((size==1)&hit).mean()),
            'incorrect_singleton_rate':float(((size==1)&~hit).mean())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    receipt=json.loads((a.out/'receipt.json').read_text())
    for name,digest in receipt['outputs'].items():assert sha(a.out/name)==digest
    frame=pd.read_csv(a.out/'replicate_metrics.csv');draws=pd.read_csv(a.out/'calibration_draws.csv')
    pairs=json.loads((a.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json').read_text())['pairs']
    index=['dataset','shift','seed','planned_labels','regime','replicate']
    observed=frame.set_index(index+['method']);counts=draws.set_index(index+['design'])
    assert observed.index.is_unique and counts.index.is_unique
    checked=0;draw_checked=0;maximum_error=0.
    for pair in pairs:
        path=a.lab_root/pair['prediction_path'];assert sha(path)==pair['prediction_sha256']
        f=pd.read_parquet(path);pool=f[f.role=='target_calibration_pool'].reset_index(drop=True)
        test=f[f.role=='target_test'].reset_index(drop=True);assert not set(pool.row_id)&set(test.row_id)
        p=pool.prob_1.to_numpy(float);y=pool.label.to_numpy(int)
        for n in [100,300]:
            for regime in ['uniform','outcome_depleted','risk_priority']:
                w={'uniform':np.ones(len(p)),'outcome_depleted':np.where(y==1,.2,1.),'risk_priority':.1+.9*p}[regime]
                prob=waterfill(w,2*n);assert abs(prob.sum()-2*n)<1e-7
                for rep in range(5):
                    rng=np.random.default_rng(seed('jbi-labelchannel-v1',pair,n,regime,rep))
                    eligible=np.where(rng.uniform(size=len(pool))<prob)[0]
                    ix=rng.permutation(eligible)[:n]
                    srng=np.random.default_rng(seed('jbi-sentinel-v1',pair,n,regime,rep))
                    sx=srng.permutation(len(pool))[:n]
                    key=(pair['dataset'],pair['shift'],pair['seed'],n,regime,rep);computed={}
                    for design,indices in [('available',ix),('sentinel',sx)]:
                        cal=pool.iloc[indices];c=counts.loc[(*key,design)]
                        assert c.n_calibration==len(cal) and c.n_class1==cal.label.sum()
                        assert c.n_class0==len(cal)-cal.label.sum()
                        assert c.eligible_n==len(eligible) and c.quota_shortfall==max(0,n-len(ix))
                        assert c.ids_sha256==hashlib.sha256('\x1f'.join(cal.row_id.astype(str)).encode()).hexdigest()
                        draw_checked+=1
                        for mondrian in [False,True]:
                            method=design+('_mondrian' if mondrian else '_split')
                            computed[method]=metrics(cal,test,mondrian)
                    computed['known_design_rule']=computed['sentinel_mondrian' if regime=='risk_priority' else 'available_mondrian']
                    for method,expected in computed.items():
                        row=observed.loc[(*key,method)]
                        errors=[abs(float(row[k])-v) for k,v in expected.items()]
                        maximum_error=max(maximum_error,max(errors));assert max(errors)<1e-12
                        checked+=1
    assert checked==len(frame)==9000 and draw_checked==len(draws)==3600
    result={'status':'PASS','metric_rows':checked,'calibration_draws':draw_checked,
            'maximum_absolute_metric_error':maximum_error,
            'scope':'Active-set propensity normalization; deterministic eligibility and draws; exact row IDs/class counts; independent sorted ranks; every membership metric and known-design rule. No clinical-utility or real-observation-mechanism validation.',
            'code_sha256':sha(Path(__file__))}
    (a.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
