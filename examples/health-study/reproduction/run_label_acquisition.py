"""Prespecified controlled label-channel experiment on frozen health predictions.

This compares calibration-label protocols, not clinical benefit or annotation
costs. Test labels only evaluate sets. Run under the CPU resource guard.
"""
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

BUDGETS=(100,300)
REGIMES=('uniform','outcome_depleted','risk_priority')
METHODS=('available_split','available_mondrian','sentinel_split','sentinel_mondrian','known_design_rule')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')


def availability(p,y,regime,n):
    if regime=='uniform':w=np.ones(len(p))
    elif regime=='outcome_depleted':w=np.where(y==1,.2,1.)
    elif regime=='risk_priority':w=.1+.9*p
    else:raise ValueError(regime)
    # Eligibility has expected size 2*n; a uniform sample of n eligible rows is
    # used. This avoids repeatedly redrawing a favorable availability mask.
    target=min(2*n,len(w));lo=0.;hi=1/min(w)
    for _ in range(60):
        mid=(lo+hi)/2
        if np.minimum(1.,mid*w).sum()<target:lo=mid
        else:hi=mid
    return np.minimum(1.,((lo+hi)/2)*w)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'receipt.json').exists(),'Completed experiment is immutable.'
    protocol={'created_at':datetime.now(timezone.utc).isoformat(),
      'question':'Does the target-label observation channel determine when split versus classwise target recalibration is appropriate?',
      'basis':'R1.2 benchmark quality; R1.3 information requirements; R2.major2/4 and R2.minor5 biomedical maintenance context.',
      'design':'Controlled counterfactual observation mechanisms; not estimated real health-system missingness.',
      'source_model':'All 60 verified primary elastic-net condition-seed prediction files; no refitting or tuning.',
      'budgets':BUDGETS,'replicates':5,'regimes':REGIMES,
      'selection_weights':{'uniform':'1','outcome_depleted':'0.2 for Y=1, 1 for Y=0; retrospective outcome-availability stress test, not a prospective sampling action',
                           'risk_priority':'0.1 + 0.9*p_source(Y=1|X); only preexisting model probabilities'},
      'eligibility':'Independent Bernoulli availability, c*w clipped at one; c fixes expected eligible count at 2*n. Sample n eligible rows uniformly; use all if insufficient. Never redraw or omit a failed quota.',
      'sentinel':'Alternative feature/outcome-blind uniform calibration sample of n from the full pool; not restricted to available-label rows. Same planned label count, independently derived seed.',
      'known_design_rule':'Use available-label Mondrian under uniform or outcome-only availability; under risk-prioritized availability use sentinel Mondrian. Collection-channel identity is prespecified, not learned from test results.',
      'primary_outcomes':['class_0_coverage','class_1_coverage','positive_class_shortfall'],
      'secondary_outcomes':['marginal coverage','mean set size','empty/singleton/two-label rates','correct and incorrect singleton masses','class counts and infinite thresholds'],
      'nominal_coverage':.9,'threshold_rule':'ceil((n+1)*.9) sorted rank; infinity for overflow/absent class; ties included; no argmax fallback',
      'summary':'Draw means, seed means, condition means, equal-dataset means; no independent-deployment inference.',
      'boundary':'Known-design benchmark policy, not a learned clinical selector or utility study. Label counts concern calibration inputs, not monetary/total acquisition cost. All conditions and all draws retained.',
      'code_sha256':sha(Path(__file__))}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    interface=Path(__file__).parent/'benchmark'
    if not interface.exists():interface=Path(__file__).parent/'reproduction/benchmark'
    sys.path.insert(0,str(interface));from api import CalibrationInput,fit_split,set_metrics
    provenance=a.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    pairs=json.loads(provenance.read_text())['pairs'];assert len(pairs)==60
    rows=[];draws=[]
    for pair in pairs:
        path=a.lab_root/pair['prediction_path'];assert sha(path)==pair['prediction_sha256']
        f=pd.read_parquet(path);pool=f[f.role=='target_calibration_pool'].reset_index(drop=True)
        test=f[f.role=='target_test'].reset_index(drop=True)
        assert not set(pool.row_id)&set(test.row_id)
        p=pool.prob_1.to_numpy(dtype=float);y=pool.label.to_numpy(int)
        tp=test.prob_1.to_numpy(dtype=float);ty=test.label.to_numpy(int)
        common={k:pair[k] for k in ('dataset','shift','seed')}
        for n in BUDGETS:
            assert n<=len(pool)
            for regime in REGIMES:
                prob=availability(p,y,regime,n);assert abs(prob.sum()-2*n)<1e-7
                for rep in range(5):
                    rng=np.random.default_rng(seed_for('jbi-labelchannel-v1',pair['dataset'],pair['shift'],pair['seed'],n,regime,rep))
                    eligible=np.flatnonzero(rng.random(len(pool))<prob)
                    idx=rng.permutation(eligible)[:n]
                    srng=np.random.default_rng(seed_for('jbi-sentinel-v1',pair['dataset'],pair['shift'],pair['seed'],n,regime,rep))
                    sentinel=srng.permutation(len(pool))[:n]
                    base=dict(**common,planned_labels=n,regime=regime,replicate=rep)
                    for name,indices in [('available',idx),('sentinel',sentinel)]:
                        cal=pool.iloc[indices];cy=cal.label.to_numpy(int)
                        d=CalibrationInput(np.array([]),np.array([],int),cal.prob_1.to_numpy(),cy,.1,rep)
                        counts=np.bincount(cy,minlength=2)
                        draws.append(dict(**base,design=name,n_calibration=len(cal),n_class0=int(counts[0]),n_class1=int(counts[1]),
                                          eligible_n=len(eligible),quota_shortfall=max(0,n-len(idx)),
                                          ids_sha256=hashlib.sha256('\x1f'.join(cal.row_id.astype(str)).encode()).hexdigest()))
                        for mondrian in (False,True):
                            fit=fit_split(d,mondrian=mondrian);sets=fit.predict_sets(tp);metric=set_metrics(sets,ty)
                            sizes=sets.sum(axis=1);hit=sets[np.arange(len(ty)),ty]
                            rows.append(dict(**base,method=name+('_mondrian' if mondrian else '_split'),n_calibration=len(cal),
                                             n_class0=int(counts[0]),n_class1=int(counts[1]),**metric,
                                             positive_class_shortfall=max(0.,.9-metric['class_1_coverage']),
                                             correct_singleton_rate=float(((sizes==1)&hit).mean()),
                                             incorrect_singleton_rate=float(((sizes==1)&~hit).mean())))
                    chosen='sentinel_mondrian' if regime=='risk_priority' else 'available_mondrian'
                    result=next(r for r in reversed(rows) if all(r[k]==v for k,v in base.items()) and r['method']==chosen)
                    rows.append(dict(result,method='known_design_rule'))
        print('label channels',pair['dataset'],pair['shift'],pair['seed'],flush=True)
    metrics=pd.DataFrame(rows);draw=pd.DataFrame(draws);assert len(metrics)==9000 and len(draw)==3600
    metrics.to_csv(a.out/'replicate_metrics.csv',index=False);draw.to_csv(a.out/'calibration_draws.csv',index=False)
    cols=['coverage','class_0_coverage','class_1_coverage','positive_class_shortfall','class_coverage_gap','mean_set_size',
          'empty_set_rate','singleton_rate','full_set_rate','correct_singleton_rate','incorrect_singleton_rate']
    condition=metrics.groupby(['dataset','shift','seed','planned_labels','regime','method'])[cols].mean().groupby(['dataset','shift','planned_labels','regime','method']).mean().reset_index()
    condition.to_csv(a.out/'condition_summary.csv',index=False)
    ds=condition.groupby(['dataset','planned_labels','regime','method'])[cols].mean().reset_index();ds.to_csv(a.out/'dataset_summary.csv',index=False)
    eq=ds.groupby(['planned_labels','regime','method'])[cols].mean().reset_index();eq.to_csv(a.out/'equal_dataset_summary.csv',index=False)
    receipt={'status':'completed','pairs':60,'rows':9000,'draws':3600,'quota_shortfall_draws':int((draw.quota_shortfall>0).sum()),
             'protocol_sha256':sha(a.out/'protocol.json'),'input_manifest_sha256':sha(provenance),
             'outputs':{p.name:sha(p) for p in a.out.glob('*.csv')}}
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(eq[['planned_labels','regime','method','coverage','class_1_coverage','mean_set_size']].to_string(index=False))


if __name__=='__main__':main()
