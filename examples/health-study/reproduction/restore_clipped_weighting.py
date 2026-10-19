"""Retain the submitted clipped-domain-weighting heuristic on corrected roles.

One deterministic 10% unlabeled target draw per model seed (draw zero of the
primary budget design). This is not a finite-sample weighted-validity claim.
"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd


def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    protocol={'question':'Retain original clipped-domain-weighting comparator, correcting roles and empty-set handling.',
              'seeds':list(range(5)),'conditions':12,'target_fraction':.1,'target_draw':0,
              'target_outcomes_used_for_weights':False,'target_test_features_used_for_weights':False,
              'threshold':'first sorted true source score with cumulative normalized weight at least .9; original heuristic, no candidate infinity mass',
              'fallback_argmax':False,'claim':'Empirical heuristic; no distribution-free validity asserted.',
              'code_sha256':sha(Path(__file__))}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    interface=Path(__file__).parent/'benchmark'
    if not interface.exists():interface=Path(__file__).parent/'reproduction/benchmark'
    sys.path.insert(0,str(interface));from api import set_metrics
    gen=load('clip_roles',a.lab_root/'exp/source_role_outcome_blind_20260725/generate.py')
    ref=load('clip_reference',a.lab_root/'exp/row_level_remediation_20260725/generate_row_level_predictions.py')
    sys.path.insert(0,str(a.lab_root))
    from scripts.run_public_shift_benchmark_v2 import prepare_brfss_multiyear,prepare_nhanes_multicycle,prepare_uci_with_pseudo
    from scripts.run_large_scale_conformal import estimate_density_ratio_weights
    manifest=a.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    pairs=json.loads(manifest.read_text())['pairs'];lookup={(p['dataset'],p['shift'],p['seed']):p for p in pairs}
    rows=[]
    for construct in (prepare_brfss_multiyear,prepare_nhanes_multicycle,prepare_uci_with_pseudo):
        ds=ref.attach_stable_ids(construct())
        for shift in (s for s in ds.shifts if s.name!='pseudo_random'):
            for seed in range(5):
                split,_=gen.outcome_blind_shift_split(ds,shift,seed)
                pair=lookup[(ds.name,shift.name,seed)];path=a.lab_root/pair['prediction_path']
                assert sha(path)==pair['prediction_sha256'];f=pd.read_parquet(path)
                s=f[f.role=='source_calibration'].reset_index(drop=True)
                t=f[f.role=='target_test'].reset_index(drop=True)
                assert s.label.tolist()==np.asarray(split['y_source_cal']).tolist()
                assert t.label.tolist()==np.asarray(split['y_test']).tolist()
                payload='|'.join(map(str,('jbi-budget-v1',ds.name,shift.name,seed,0)))
                drawseed=int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big')
                n=max(1,int(round(.1*(len(split['X_target_pool'])+len(t)))))
                indices=np.random.default_rng(drawseed).permutation(len(split['X_target_pool']))[:n]
                weights=estimate_density_ratio_weights(ds,split['X_source_cal'],split['X_target_pool'].iloc[indices],seed+1000)
                assert np.isfinite(weights).all() and weights.min()>=.05 and weights.max()<=20
                scores=np.where(s.label.to_numpy()==0,s.prob_1.to_numpy(),1-s.prob_1.to_numpy())
                order=np.argsort(scores);cum=np.cumsum(weights[order])/weights.sum()
                k=int(np.searchsorted(cum,.9,side='left'));q=float(scores[order[k]])
                # Independent weighted empirical-CDF inversion by scalar mass search.
                below=weights[scores<q].sum()/weights.sum();through=weights[scores<=q].sum()/weights.sum()
                assert below<.9+1e-12 and through>=.9-1e-12
                tp=t.prob_1.to_numpy();sets=np.column_stack([tp<=q,1-tp<=q]);metric=set_metrics(sets,t.label.to_numpy(int))
                rows.append(dict(dataset=ds.name,shift=shift.name,seed=seed,replicate=0,
                                 n_source_labels=len(s),n_unlabeled_target=n,threshold=q,
                                 **metric,shortfall=max(0.,.9-metric['coverage']),
                                 prediction_sha256=pair['prediction_sha256']))
                pd.DataFrame(rows).to_csv(a.out/'metrics.partial.csv',index=False)
                print('clipped weighting',ds.name,shift.name,seed,flush=True)
    f=pd.DataFrame(rows);assert len(f)==60;f.to_csv(a.out/'metrics.csv',index=False)
    columns=['coverage','shortfall','mean_set_size','class_coverage_gap','empty_set_rate','singleton_rate','full_set_rate']
    g=f.groupby(['dataset','shift'])[columns].mean().reset_index();g.to_csv(a.out/'condition_summary.csv',index=False)
    eq=g.groupby('dataset')[columns].mean().mean();eq.to_frame('equal_dataset_mean').to_csv(a.out/'summary.csv')
    (a.out/'metrics.partial.csv').unlink()
    receipt={'status':'completed','fits':60,'all_conditions':True,'target_test_outcomes_for_weighting':False,
             'weighted_cdf_threshold_reconstruction':True,'protocol_sha256':sha(a.out/'protocol.json'),
             'outputs':{p.name:sha(p) for p in a.out.glob('*.csv')}}
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(eq.to_string())


if __name__=='__main__':main()
