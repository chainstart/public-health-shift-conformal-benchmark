"""Independent reconstruction of all high-budget OT weights and metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def seed(*parts):
    raw='\x1f'.join(('ara.shift_aware_baselines.draw.v1',*(str(x) for x in parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lab-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out
    provenance=args.lab_root/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
    pairs=json.loads(provenance.read_text())['pairs']
    weights=np.load(out/'weights-steps4000.npy',allow_pickle=False)
    rows=pd.read_csv(out/'replicate_metrics.csv',float_precision='round_trip')
    trace=pd.read_csv(out/'optimization_trace.csv',float_precision='round_trip');trace=trace[trace.step==4000].set_index('case')
    assert weights.shape==(300,300) and len(rows)==300
    maximum_objective_error=0.;position=0
    for pair in pairs:
        path=args.lab_root/pair['prediction_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pair['prediction_sha256']
        f=pd.read_parquet(path)
        source=f[f.role=='source_calibration'].reset_index(drop=True)
        pool=f[f.role=='target_calibration_pool'].reset_index(drop=True)
        test=f[f.role=='target_test'].reset_index(drop=True)
        for replicate in range(5):
            parts=(pair['dataset'],pair['shift'],pair['seed'],replicate)
            si=np.random.default_rng(seed(*parts,'source')).choice(len(source),300,replace=False)
            ti=np.random.default_rng(seed(*parts,'target')).choice(len(pool),300,replace=False)
            calibration=source.iloc[si];w=weights[position]
            assert np.isfinite(w).all() and (w>=0).all() and abs(w.sum()-300)<.001
            scores=np.where(calibration.label==0,calibration.prob_1.to_numpy(dtype=float),1-calibration.prob_1.to_numpy(dtype=float))
            order=np.argsort(scores,kind='stable');mass=.9*(w.sum()+1)
            if mass>w.sum()+1e-12*max(1,w.sum()+1):
                threshold=np.inf
            else:
                index=np.searchsorted(np.cumsum(w[order]),mass-1e-12*max(1,w.sum()+1))
                threshold=scores[order[min(index,299)]]
            row=rows.iloc[position]
            assert all(row[k]==v for k,v in zip(['dataset','shift','seed','replicate'],parts))
            assert row.threshold==threshold or (np.isinf(row.threshold) and np.isinf(threshold))
            p=test.prob_1.to_numpy(dtype=float);y=test.label.to_numpy(dtype=int)
            include0=p<=threshold;include1=1-p<=threshold
            covered=np.where(y==0,include0,include1);size=include0.astype(int)+include1.astype(int)
            metrics={'coverage':covered.mean(),'class_0_coverage':covered[y==0].mean(),
                     'class_1_coverage':covered[y==1].mean(),'mean_set_size':size.mean(),
                     'empty_set_rate':(size==0).mean(),'singleton_rate':(size==1).mean(),'full_set_rate':(size==2).mean()}
            assert all(abs(row[k]-v)<1e-12 for k,v in metrics.items())
            original_grid=np.linspace(0,1,401)
            grid=original_grid.astype(np.float32).astype(float)
            # Reconstruct the specified float32 discretization independently.
            band=max(.01,np.std(scores,ddof=1)*300**(-.2))
            kernel=(np.exp(-.5*((original_grid[None,:]-scores[:,None])/band)**2)/(np.sqrt(2*np.pi)*band)).astype(np.float32).astype(float)
            target=pool.iloc[ti].prob_1.to_numpy(dtype=float)
            lower=(np.maximum(target,1-target)[:,None]<=original_grid[None,:]).mean(axis=0).astype(np.float32).astype(float)
            upper=(np.minimum(target,1-target)[:,None]<=original_grid[None,:]).mean(axis=0).astype(np.float32).astype(float)
            normalized=w/w.sum();cdf=normalized@(scores[:,None]<=original_grid[None,:]);density=normalized@kernel
            objective=.5*np.trapezoid(density*(abs(cdf-lower)+abs(cdf-upper)+upper-lower),grid)
            error=abs(objective-trace.loc[position,'objective']);maximum_objective_error=max(maximum_objective_error,error)
            assert error<2e-5,error
            position+=1
    receipt={'status':'passed','cases':position,'method':'Independent NumPy reconstruction; no solver imports',
             'maximum_objective_error':maximum_objective_error,'weights_sha256':hashlib.sha256((out/'weights-steps4000.npy').read_bytes()).hexdigest(),
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
