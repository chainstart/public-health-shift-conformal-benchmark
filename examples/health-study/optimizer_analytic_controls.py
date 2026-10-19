"""Independent global solutions for tractable OT and LR-QR controls.

OT: two score support points make the discretized objective piecewise
quadratic in their total mass. Enumerate all breakpoints and stationary
points for a global simplex solution. LR-QR: intercept-only identical
covariates give loss(h) + lambda*(beta*h-1)^2 - lambda; a score quantile
with beta=1/h attains the global lower bound whenever beta is feasible.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from validate_optimizer_stability import load, ot_fit, sha


def scalar_ot_global(problem):
    grid = np.asarray(problem['grid'], dtype=float)
    c0 = np.asarray(problem['indicator'][0], dtype=float)
    c1 = np.asarray(problem['indicator'][-1], dtype=float)
    d0 = np.asarray(problem['kernel'][0], dtype=float)
    d1 = np.asarray(problem['kernel'][-1], dtype=float)
    lower = np.asarray(problem['target_max_cdf'], dtype=float)
    upper = np.asarray(problem['target_min_cdf'], dtype=float)
    ca, cb, da, db = c0-c1, c1, d0-d1, d1
    cuts = {0., 1.}
    mask = ca != 0
    for target in [lower, upper]:
        cuts.update(float(x) for x in ((target[mask]-cb[mask])/ca[mask]) if 0 < x < 1)
    cuts = sorted(cuts)
    candidates = list(cuts)
    for left, right in zip(cuts[:-1], cuts[1:]):
        mid = (left+right)/2
        signs0 = np.sign(cb+ca*mid-lower)
        signs1 = np.sign(cb+ca*mid-upper)
        sa = (signs0+signs1)*ca
        sb = signs0*(cb-lower)+signs1*(cb-upper)+upper-lower
        quadratic = 0.5*np.trapezoid(da*sa,grid)
        linear = 0.5*np.trapezoid(da*sb+db*sa,grid)
        if quadratic > 0:
            critical = -linear/(2*quadratic)
            if left < critical < right:
                candidates.append(float(critical))
    def objective(w):
        cdf=cb+ca*w;density=db+da*w
        return float(0.5*np.trapezoid(density*(abs(cdf-lower)+abs(cdf-upper)+upper-lower),grid))
    values = [objective(w) for w in candidates]
    index = int(np.argmin(values))
    return candidates[index], values[index], objective, len(candidates)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-code',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    otcode=args.reference_code/'exp/shift_aware_baselines_20260813/run.py'
    lrcode=args.reference_code/'exp/lrqr_baseline_20260824/run.py'
    ot=load('control_ot',otcode,Path('/unused'))
    lr=load('control_lr',lrcode,Path('/unused'))
    scenarios=[(.1,.9,[.15,.65]),(.3,.55,[.12,.60]),(.25,.75,[.10,.65])]
    problems=[]
    for a,b,target in scenarios:
        scores=np.r_[np.full(150,a),np.full(150,b)]
        p=np.r_[np.full(90,target[0]),np.full(210,target[1])]
        problems.append(ot.prepare_ot_problem(scores,p))
    rows=[]
    for steps in [1000,4000,8000]:
        weights,trace=ot_fit(problems,steps,None)
        for i,problem in enumerate(problems):
            exact_w,exact_value,objective,candidates=scalar_ot_global(problem)
            fitted_mass=float(weights[i,:150].sum()/weights[i].sum())
            independent_value=objective(fitted_mass)
            torch_value=next(x['objective'] for x in trace if x['case']==i and x['step']==steps)
            assert abs(torch_value-independent_value)<2e-6
            assert independent_value>=exact_value-1e-9
            rows.append({'control':i,'steps':steps,'exact_support0_mass':exact_w,'fitted_support0_mass':fitted_mass,
                'exact_global_objective':exact_value,'fitted_objective':independent_value,
                'global_objective_gap':independent_value-exact_value,'enumerated_candidates':candidates,
                'objective_implementation_error':abs(torch_value-independent_value)})
    pd.DataFrame(rows).to_csv(args.out/'ot_global_controls.csv',index=False)
    scores=np.linspace(.05,.95,300)
    q=float(np.sort(scores)[269])
    loss=float(np.where(scores>=q,.9*(scores-q),.1*(q-scores)).mean())
    lambdas=np.repeat([.1,.4,.8],3);starts=np.tile([.25,1.,4.],3)
    x=np.ones((9,300,1));s=np.tile(scores,(9,1));rows=[]
    for steps in [2000,8000]:
        theta,beta=lr.batched_adam(x,s,x,x,lambdas,starts,steps)
        for i,lam in enumerate(lambdas):
            h=float(theta[i,0]);b=float(beta[i])
            observed=float(np.where(scores>=h,.9*(scores-h),.1*(h-scores)).mean()+lam*(b*h-1)**2-lam)
            implemented,*_=lr.objective_and_gradient(theta[i],b,lam,x[i],scores,x[i],x[i])
            optimum=loss-lam
            assert abs(observed-implemented)<1e-12
            assert observed>=optimum-1e-12
            rows.append({'steps':steps,'lambda':lam,'initial_beta':starts[i],'exact_h':q,'exact_beta':1/q,
                'fitted_h':h,'fitted_beta':b,'global_objective':optimum,'fitted_objective':observed,
                'global_objective_gap':observed-optimum,'product_residual':abs(b*h-1)})
    pd.DataFrame(rows).to_csv(args.out/'lrqr_global_controls.csv',index=False)
    receipt={'status':'objective_fidelity_verified_optimization_gaps_reported','ot_controls':9,'lrqr_controls':18,
             'optimizer_global_correctness':'Not inferred for health-data problems; analytic global minima apply only to these controls.',
             'script_sha256':sha(Path(__file__)),'ot_reference_sha256':sha(otcode),'lrqr_reference_sha256':sha(lrcode)}
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('OT controls\n',pd.read_csv(args.out/'ot_global_controls.csv').to_string(index=False))
    print('LR-QR controls\n',pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
