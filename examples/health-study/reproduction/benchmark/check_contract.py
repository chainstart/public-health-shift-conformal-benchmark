"""Checks for finite ranks, ties, absent classes and hidden evaluation outcomes."""
import numpy as np
from api import finite_quantile, fit_split, CalibrationInput, BUILTINS, evaluate_method
from example_method import METHOD


def main():
    assert np.isinf(finite_quantile([], .1))
    assert np.isinf(finite_quantile(np.arange(8.), .1))
    assert finite_quantile(np.arange(9.), .1) == 8
    assert finite_quantile([.2]*19, .1) == .2
    d=CalibrationInput(np.full(10,.2),np.zeros(10,int),np.full(10,.2),np.zeros(10,int),.1,0)
    m=fit_split(d, mondrian=True)
    assert np.isinf(m.threshold_1)
    assert m.predict_sets(np.array([.2]))[0,0]  # equality includes ties
    # A plug-in has no field containing test labels, and source-only receives
    # no target labels. Evaluation of the example agrees with the built-in.
    args=(np.array([.1,.8,.2,.7]*5),np.array([0,1,0,1]*5),np.array([.3,.6]*10),
          np.array([0,1]*10),np.array([.2,.9,.6]),np.array([0,1,1]))
    assert evaluate_method(METHOD,*args)==evaluate_method(BUILTINS['source_split'],*args)
    changed=list(args);changed[-1]=1-changed[-1]
    # Fitting/prediction cannot change when only external evaluation labels change.
    p=METHOD.fit(CalibrationInput(args[0],args[1],args[2],None,.1,0)).predict_sets(args[4])
    q=METHOD.fit(CalibrationInput(args[0],args[1],args[2],None,.1,0)).predict_sets(changed[4])
    assert np.array_equal(p,q)
    print('finite-rank and extension contract checks passed')


if __name__=='__main__': main()
