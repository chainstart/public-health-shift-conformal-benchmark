"""Small binary-conformal extension interface; evaluation labels stay in the runner."""
from dataclasses import dataclass
from typing import Callable, Protocol
import numpy as np


def finite_quantile(scores, alpha=0.1):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('Calibration scores must be a finite one-dimensional array')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0,1)')
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if rank > len(scores) or not len(scores):
        return float('inf')
    return float(np.partition(scores, rank - 1)[rank - 1])


def true_scores(probabilities, labels):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Aligned binary calibration labels are required')
    return np.where(y == 0, p, 1-p)


@dataclass(frozen=True)
class CalibrationInput:
    source_probabilities: np.ndarray
    source_labels: np.ndarray
    target_probabilities: np.ndarray
    target_labels: np.ndarray | None
    alpha: float
    seed: int


class FittedMethod(Protocol):
    def predict_sets(self, probabilities: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ThresholdSets:
    threshold_0: float
    threshold_1: float

    def predict_sets(self, probabilities):
        p = np.asarray(probabilities, dtype=float)
        return np.column_stack([p <= self.threshold_0, 1-p <= self.threshold_1])


@dataclass(frozen=True)
class MethodSpec:
    name: str
    uses_target_labels: bool
    fit: Callable[[CalibrationInput], FittedMethod]


def fit_split(data, domain='target', mondrian=False):
    if domain == 'target':
        p, y = data.target_probabilities, data.target_labels
        if y is None:
            raise ValueError('This method requires target labels')
    elif domain == 'source':
        p, y = data.source_probabilities, data.source_labels
    else:
        raise ValueError('Unknown calibration domain')
    scores = true_scores(p, y)
    if mondrian:
        return ThresholdSets(*(finite_quantile(scores[y == c], data.alpha) for c in (0, 1)))
    q = finite_quantile(scores, data.alpha)
    return ThresholdSets(q, q)


BUILTINS = {
    'source_split': MethodSpec('source_split', False, lambda d: fit_split(d, 'source')),
    'source_mondrian': MethodSpec('source_mondrian', False, lambda d: fit_split(d, 'source', True)),
    'target_split': MethodSpec('target_split', True, lambda d: fit_split(d)),
    'target_mondrian': MethodSpec('target_mondrian', True, lambda d: fit_split(d, mondrian=True)),
}


def evaluate_method(spec, source_p, source_y, target_p, target_y, test_p, test_y, alpha=0.1, seed=0):
    # A source-only plug-in receives target probabilities but not target outcomes.
    data = CalibrationInput(np.asarray(source_p).copy(), np.asarray(source_y).copy(),
                            np.asarray(target_p).copy(),
                            np.asarray(target_y).copy() if spec.uses_target_labels else None,
                            alpha, seed)
    fitted = spec.fit(data)
    sets = np.asarray(fitted.predict_sets(np.asarray(test_p).copy()))
    if sets.shape != (len(test_y), 2) or sets.dtype != bool:
        raise ValueError('Plug-ins must return an n_test by 2 boolean membership matrix')
    return set_metrics(sets, test_y)


def set_metrics(sets, labels):
    y = np.asarray(labels, dtype=int)
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(axis=1)
    c = [float(covered[y == k].mean()) if np.any(y == k) else float('nan') for k in (0,1)]
    return {'coverage':float(covered.mean()), 'mean_set_size':float(size.mean()),
            'empty_set_rate':float((size==0).mean()), 'singleton_rate':float((size==1).mean()),
            'full_set_rate':float((size==2).mean()), 'class_0_coverage':c[0],
            'class_1_coverage':c[1], 'class_coverage_gap':abs(c[1]-c[0])}
