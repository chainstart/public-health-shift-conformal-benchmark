"""User extension example: reproduce source split through the documented API."""
from benchmark.api import MethodSpec, ThresholdSets, finite_quantile, true_scores


def fit(data):
    assert data.target_labels is None
    q = finite_quantile(true_scores(data.source_probabilities, data.source_labels), data.alpha)
    return ThresholdSets(q, q)


METHOD = MethodSpec('example_source_split', False, fit)
