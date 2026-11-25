"""Fold-local evaluation and grouped importance for tabular rankings."""
from itertools import combinations
from numbers import Integral

import numpy as np
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state, _safe_indexing
from sklearn.utils.validation import check_array, check_consistent_length


def compare_rankings(X, y, rankers, estimator, splits, *, scoring=None):
    """Evaluate fixed configurations on shared splits. Selection stays inside training folds.

    rankers maps names to FeatureRanker instances (or None for all columns).
    estimator can itself be a Pipeline. Pass precomputed group/time-respecting
    splits as needed. The returned arrays retain paired fold order.
    """
    check_consistent_length(X, y)
    folds = list(splits)
    if len(folds) < 2:
        raise ValueError("At least two evaluation folds are required for stability")
    result = {}
    for name, ranker in rankers.items():
        scores, supports = [], []
        for train, test in folds:
            if np.intersect1d(train, test).size:
                raise ValueError("Training and evaluation rows must be disjoint")
            x_train, x_test = _safe_indexing(X, train), _safe_indexing(X, test)
            y_train, y_test = _safe_indexing(y, train), _safe_indexing(y, test)
            if ranker is None:
                mask = np.ones(X.shape[1], dtype=bool)
            else:
                fitted = clone(ranker).fit(x_train, y_train)
                mask = fitted.get_support()
                x_train, x_test = fitted.transform(x_train), fitted.transform(x_test)
            predictor = clone(estimator).fit(x_train, y_train)
            scores.append(check_scoring(predictor, scoring=scoring)(predictor, x_test, y_test))
            supports.append(mask)
        overlaps = [np.count_nonzero(a & b) / np.count_nonzero(a | b)
                    for a, b in combinations(supports, 2)]
        result[name] = {"fold_scores": np.array(scores),
                        "selection_frequency": np.mean(supports, axis=0),
                        "mean_jaccard": float(np.mean(overlaps))}
    return result


def grouped_permutation_importance(estimator, X, y, groups, *, scoring=None, n_repeats=5,
                                   random_state=42):
    """Change in a fitted predictor's held-out score when groups of columns are shuffled together.

    groups maps group names to zero-based feature indices. Each repeat uses
    the same row permutation across groups to support paired comparisons.
    """
    check_consistent_length(X, y)
    x = check_array(X, dtype=None)
    if not isinstance(n_repeats, Integral) or n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    scorer = check_scoring(estimator, scoring=scoring)
    # Keep column names and dtypes for pipelines that select columns by name.
    named = hasattr(X, "iloc")
    baseline = scorer(estimator, X, y)
    rng = check_random_state(random_state)
    permutations = [rng.permutation(len(x)) for _ in range(n_repeats)]
    result = {}
    for name, group in groups.items():
        indices = np.asarray(group)
        if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu"
                or indices.min() < 0 or indices.max() >= x.shape[1]
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("Groups must contain distinct valid column indices")
        changes = []
        for order in permutations:
            shuffled = X.copy() if named else x.copy()
            if named:
                for i in indices:
                    shuffled.iloc[:, i] = X.iloc[order, i].to_numpy()
            else:
                shuffled[:, indices] = x[order][:, indices]
            changes.append(baseline - scorer(estimator, shuffled, y))
        result[name] = np.array(changes)
    return result
