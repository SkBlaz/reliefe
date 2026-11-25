"""Dense numeric filters. Each function returns a value per input column."""
import numpy as np
from .native import (concordance_scores, greedy_information, original_scores,
                     regression_scores, scale_columns)


def original_relief(x, y):
    """Kira and Rendell's binary update with every row as a query."""
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2 or min(counts) < 2:
        raise ValueError("Original Relief requires two classes with at least two rows each")
    return original_scores(scale_columns(x), y)


def rrelieff(x, y, k=10, sigma=50.0):
    """Regression ReliefF using rank-weighted nearest neighbors and range differences."""
    if not isinstance(k, (int, np.integer)) or k < 1 or not np.isfinite(sigma) or sigma <= 0 or len(y) < 2:
        raise ValueError("RReliefF requires positive k, sigma and at least two rows")
    k = min(k, len(y) - 1)
    ranks = np.arange(1, k + 1, dtype=float)
    # Subtract the nearest neighbor's log weight before exponentiation.
    influence = np.ones(k)
    with np.errstate(over="ignore"):
        influence[1:] = np.exp(-((ranks[1:] - 1) / sigma) * ((ranks[1:] + 1) / sigma))
    influence /= influence.sum()
    target = scale_columns(y.reshape(-1, 1)).ravel()
    return regression_scores(scale_columns(x), target, influence)


def quantile_codes(x, bins):
    """Discretize each column and return one contiguous row of codes per feature."""
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    columns = []
    for col in x.T:
        levels = np.unique(col)
        if len(levels) <= bins:
            columns.append(np.searchsorted(levels, col))
            continue
        cuts = np.unique(np.quantile(col, np.arange(1, bins) / bins))
        columns.append(np.searchsorted(cuts, col, side="left"))
    return np.asarray(columns, dtype=np.int64)


def information_order(x, y, method, bins=8):
    codes = quantile_codes(x, bins)
    return greedy_information(codes, y, int(y.max()) + 1, int(codes.max()) + 1,
                              method != "mrmr", method == "cmim", method == "cife")


def laplacian_score(x, k=5, heat_width=1.0):
    """Laplacian Score on a symmetric heat-weighted nearest-neighbor graph.

    Returns the native graph energy divided by degree-weighted variance.
    Lower values rank first. Constant columns have undefined scores.
    """
    from sklearn.neighbors import kneighbors_graph
    if not isinstance(k, (int, np.integer)) or not 1 <= k < x.shape[0] or heat_width <= 0:
        raise ValueError("Laplacian Score requires 1 <= k < n and positive heat_width")
    graph = kneighbors_graph(x, k, mode="distance", include_self=False)
    graph.data = np.exp(-np.square(graph.data) / heat_width)
    graph = graph.maximum(graph.T)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    if degree.sum() == 0:
        raise ValueError("Graph has zero total weight; increase heat_width")
    centered = x - np.average(x, axis=0, weights=degree)
    denom = degree @ (centered ** 2)
    energy = np.sum(centered * (degree[:, None] * centered - graph @ centered), axis=0)
    return np.divide(energy, denom, out=np.full(x.shape[1], np.nan), where=denom > 0)


def survival_concordance(x, y, tied_tol=1e-8):
    """Univariate Harrell concordance for (event, time) pairs; greater x means greater risk."""
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("Expected two target columns: event indicator and duration")
    if (not np.isfinite(y).all() or not np.isin(y[:, 0], [0, 1]).all()
            or (y[:, 1] < 0).any() or tied_tol < 0):
        raise ValueError("Expected binary event indicators, finite nonnegative times and tolerance")
    event, time = y[:, 0].astype(bool), y[:, 1]
    return concordance_scores(x, event, time, tied_tol)


def rank_association(x, y, method):
    """Absolute correlations for regression; Kruskal-Wallis H for classification."""
    from scipy.stats import kendalltau, kruskal, rankdata
    scores = np.zeros(x.shape[1])
    for j, col in enumerate(x.T):
        if np.ptp(col) == 0:
            continue
        if method == "kruskal":
            scores[j] = kruskal(*(col[y == label] for label in np.unique(y))).statistic
        elif np.ptp(y) > 0:
            if method == "kendall":
                scores[j] = abs(kendalltau(col, y)[0])
            else:
                a, b = (rankdata(col), rankdata(y)) if method == "spearman" else (col, y)
                scores[j] = abs(np.corrcoef(a, b)[0, 1])
    return scores
