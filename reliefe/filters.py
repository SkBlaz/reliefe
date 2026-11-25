"""Dense numeric filters. Each function returns a value per input column."""
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler


def original_relief(x, y):
    """Kira and Rendell's binary update with every row as a query."""
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2 or min(counts) < 2:
        raise ValueError("Original Relief requires two classes with at least two rows each")
    values = MinMaxScaler().fit_transform(x)
    scores = np.zeros(values.shape[1])
    for i, row in enumerate(values):
        squared = (values - row) ** 2
        distances = squared.sum(axis=1)
        distances[i] = np.inf
        hit = np.argmin(np.where(y == y[i], distances, np.inf))
        miss = np.argmin(np.where(y != y[i], distances, np.inf))
        scores += squared[miss] - squared[hit]
    return scores / len(y)


def rrelieff(x, y, k=10, sigma=50.0):
    """Regression ReliefF using rank-weighted nearest neighbors and range differences."""
    if not isinstance(k, (int, np.integer)) or k < 1 or sigma <= 0 or len(y) < 2:
        raise ValueError("RReliefF requires positive k, sigma and at least two rows")
    k = min(k, len(y) - 1)
    values = MinMaxScaler().fit_transform(x)
    target = MinMaxScaler().fit_transform(y.reshape(-1, 1)).ravel()
    ranks = np.arange(1, k + 1, dtype=float)
    # Subtract the nearest neighbor's log weight before exponentiation.
    influence = np.exp(-((ranks - 1) / sigma) * ((ranks + 1) / sigma))
    influence /= influence.sum()
    feature_diff = np.zeros(x.shape[1])
    joint = np.zeros_like(feature_diff)
    target_diff = 0.
    for i, row in enumerate(values):
        diffs = np.abs(values - row)
        distances = diffs.sum(axis=1)
        distances[i] = np.inf
        neighbors = np.argsort(distances, kind="stable")[:k]
        weighted = influence * np.abs(target[i] - target[neighbors])
        target_diff += weighted.sum()
        joint += weighted @ diffs[neighbors]
        feature_diff += influence @ diffs[neighbors]
    if target_diff <= 0 or target_diff >= len(y):
        return np.zeros_like(feature_diff)
    return joint / target_diff - (feature_diff - joint) / (len(y) - target_diff)


def mrmr_order(x, y, bins=8):
    """Greedy mutual information difference with empirical quantile discretization."""
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    columns = []
    for col in x.T:
        cuts = np.unique(np.quantile(col, np.arange(1, bins) / bins))
        columns.append(np.searchsorted(cuts, col, side="left"))
    relevance = np.array([mutual_info_score(y, col) for col in columns])
    redundancy = np.zeros(len(columns))
    remaining = np.ones(len(columns), dtype=bool)
    order = []
    for step in range(len(columns)):
        criterion = relevance - redundancy / max(step, 1)
        best = np.argmax(np.where(remaining, criterion, -np.inf))
        remaining[best] = False
        order.append(best)
        for j in np.flatnonzero(remaining):
            redundancy[j] += mutual_info_score(columns[best], columns[j])
    return np.argsort(order) + 1
