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


def information_order(x, y, method, bins=8):
    """Greedy discrete information criteria with empirical quantile discretization."""
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    columns = []
    for col in x.T:
        cuts = np.unique(np.quantile(col, np.arange(1, bins) / bins))
        columns.append(np.searchsorted(cuts, col, side="left"))
    relevance = np.array([mutual_info_score(y, col) for col in columns])
    penalty = np.zeros(len(columns))
    worst = np.full(len(columns), -np.inf)
    classes, counts = np.unique(y, return_counts=True)
    class_rows = [(y == c, count / len(y)) for c, count in zip(classes, counts)]
    remaining = np.ones(len(columns), dtype=bool)
    order = []
    for step in range(len(columns)):
        if method == "cmim" and step:
            criterion = relevance - worst
        elif method == "cife":
            criterion = relevance - penalty
        else:
            criterion = relevance - penalty / max(step, 1)
        best = np.argmax(np.where(remaining, criterion, -np.inf))
        remaining[best] = False
        order.append(best)
        for j in np.flatnonzero(remaining):
            change = mutual_info_score(columns[best], columns[j])
            if method != "mrmr":
                conditional = sum(prob * mutual_info_score(columns[best][rows], columns[j][rows])
                                  for rows, prob in class_rows)
                change -= conditional
            penalty[j] += change
            worst[j] = max(worst[j], change)
    return np.argsort(order) + 1


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
    pairs = 0
    credit = np.zeros(x.shape[1])
    for i in np.flatnonzero(event):
        later = (time > time[i]) | ((time == time[i]) & ~event)
        delta = x[i] - x[later]
        credit += (delta > tied_tol).sum(axis=0) + .5 * (np.abs(delta) <= tied_tol).sum(axis=0)
        pairs += later.sum()
    if pairs == 0:
        raise ValueError("No comparable survival pairs")
    return credit / pairs


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
