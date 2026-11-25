"""Numba split criteria following CORElearn 1.56.0 estimator.cpp."""

from math import lgamma
import numpy as np
from numba import njit


CRITERIA = """infgain gainratio mdl gini myopicrelieff accuracy dkm impurityeuclid
impurityhellinger uniformdkm uniformgini uniforminf uniformaccuracy equaldkm equalgini
equalinf equalhellinger disthellinger distauc distangle disteuclid gainratiocost dkmcost""".split()


@njit(cache=True)
def _entropy(p):
    nonzero = p[p > 0]
    return -np.sum(nonzero * np.log2(nonzero))


@njit(cache=True)
def _mdl(counts):
    classes = len(counts)
    bits = lgamma(counts.sum() + classes) - lgamma(classes)
    for v in counts:
        bits -= lgamma(v + 1.0)
    return bits / np.log(2.0)


@njit(cache=True)
def _pair_distance(a, b, method):
    if method == "distangle":
        return np.linalg.norm(a / np.linalg.norm(a) - b / np.linalg.norm(b)) / np.sqrt(
            2.0
        )
    if method == "distauc":
        total = 0.0
        for i in range(len(a)):
            for j in range(i):
                total += abs(a[i] * b[j] - b[i] * a[j])
        return total
    if method == "disthellinger" or method == "equalhellinger":
        return np.linalg.norm(np.sqrt(a) - np.sqrt(b))
    return np.linalg.norm(a - b) / np.sqrt(2.0)


@njit(cache=True)
def _impurity(p, method, costs):
    if method.endswith("cost"):
        adjusted = np.zeros_like(p)
        for i in range(len(p)):
            if p[i] < 1:
                adjusted[i] = (
                    p[i] * (np.dot(costs[i], p) - costs[i, i] * p[i]) / (1 - p[i])
                )
        if adjusted.sum() > 0:
            p = adjusted / adjusted.sum()
        elif method == "dkmcost":
            return np.nan
    if "dkm" in method:
        q = p.max()
        return 2 * np.sqrt(q * (1 - q))
    if "gini" in method or method == "myopicrelieff":
        return np.dot(p, p)
    if "accuracy" in method:
        return p.max()
    return _entropy(p)


@njit(cache=True)
def _score(table, method, costs, multiclass):
    counts = table.sum(axis=0)
    sizes = table.sum(axis=1)
    n, classes = counts.sum(), len(counts)
    weights = sizes / n
    dist = table / sizes.reshape((-1, 1))
    prior = counts / n
    if method == "mdl":
        return (_mdl(counts) - sum_mdl(table)) / n
    if method == "equalhellinger":
        total = 0.0
        for i in range(len(sizes)):
            for j in range(i):
                total += _pair_distance(dist[i], dist[j], method)
        return total * 2 / (len(sizes) * (len(sizes) - 1))
    if method.startswith("dist") or method.startswith("impurity"):
        distributions = table / counts
        total = np.zeros(len(sizes)) if method.startswith("impurity") else np.zeros(1)
        pairs = 0
        for i in range(classes):
            for j in range(i if multiclass <= 2 else 1):
                if method.startswith("impurity"):
                    a = dist[:, i]
                    b = dist[:, j] if multiclass <= 2 else 1 - a
                    value = (
                        (np.sqrt(a) - np.sqrt(b)) ** 2
                        if method.endswith("hellinger")
                        else (a - b) ** 2
                    )
                else:
                    a = distributions[:, i].copy()
                    b = (
                        distributions[:, j]
                        if multiclass <= 2
                        else (sizes - table[:, i]) / (n - counts[i])
                    )
                    value = np.array([_pair_distance(a, b.copy(), method)])
                if multiclass in (1, 3):
                    total += value
                else:
                    total = np.maximum(total, value)
                pairs += 1
        if multiclass in (1, 3):
            total /= pairs
        return np.sqrt(total.sum()) if method.startswith("impurity") else total[0]
    baseline = _impurity(prior, method, costs)
    if method.startswith("uniform"):
        adjusted = table / counts
        masses = adjusted.sum(axis=1)
        dist = adjusted / masses.reshape((-1, 1))
        weights = masses / classes
    child = np.empty(len(weights))
    for i in range(len(child)):
        child[i] = _impurity(dist[i], method, costs)
    weighted = np.dot(weights, child)
    if method == "myopicrelieff":
        equal = weights**2
        return (np.dot(equal, child) - equal.sum() * baseline) / (
            baseline * (1 - baseline)
        )
    if method.startswith("equal"):
        return child.sum() if method == "equalgini" else -child.sum()
    if method.startswith("uniform"):
        if method == "uniformaccuracy":
            return weighted
        return weighted if method == "uniformgini" else 1 - weighted
    if method in ("gini", "accuracy"):
        return weighted - baseline
    gain = baseline - weighted
    return gain / _entropy(weights) if method.startswith("gainratio") else gain


@njit(cache=True)
def sum_mdl(table):
    total = 0.0
    for counts in table:
        total += _mdl(counts)
    return total


@njit(cache=True)
def split_scores(x, y, is_category, method, costs, min_size, multiclass):
    classes = int(y.max()) + 1
    scores = np.full(x.shape[1], -np.finfo(np.float64).max)
    for f in range(x.shape[1]):
        col = x[:, f]
        levels = np.unique(col)
        if len(levels) < 2:
            continue
        codes = np.searchsorted(levels, col)
        table = np.zeros((len(levels), classes))
        for i in range(len(y)):
            table[codes[i], y[i]] += 1
        if is_category[f]:
            scores[f] = _score(table, method, costs, multiclass)
        else:
            split = np.zeros((2, classes))
            split[1] = table.sum(axis=0)
            for t in range(len(levels) - 1):
                split[0] += table[t]
                split[1] -= table[t]
                if split[0].sum() >= min_size and split[1].sum() > min_size:
                    score = _score(split, method, costs, multiclass)
                    if score > scores[f]:
                        scores[f] = score
    return scores
