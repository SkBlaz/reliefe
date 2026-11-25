"""Neighborhood scoring conventions used by the CORElearn 1.56 evaluators."""

import numpy as np
from numba import njit


@njit(cache=True)
def _sift(heap, distances, root, end):
    while root * 2 + 1 < end:
        child = root * 2 + 1
        if child + 1 < end and distances[heap[child + 1]] < distances[heap[child]]:
            child += 1
        if distances[heap[root]] <= distances[heap[child]]:
            break
        heap[root], heap[child] = heap[child], heap[root]
        root = child


@njit(cache=True)
def _neighbors(rows, distances):
    # The reference resolves exact ties by extracting from a binary min-heap.
    heap = rows.copy()
    for root in range(len(heap) // 2 - 1, -1, -1):
        _sift(heap, distances, root, len(heap))
    for end in range(len(heap) - 1, 0, -1):
        heap[0], heap[end] = heap[end], heap[0]
        _sift(heap, distances, 0, end)
    return heap[::-1]


@njit(cache=True)
def _ramp(diff, span, equal, different):
    if span == 0.0:
        return 0.0
    if diff >= span * different:
        return 1.0
    if diff <= span * equal:
        return 0.0
    return (diff - span * equal) / (span * (different - equal))


@njit(cache=True)
def _diff(x, i, spans, cat, equal, different):
    diff = np.empty_like(x)
    for j in range(len(x)):
        for f in range(x.shape[1]):
            if cat[f]:
                diff[j, f] = x[j, f] != x[i, f]
            else:
                diff[j, f] = _ramp(abs(x[i, f] - x[j, f]), spans[f], equal, different)
    return diff


@njit(cache=True)
def _weight(method, rank, distance, previous, factor, sigma, min_nonzero):
    if method.endswith("sqrdistance"):
        return 1 / distance**2 if distance > 0 else 2 / min_nonzero**2
    if method.endswith("distance"):
        return 1 / distance if distance > 0 else 2 / min_nonzero
    if method.endswith("equalk") or method.endswith("bestk"):
        return 1.0
    return factor if previous == distance else np.exp(-((rank / sigma) ** 2))


@njit(cache=True)
def _class_weights(prior, costs, method):
    n = len(prior)
    weights = np.zeros((n, n))
    exp_cost = (costs @ prior) / (1 - prior)
    avg_cost = costs.sum(axis=1) / (n - 1)
    for i in range(n):
        for c in range(n):
            if method == "relieffavgc":
                weights[i, c] = -avg_cost[i] if c == i else costs[i, c] / (n - 1)
            elif method == "relieffexpc":
                weights[i, c] = (
                    -exp_cost[i] if c == i else prior[c] * costs[i, c] / (1 - prior[i])
                )
            elif method == "relieffpa":
                weights[i, c] = (
                    -1.0 if c == i else avg_cost[c] / (avg_cost.sum() - avg_cost[i])
                )
            elif method == "relieffpe":
                p = prior * exp_cost
                weights[i, c] = -1.0 if c == i else p[c] / (p.sum() - p[i])
            else:
                weights[i, c] = -1.0 if c == i else prior[c] / (1 - prior[i])
    return weights


@njit(cache=True)
def class_neighbors(x, y, cat, costs, method, k, sigma, equal, different, queries):
    n, p = x.shape
    freq = np.bincount(y)
    priors = freq / n
    factors = _class_weights(priors, costs, method)
    spans = np.array([np.ptp(x[:, f]) for f in range(p)])
    limits = freq.max() if method == "relieffbestk" else 1
    scores = np.zeros((limits, p))
    smoothed = (freq + 1.0) / (n + len(freq))
    kukar = (costs @ smoothed) / (1 - smoothed)
    kukar /= np.dot(kukar, smoothed)
    for i in queries:
        delta = _diff(x, i, spans, cat, equal, different)
        d = delta.sum(axis=1)
        if method in ("relief", "reliefkukar"):
            hit = np.argmin(np.where((y == y[i]) & (np.arange(n) != i), d, np.inf))
            miss = np.argmin(np.where(y != y[i], d, np.inf))
            scores[0] += (
                delta[miss] * (kukar[y[i]] if method == "reliefkukar" else 1.0)
                - delta[hit]
            )
            continue
        for c in range(len(freq)):
            order = _neighbors(np.flatnonzero((y == c) & (d > 0)), d)
            numerator, denom, factor, previous = np.zeros(p), 0.0, 1.0, -1.0
            for rank in range(min(len(order), k)):
                j = order[rank]
                factor = _weight(method, rank, d[j], previous, factor, sigma, 1.0)
                previous = d[j]
                value = delta[j] / d[j] if method == "relieffmerit" else delta[j]
                numerator += value * factor
                denom += factor
                if limits > 1:
                    scores[rank] += factors[y[i], c] * numerator / denom
            if denom:
                amount = factors[y[i], c] * numerator / denom
                if limits == 1:
                    scores[0] += amount
                else:
                    for t in range(min(len(order), k), limits):
                        scores[t] += amount
    return np.array([scores[:, f].max() / len(queries) for f in range(p)])


@njit(cache=True)
def reg_neighbors(x, y, cat, method, k, sigma, equal, different, queries):
    n, p = x.shape
    spans = np.array([np.ptp(x[:, f]) for f in range(p)])
    yrange = np.ptp(y)
    size = min(k, n - 1)
    levels = size if method.endswith("bestk") else 1
    a, b, t = np.zeros((levels, p)), np.zeros((levels, p)), np.zeros(levels)
    for i in queries:
        delta = _diff(x, i, spans, cat, equal, different)
        d = delta.sum(axis=1)
        order = _neighbors(np.arange(n), d).copy()
        pos = np.where(order == i)[0][0]
        order[pos] = order[0]
        order = order[1 : size + 1]
        positive = d[order][d[order] > 0]
        minimum = positive.min() if len(positive) else 1.0
        wa, wb = np.zeros(p), np.zeros(p)
        wt, denom, previous, weight = 0.0, 0.0, -1.0, 1.0
        for rank in range(size):
            j = order[rank]
            weight = _weight(method, rank, d[j], previous, weight, sigma, minimum)
            previous = d[j]
            dy = _ramp(abs(y[i] - y[j]), yrange, equal, different)
            wa += weight * delta[j]
            wb += weight * dy * delta[j]
            wt += weight * dy
            denom += weight
            if levels > 1 or rank == size - 1:
                slot = rank if levels > 1 else 0
                a[slot] += wa / denom
                b[slot] += wb / denom
                t[slot] += wt / denom
    out = np.full(p, -1.0)
    for level in range(levels):
        if 0 < t[level] < len(queries):
            out = np.maximum(
                out,
                b[level] / t[level] - (a[level] - b[level]) / (len(queries) - t[level]),
            )
    return out
