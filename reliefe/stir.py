"""Binary STIR pseudo-t ranking, using the authors' Manhattan-distance formulation."""
import numpy as np
from numba import njit

from .native import scale_columns


@njit(cache=True)
def _global_radius(x, sd_fraction):
    count, mean, m2 = 0, 0., 0.
    for i in range(len(x)-1):
        for j in range(i+1, len(x)):
            distance = np.abs(x[i]-x[j]).sum()
            count += 1
            delta = distance - mean
            mean += delta / count
            m2 += delta * (distance - mean)
    return mean - sd_fraction * np.sqrt(m2 / (count-1))


@njit(cache=True)
def _stir(x, y, method, k, sd_fraction):
    n, p = x.shape
    global_radius = _global_radius(x, sd_fraction) if method == "stir-surf" else 0.
    # Each eligible query contributes equal total weight to each group.
    means = np.zeros((2, p))
    within = np.zeros((2, p))
    counts = np.zeros(2, np.int64)
    eligible = 0
    for i in range(n):
        diff = np.abs(x - x[i])
        distances = diff.sum(axis=1)
        others = np.arange(n) != i
        if method == "stir-relieff":
            order = np.argsort(distances, kind="mergesort")
            hits = order[(y[order] == y[i]) & others[order]][:k]
            misses = order[y[order] != y[i]][:k]
        else:
            radius = global_radius
            if method == "stir-multisurf":
                d = distances[others]
                radius = d.mean() - sd_fraction * d.std() * np.sqrt(len(d)/(len(d)-1))
            near = (distances < radius) & others
            hits = np.where(near & (y == y[i]))[0]
            misses = np.where(near & (y != y[i]))[0]
            if len(hits) == 0 or len(misses) < 2:
                continue
        eligible += 1
        for g in range(2):
            rows = hits if g == 0 else misses
            counts[g] += len(rows)
            subset = diff[rows]
            mean = subset.sum(axis=0) / len(rows)
            change = mean - means[g]
            means[g] += change / eligible
            within[g] += ((subset - mean)**2).sum(axis=0) / len(rows) + change * (mean - means[g])
    if eligible == 0:
        return np.full(p, np.nan)
    # The reference divides query-weighted sums by all training rows, including
    # queries skipped by the radius rule, and pools using directed pair counts.
    scores = means * (eligible / n)
    variance = (within + eligible * (means - scores)**2) / n
    pooled = ((counts[0]-1)*variance[0] + (counts[1]-1)*variance[1]) / (counts.sum()-2)
    error = np.sqrt(pooled * (1./counts[0] + 1./counts[1]))
    return (scores[1] - scores[0]) / error


def stir_scores(method, x, y, k=5, sd_fraction=.5):
    counts = np.bincount(y)
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("STIR requires two classes with at least two observations each")
    if not isinstance(k, (int, np.integer)) or k < 1 or not np.isfinite(sd_fraction) or sd_fraction < 0:
        raise ValueError("Expected positive integer k and finite nonnegative sd_fraction")
    return _stir(scale_columns(x), y, method, min(k, counts.min()-1), sd_fraction)
