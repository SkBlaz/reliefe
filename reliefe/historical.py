"""Kononenko (1994) RELIEF-A through RELIEF-F for discrete attributes."""
import numpy as np
from numba import njit

METHODS_1994 = tuple("relief-1994-" + letter for letter in "abcdef")


def categorical_relief(x, y, variant, k=1):
    """Use all queries with equal weight; class frequencies are training-set frequencies."""
    labels, classes, counts = np.unique(y, return_inverse=True, return_counts=True)
    classes = classes.ravel()
    if variant in "abcd" and len(labels) != 2:
        raise ValueError("RELIEF-A through D require two classes")
    if not isinstance(k, (int, np.integer)) or not 1 <= k < min(counts):
        raise ValueError("Require 1 <= k < smallest class size")
    # Per attribute, represent each observed category by a compact code.
    # A negative code denotes a missing value.
    codes = np.full(x.shape, -1, dtype=int)
    tables = []
    for j, col in enumerate(x.T):
        present = ~np.isnan(col)
        categories, encoded = np.unique(col[present], return_inverse=True)
        codes[present, j] = encoded
        table = np.zeros((len(labels), max(1, len(categories))))
        np.add.at(table, (classes[present], encoded), 1)
        total = table.sum(axis=1, keepdims=True)
        # A class with no observations uses the pooled training distribution.
        pooled = table.sum(axis=0) / max(1, present.sum())
        table = np.divide(table, total, out=np.broadcast_to(pooled, table.shape).copy(), where=total > 0)
        tables.append(table)
    offsets = np.r_[0, np.cumsum([table.shape[1] for table in tables])].astype(np.int64)
    return _historical_scores(codes, classes, counts, np.hstack(tables), offsets, variant, k)


@njit(cache=True)
def _historical_scores(codes, classes, counts, probabilities, offsets, variant, k):
    n, p = codes.shape
    positive, negative = np.zeros(p), np.zeros(p)
    positive_counts, negative_counts = np.zeros(p), np.zeros(p)
    for i in range(n):
        diff = np.zeros((n, p))
        for j in range(n):
            for f in range(p):
                a, b = codes[i, f], codes[j, f]
                lo, hi = offsets[f], offsets[f + 1]
                if a >= 0 and b >= 0:
                    diff[j, f] = a != b
                elif variant == "b" or variant == "c":
                    diff[j, f] = 1 - 1 / (hi - lo)
                elif a >= 0:
                    diff[j, f] = 1 - probabilities[classes[j], lo + a]
                elif b >= 0:
                    diff[j, f] = 1 - probabilities[classes[i], lo + b]
                else:
                    left = probabilities[classes[i], lo:hi]
                    right = probabilities[classes[j], lo:hi]
                    # A completely unobserved column has no training distribution.
                    diff[j, f] = 1 - np.dot(left, right) if left.sum() > 0 else 0.
        distances = diff.sum(axis=1)
        distances[i] = np.inf
        order = np.argsort(distances, kind="mergesort")
        seen = np.zeros(len(counts), dtype=np.int64)
        misses = 0
        for j in order:
            if j == i:
                continue
            c = classes[j]
            hit = c == classes[i]
            if seen[c] >= k or (not hit and variant != "f" and misses >= k):
                continue
            seen[c] += 1
            if not hit:
                misses += 1
            for f in range(p):
                if variant == "c" and (codes[i, f] < 0 or codes[j, f] < 0):
                    continue
                if hit:
                    negative[f] += diff[j, f]
                    negative_counts[f] += 1
                elif variant == "f":
                    positive[f] += (counts[c] / n) * diff[j, f]
                else:
                    positive[f] += diff[j, f]
                    positive_counts[f] += 1
        if variant == "f":
            positive_counts += k
    return positive / np.maximum(positive_counts, 1) - negative / np.maximum(negative_counts, 1)
