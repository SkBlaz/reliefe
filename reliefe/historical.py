"""Kononenko (1994) RELIEF-A through RELIEF-F for discrete attributes."""
import numpy as np

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
    positive = np.zeros(x.shape[1])
    negative = np.zeros_like(positive)
    positive_counts = np.zeros_like(positive)
    negative_counts = np.zeros_like(positive)
    for i, row in enumerate(codes):
        observed = (codes >= 0) & (row >= 0)
        diff = (codes != row).astype(float)
        for j in np.flatnonzero(~observed.all(axis=0)):
            table = tables[j]
            unknown = ~observed[:, j]
            if (codes[:, j] < 0).all():
                diff[:, j] = 0
            elif variant in "bc":
                diff[unknown, j] = 1 - 1 / table.shape[1]
            elif row[j] >= 0:
                diff[unknown, j] = 1 - table[classes[unknown], row[j]]
            else:
                present = codes[:, j] >= 0
                prior = table[classes[i]]
                diff[present, j] = 1 - prior[codes[present, j]]
                diff[~present, j] = 1 - table[classes[~present]] @ prior
        distance = diff.sum(axis=1)
        distance[i] = np.inf
        order = np.argsort(distance, kind="stable")
        hit = order[classes[order] == classes[i]][:k]
        if variant == "c":
            negative += (diff[hit] * observed[hit]).sum(axis=0)
            negative_counts += observed[hit].sum(axis=0)
        else:
            negative += diff[hit].sum(axis=0)
            negative_counts += k
        if variant == "f":
            for c in np.flatnonzero(np.arange(len(labels)) != classes[i]):
                miss = order[classes[order] == c][:k]
                # Unconditional P(C) in the original 1994 multiclass equation.
                positive += (counts[c] / len(y)) * diff[miss].sum(axis=0)
            positive_counts += k
        else:
            miss = order[classes[order] != classes[i]][:k]
            if variant == "c":
                positive += (diff[miss] * observed[miss]).sum(axis=0)
                positive_counts += observed[miss].sum(axis=0)
            else:
                positive += diff[miss].sum(axis=0)
                positive_counts += k
    return positive / np.maximum(positive_counts, 1) - negative / np.maximum(negative_counts, 1)
