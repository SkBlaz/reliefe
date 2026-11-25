"""Marginal feature screening by distance and kernel dependence."""

import numpy as np
from numba import njit


@njit(cache=True)
def _centered_matrix(values, categorical, gaussian, bandwidth):
    n = values.shape[0]
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if categorical:
                distance = 1.0 if values[i, 0] != values[j, 0] else 0.0
            else:
                distance = np.linalg.norm(values[i] - values[j])
            matrix[i, j] = distance
            matrix[j, i] = distance
    if gaussian:
        if categorical:
            matrix = 1.0 - matrix
        else:
            if bandwidth == 0.0:
                flat = matrix.ravel()
                positive = flat[flat > 0]
                bandwidth = np.median(positive) if len(positive) else 1.0
            matrix = np.exp(-0.5 * (matrix / bandwidth) ** 2)
    means = matrix.sum(axis=0) / n
    grand = means.mean()
    for i in range(n):
        for j in range(n):
            matrix[i, j] += grand - means[i] - means[j]
    return matrix


@njit(cache=True)
def _screen(x, y, categorical, gaussian, bandwidth, target_bandwidth):
    target = _centered_matrix(y, categorical, gaussian, target_bandwidth)
    target_norm = np.linalg.norm(target)
    scores = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        feature = _centered_matrix(x[:, j : j + 1], False, gaussian, bandwidth)
        product = np.sum(feature * target)
        if gaussian:
            # Biased empirical HSIC with the (n-1)^2 normalization.
            scores[j] = max(0.0, product) / (len(x) - 1) ** 2
        else:
            norm = np.linalg.norm(feature) * target_norm
            scores[j] = max(0.0, product / norm) if norm else 0.0
    return scores


def dependence_scores(x, y, task, method, bandwidth=None, target_bandwidth=None):
    """DC-SIS returns squared distance correlation. HSIC uses a Gaussian input
    kernel and either a class-equality kernel or a Gaussian response kernel.
    Missing Gaussian widths use the median nonzero pairwise distance.
    """
    gaussian = method == "hsic"
    if len(x) < 2 or any(
        v is not None and (not gaussian or not np.isfinite(v) or v <= 0)
        for v in (bandwidth, target_bandwidth)
    ):
        raise ValueError("Expected two or more rows and positive HSIC bandwidths")
    target = np.ascontiguousarray(y, dtype=np.float64).reshape(len(x), -1)
    if not np.isfinite(target).all():
        raise ValueError("Dependence screening requires finite targets")
    return _screen(
        x,
        target,
        task == "classification",
        gaussian,
        bandwidth or 0.0,
        target_bandwidth or 0.0,
    )
