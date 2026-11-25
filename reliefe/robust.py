"""Robust feature selection using joint L2,1 residual and coefficient penalties."""

import numpy as np
from numba import njit


@njit(cache=True)
def _row_norms(matrix):
    return np.sqrt(np.sum(matrix * matrix, axis=1))


@njit(cache=True)
def _rfs(x, target, gamma, tolerance, max_iter):
    n, p = x.shape
    feature_scale = np.ones(p)
    residual_scale = np.ones(n)
    previous = np.inf
    scores = np.zeros(p)
    for _ in range(max_iter):
        # Eliminate the identity block of the augmented [X, gamma I] system.
        weighted_x = x * feature_scale
        system = weighted_x @ x.T
        for i in range(n):
            system[i, i] += gamma**2 * residual_scale[i] + 1e-6
        dual = np.linalg.solve(system, target)
        coefficients = weighted_x.T @ dual
        norms = _row_norms(coefficients)
        scores = norms**2
        objective = _row_norms(x @ coefficients - target).sum() + gamma * norms.sum()
        if abs(previous - objective) < tolerance:
            break
        previous = objective
        feature_scale = 2.0 * np.maximum(norms, 1e-16)
        residual_scale = 2.0 * np.maximum(
            gamma * residual_scale * _row_norms(dual), 1e-16
        )
    return scores


def robust_scores(x, y, task, gamma=1.0, tolerance=1e-3, max_iter=1000):
    """Return squared coefficient row norms, using indicator columns for class labels."""
    if (
        not np.isfinite(gamma)
        or gamma <= 0
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("gamma must be positive and tolerance must be nonnegative")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if task == "classification":
        target = np.eye(y.max() + 1)[y]
    else:
        target = np.ascontiguousarray(y, dtype=np.float64).reshape(len(x), -1)
        if not np.isfinite(target).all():
            raise ValueError("RFS requires finite responses")
    return _rfs(x, target, gamma, tolerance, max_iter)
