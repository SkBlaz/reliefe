"""Zhao and Liu's three SPEC scores on a symmetric sample affinity graph."""

import numpy as np
from numba import njit


@njit(cache=True)
def _rbf_graph(x, gamma):
    n = len(x)
    graph = np.eye(n)
    for i in range(n):
        for j in range(i):
            difference = x[i] - x[j]
            graph[i, j] = np.exp(-gamma * np.dot(difference, difference))
            graph[j, i] = graph[i, j]
    return graph


@njit(cache=True)
def _spec_scores(x, graph, style, clusters):
    roots = np.sqrt(graph.sum(axis=1))
    inv = np.zeros(len(x))
    active = roots > 0
    inv[active] = 1.0 / roots[active]
    laplacian = np.diag(active.astype(np.float64)) - inv.reshape(-1, 1) * graph * inv
    dc = roots / np.linalg.norm(roots)
    operator = laplacian
    if style == "spec-cluster":
        # Lift the known constant eigenvector above the Laplacian spectrum.
        # Average a tied boundary eigenspace over its interchangeable directions.
        eigenvalues, vectors = np.linalg.eigh(laplacian + 3.0 * np.outer(dc, dc))
        cutoff = eigenvalues[clusters - 2]
        tied = np.abs(eigenvalues - cutoff) < 1e-12
        lower = eigenvalues < cutoff - 1e-12
        included = lower.astype(np.float64) + tied * (
            (clusters - 1 - lower.sum()) / tied.sum()
        )
        operator = (vectors * (included * (2 - eigenvalues))) @ vectors.T
    result = np.empty(x.shape[1])
    for j in range(x.shape[1]):
        scale = np.max(np.abs(x[:, j]))
        f = roots * (x[:, j] / scale) if scale else np.zeros(len(x))
        original_norm = np.linalg.norm(f)
        if style == "spec":
            # Remove the constant graph signal before normalization.
            f -= dc * np.dot(f, dc)
        norm = np.linalg.norm(f)
        if norm <= 100 * np.finfo(np.float64).eps * original_norm:
            result[j] = np.nan
            continue
        f /= norm
        result[j] = np.dot(f, operator @ f)
    return result


def spectral_scores(x, method, affinity=None, gamma=1.0, clusters=2):
    """Use exp(-gamma * squared distance) by default, or supply a sample affinity matrix."""
    if len(x) < 2 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("SPEC requires two rows and positive finite gamma")
    if not isinstance(clusters, (int, np.integer)) or not 2 <= clusters <= len(x):
        raise ValueError("clusters must be between two and the sample count")
    if affinity is None:
        graph = _rbf_graph(x, gamma)
    else:
        graph = np.asarray(affinity, dtype=np.float64)
        if (
            graph.shape != (len(x), len(x))
            or not np.isfinite(graph).all()
            or np.any(graph < 0)
            or not np.array_equal(graph, graph.T)
            or not graph.any()
        ):
            raise ValueError(
                "Expected a finite nonnegative symmetric affinity matrix with positive total weight"
            )
    return _spec_scores(x, graph, method, clusters)
