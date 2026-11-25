"""Discrete forward ranking criteria from FEAST and praznik."""

import numpy as np
from numba import njit


CRITERIA = {
    "weighted-mim": "weighted mutual information maximisation",
    "weighted-cmim": "weighted conditional mutual information maximisation",
    "weighted-cmi": "weighted conditional mutual information",
    "weighted-jmi": "weighted joint mutual information",
    "weighted-disr": "weighted double input symmetrical relevance",
    "mifs": "mutual information with a redundancy penalty",
    "icap": "interaction capping",
    "condred": "conditional redundancy",
    "beta-gamma": "parameterized relevance and redundancy",
    "disr": "double input symmetrical relevance",
    "jmim": "joint mutual information maximisation",
    "njmim": "normalised joint mutual information maximisation",
    "cmi": "mutual information conditional on all selected columns",
    "jim": "joint Gini impurity gain",
}


@njit(cache=True)
def _joint(a, b):
    # Compact observed states at each merge, even when conditioning on many columns.
    keys = a * (b.max() + 1) + b
    return np.searchsorted(np.unique(keys), keys)


@njit(cache=True)
def _entropy(a, weights):
    counts = np.bincount(a)
    if weights is not None:
        # FEAST weights the information of each observation, using the
        # empirical frequencies of the original (unweighted) contingency table.
        return -np.mean(weights * np.log(counts[a] / len(a)))
    freq = counts[counts > 0] / len(a)
    return -np.sum(freq * np.log(freq))


@njit(cache=True)
def _purity(a, y):
    groups = np.bincount(a)
    # Accumulate only observed pairs. The full Cartesian product can be large.
    keys = a * (y.max() + 1) + y
    states = np.unique(keys)
    counts = np.bincount(np.searchsorted(states, keys)).astype(np.float64)
    return np.sum(counts**2 / groups[states // (y.max() + 1)]) / len(y)


@njit(cache=True)
def forward_ranks(columns, y, method, beta, gamma, weights):
    p = len(columns)
    target_entropy = _entropy(y, weights)
    marginal = np.array([_entropy(col, weights) for col in columns])
    with_target = np.array([_entropy(_joint(col, y), weights) for col in columns])
    relevance = marginal + target_entropy - with_target
    criterion = relevance.copy()
    if method == "jim":
        criterion = np.array([_purity(col, y) for col in columns])
    totals = np.zeros(p)
    minima = np.full(p, np.inf)
    ranks = np.zeros(p, dtype=np.int64)
    if method == "cmim":
        minima = relevance.copy()
    selected_state = np.zeros(len(y), dtype=np.int64)
    for step in range(p):
        # Entropies with the same theoretical value can differ by a few ULPs.
        maximum = np.max(criterion)
        best = np.flatnonzero(criterion >= maximum - 1e-13)[0]
        ranks[best] = step + 1
        criterion[best] = -np.inf
        if method == "mim":
            continue
        if method == "cmi":
            selected_state = _joint(selected_state, columns[best])
        for j in range(p):
            if ranks[j]:
                continue
            partner = selected_state if method == "cmi" else columns[best]
            pair = _joint(partner, columns[j])
            if method == "jim":
                totals[j] += _purity(pair, y)
                criterion[j] = totals[j]
                continue
            pair_entropy = _entropy(pair, weights)
            triple_entropy = _entropy(_joint(pair, y), weights)
            joint_mi = pair_entropy + target_entropy - triple_entropy
            if method == "cmi":
                # I((S,j);Y) differs from I(j;Y|S) by the same constant for all j.
                criterion[j] = joint_mi
            elif method == "cmim":
                minima[j] = min(minima[j], joint_mi - relevance[best])
                criterion[j] = minima[j]
            elif method == "jmi":
                totals[j] += joint_mi
                criterion[j] = totals[j]
            elif method in ("disr", "jmim", "njmim"):
                ratio = joint_mi
                if method != "jmim":
                    ratio = joint_mi / triple_entropy if triple_entropy > 0 else 0.0
                totals[j] += ratio
                minima[j] = min(minima[j], ratio)
                criterion[j] = totals[j] if method == "disr" else minima[j]
            else:
                redundancy = marginal[j] + marginal[best] - pair_entropy
                conditional = (
                    with_target[j] + with_target[best] - target_entropy - triple_entropy
                )
                penalty = beta * redundancy - gamma * conditional
                totals[j] += max(0.0, penalty) if method == "icap" else penalty
                criterion[j] = relevance[j] - totals[j]
    return ranks


def rank_discrete(columns, y, method, **params):
    weights = None
    if method.startswith("weighted-"):
        method = method[len("weighted-") :]
        weights = np.asarray(
            params.pop("sample_weight", np.ones(len(y))), dtype=np.float64
        )
        if (
            weights.shape != y.shape
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or weights.max() == 0
        ):
            raise ValueError(
                "sample_weight must contain one finite nonnegative weight per row and positive total"
            )
        # Every criterion is invariant to positive rescaling of all weights.
        weights = weights / weights.max()
    beta = params.pop("beta", 1.0) if method in ("mifs", "beta-gamma") else 1.0
    gamma = params.pop("gamma", 1.0) if method == "beta-gamma" else 1.0
    if params or not np.isfinite(beta) or not np.isfinite(gamma):
        raise ValueError(
            "Expected finite coefficients and supported criterion parameters"
        )
    if method == "mifs":
        gamma = 0.0
    if method == "condred":
        beta = 0.0
    return forward_ranks(columns, y, method, float(beta), float(gamma), weights)
