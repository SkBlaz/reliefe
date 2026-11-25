"""Regression split scores with node-local models and training-only split choice."""

import numpy as np
from numba import njit


@njit(cache=True)
def _linear_predict(design, y, simplify):
    n, p = design.shape
    mask = np.ones(p, np.bool_)
    coefficients = np.linalg.lstsq(design, y, rcond=1e-5)[0]
    best = design @ coefficients
    if simplify:
        error = np.abs(y - best).mean() * (n + p) / (n - p)
        for size in range(p - 1, 0, -1):
            best_trial = np.inf
            removed = -1
            fitted = design[:, mask].copy() @ coefficients
            for j, col in enumerate(np.where(mask)[0]):
                trial = fitted - design[:, col] * coefficients[j]
                score = np.abs(y - trial).mean() * (n + size) / (n - size)
                if score < best_trial:
                    removed, best_trial = col, score
            mask[removed] = False
            reduced = design[:, mask].copy()
            coefficients = np.linalg.lstsq(reduced, y, rcond=1e-5)[0]
            if best_trial <= error:
                error = best_trial
                best = reduced @ coefficients
    return np.minimum(y.max(), np.maximum(y.min(), best))


@njit(cache=True)
def _loss(design, y, model, absolute):
    if model == 2:
        predicted = np.full(len(y), np.sort(y)[len(y) // 2])
    elif model == 1 or len(y) <= design.shape[1]:
        predicted = np.full(len(y), y.mean())
    else:
        predicted = _linear_predict(design, y, model == 5)
    err = y - predicted
    return np.abs(err).sum() if absolute else np.dot(err, err)


@njit(cache=True)
def _split_loss(design, y, left, model, absolute):
    return (
        _loss(design[left], y[left], model, absolute)
        + _loss(design[~left], y[~left], model, absolute)
    ) / len(y)


@njit(cache=True)
def regression_splits(
    x, y, categorical, method, model, min_size, candidates, exhaustive
):
    n, p = x.shape
    design = np.ones((n, 1 + (~categorical).sum()))
    design[:, :-1] = x[:, ~categorical]
    mean_only = method == "mseofmean"
    scores = np.full(p, -np.finfo(np.float64).max)
    for f in range(p):
        levels = np.unique(x[:, f])
        codes = np.searchsorted(levels, x[:, f])
        if categorical[f] and mean_only:
            means = np.array([y[codes == v].mean() for v in range(len(levels))])
            codes = np.argsort(np.argsort(means))[codes]
        if categorical[f] and not mean_only and len(levels) > exhaustive:
            selected = np.zeros(len(levels), np.bool_)
            for _ in range(len(levels) - 1):
                best_loss, add = np.inf, -1
                for level in np.flatnonzero(~selected):
                    left = selected[codes] | (codes == level)
                    loss = _split_loss(design, y, left, model, method == "maeofmodel")
                    if loss < best_loss:
                        best_loss, add = loss, level
                selected[add] = True
                scores[f] = max(scores[f], -best_loss)
            continue
        if categorical[f] and not mean_only:
            # Exhaustive binary partitions for small-cardinality categoricals.
            total = 2 ** (len(levels) - 1)
        else:
            total = len(levels)
        for idx in range(1, total):
            if not categorical[f] and not mean_only and not candidates[f, idx]:
                continue
            left = (
                ((idx >> codes) & 1) != 0
                if categorical[f] and not mean_only
                else codes < idx
            )
            if (
                not mean_only
                and not categorical[f]
                and min(left.sum(), n - left.sum()) < min_size
            ):
                continue
            loss = _split_loss(
                design, y, left, 1 if mean_only else model, method == "maeofmodel"
            )
            scores[f] = max(scores[f], -loss)
    return scores
