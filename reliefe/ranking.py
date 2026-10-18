"""A bounded, JSON-compatible classification interface for agents and scripts."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .reliefe_classes import ReliefE


MAX_SAMPLES = 512
MAX_FEATURES = 1024
MAX_CELLS = 131072


def rank_features(
    x: List[List[float]],
    y: List[Union[int, str]],
    feature_names: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    num_iter: int = 128,
    k: int = 1,
) -> Dict[str, Any]:
    """Rank numeric features for classification using direct Relief distances.

    Supply a rectangular finite numeric matrix and one integer/string class
    label per row. Each of at least two classes must have two or more rows.
    Limits: 512 rows, 1024 columns, 131072 cells, 512 iterations. k must be
    smaller than the smallest class size. Columns are scaled by their range;
    constant columns stay zero. Missing values must be imputed beforehand.

    Return scores in original column order plus a descending ranking with
    zero-based indices and optional names. Ties retain input column order.
    top_k limits the ranking only. num_iter is capped at the number of rows.
    Scores can be negative and are not probabilities or causal effects.
    This bounded interface is for small classification tables; use ReliefE
    directly for larger/sparse data, embeddings, and multi-label targets.
    """
    values = np.asarray(x)
    if values.ndim != 2 or not 4 <= values.shape[0] <= MAX_SAMPLES:
        raise ValueError("x must be a rectangular matrix with 4 to 512 rows.")
    n_samples, n_features = values.shape
    if not 1 <= n_features <= MAX_FEATURES or values.size > MAX_CELLS:
        raise ValueError("x must have 1 to 1024 columns and at most 131072 cells.")
    if values.dtype.kind not in "iuf":
        raise ValueError("x must contain only finite real numbers.")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("x must contain only finite real numbers.")
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.shape[0] != n_samples:
        raise ValueError("y must contain exactly one class label per row of x.")
    if labels.dtype.kind not in "iuU":
        raise ValueError("y must contain integer or string class labels.")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        raise ValueError("y must have at least two classes with at least two rows each.")
    for name, value, upper in (("num_iter", num_iter, MAX_SAMPLES),
                               ("k", k, int(counts.min()) - 1),
                               ("top_k", n_features if top_k is None else top_k, n_features)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= upper:
            raise ValueError("{} must be an integer between 1 and {}.".format(name, upper))
    if feature_names is not None:
        if isinstance(feature_names, str) or len(feature_names) != n_features or any(not isinstance(name, str) for name in feature_names):
            raise ValueError("feature_names must contain one string per column.")
        if len(set(feature_names)) != n_features:
            raise ValueError("feature_names must be unique.")

    # Divide first by magnitude to avoid overflow/underflow when taking ranges.
    magnitudes = np.max(np.abs(values), axis=0)
    values = values / np.where(magnitudes == 0, 1.0, magnitudes)
    ranges = np.ptp(values, axis=0)
    values = (values - np.min(values, axis=0)) / np.where(ranges == 0, 1.0, ranges)
    iterations = min(int(num_iter), n_samples)
    model = ReliefE(num_iter=iterations, k=int(k), normalize_descriptive=False,
                    embedding_based_distances=False, samples=n_samples,
                    sparsity_threshold=1.0, verbose=False)
    model.fit(values, labels, embedding_method="SVD")
    scores = model.feature_importances_
    if not np.isfinite(scores).all():
        raise RuntimeError("ReliefE produced non-finite feature scores.")
    order = np.argsort(-scores, kind="stable")[:top_k]
    ranking = [{"index": int(index), "score": float(scores[index]),
                "name": None if feature_names is None else feature_names[index]}
               for index in order]
    return {"scores": scores.tolist(), "ranking": ranking,
            "n_samples": n_samples, "n_features": n_features,
            "num_iter": iterations, "k": int(k),
            "method": "relief-direct", "normalization": "column-range"}
