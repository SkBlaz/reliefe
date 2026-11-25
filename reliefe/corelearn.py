"""Native evaluators of the criteria catalogued in CORElearn 1.56.0."""

import numpy as np

from .core_neighbors import class_neighbors, reg_neighbors
from .core_regression import regression_splits
from .split_scores import CRITERIA, split_scores

CLASSIFICATION = """ReliefFequalK ReliefFexpRank ReliefFbestK Relief InfGain
GainRatio MDL Gini MyopicReliefF Accuracy ReliefFmerit ReliefFdistance
ReliefFsqrDistance DKM ReliefFexpC ReliefFavgC ReliefFpe ReliefFpa ReliefFsmp
GainRatioCost DKMcost ReliefKukar MDLsmp ImpurityEuclid ImpurityHellinger
UniformDKM UniformGini UniformInf UniformAccuracy EqualDKM EqualGini EqualInf
EqualHellinger DistHellinger DistAUC DistAngle DistEuclid""".split()
REGRESSION = """RReliefFequalK RReliefFexpRank RReliefFbestK RReliefFwithMSE
MSEofMean MSEofModel MAEofModel RReliefFdistance RReliefFsqrDistance""".split()
CORE_METHODS = {"core-" + name.lower(): name for name in CLASSIFICATION + REGRESSION}


def _rng(seed):
    """Initialize MT19937 with the version used for the original reference scores."""
    if seed is None:
        return np.random.RandomState()
    state = []
    value = int(seed)
    for i in range(675):
        value = (69069 * value + 1) & 0xFFFFFFFF
        if i > 50:
            state.append(value)
    rng = np.random.RandomState()
    rng.set_state(("MT19937", np.asarray(state, np.uint32), 624, 0, 0.0))
    return rng


def _draw(rng, upper):
    return int(int(rng.randint(0, 2**32, dtype=np.uint32)) * (upper / 2**32))


def _sample(rng, pool, count):
    pool = pool.copy()
    result = []
    for i in range(count):
        pos = _draw(rng, len(pool) - i)
        result.append(pool[pos])
        pool[pos] = pool[len(pool) - i - 1]
    return result


def _queries(rng, y, n, costs=None):
    if costs is None:
        return (
            np.arange(n)
            if n == len(y)
            else np.asarray(_sample(rng, np.arange(len(y)), n))
        )
    prior = np.bincount(y) / len(y)
    expected = np.array(
        [
            sum(prior[j] * costs[i, j] for j in range(len(prior)) if i != j)
            / (1 - prior[i])
            for i in range(len(prior))
        ]
    )
    weights = prior * expected
    weights /= sum(weights)
    indices = []
    cumulative = 0.0
    for c, w in enumerate(weights):
        rows = np.flatnonzero(y == c)
        cumulative += w
        repeats = int(w * n) // len(rows)
        indices.extend(rows.tolist() * repeats)
        count = int(cumulative * n) - len(indices)
        indices.extend(_sample(rng, rows, count))
    while len(indices) < n:
        indices.append(_draw(rng, len(y)))
    return np.array(indices)


def core_scores(
    method,
    x,
    y,
    task,
    random_state,
    cost_matrix=None,
    categorical_features=(),
    kNearestEqual=10,
    kNearestExpRank=70,
    quotientExpRankDistance=20.0,
    numAttrProportionEqual=0.04,
    numAttrProportionDifferent=0.1,
    minNodeWeightEst=2.0,
    multiclassEvaluation=1,
    modelTypeReg=5,
    ReliefIterations=0,
    maxValues4Exhaustive=7,
    discretizationSample=50,
):
    method = method[5:]
    categorical = np.zeros(x.shape[1], dtype=np.bool_)
    indices = np.asarray(categorical_features)
    if indices.size:
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or indices.min() < 0
            or indices.max() >= len(categorical)
        ):
            raise ValueError("Expected zero-based categorical column indices")
        categorical[indices] = True
    classes = len(np.unique(y)) if task == "classification" else 0
    costs = (
        1.0 - np.eye(classes)
        if cost_matrix is None
        else np.asarray(cost_matrix, dtype=float)
    )
    if cost_matrix is not None and (
        not classes or costs.shape != (classes, classes) or not np.isfinite(costs).all()
    ):
        raise ValueError(
            "Expected a finite classification cost matrix in sorted class order"
        )
    if (
        kNearestEqual < 1
        or kNearestExpRank < 1
        or quotientExpRankDistance <= 0
        or maxValues4Exhaustive < 1
        or discretizationSample < 0
        or minNodeWeightEst < 1
        or not 0 <= numAttrProportionEqual < numAttrProportionDifferent
        or multiclassEvaluation not in (1, 2, 3, 4)
        or modelTypeReg not in (1, 2, 3, 5)
    ):
        raise ValueError("Invalid neighbor, model or split parameters")
    rng = _rng(random_state)
    if method == "mdlsmp":
        scores = np.empty(x.shape[1])
        for f in range(x.shape[1]):
            rows = (
                np.arange(len(y)) if categorical[f] else _queries(rng, y, len(y), costs)
            )
            scores[f] = split_scores(
                x[rows, f : f + 1],
                y[rows],
                categorical[f : f + 1],
                "mdl",
                costs,
                minNodeWeightEst,
                multiclassEvaluation,
            )[0]
        return scores
    if method in CRITERIA:
        return split_scores(
            x, y, categorical, method, costs, minNodeWeightEst, multiclassEvaluation
        )
    candidates = np.ones((x.shape[1], len(x)), dtype=bool) if not classes else None
    if method in ("mseofmean", "mseofmodel", "maeofmodel"):
        if method != "mseofmean" and discretizationSample:
            for f in np.flatnonzero(~categorical):
                count = len(np.unique(x[:, f]))
                if count > discretizationSample:
                    chosen = sorted(
                        _sample(rng, np.arange(count), discretizationSample)
                    )
                    candidates[f] = False
                    candidates[f, chosen[1:]] = True
        return regression_splits(
            x,
            y,
            categorical,
            method,
            modelTypeReg,
            minNodeWeightEst,
            candidates,
            maxValues4Exhaustive,
        )
    n = len(y)
    iterations = (
        int(np.log(n))
        if ReliefIterations == -1
        else int(np.sqrt(n))
        if ReliefIterations == -2
        else min(n, ReliefIterations)
        if ReliefIterations > 0
        else n
    )
    rows = _queries(rng, y, iterations, costs if method == "relieffsmp" else None)
    k = (
        n
        if method.endswith("bestk")
        else kNearestEqual
        if method.endswith("equalk")
        else kNearestExpRank
    )
    args = (
        k,
        quotientExpRankDistance,
        numAttrProportionEqual,
        numAttrProportionDifferent,
        rows,
    )
    if classes:
        return class_neighbors(x, y, categorical, costs, method, *args)
    if method == "rrelieffwithmse":
        relief = reg_neighbors(x, y, categorical, "rrelieffexprank", *args)
        mse = regression_splits(
            x,
            y,
            categorical,
            "mseofmean",
            1,
            minNodeWeightEst,
            candidates,
            maxValues4Exhaustive,
        )
        weight = np.exp(-((50 / n) ** 2))
        return weight * (relief - relief.min()) / np.ptp(relief) + (1 - weight) * (
            mse - mse.min()
        ) / np.ptp(mse)
    return reg_neighbors(x, y, categorical, method, *args)
