"""Native Numba implementations of neighborhood and information scoring rules.

Origins and reference implementations are listed in the README catalog.
The radius methods hold one row of distances at a time.
"""
import numpy as np
from numba import njit


RADIUS_METHODS = ("surf", "surf-star", "multisurf", "multisurf-star", "swrf",
                  "swrf-star", "multiswrf", "multiswrf-star", "multiswrfdb",
                  "multiswrfdb-star")


@njit(cache=True)
def scale_columns(x):
    values = np.empty_like(x)
    for f in range(x.shape[1]):
        col = x[:, f]
        mag = np.max(np.abs(col))
        scaled = col / max(mag, np.finfo(np.float64).tiny)
        low, high = scaled.min(), scaled.max()
        values[:, f] = (scaled - low) / (high - low if high > low else 1.)
    return values


@njit(cache=True)
def original_scores(x, y):
    scores = np.zeros(x.shape[1])
    for i in range(len(y)):
        hit, miss = -1, -1
        hit_distance, miss_distance = np.inf, np.inf
        for j in range(len(y)):
            if j == i:
                continue
            delta = x[i] - x[j]
            distance = np.dot(delta, delta)
            if y[j] == y[i] and distance < hit_distance:
                hit, hit_distance = j, distance
            elif y[j] != y[i] and distance < miss_distance:
                miss, miss_distance = j, distance
        scores += (x[i] - x[miss]) ** 2 - (x[i] - x[hit]) ** 2
    return scores / len(y)


@njit(cache=True)
def regression_scores(x, y, influence):
    different_target = 0.
    different_feature = np.zeros(x.shape[1])
    different_both = np.zeros(x.shape[1])
    for i in range(len(y)):
        distances = np.sum(np.abs(x - x[i]), axis=1)
        distances[i] = np.inf
        order = np.argsort(distances, kind="mergesort")
        for rank in range(len(influence)):
            j = order[rank]
            delta = np.abs(x[i] - x[j])
            weight = influence[rank]
            target_weight = weight * abs(y[i] - y[j])
            different_target += target_weight
            different_feature += weight * delta
            different_both += target_weight * delta
    if not 0 < different_target < len(y):
        return np.zeros(x.shape[1])
    return different_both / different_target - (different_feature - different_both) / (len(y) - different_target)


@njit(cache=True)
def _row_distances(x, i, categorical):
    distances = np.zeros(x.shape[0])
    for j in range(x.shape[0]):
        for f in range(x.shape[1]):
            if categorical[f]:
                distances[j] += x[i, f] != x[j, f]
            else:
                distances[j] += abs(x[i, f] - x[j, f])
    return distances


@njit(cache=True)
def radius_scores(x, labels, categorical, local, deadband, spatial, far, invert_far, regression):
    """SURF/SWRF family. Per-zone updates use group counts, with a sigmoid for SWRF."""
    n, p = x.shape
    nclasses = 2 if regression else int(labels.max()) + 1
    target_radius = labels.std() * np.sqrt(n / (n - 1))
    mean, std = 0., 0.
    if not local:
        # Welford update on unordered pairs avoids materializing the n-by-n matrix.
        count, m2 = 0, 0.
        for i in range(n - 1):
            distances = _row_distances(x, i, categorical)
            for d in distances[i + 1:]:
                count += 1
                change = d - mean
                mean += change / count
                m2 += change * (d - mean)
        # Dense ReBATE includes the zero diagonal in its global distance moments.
        proportion = (n - 1) / n
        variance = proportion * (m2 / count) + proportion * (1 - proportion) * mean ** 2
        mean *= proportion
        std = np.sqrt(max(0., variance))

    scores = np.zeros(p)
    for i in range(n):
        distances = _row_distances(x, i, categorical)
        if local:
            others = distances[np.arange(n) != i]
            mean, std = others.mean(), others.std()
        width = std * .5 if deadband else 0.
        sums = np.zeros((2, nclasses, p))
        counts = np.zeros((2, nclasses), dtype=np.int64)
        own = 0 if regression else int(labels[i])
        for j in range(n):
            if i == j:
                continue
            d = distances[j]
            if d < mean - width:
                zone, boundary = 0, mean - width
            elif far and d > mean + width:
                zone, boundary = 1, mean + width
            else:
                continue
            group = int(abs(labels[i] - labels[j]) >= target_radius) if regression else int(labels[j])
            counts[zone, group] += 1
            weight = np.tanh(2 * abs(d - boundary) / std) if spatial and std > 0 else 1.
            for f in range(p):
                delta = (1. if x[i, f] != x[j, f] else 0.) if categorical[f] else abs(x[i, f] - x[j, f])
                value = weight * delta
                if zone == 1:
                    if invert_far:
                        value = -value
                    elif categorical[f]:
                        value = weight * (1 - delta)
                    else:
                        value = 1 - value
                sums[zone, group, f] += value
        for zone in range(2):
            hits = counts[zone, own]
            misses = counts[zone].sum() - hits
            if hits:
                scores -= sums[zone, own] / hits
            if misses:
                for c in range(nclasses):
                    if c != own:
                        factor = (nclasses - 1) * counts[zone, c] / (misses * misses)
                        scores += factor * sums[zone, c]
    return scores / n


def native_radius(method, x, y, regression, categorical_features=()):
    if len(y) < 2:
        raise ValueError("Neighborhood ranking requires at least two rows")
    categorical = np.zeros(x.shape[1], dtype=bool)
    for index in categorical_features:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(categorical):
            raise ValueError("Expected zero-based categorical column indices")
        categorical[index] = True
    if regression and np.ptp(y) == 0:
        return np.zeros(x.shape[1])
    values = scale_columns(x)
    values[:, categorical] = x[:, categorical]
    return radius_scores(values, y.astype(np.float64), categorical,
                         method.startswith("multi"), method.startswith("multisurf") or "db" in method,
                         "swrf" in method, method.endswith("-star"), method == "surf-star", regression)


@njit(cache=True)
def mutual_information(a, b, a_size, b_size):
    """Empirical mutual information of compact nonnegative integer codes, in nats."""
    counts = np.zeros((a_size, b_size), dtype=np.int64)
    a_counts = np.zeros(a_size, dtype=np.int64)
    b_counts = np.zeros(b_size, dtype=np.int64)
    for i in range(len(a)):
        counts[a[i], b[i]] += 1
        a_counts[a[i]] += 1
        b_counts[b[i]] += 1
    score = 0.
    for i in range(a_size):
        for j in range(b_size):
            count = counts[i, j]
            if count:
                score += count / len(a) * np.log((count * len(a)) / (a_counts[i] * b_counts[j]))
    return score


@njit(cache=True)
def greedy_information(columns, y, classes, bins, conditional, worst_case, cumulative):
    p = len(columns)
    relevance = np.empty(p)
    for j in range(p):
        relevance[j] = mutual_information(columns[j], y, bins, classes)
    penalty, worst = np.zeros(p), np.full(p, -np.inf)
    ranks = np.zeros(p, dtype=np.int64)
    for step in range(p):
        if worst_case and step:
            criterion = relevance - worst
        else:
            criterion = relevance - penalty / (1 if cumulative else max(step, 1))
        criterion[ranks > 0] = -np.inf
        best = np.argmax(criterion)
        ranks[best] = step + 1
        for j in range(p):
            if ranks[j]:
                continue
            change = mutual_information(columns[best], columns[j], bins, bins)
            if conditional:
                for c in range(classes):
                    rows = y == c
                    change -= rows.sum() / len(y) * mutual_information(
                        columns[best][rows], columns[j][rows], bins, bins)
            penalty[j] += change
            worst[j] = max(worst[j], change)
    return ranks


@njit(cache=True)
def concordance_scores(x, event, time, tied_tol):
    pairs = 0
    credit = np.zeros(x.shape[1])
    for i in range(len(event)):
        if event[i]:
            for j in range(len(event)):
                if time[j] > time[i] or (time[j] == time[i] and not event[j]):
                    pairs += 1
                    for f in range(x.shape[1]):
                        delta = x[i, f] - x[j, f]
                        if delta > tied_tol:
                            credit[f] += 1
                        elif abs(delta) <= tied_tol:
                            credit[f] += .5
    if not pairs:
        raise ValueError("No comparable survival pairs")
    return credit / pairs


@njit(cache=True)
def irelief_scores(x, y, max_iter, width, tolerance):
    """Sun and Li: conditional expected margins and a nonnegative unit L2 update."""
    n, p = x.shape
    weights = np.ones(p) / p
    for _ in range(max_iter):
        margin = np.zeros(p)
        for i in range(n):
            diffs = np.abs(x - x[i])
            distances = diffs @ weights
            distances[i] = np.inf
            # A common shift cancels in the hit, miss and inlier probabilities.
            affinity = np.exp(-(distances - distances.min()) / width)
            hits = affinity * (y == y[i])
            misses = affinity * (y != y[i])
            h, m = hits.sum(), misses.sum()
            if h > 0 and m > 0:
                margin += (h / (h + m)) * ((misses / m - hits / h) @ diffs)
        updated = np.maximum(margin, 0)
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated /= norm
        change = np.abs(updated - weights).sum()
        weights = updated
        if change < tolerance:
            break
    return weights


@njit(cache=True)
def mss_scores(x, y, k):
    """Neighbor differences above the average difference of the other attributes."""
    n, p = x.shape
    scores = np.zeros(p)
    counts = np.bincount(y)
    for i in range(n):
        diffs = np.abs(x - x[i])
        distances = diffs.sum(axis=1)
        distances[i] = np.inf
        selected = np.zeros(len(counts), dtype=np.int64)
        for j in np.argsort(distances, kind="mergesort"):
            c = y[j]
            if j == i or selected[c] >= k:
                continue
            selected[c] += 1
            weight = -1. if c == y[i] else counts[c] / (n - counts[y[i]])
            for f in range(p):
                background = (distances[j] - diffs[j, f]) / (p - 1)
                scores[f] += weight * max(0., diffs[j, f] - background)
    return scores / (n * k)


def native_margin(method, x, y, *, k=5, max_iter=100, width=2., tolerance=1e-6):
    counts = np.bincount(y)
    if counts.min() < 2:
        raise ValueError("Each class needs at least two rows")
    x = scale_columns(x)
    if method == "irelief":
        if (not isinstance(max_iter, (int, np.integer)) or max_iter < 1 or not np.isfinite(width)
                or width <= 0 or not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Expected positive iteration count and width, and nonnegative tolerance")
        return irelief_scores(x, y, max_iter, width, tolerance)
    if x.shape[1] < 2 or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("ReliefMSS requires multiple columns and positive integer k")
    return mss_scores(x, y, min(k, counts.min() - 1))


@njit(cache=True)
def categorical_impurity(codes, y, bins, classes, method):
    """Information gain, gain ratio, symmetrical uncertainty and reduction in Gini."""
    n = len(y)
    label_counts = np.bincount(y, minlength=classes)
    py = label_counts / n
    entropy_y = -np.sum(py * np.log(py))
    scores = np.zeros(len(codes))
    for f in range(len(codes)):
        a = codes[f]
        counts = np.bincount(a, minlength=bins)
        if method == "gini":
            joint = np.zeros((bins, classes))
            for i in range(n):
                joint[a[i], y[i]] += 1
            for b in range(bins):
                if counts[b]:
                    scores[f] += np.sum(joint[b] ** 2) / (n * counts[b])
            scores[f] -= np.sum(py * py)
        else:
            gain = mutual_information(a, y, bins, classes)
            px = counts[counts > 0] / n
            entropy_x = -np.sum(px * np.log(px))
            if method == "gain-ratio":
                scores[f] = gain / entropy_x if entropy_x > 0 else 0.
            elif method == "symmetrical-uncertainty":
                scores[f] = 2 * gain / (entropy_x + entropy_y)
            else:
                scores[f] = gain
    return scores


@njit(cache=True)
def _average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[start]] == values[order[stop]]:
            stop += 1
        for pos in range(start, stop):
            ranks[order[pos]] = .5 * (start + 1 + stop)
        start = stop
    return ranks


@njit(cache=True)
def univariate_scores(x, y, method, regression):
    n, p = x.shape
    scores = np.zeros(p)
    if method == "variance":
        for j in range(p):
            scores[j] = x[:, j].var()
        return scores
    if regression:
        response = _average_ranks(y) if method == "spearman" else y
        response = response - response.mean()
        y_norm = np.linalg.norm(response)
        for j in range(p):
            col = _average_ranks(x[:, j]) if method == "spearman" else x[:, j]
            centered = col - col.mean()
            norm = np.linalg.norm(centered) * y_norm
            r = min(1., abs(np.dot(centered, response)) / norm) if norm else 0.
            if method == "f":
                scores[j] = r*r*(n-2)/(1-r*r) if r < 1 else np.finfo(np.float64).max
            else:
                scores[j] = r
        return scores
    labels = y.astype(np.int64)
    counts = np.bincount(labels)
    for j in range(p):
        col = x[:, j]
        totals = np.bincount(labels, weights=col, minlength=len(counts))
        if method == "chi2":
            expected = counts * col.mean()
            scores[j] = np.sum((totals - expected) ** 2 / expected) if col.sum() else np.nan
        else:
            means = totals / counts
            within = np.sum((col - means[labels]) ** 2)
            between = np.sum(counts * (means - col.mean()) ** 2)
            if within:
                scores[j] = (between / (len(counts)-1)) / (within / (n-len(counts)))
            else:
                scores[j] = np.inf if between else np.nan
    return scores


def native_univariate(method, x, y, task):
    if method == "chi2" and x.min() < 0:
        raise ValueError("Chi-square requires nonnegative features")
    if method == "f" and len(x) <= (2 if task == "regression" else len(np.unique(y))):
        raise ValueError("The F statistic requires residual degrees of freedom")
    target = np.zeros(len(x)) if method == "variance" else y.astype(np.float64)
    return univariate_scores(x, target, method, task == "regression")


@njit(cache=True)
def iterative_scores(x, y, min_incl, max_iter, tolerance):
    """Draper et al. hypersphere updates with all observations as queries."""
    n, p = x.shape
    weights = np.ones(p)
    previous = np.zeros(p)
    for _ in range(max_iter):
        update = np.zeros(p)
        for i in range(n):
            differences = np.abs(x - x[i])
            distances = differences @ np.abs(weights)
            hits = y == y[i]
            hits[i] = False
            misses = y != y[i]
            radius = max(np.sort(distances[hits])[min_incl - 1], np.sort(distances[misses])[min_incl - 1])
            if radius == 0:
                continue
            kernel = np.maximum(0., 1 - (distances / radius)**2)
            h = kernel * hits
            m = kernel * misses
            update += (m / max(m.sum(), np.finfo(np.float64).eps)
                       - h / max(h.sum(), np.finfo(np.float64).eps)) @ differences
        weights += update
        if np.abs(update - previous).sum() < tolerance:
            break
        previous = update
    return weights


@njit(cache=True)
def nearest_class_scores(x, y, k, mean_deviation):
    """ReliefF per-class updates; mu-Relief orders by deviation from class mean distance."""
    n, p = x.shape
    counts = np.bincount(y)
    scores = np.zeros(p)
    for i in range(n):
        diff = np.abs(x - x[i])
        distances = diff.sum(axis=1)
        for c in range(len(counts)):
            members = np.flatnonzero((y == c) & (np.arange(n) != i))
            priorities = distances[members]
            if mean_deviation:
                priorities = -np.abs(priorities - priorities.mean())
            order = np.argsort(priorities, kind="mergesort")[:k]
            factor = -1. if c == y[i] else counts[c] / (n - counts[y[i]])
            for j in members[order]:
                scores += factor * diff[j] / len(order)
    return scores / n


def native_subset(method, x, y, random_state, k=5, keep_fraction=.5, subset_size=5,
                  repetitions=20, max_iter=100, tolerance=.001):
    """TuRF elimination, bagged feature subspaces, and two iterative neighbor schemes."""
    min_class = np.bincount(y).min()
    if min_class < 2 or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("Require positive k and at least two rows per class")
    k = min(k, min_class - 1)
    x = scale_columns(x)
    if method == "iterative-relief":
        if (not isinstance(max_iter, (int, np.integer)) or max_iter < 1
                or not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Expected positive max_iter and nonnegative tolerance")
        return iterative_scores(x, y, k, max_iter, tolerance)
    if method == "mu-relief":
        return nearest_class_scores(x, y, k, True)
    p = x.shape[1]
    if method == "turf":
        if not 0 < keep_fraction < 1:
            raise ValueError("keep_fraction must be between zero and one")
        remaining = np.arange(p)
        ranking = np.ones(p, dtype=np.int64)
        tiers = []
        while len(remaining) > 1:
            scores = nearest_class_scores(x[:, remaining], y, k, False)
            order = np.argsort(-scores, kind="stable")
            keep = max(1, int(len(remaining) * keep_fraction))
            tiers.append(remaining[order[keep:]])
            remaining = remaining[order[:keep]]
        for tier, eliminated in enumerate(reversed(tiers), start=2):
            ranking[eliminated] = tier
        return ranking
    if (not isinstance(subset_size, (int, np.integer)) or subset_size < 1
            or not isinstance(repetitions, (int, np.integer)) or repetitions < 1):
        raise ValueError("Expected positive subset_size and repetitions")
    rng = np.random.RandomState(random_state)
    size = min(subset_size, p)
    result = np.full(p, -np.inf)
    for _ in range(repetitions):
        permutation = rng.permutation(p)
        for start in range(0, p, size):
            subset = permutation[start:start + size]
            if len(subset) < size:
                subset = np.r_[subset, rng.choice(permutation[:start], size - len(subset), replace=False)]
            scores = nearest_class_scores(x[:, subset], y, k, False)
            result[subset] = np.maximum(result[subset], scores)
    return result
