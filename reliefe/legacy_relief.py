"""Numba implementations of the scoring conventions in Relief.jl 0.2.0."""

import hashlib
import numpy as np
from numba import njit
from sklearn.model_selection import StratifiedKFold

_EPS = np.finfo(np.float64).eps
OPTIONS = {
    "reliefmss": {"m": -1, "k": 10},
    "boostedsurf": {"m": -1, "phi": 3},
    "iterative-relief": {"m": -1, "min_incl": 3, "max_iter": 100},
    "irelief": {
        "max_iter": 1000,
        "k_width": 2.0,
        "conv_condition": 1e-6,
        "initial_w_div": -1.0,
    },
    "ec-relieff": {"m": -1, "k": 10},
}


class _Sampler:
    """StatsBase sampling without replacement using the Julia 1.10 Xoshiro stream."""

    def __init__(self, seed):
        seed = int(np.random.SeedSequence().entropy if seed is None else seed)
        data = seed.to_bytes(max(4, 4 * ((seed.bit_length() + 31) // 32)), "little")
        self.state = list(
            map(int, np.frombuffer(hashlib.sha256(data).digest(), dtype="<u8"))
        )

    def index(self, n):
        mask = (1 << 64) - 1
        a, b, c, d = self.state
        v = (a + d) & mask
        value = (((v << 23) | (v >> 41)) + a) & mask
        t = (b << 17) & mask
        c ^= a
        d ^= b
        self.state = [a ^ d, b ^ c, c ^ t, ((d << 45) | (d >> 19)) & mask]
        product = value * n
        if (product & mask) < (1 << 64) % n:
            return self.index(n)
        return product >> 64

    def sample(self, n, k):
        if k < 3:
            first = self.index(n)
            if k == 1:
                return np.array([first])
            second = self.index(n - 1)
            return np.array([first, n - 1 if second == first else second])
        if n >= 24 * k:
            picked = []
            while len(picked) < k:
                i = self.index(n)
                if i not in picked:
                    picked.append(i)
            return np.array(picked)
        rows = np.arange(n)
        for i in range(k):
            j = i + self.index(n - i)
            rows[i], rows[j] = rows[j], rows[i]
        return rows[:k]


@njit(cache=True)
def _nearest(x, y, rows, k, discrete, mss):
    n, p = x.shape
    spans = np.array([np.ptp(x[:, f]) + _EPS for f in range(p)])
    freq = np.bincount(y)
    weights = np.zeros(p)
    for i in rows:
        raw = np.abs(x - x[i])
        distances = raw.sum(axis=1)
        distances[i] = np.inf
        for c in range(len(freq)):
            members = np.flatnonzero(y == c)
            order = np.argsort(distances[members], kind="mergesort")[:k]
            factor = -1.0 if c == y[i] else freq[c] / (n - freq[y[i]])
            for j in members[order]:
                normalized = raw[j] / spans
                diff = (raw[j] != 0).astype(np.float64) if discrete else normalized
                if mss:
                    bg = (normalized.sum() - normalized) / (p - 1)
                    delta = factor * diff - bg if c != y[i] else bg - diff
                    weights += np.where(normalized > bg, delta, 0.0)
                else:
                    weights += factor * diff
    return weights / (len(rows) * (1 if discrete and not mss else k))


@njit(cache=True)
def _boosted(x, y, rows, discrete):
    n, p = x.shape
    spans = np.array([np.ptp(x[:, f]) + _EPS for f in range(p)])
    weights = np.zeros(p)
    for i in rows:
        raw = np.abs(x - x[i])
        d = raw.sum(axis=1)
        other = np.arange(n) != i
        d0 = d[other]
        width = d0.std() * np.sqrt(len(d0) / (len(d0) - 1)) / 2
        near = (d < d0.mean() - width) & other
        far = (d > d0.mean() + width) & other
        hit = y == y[i]
        if not (
            np.any(near & hit)
            and np.any(near & ~hit)
            and np.any(far & hit)
            and np.any(far & ~hit)
        ):
            continue
        diff = (raw != 0).astype(np.float64) if discrete else raw / spans
        for zone in range(2):
            selected = near if zone == 0 else far
            h = selected & hit
            m = selected & ~hit
            update = -diff[h].sum(axis=0) / h.sum()
            for c in range(y.max() + 1):
                members = m & (y == c)
                update += diff[members].sum(axis=0) * members.sum() / m.sum() ** 2
            weights += update if zone == 0 else -update
    return weights / n


@njit(cache=True)
def _irelief(x, y, max_iter, width, tolerance, initial, discrete):
    w = np.ones(x.shape[1]) / (x.shape[1] if initial == -1 else initial)
    for _ in range(max_iter):
        margin = np.zeros_like(w)
        for i in range(len(x)):
            raw = np.abs(x - x[i])
            diff = (raw != 0).astype(np.float64) if discrete else raw
            kernel = np.exp(-(raw @ np.abs(w)) / width)
            kernel[i] = 0.0
            h = kernel * (y == y[i])
            m = kernel * (y != y[i])
            margin += (1.0 - m.sum() / (h.sum() + m.sum() + _EPS)) * (
                (m / (m.sum() + _EPS) - h / (h.sum() + _EPS)) @ diff
            )
        candidate = np.maximum(0.0, margin)
        candidate /= np.linalg.norm(candidate) + _EPS
        change = np.sum(np.abs(w - candidate))
        w = candidate
        if change < tolerance:
            break
    return w


@njit(cache=True)
def _iteration(x, y, rows, w, k, discrete):
    update = np.zeros_like(w)
    for i in rows:
        raw = np.abs(x - x[i])
        d = raw @ np.abs(w)
        h = y == y[i]
        h[i] = False
        m = y != y[i]
        radius = max(np.sort(d[h])[k - 1], np.sort(d[m])[k - 1])
        if radius == 0:
            continue
        kernel = np.maximum(0.0, 1.0 - (d / radius) ** 2)
        hits = kernel * h
        misses = kernel * m
        diff = (raw != 0).astype(np.float64) if discrete else raw
        update += (misses / (misses.sum() + _EPS) - hits / (hits.sum() + _EPS)) @ diff
    return update


@njit(cache=True)
def _nb_accuracy(x, y, folds):
    score = 0.0
    for f in range(5):
        train = x[folds != f]
        labels = y[folds != f]
        test = x[folds == f]
        # Match GaussianNB's training-fold variance smoothing and class priors.
        smoothing = 1e-9 * max(
            np.array([train[:, j].var() for j in range(train.shape[1])])
        )
        logp = np.empty((len(test), int(y.max()) + 1))
        for c in range(logp.shape[1]):
            v = train[labels == c]
            mean = v.sum(axis=0) / len(v)
            var = ((v - mean) ** 2).sum(axis=0) / len(v) + smoothing
            logp[:, c] = np.log(len(v) / len(train)) - 0.5 * (
                np.log(2 * np.pi * var).sum() + ((test - mean) ** 2 / var).sum(axis=1)
            )
        score += (np.argmax(logp, axis=1) == y[folds == f]).mean()
    return score / 5


def _evaporate(x, y, k, m, discrete, sampler):
    # The original scales raw-vector entropies and a joint frequency entropy.
    def entropy(values):
        nonzero = values[values > 0]
        return -np.dot(nonzero, np.log(nonzero))

    mu = np.empty(x.shape[1])
    for j, col in enumerate(x.T):
        counts = np.unique(np.c_[col, y], axis=0, return_counts=True)[1]
        mu[j] = (entropy(col) + entropy(y + 1) + entropy(counts / len(y))) / entropy(
            col
        )
    folds = np.zeros(len(y), dtype=np.int64)
    for f, (_, test) in enumerate(StratifiedKFold(5).split(x, y)):
        folds[test] = f
    active = np.arange(x.shape[1])
    ranks = np.ones(x.shape[1], dtype=int)
    while len(active) > 1:
        cols = x[:, active]
        w = _nearest(cols, y, sampler.sample(len(y), m), k, discrete, False)
        energy = (w.max() - w) / w.max()
        removed = -1
        for temp in np.arange(7) * 0.1 + 0.7:
            j = np.argsort(energy - temp * mu[active], kind="stable")[-1]
            if _nb_accuracy(cols[:, np.arange(len(active)) != j], y, folds) > 0:
                removed = j
        if removed < 0:
            raise ValueError(
                "No candidate removal achieved positive validation accuracy"
            )
        ranks[active[removed]] = len(active)
        active = np.delete(active, removed)
    return ranks


def legacy_scores(method, x, y, seed, f_type="continuous", **params):
    defaults = OPTIONS[method]
    if f_type not in ("continuous", "discrete") or params.keys() - defaults.keys():
        raise ValueError("Unknown feature type or algorithm option")
    opt = dict(defaults, **params)
    for name, default in defaults.items():
        if isinstance(default, int) and (
            not isinstance(opt[name], (int, np.integer))
            or (opt[name] <= 0 and not (name == "m" and opt[name] == -1))
        ):
            raise ValueError("Invalid integer option: " + name)
    n = len(y)
    m = opt.get("m", -1)
    m = n if m == -1 else m
    if m > n or opt.get("k_width", 1.0) <= 0:
        raise ValueError("Invalid number of queries or kernel width")
    discrete = f_type == "discrete"
    sampler = _Sampler(seed)
    if method == "irelief":
        return _irelief(
            x,
            y,
            opt["max_iter"],
            opt["k_width"],
            opt["conv_condition"],
            opt["initial_w_div"],
            discrete,
        )
    if method == "boostedsurf":
        return _boosted(x, y, sampler.sample(n, m), discrete)
    k = min(opt.get("k", opt.get("min_incl", 1)), np.bincount(y).min() - 1)
    if k < 1:
        raise ValueError("Each class requires at least two rows")
    if method == "reliefmss":
        return _nearest(x, y, sampler.sample(n, m), k, discrete, True)
    if method == "ec-relieff":
        return _evaporate(x, y, k, m, discrete, sampler)
    w = np.ones(x.shape[1])
    prev = np.zeros_like(w)
    for _ in range(opt["max_iter"]):
        step = _iteration(x, y, sampler.sample(n, m), w, k, discrete)
        w += step
        if np.sum(np.abs(step - prev)) < 0.001:
            break
        prev = step
    return w
