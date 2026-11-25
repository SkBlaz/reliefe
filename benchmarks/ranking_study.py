"""Reproducible Linux benchmark. Run with uv run python benchmarks/ranking_study.py."""
import json
import os
import platform
import resource
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold, StratifiedKFold

from reliefe import FeatureRanker, compare_rankings

ROOT = Path(__file__).resolve().parents[1]
CASES = ("additive", "xor", "redundant", "noise", "madelon", "medical", "wide-csr")
SEED = 17


def dataset(case):
    rng = np.random.RandomState(SEED)
    if case in ("madelon", "medical"):
        folder = "mcc" if case == "madelon" else "mlc"
        data = loadmat(ROOT / "data" / folder / (case + ".mat"))
        x = sparse.csr_matrix(data["input_space"])
        rows = rng.choice(x.shape[0], 144, replace=False)
        target = data["target_space"]
        y = sparse.csr_matrix(target)[rows].toarray() if case == "medical" else target.ravel()[rows]
        return x[rows], y, []
    if case == "wide-csr":
        x = sparse.random(32, 50000, density=.0005, format="csr", random_state=rng)
        return x, np.arange(32) % 2, []
    x = rng.randint(0, 2, size=(240, 16))
    target = x[:, 0] ^ x[:, 1] if case == "xor" else x[:, 0] + x[:, 1] > 0
    if case == "redundant":
        x[:, 2] = x[:, 0]
    if case == "noise":
        target = rng.randint(0, 2, size=x.shape[0])
    return x, target, [] if case == "noise" else [0, 1]


def measure(case):
    x, y, signal = dataset(case)
    multilabel = case == "medical"
    descriptive = dict(shape=list(x.shape), density=(x.nnz if sparse.issparse(x) else
                       np.count_nonzero(x)) / float(np.prod(x.shape)))
    method = "reliefe" if multilabel else "relieff"
    params = dict(num_iter=8, k=2)
    if multilabel:
        params.update(embedding_method="SVD", mlc_distance="hamming", samples=32,
                      latent_dimension=4, LD_friendly=False)
    ranker = FeatureRanker(method, method_params=params, top_k=4, random_state=SEED)
    times = []
    for _ in range(3):
        start = perf_counter()
        ranker.fit(x, y)
        times.append(perf_counter() - start)
    result = dict(descriptive, method=method, cold_fit_seconds=times[0], warm_fit_seconds=times[1:],
                  effective_parameters=params, top_four=ranker.order_[:4].tolist())
    if case != "wide-csr":
        rankers = {"relief": ranker, "all": None, "random": FeatureRanker("random", top_k=4, random_state=SEED),
                   "statistical": FeatureRanker("per-target" if multilabel else "f", top_k=4, random_state=SEED)}
        cv = KFold(3, shuffle=True, random_state=SEED) if multilabel else StratifiedKFold(
            3, shuffle=True, random_state=SEED)
        predictor = ExtraTreesClassifier(n_estimators=32, random_state=SEED)
        result["scoring"] = "f1_micro" if multilabel else "balanced_accuracy"
        comparison = compare_rankings(x, y, rankers, predictor, cv.split(x, y), scoring=result["scoring"])
        result["fold_results"] = {}
        for name, metrics in comparison.items():
            detail = {"scores": metrics["fold_scores"].tolist(), "mean_jaccard": metrics["mean_jaccard"]}
            if signal:
                frequency = metrics["selection_frequency"]
                # Give credit to either copy for recovering the first signal group.
                detail["mean_signal_recall"] = float(frequency[signal].mean())
                if case == "redundant":
                    selected = metrics["fold_support"]
                    detail["mean_group_recall"] = float(((selected[:, 0] | selected[:, 2]).mean()
                                                        + frequency[1]) / 2)
            result["fold_results"][name] = detail
    result["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(measure(sys.argv[1]), allow_nan=False))
    else:
        env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NUMBA_DISABLE_JIT="0")
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
        report = {"platform": platform.platform(), "cpu": cpu, "python": sys.version.split()[0], "seed": SEED,
                  "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "threads_per_backend": 1, "versions": {p: version(p) for p in (
                      "numpy", "scipy", "scikit-learn", "numba", "umap-learn")}, "cases": {}}
        for case in CASES:
            output = subprocess.check_output([sys.executable, __file__, case], env=env, text=True)
            report["cases"][case] = json.loads(output)
        print(json.dumps(report, indent=2, allow_nan=False))
