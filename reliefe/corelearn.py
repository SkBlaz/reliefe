"""Optional bridge to the CORElearn authors' R implementations."""
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np


CLASSIFICATION = """ReliefFequalK ReliefFexpRank ReliefFbestK Relief InfGain
GainRatio MDL Gini MyopicReliefF Accuracy ReliefFmerit ReliefFdistance
ReliefFsqrDistance DKM ReliefFexpC ReliefFavgC ReliefFpe ReliefFpa ReliefFsmp
GainRatioCost DKMcost ReliefKukar MDLsmp ImpurityEuclid ImpurityHellinger
UniformDKM UniformGini UniformInf UniformAccuracy EqualDKM EqualGini EqualInf
EqualHellinger DistHellinger DistAUC DistAngle DistEuclid""".split()
REGRESSION = """RReliefFequalK RReliefFexpRank RReliefFbestK RReliefFwithMSE
MSEofMean MSEofModel MAEofModel RReliefFdistance RReliefFsqrDistance""".split()
CORE_METHODS = {"core-" + name.lower(): name for name in CLASSIFICATION + REGRESSION}


def core_scores(method, x, y, task, random_state, cost_matrix=None, categorical_features=(), **options):
    """Exchange numeric buffers with a local R process. No shell or R expression interpolation."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise ImportError("CORElearn methods require Rscript and the R package CORElearn")
    categorical = np.asarray(categorical_features)
    if categorical.size and (categorical.ndim != 1 or categorical.dtype.kind not in "iu"
                             or categorical.min() < 0 or categorical.max() >= x.shape[1]):
        raise ValueError("categorical_features must be a sequence of zero-based column indices")
    with TemporaryDirectory(prefix="reliefe-") as folder:
        path = Path(folder)
        np.asarray(x, dtype="<f8").tofile(path / "x.bin")
        np.asarray(y, dtype="<f8").tofile(path / "y.bin")
        if cost_matrix is not None:
            costs = np.asarray(cost_matrix, dtype="<f8")
            n = len(np.unique(y))
            if task != "classification" or costs.shape != (n, n) or not np.isfinite(costs).all():
                raise ValueError("cost_matrix requires a finite square matrix in sorted class order")
            costs.tofile(path / "cost.bin")
        args = [rscript, "--vanilla", str(Path(__file__).with_name("corelearn.R")),
                folder, CORE_METHODS[method], task, str(len(y)), str(x.shape[1]),
                str(random_state), ",".join(map(str, categorical))]
        for name, value in options.items():
            if not isinstance(value, (str, int, float, bool)) or name in {
                    "formula", "data", "estimator", "costMatrix", "outputNumericSplits"}:
                raise ValueError("Expected a scalar CORElearn option: " + name)
            args.extend([name, ("Y" if value else "N") if isinstance(value, bool) else str(value)])
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("CORElearn failed: " + result.stderr.strip())
        return np.fromfile(path / "scores.bin", dtype="<f8")
