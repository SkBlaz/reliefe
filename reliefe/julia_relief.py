"""Adapter for complementary algorithms in Relief.jl 0.2.0."""
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np


# Optional positional arguments, in the reference implementation's order.
JULIA_OPTIONS = {
    "reliefmss": {"m": -1, "k": 10},
    "boostedsurf": {"m": -1, "phi": 3},
    "iterative-relief": {"m": -1, "min_incl": 3, "max_iter": 100},
    "irelief": {"max_iter": 1000, "k_width": 2., "conv_condition": 1e-6, "initial_w_div": -1.},
    "ec-relieff": {"m": -1, "k": 10},
}


def julia_scores(method, x, y, seed, f_type="continuous", **params):
    executable = shutil.which("julia")
    if executable is None:
        raise ImportError("Install Julia and Relief.jl 0.2.0 to use " + method)
    defaults = JULIA_OPTIONS[method]
    if params.keys() - defaults.keys() or f_type not in ("continuous", "discrete"):
        raise ValueError("Unknown Relief.jl option or feature type")
    options = []
    for key, default in defaults.items():
        value = params.get(key, default)
        if isinstance(default, int) and (not isinstance(value, (int, np.integer))
                                         or (value <= 0 and (key != "m" or value != -1))):
            raise ValueError("Invalid integer option: " + key)
        if key == "m" and value > len(y):
            raise ValueError("m cannot exceed the training row count")
        if key == "k_width" and value <= 0:
            raise ValueError("k_width must be positive")
        options.append(str(value))
    with TemporaryDirectory(prefix="reliefe-julia-") as folder:
        path = Path(folder)
        np.asarray(x, dtype="<f8").tofile(path / "features")
        np.asarray(y + 1, dtype="<i8").tofile(path / "labels")
        args = [executable, "--startup-file=no", str(Path(__file__).with_name("julia_relief.jl")),
                folder, method, str(x.shape[1]), str(seed), f_type] + options
        run = subprocess.run(args, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError("Relief.jl failed: " + run.stderr.strip())
        return np.fromfile(path / "scores", dtype="<f8")
