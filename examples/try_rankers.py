"""Try feature rankers on deterministic example data with known feature names."""
import argparse
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from reliefe import FeatureRanker, classification_strategies, list_methods


DEFAULTS = {
    "multiclass": ("f", "relief", "mrmr", "forest"),
    "multilabel": ("f", "numba-multisurf", "cmim"),
    "regression": ("pearson", "rrelieff", "forest"),
    "multioutput": ("per-target",),
    "unsupervised": ("variance", "laplacian"),
    "survival": ("survival-concordance",),
}


def dataset(task):
    rng = np.random.RandomState(42)
    x = rng.uniform(size=(60, 6))
    y = np.digitize(x[:, 0], [1/3, 2/3])
    labels = np.c_[x[:, 0] > .5, x[:, 1] > .5, (x[:, 2] > .5) ^ (x[:, 3] > .5)]
    response = x[:, 0] + x[:, 1]**2
    targets = {"multiclass": y, "multilabel": labels, "regression": response,
               "multioutput": np.c_[response, x[:, 2]], "unsupervised": None,
               "survival": np.c_[rng.rand(len(x)) > .2, 3 - response]}
    return x, targets[task]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="*", help="public method IDs")
    parser.add_argument("--task", choices=DEFAULTS, default="multiclass")
    parser.add_argument("--strategy", choices=("auto", "label-powerset"), default="auto",
                        help="for multilabel data, optionally rank against joint label combinations")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="show compatible methods and target strategies")
    group.add_argument("--all", action="store_true", help="try every compatible method, including installed extras")
    args = parser.parse_args(argv)
    task = args.task
    if task in ("multiclass", "multilabel"):
        strategies = classification_strategies(task)
    else:
        strategies = {m: "direct" for m, spec in list_methods().items() if task in spec.tasks}
    if args.strategy == "label-powerset":
        if task != "multilabel":
            parser.error("label-powerset requires --task multilabel")
        strategies = {m: args.strategy for m, spec in list_methods().items() if "classification" in spec.tasks}
    if args.list:
        for method in sorted(strategies):
            print(method, strategies[method], sep="\t")
        return 0
    if args.all and args.methods:
        parser.error("choose individual methods or --all")
    methods = sorted(strategies) if args.all else args.methods or DEFAULTS[task]
    unknown = set(methods) - strategies.keys()
    if unknown:
        parser.error("unsupported for " + task + ": " + ", ".join(sorted(unknown)))
    x, y = dataset(task)
    failed = False
    for method in methods:
        estimator = None
        if method == "estimator":
            tree = DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
            estimator = tree(max_depth=3, random_state=42)
        params = {"embedding_method": "SVD", "mlc_distance": "hamming"} if method == "reliefe" else {}
        ranker = FeatureRanker(method, task="classification" if task == "multiclass" else task,
                               estimator=estimator, method_params=params, target_strategy=args.strategy)
        try:
            ranker.fit(x, y)
        except ImportError as exc:
            print(f"{method}\tskipped\t{exc}", file=sys.stderr)
            failed |= not args.all
        except (ValueError, RuntimeError) as exc:
            print(f"{method}\tfailed\t{exc}", file=sys.stderr)
            failed = True
        else:
            order = " ".join(f"x{i}" for i in ranker.order_)
            print(f"{method}\t{ranker.target_strategy_}\t{order}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
