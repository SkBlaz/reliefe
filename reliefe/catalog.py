"""Discover rankers and the targets and input formats accepted by each."""
from typing import NamedTuple

from .information import CRITERIA
from .native import RADIUS_METHODS
from .legacy_relief import OPTIONS as RELIEF_OPTIONS
from .historical import METHODS_1994
from .corelearn import CLASSIFICATION, CORE_METHODS


class Method(NamedTuple):
    tasks: tuple
    sparse: bool
    meaning: str


C = ("classification",)
R = ("regression",)
CR = C + R
REBATE = {
    "rebate-relieff": "ReliefF", "surf": "SURF", "surf-star": "SURFstar",
    "multisurf": "MultiSURF", "multisurf-star": "MultiSURFstar",
    "swrf": "SWRF", "swrf-star": "SWRFstar", "multiswrf": "MultiSWRF",
    "multiswrf-star": "MultiSWRFstar", "multiswrfdb": "MultiSWRFDB",
    "multiswrfdb-star": "MultiSWRFDBstar", "mu-relief": "MuRelief",
    "turf": "TURF", "vls": "VLS", "iter": "Iter",
}

_METHODS = {
    "stir-relieff": Method(C, False, "binary STIR pooled pseudo-t statistic"),
    "stir-surf": Method(C, False, "binary STIR pooled pseudo-t statistic"),
    "stir-multisurf": Method(C, False, "binary STIR pooled pseudo-t statistic"),
    "numba-variance": Method(("unsupervised",), False, "population variance"),
    "numba-f": Method(CR, False, "univariate F statistic"),
    "numba-chi2": Method(C, False, "chi-square statistic on nonnegative features"),
    "numba-pearson": Method(R, False, "absolute Pearson correlation"),
    "numba-spearman": Method(R, False, "absolute Spearman correlation"),
    "numba-iterative-relief": Method(C, False, "iterative hypersphere Relief weight"),
    "numba-mu-relief": Method(C, False, "class-mean distance deviation Relief weight"),
    "numba-turf": Method(C, False, "iterative ReliefF elimination tier"),
    "numba-vls": Method(C, False, "maximum subspace ReliefF weight"),
    "numba-irelief": Method(C, False, "iterated nonnegative margin weight"),
    "numba-reliefmss": Method(C, False, "above-background ReliefF difference"),
    "numba-info-gain": Method(C, False, "empirical information gain in nats"),
    "numba-gain-ratio": Method(C, False, "information gain divided by feature entropy"),
    "numba-symmetrical-uncertainty": Method(C, False, "symmetric normalized mutual information"),
    "numba-gini": Method(C, False, "reduction in Gini impurity"),
    "relief": Method(C, False, "binary squared-difference Relief weight"),
    "relieff": Method(C, True, "direct multiclass ReliefF weight"),
    "reliefe": Method(C + ("multilabel",), True, "embedding-neighborhood ReliefE weight"),
    "rrelieff": Method(R, False, "regression ReliefF weight"),
    "laplacian": Method(("unsupervised",), False, "normalized graph energy (lower is better)"),
    "per-target": Method(("multioutput", "multilabel"), True, "mean per-target rank (lower is better)"),
    "shadow-forest": Method(CR, False, "fraction of repetitions beating the strongest shuffled feature"),
    "reliefseq": Method(C, True, "maximum ReliefF weight across neighborhood sizes"),
    "dc-sis": Method(CR + ("multilabel", "multioutput"), False, "squared distance correlation"),
    "hsic": Method(CR + ("multilabel", "multioutput"), False, "biased empirical Hilbert-Schmidt independence criterion"),
    "survival-concordance": Method(("survival",), False, "absolute Somers D from censored concordance"),
    "variance": Method(("unsupervised",), True, "population variance"),
    "pearson": Method(R, False, "absolute Pearson correlation"),
    "spearman": Method(R, False, "absolute Spearman rank correlation"),
    "kendall": Method(R, False, "absolute Kendall tau-b"),
    "kruskal": Method(C, False, "Kruskal-Wallis H statistic"),
    "jmi": Method(C, False, "joint mutual information selection rank"),
    "cmim": Method(C, False, "conditional mutual information maximization rank"),
    "cife": Method(C, False, "conditional infomax feature extraction rank"),
    "boruta": Method(CR, False, "BorutaPy all-relevant selection rank"),
    "f": Method(CR, True, "univariate F statistic"),
    "chi2": Method(C, True, "chi-square statistic on nonnegative features"),
    "mi": Method(CR, True, "estimated mutual information in nats"),
    "mrmr": Method(C, False, "greedy MID selection rank"),
    "estimator": Method(CR, True, "importance from a caller-supplied estimator"),
    "forest": Method(CR, True, "mean decrease in impurity"),
    "linear": Method(CR, True, "mean absolute standardized coefficient"),
    "rfe": Method(CR, True, "recursive elimination rank"),
    "permutation": Method(CR, False, "validation score decrease"),
    "random": Method(("unsupervised",), True, "seeded random baseline"),
}
_METHODS.update({name: Method(C, False, meaning + " selection rank") for name, meaning in CRITERIA.items()})
_METHODS.update({key: Method(CR, False, "ReBATE " + name + " weight") for key, name in REBATE.items()})
_METHODS.update({key: Method(C if name in CLASSIFICATION else R, False,
                             "CORElearn " + name + " attribute evaluation")
                 for key, name in CORE_METHODS.items()})
_METHODS.update({key: Method(C, False, "1994 categorical " + key + " weight")
                 for key in METHODS_1994})
_METHODS.update({key: Method(C, False, "native " + key + (" rank" if key == "ec-relieff" else " weight"))
                 for key in RELIEF_OPTIONS})


_METHODS.update({"numba-" + key: Method(CR, False, "native " + key + " neighborhood weight")
                 for key in RADIUS_METHODS})


def list_methods():
    """Return a fresh mapping of public method names to immutable capability records."""
    return dict(_METHODS)


def classification_strategies(target):
    """Map compatible method IDs to their multiclass or multilabel scoring strategy."""
    if target not in ("multiclass", "multilabel"):
        raise ValueError("target must be multiclass or multilabel")
    binary_only = {"relief", *METHODS_1994[:4], "stir-relieff", "stir-surf", "stir-multisurf"}
    result = {}
    for name, spec in _METHODS.items():
        if target == "multilabel" and "multilabel" in spec.tasks:
            result[name] = "binary-relevance" if name == "per-target" else "direct"
        elif "classification" in spec.tasks:
            if target == "multilabel":
                result[name] = "binary-relevance"
            else:
                result[name] = "one-vs-rest" if name in binary_only else "direct"
    return result
