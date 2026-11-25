"""Discover rankers and the targets and input formats accepted by each."""
from typing import NamedTuple

from .julia_relief import JULIA_OPTIONS
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
    "relief": Method(C, False, "binary squared-difference Relief weight"),
    "relieff": Method(C, True, "direct multiclass ReliefF weight"),
    "reliefe": Method(C + ("multilabel",), True, "embedding-neighborhood ReliefE weight"),
    "rrelieff": Method(R, False, "regression ReliefF weight"),
    "laplacian": Method(("unsupervised",), False, "normalized graph energy (lower is better)"),
    "per-target": Method(("multioutput", "multilabel"), True, "mean per-target rank (lower is better)"),
    "shadow-forest": Method(CR, False, "fraction of repetitions beating the strongest shuffled feature"),
    "reliefseq": Method(C, True, "maximum ReliefF weight across neighborhood sizes"),
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
_METHODS.update({key: Method(CR, False, "ReBATE " + name + " weight") for key, name in REBATE.items()})
_METHODS.update({key: Method(C if name in CLASSIFICATION else R, False,
                             "CORElearn " + name + " attribute evaluation")
                 for key, name in CORE_METHODS.items()})
_METHODS.update({key: Method(C, False, "1994 categorical " + key + " weight")
                 for key in METHODS_1994})
_METHODS.update({key: Method(C, False, "Relief.jl " + key + (" rank" if key == "ec-relieff" else " weight"))
                 for key in JULIA_OPTIONS})


def list_methods():
    """Return a fresh mapping of public method names to immutable capability records."""
    return dict(_METHODS)
