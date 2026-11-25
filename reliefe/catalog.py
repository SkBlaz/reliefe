"""Discover rankers and the targets and input formats accepted by each."""
from typing import NamedTuple

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
    "variance": Method(("unsupervised",), True, "population variance"),
    "f": Method(CR, True, "univariate F statistic"),
    "chi2": Method(C, True, "chi-square statistic on nonnegative features"),
    "mi": Method(CR, True, "estimated mutual information in nats"),
    "mrmr": Method(C, False, "greedy MID selection rank"),
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


def list_methods():
    """Return a fresh mapping of public method names to immutable capability records."""
    return dict(_METHODS)
