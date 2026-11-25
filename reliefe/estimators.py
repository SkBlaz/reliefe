"""Feature ranking and selection for scikit-learn pipelines."""
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (RFE, chi2, f_classif, f_regression,
                                       mutual_info_classif, mutual_info_regression)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted

from .catalog import REBATE, list_methods
from .corelearn import CORE_METHODS, core_scores
from .filters import laplacian_score, mrmr_order, original_relief, rrelieff
from .reliefe_classes import ReliefE


def _rebate(method, x, y, classification, params, seed):
    try:
        import skrebate
    except ImportError as exc:
        raise ImportError("Install reliefe[rebate] with Python >= 3.9 for ReBATE methods") from exc
    target = ("binary" if len(np.unique(y)) == 2 else "multiclass") if classification else "continuous"
    if method in ("turf", "vls", "iter"):
        base = params.pop("base_method", "rebate-relieff")
        base_params = params.pop("base_params", {})
        if base not in REBATE or base in ("turf", "vls", "iter"):
            raise ValueError("base_method must name a ReBATE core method")
        scorer = getattr(skrebate, REBATE[base])(label_type=target, **base_params)
        if method == "vls":
            params.setdefault("size_feature_subset", min(5, x.shape[1]))
            params.setdefault("random_state", seed)
        if method == "turf":
            params.setdefault("num_scores_to_return", max(1, x.shape[1] // 2))
        backend = getattr(skrebate, REBATE[method])(scorer, **params)
    else:
        backend = getattr(skrebate, REBATE[method])(label_type=target, **params)
    backend.fit(x.copy(), y.copy())
    return backend.feature_importances_


class FeatureRanker(TransformerMixin, BaseEstimator):
    """Rank columns and optionally select the top k in original column order.

    task='auto' follows sklearn target typing. Set task='regression' for
    integer-valued responses. method_params holds backend-specific options.
    Ordinal methods expose ranks_ and scores_=None. Other methods expose raw
    scores_ in input order. Equal scores get equal ranks, then ties at the
    selection boundary retain column order.
    """
    def __init__(self, method="relieff", *, top_k=None, task="auto", random_state=42,
                 estimator=None, method_params=None):
        self.method = method
        self.top_k = top_k
        self.task = task
        self.random_state = random_state
        self.estimator = estimator
        self.method_params = method_params

    def fit(self, X, y=None):
        methods = list_methods()
        if self.method not in methods:
            raise ValueError("Unknown method: " + str(self.method))
        spec = methods[self.method]
        names = getattr(X, "columns", None)
        x = check_array(X, accept_sparse="csr" if spec.sparse else False, dtype=np.float64)
        k = x.shape[1] if self.top_k is None else self.top_k
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= x.shape[1]:
            raise ValueError("top_k must be between 1 and the feature count")
        task = self.task
        if y is not None:
            y = y.toarray() if sparse.issparse(y) else np.asarray(y)
            check_consistent_length(x, y)
        if task == "auto":
            if spec.tasks == ("unsupervised",):
                task = "unsupervised"
            else:
                kind = type_of_target(y) if y is not None else "missing"
                task = {"binary": "classification", "multiclass": "classification",
                        "continuous": "regression", "multilabel-indicator": "multilabel",
                        "multiclass-multioutput": "multioutput", "continuous-multioutput": "multioutput"}.get(kind)
        if task not in spec.tasks or (task != "unsupervised" and y is None):
            raise ValueError(f"{self.method} supports {spec.tasks}; received task {task}")
        if self.estimator is not None and self.method not in ("rfe", "permutation"):
            raise ValueError("estimator is supported by rfe and permutation")
        if task == "classification":
            if type_of_target(y) not in ("binary", "multiclass") or len(np.unique(y)) < 2:
                raise ValueError("Classification requires at least two distinct classes")
            _, y = np.unique(y, return_inverse=True)
            y = y.ravel()
        if task == "regression":
            y = check_array(y, ensure_2d=False, dtype=float)
            if y.ndim != 1:
                raise ValueError("Regression requires a one-dimensional response")
        if task == "multilabel" and type_of_target(y) != "multilabel-indicator":
            raise ValueError("Expected a binary indicator matrix for multilabel ranking")
        raw = self._fit_scores(x, y, task, dict(self.method_params or {}))
        values = np.asarray(raw, dtype=float)
        if values.shape != (x.shape[1],):
            raise ValueError("Backend returned an incorrect number of feature values")
        ordinal = self.method in ("mrmr", "rfe")
        # Undefined statistics (e.g. F for a constant column) sort last.
        ascending = ordinal or self.method in ("laplacian", "per-target")
        order_values = values if ascending else -values
        order_values = np.where(np.isnan(order_values), np.inf, order_values)
        self.order_ = np.argsort(order_values, kind="stable")
        self.ranks_ = rankdata(order_values, method="min").astype(int)
        self.scores_ = None if ordinal else values
        self.support_ = np.zeros(x.shape[1], dtype=bool)
        self.support_[self.order_[:k]] = True
        self.n_features_in_ = x.shape[1]
        self.feature_names_in_ = None if names is None else np.asarray(names, dtype=object)
        self.task_ = task
        self.score_meaning_ = spec.meaning
        return self

    def _fit_scores(self, x, y, task, params):
        m = self.method
        classification = task == "classification"
        if m == "laplacian":
            return laplacian_score(x, **params)
        if m == "per-target":
            base = params.pop("base_method", "f")
            base_task = params.pop("base_task", "classification" if task == "multilabel" else "auto")
            if y.ndim != 2 or base == m:
                raise ValueError("per-target requires a target matrix and a single-output base method")
            ranks = []
            for target in y.T:
                if np.unique(target).size < 2:
                    ranks.append(np.full(x.shape[1], (x.shape[1] + 1) / 2))
                else:
                    model = FeatureRanker(base, task=base_task, random_state=self.random_state,
                                          method_params=params).fit(x, target)
                    ranks.append(rankdata(model.ranks_, method="average"))
            self.per_target_ranks_ = np.asarray(ranks)
            return self.per_target_ranks_.mean(axis=0)
        if m == "shadow-forest":
            repeats = params.pop("n_repeats", 10)
            if not isinstance(repeats, int) or repeats < 1:
                raise ValueError("n_repeats must be positive")
            rng = check_random_state(self.random_state)
            hits = np.zeros(x.shape[1])
            forest = RandomForestClassifier if classification else RandomForestRegressor
            for _ in range(repeats):
                shadows = np.column_stack([rng.permutation(col) for col in x.T])
                model = forest(random_state=rng.randint(2**31 - 1), **params).fit(np.c_[x, shadows], y)
                real, null = np.split(model.feature_importances_, 2)
                hits += real > null.max()
            return hits / repeats
        if m == "reliefseq":
            limit = np.unique(y, return_counts=True)[1].min() - 1
            k_min, k_max = params.pop("k_min", 1), params.pop("k_max", min(10, limit))
            if not (isinstance(k_min, (int, np.integer)) and isinstance(k_max, (int, np.integer))
                    and 1 <= k_min <= k_max <= limit):
                raise ValueError("Require 1 <= k_min <= k_max < smallest class size")
            weights = [ReliefE(k=k, random_state=self.random_state, **params).fit(x, y).feature_importances_
                       for k in range(k_min, k_max + 1)]
            self.best_k_ = k_min + np.argmax(weights, axis=0)
            return np.max(weights, axis=0)
        if m in CORE_METHODS:
            return core_scores(m, x, y, task, self.random_state, **params)
        if m in REBATE:
            return _rebate(m, x, y, classification, params, self.random_state)
        if m in ("relieff", "reliefe"):
            embedding = params.pop("embedding_method", "auto")
            model = ReliefE(embedding_based_distances=m == "reliefe",
                            random_state=self.random_state, **params)
            return model.fit(x, y, embedding_method=embedding).feature_importances_
        if m == "relief":
            return original_relief(x, y, **params)
        if m == "rrelieff":
            return rrelieff(x, y, **params)
        if m == "mrmr":
            return mrmr_order(x, y, **params)
        if m == "mi":
            score = mutual_info_classif if classification else mutual_info_regression
            return score(x, y, random_state=self.random_state, **params)
        if m in ("f", "chi2"):
            score = chi2 if m == "chi2" else (f_classif if classification else f_regression)
            return score(x, y, **params)[0]
        if m == "variance":
            return StandardScaler(with_mean=False, **params).fit(x).var_
        if m == "random":
            return check_random_state(self.random_state).uniform(size=x.shape[1], **params)
        forest = RandomForestClassifier if classification else RandomForestRegressor
        if m == "forest":
            return forest(random_state=self.random_state, **params).fit(x, y).feature_importances_
        if m == "linear":
            model = LogisticRegression(max_iter=1000, **params) if classification else Ridge(solver="lsqr", **params)
            scaled = StandardScaler(with_mean=False).fit_transform(x)
            return np.abs(np.atleast_2d(model.fit(scaled, y).coef_)).mean(axis=0)
        model = clone(self.estimator) if self.estimator is not None else forest(random_state=self.random_state)
        if m == "rfe":
            params.setdefault("n_features_to_select", 1)
            return RFE(model, **params).fit(x, y).ranking_
        test_size = params.pop("test_size", 0.25)
        train, valid = train_test_split(np.arange(len(y)), random_state=self.random_state,
                                        test_size=test_size, stratify=y if classification else None)
        model.fit(x[train], y[train])
        self.validation_indices_ = valid
        self.scoring_ = params.get("scoring")
        self.estimator_ = model
        result = permutation_importance(model, x[valid], y[valid],
                                        random_state=self.random_state, **params)
        self.permutation_std_ = result.importances_std
        return result.importances_mean

    def ranking(self):
        """Return ordered feature identities, native scores, tied ranks and selection."""
        check_is_fitted(self, "support_")
        return [{"index": int(i),
                 "name": None if self.feature_names_in_ is None else self.feature_names_in_[i],
                 "score": None if self.scores_ is None else float(self.scores_[i]),
                 "rank": int(self.ranks_[i]), "selected": bool(self.support_[i])}
                for i in self.order_]

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def transform(self, X):
        check_is_fitted(self, "support_")
        names = getattr(X, "columns", None)
        if self.feature_names_in_ is not None and not np.array_equal(names, self.feature_names_in_):
            raise ValueError("Input feature names or order differ from fit")
        values = check_array(X, accept_sparse="csr", dtype=None)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from fit")
        return values[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        if names is None:
            names = np.array(["x%d" % i for i in range(self.n_features_in_)])
        if len(names) != self.n_features_in_ or (self.feature_names_in_ is not None
                                                 and not np.array_equal(names, self.feature_names_in_)):
            raise ValueError("Input feature names differ from fit")
        return names[self.support_]
