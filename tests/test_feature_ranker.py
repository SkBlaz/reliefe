"""Public estimator contracts and checks against independent ranking backends."""
import importlib.util
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from reliefe import FeatureRanker, list_methods
from reliefe.catalog import REBATE


class NamedTable:
    def __init__(self, data, names):
        self.data = data
        self.columns = names

    def __array__(self, dtype=None, **kwargs):
        return np.asarray(self.data, dtype=dtype)


class RankerTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.RandomState(41).uniform(size=(30, 6))
        self.y = np.repeat([0, 1], 15)
        self.x[:, -1] += self.y
        self.response = self.x[:, -1] + self.x[:, 0]
        self.names = list("abcdef")

    def test_catalog_and_target_matrix(self):
        for m, spec in list_methods().items():
            if m in REBATE and importlib.util.find_spec("skrebate") is None:
                continue
            for task in spec.tasks:
                with self.subTest(m=m, task=task):
                    target = {"classification": self.y, "multilabel": np.c_[self.y, self.x[:, 0] > .5],
                              "regression": self.response, "unsupervised": None}[task]
                    params = {"mlc_distance": "hamming", "embedding_method": "SVD"} if m == "reliefe" else {}
                    model = FeatureRanker(m, task=task, method_params=params, top_k=3).fit(self.x, target)
                    np.testing.assert_array_equal(np.sort(model.order_), np.arange(6))
                    self.assertEqual(len(model.ranking()), 6)
                    self.assertEqual(model.transform(self.x).shape, (30, 3))
                    if model.scores_ is not None:
                        self.assertTrue(np.isfinite(model.scores_).all())

    def test_native_reference_and_sparse_names(self):
        expected = f_classif(self.x, self.y)[0]
        for x in (self.x, sparse.csc_matrix(self.x), NamedTable(self.x, self.names)):
            ranker = FeatureRanker("f", top_k=2).fit(x, self.y)
            np.testing.assert_allclose(ranker.scores_, expected)
            kept = np.sort(np.argsort(-expected)[:2])
            np.testing.assert_array_equal(ranker.get_support(indices=True), kept)
            names = self.names if isinstance(x, NamedTable) else ["x%d" % i for i in range(6)]
            np.testing.assert_array_equal(ranker.get_feature_names_out(), np.array(names)[kept])
            self.assertEqual(ranker.transform(x).shape, (30, 2))
        with self.assertRaises(ValueError):
            ranker.transform(NamedTable(self.x, self.names[::-1]))
        with self.assertRaises(ValueError):
            ranker.get_feature_names_out(["wrong"])
        with self.assertRaises(ValueError):
            FeatureRanker("f").fit(self.x, self.y).transform(self.x[:, :2])

    def test_hand_calculated_relief(self):
        x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        np.testing.assert_allclose(FeatureRanker("relief").fit(x, [0, 0, 1, 1]).scores_, [1., -1.])
        # Pairs: 0->1, 1->0, 2->3, 3->2. y differences: 0,0,1,1.
        x = np.array([[0., 0.], [0., 0.], [1., 0.], [1., 1.]])
        model = FeatureRanker("rrelieff", task="regression", method_params={"k": 1})
        # Break the equidistant tie for row 2 by increasing its first coordinate distance
        # via a duplicate identity column.
        x = np.c_[x, x[:, 0]]
        np.testing.assert_allclose(model.fit(x, [0, 0, 0, 1]).scores_, [0, 1, 0])
        np.testing.assert_allclose(model.fit(x, np.ones(4)).scores_, 0)
        with self.assertRaises(ValueError):
            FeatureRanker("relief").fit(x, [0, 1, 2, 2])
        with self.assertRaises(ValueError):
            FeatureRanker("rrelieff", task="regression", method_params={"k": 0}).fit(x, self.y[:4])
        with self.assertRaises(ValueError):
            FeatureRanker("mrmr", method_params={"bins": 1}).fit(self.x, self.y)

    def test_pipeline_ties_and_refit(self):
        pipeline = make_pipeline(FeatureRanker(), LogisticRegression())
        search = GridSearchCV(pipeline, {"featureranker__method": ["f", "forest"],
                                         "featureranker__top_k": [1, 2]}, cv=2)
        search.fit(self.x, self.y)
        self.assertGreater(search.best_score_, .8)
        fitted = FeatureRanker("variance", top_k=1).fit(np.ones((4, 3)))
        np.testing.assert_array_equal(fitted.ranks_, [1, 1, 1])
        np.testing.assert_array_equal(fitted.get_support(), [True, False, False])
        self.assertEqual(clone(fitted).get_params(), fitted.get_params())

    def test_invalid_inputs(self):
        cases = [("unknown", "auto", self.y), ("f", "auto", None),
                 ("f", "classification", np.ones(30)), ("mi", "regression", np.c_[self.y, self.y]),
                 ("reliefe", "multilabel", self.y)]
        for method, task, target in cases:
            with self.assertRaises(ValueError):
                FeatureRanker(method, task=task).fit(self.x, target)
        with self.assertRaises(ValueError):
            FeatureRanker(estimator=LogisticRegression()).fit(self.x, self.y)
        with self.assertRaises(ValueError):
            FeatureRanker(top_k=True).fit(self.x, self.y)
        with patch.object(FeatureRanker, "_fit_scores", return_value=np.ones(2)), self.assertRaises(ValueError):
            FeatureRanker().fit(self.x, self.y)
        with patch.dict(sys.modules, {"skrebate": None}), self.assertRaises(ImportError):
            FeatureRanker("surf").fit(self.x, self.y)
        if importlib.util.find_spec("skrebate") is not None:
            with self.assertRaises(ValueError):
                FeatureRanker("turf", method_params={"base_method": "turf"}).fit(self.x, self.y)
