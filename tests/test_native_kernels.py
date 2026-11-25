import importlib.util
import unittest

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import chi2, f_classif, f_regression

from reliefe import FeatureRanker
from reliefe import native
from reliefe.historical import METHODS_1994, _historical_scores


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.RandomState(17).uniform(size=(24, 5))
        self.y = np.repeat([0, 1, 2], 8)
        self.r = self.x[:, 0] + self.x[:, 3] ** 2

    @unittest.skipUnless(importlib.util.find_spec("skrebate"), "install rebate for upstream comparisons")
    def test_all_radius_weights_against_reference(self):
        for target in (self.y, self.r, self.y % 2):
            for method in native.RADIUS_METHODS:
                with self.subTest(method=method):
                    expected = FeatureRanker(method, method_params={"categorical_features": []}).fit(self.x, target)
                    actual = FeatureRanker("numba-" + method).fit(self.x, target)
                    np.testing.assert_allclose(actual.scores_, expected.scores_, atol=1e-12)

    def test_radius_edge_cases(self):
        params = {"categorical_features": [0, 1]}
        x = np.array([[0, 0], [1, 1], [0, 0], [1, 1], [2, 4], [1, 4]])
        for name in native.RADIUS_METHODS:
            model = FeatureRanker("numba-" + name, method_params=params).fit(x, [0, 0, 1, 1, 1, 0])
            self.assertTrue(np.isfinite(model.scores_).all())
        with self.assertRaises(ValueError):
            FeatureRanker("numba-surf", method_params={"categorical_features": [-1]}).fit(self.x, self.y)
        with self.assertRaises(ValueError):
            FeatureRanker("numba-surf", task="regression").fit([[0]], [0])
        np.testing.assert_array_equal(FeatureRanker("numba-surf", task="regression").fit(x, np.ones(6)).scores_, 0)

    def test_conditional_margins_and_mss(self):
        x = np.array([[0., 0], [0, 1], [1, 0], [1, 1]])
        y = [0, 0, 1, 1]
        mss = FeatureRanker("numba-reliefmss", method_params={"k": 1}).fit(x, y)
        np.testing.assert_allclose(mss.scores_, [1, -1])
        weights = FeatureRanker("numba-irelief").fit(x, y).scores_
        np.testing.assert_allclose(weights, [1, 0])
        for name, opts, target in [("numba-irelief", {"width": 0}, y),
                                   ("numba-reliefmss", {"k": 0}, y),
                                   ("numba-irelief", {}, [1, 1, 1, 0])]:
            with self.assertRaises(ValueError):
                FeatureRanker(name, method_params=opts).fit(x, target)

    def test_univariate_numerical_references(self):
        for name, target, expected in [
            ("variance", None, self.x.var(axis=0)),
            ("f", self.y, f_classif(self.x, self.y)[0]),
            ("f", self.r, f_regression(self.x, self.r)[0]),
            ("chi2", self.y, chi2(self.x, self.y)[0]),
            ("pearson", self.r, [abs(np.corrcoef(x, self.r)[0, 1]) for x in self.x.T]),
            ("spearman", self.r, [abs(spearmanr(x, self.r)[0]) for x in self.x.T]),
        ]:
            fitted = FeatureRanker("numba-" + name).fit(self.x, target)
            np.testing.assert_allclose(fitted.scores_, expected, rtol=1e-11, atol=1e-12)
        tied = np.round(self.x, 1)
        expected = [abs(spearmanr(col, self.r)[0]) for col in tied.T]
        np.testing.assert_allclose(FeatureRanker("numba-spearman").fit(tied, self.r).scores_, expected)
        x = np.c_[self.y, np.zeros(len(self.y))]
        self.assertTrue(np.isinf(FeatureRanker("numba-f").fit(x, self.y).scores_[0]))
        self.assertTrue(np.isnan(FeatureRanker("numba-chi2").fit(x, self.y).scores_[1]))
        with self.assertRaises(ValueError):
            FeatureRanker("numba-chi2").fit(-self.x, self.y)
        with self.assertRaises(ValueError):
            FeatureRanker("numba-f").fit(self.x[:2], [0, 1])

    def test_information_and_historical_oracles(self):
        x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = np.array([0, 0, 1, 1])
        for name, expected in [("info-gain", np.log(2)), ("gain-ratio", 1),
                               ("symmetrical-uncertainty", 1), ("gini", .5)]:
            np.testing.assert_allclose(FeatureRanker("numba-" + name).fit(x, y).scores_, [expected, 0])
        for method in ("mrmr", "cife", "jmi", "cmim"):
            np.testing.assert_array_equal(FeatureRanker(method).fit(x, y).ranks_, [1, 2])
        for method in METHODS_1994:
            np.testing.assert_allclose(FeatureRanker(method).fit(x, y).scores_,
                                       [.5 if method.endswith("-f") else 1, -1])
        for method in ("relief", "rrelieff", "survival-concordance"):
            target = np.c_[y, np.arange(4)] if method == "survival-concordance" else y
            task = "regression" if method == "rrelieff" else "auto"
            self.assertTrue(np.isfinite(FeatureRanker(method, task=task).fit(x, target).scores_).all())

    def test_compilation_mode(self):
        from numba import config
        if not config.DISABLE_JIT:
            FeatureRanker("numba-multisurf").fit(self.x, self.y)
            FeatureRanker("cmim").fit(self.x, self.y)
            FeatureRanker("relief-1994-e").fit(self.x, self.y)
            for kernel in (native.radius_scores, native.greedy_information, _historical_scores):
                self.assertTrue(kernel.nopython_signatures)

    def test_subspace_and_iteration_rules(self):
        x = np.array([[0., 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
        y = np.repeat([0, 1], 2)
        expected = FeatureRanker("relieff", method_params={"k": 1}).fit(x, y).scores_
        params = {"k": 1, "subset_size": 3, "repetitions": 2}
        model = FeatureRanker("numba-vls", method_params=params).fit(x, y)
        np.testing.assert_allclose(model.scores_, expected)
        params["subset_size"] = 2
        self.assertTrue(np.isfinite(model.fit(x, y).scores_).all())
        self.assertEqual(FeatureRanker("numba-turf").fit(x, y).ranks_[0], 1)
        for name in ("iterative-relief", "mu-relief"):
            result = FeatureRanker("numba-" + name).fit(self.x, self.y).scores_
            self.assertTrue(np.isfinite(result).all())
        constant = np.zeros_like(x)
        np.testing.assert_allclose(FeatureRanker("numba-iterative-relief").fit(constant, y).scores_, 1)
        for m, opts in [("vls", {"subset_size": 0}), ("turf", {"keep_fraction": 1}),
                        ("mu-relief", {"k": 0}), ("iterative-relief", {"max_iter": 0})]:
            with self.assertRaises(ValueError):
                FeatureRanker("numba-" + m, method_params=opts).fit(x, y)

    def test_stir_original_R_statistics(self):
        # insilico/stir 01890541, find.neighbors + stir, Manhattan, k=2,
        # sd.frac=0.5. Reorder the R STIR_T table to original column order.
        x = np.random.RandomState(3072).uniform(size=(32, 4))
        y = np.tile([0, 1], 16)
        for method, expected in [
            ("stir-relieff", [-.02888967526854, .90670789067875, .74090655996531, -.01837448691040]),
            ("stir-surf", [-1.35812814125983, 2.08136513945228, .87540473663444, .47193454667341]),
            ("stir-multisurf", [-1.36528220496856, 1.98843873043866, 1.34306397205879, .35515750431171]),
        ]:
            scores = FeatureRanker(method, method_params={"k": 2}).fit(x, y).scores_
            np.testing.assert_allclose(scores, expected, atol=1e-12)
        for options, target in [({}, self.y), ({"k": 0}, self.y % 2)]:
            with self.assertRaises(ValueError):
                FeatureRanker("stir-relieff", method_params=options).fit(self.x, target)
        scores = FeatureRanker("stir-surf", method_params={"sd_fraction": 100}).fit(x, y).scores_
        self.assertTrue(np.isnan(scores).all())
