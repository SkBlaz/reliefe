"""FEAST bd9ab70 and praznik 10.0.0 orders captured from their upstream implementations."""

import json
from pathlib import Path
import unittest

import numpy as np

from reliefe import FeatureRanker
from reliefe.information import CRITERIA


class ForwardTests(unittest.TestCase):
    def test_classification_parity(self):
        fixture = Path(__file__).parent / "fixtures/forward_ranks.json"
        for seed, orders in json.loads(fixture.read_text()).items():
            rng = np.random.RandomState(int(seed))
            x = rng.randint(0, 3, (320, 7))
            y = (x[:, 0] + x[:, 2] + rng.randint(0, 3, 320) * (x[:, 5] == 1)) % 3
            x[:, 4] = (x[:, 1] + (rng.rand(320) < 0.15)) % 3
            weights = 0.8 + 0.4 * rng.rand(len(y))
            self.assertEqual(set(orders), set(CRITERIA))
            for method, expected in orders.items():
                params = {"beta": 0.55, "gamma": 0.72} if method == "beta-gamma" else {}
                if method.startswith("weighted-"):
                    params = {"sample_weight": weights}
                with self.subTest(seed=seed, method=method):
                    ranker = FeatureRanker(method, method_params=params).fit(x, y)
                    np.testing.assert_array_equal(ranker.order_, expected)

    def test_ties_reductions_and_parameters(self):
        x = np.zeros((9, 4))
        y = np.tile(np.arange(3), 3)
        for method in CRITERIA:
            direct = FeatureRanker(method).fit(x, y)
            np.testing.assert_array_equal(direct.order_, np.arange(4))
            labels = np.column_stack([y == 1, y == 2])
            reduced = FeatureRanker(method, target_strategy="auto").fit(x, labels)
            self.assertEqual(reduced.target_strategy_, "binary-relevance")
        for params in ({"gamma": 1}, {"beta": np.nan}):
            with self.assertRaises(ValueError):
                FeatureRanker("mifs", method_params=params).fit(x, y)
        with self.assertRaises(ValueError):
            FeatureRanker(
                "weighted-mim", method_params={"sample_weight": np.zeros(len(y))}
            ).fit(x, y)


class DependenceTests(unittest.TestCase):
    def test_scores_against_centering_matrix(self):
        from scipy.spatial.distance import cdist

        rng = np.random.RandomState(84)
        x = rng.normal(size=(24, 4))
        x[:, 3] = 1
        continuous = np.column_stack([x[:, 0] ** 2, x[:, 1] * x[:, 2]])
        labels = continuous > np.median(continuous, axis=0)
        center = np.eye(len(x)) - np.ones((len(x), len(x))) / len(x)
        for task, y in (
            ("multioutput", continuous),
            ("multilabel", labels),
            ("regression", continuous[:, 0]),
            ("classification", np.arange(len(x)) % 3),
        ):
            dy = cdist(y.reshape(len(x), -1), y.reshape(len(x), -1))
            for method in ("dc-sis", "hsic"):
                ky = dy
                params = {}
                if task == "classification":
                    ky = (dy != 0).astype(float)
                if method == "hsic":
                    params = {"bandwidth": 0.8, "target_bandwidth": 1.3}
                    ky = (
                        (dy == 0)
                        if task == "classification"
                        else np.exp(-(dy**2) / (2 * 1.3**2))
                    )
                b = center @ ky @ center
                expected = []
                for col in x.T:
                    dx = np.abs(col[:, None] - col)
                    kx = dx if method == "dc-sis" else np.exp(-(dx**2) / (2 * 0.8**2))
                    a = center @ kx @ center
                    denom = (
                        np.linalg.norm(a) * np.linalg.norm(b)
                        if method == "dc-sis"
                        else (len(x) - 1) ** 2
                    )
                    expected.append(np.trace(a @ b) / denom if denom else 0.0)
                with self.subTest(task=task, method=method):
                    fitted = FeatureRanker(method, task=task, method_params=params).fit(
                        x, y
                    )
                    np.testing.assert_allclose(fitted.scores_, expected, atol=1e-12)

    def test_median_width_and_validation(self):
        from reliefe.dependence import dependence_scores

        x = np.arange(6.0).reshape(-1, 1)
        y = x[:, 0] ** 2
        score = FeatureRanker("hsic", task="regression").fit(x, y).scores_
        distances = np.abs(x - x.T)
        target_distances = np.abs(y[:, None] - y)
        params = dict(
            bandwidth=np.median(distances[distances > 0]),
            target_bandwidth=np.median(target_distances[target_distances > 0]),
        )
        explicit = FeatureRanker("hsic", task="regression", method_params=params).fit(
            x, y
        )
        np.testing.assert_allclose(score, explicit.scores_)
        np.testing.assert_array_equal(
            dependence_scores(x * 0, y * 0, "regression", "hsic"), [0.0]
        )
        with self.assertRaises(ValueError):
            dependence_scores(x, y, "regression", "dc-sis", bandwidth=-1)
        with self.assertRaises(ValueError):
            dependence_scores(x, y * np.nan, "regression", "hsic")


class SpectralTests(unittest.TestCase):
    def test_scikit_feature_reference(self):
        # Raw output from scikit-feature SPEC.spec, gamma=1, styles -1, 0, 3.
        x = np.random.RandomState(24).normal(size=(15, 5))
        scores = {
            "spec-energy": [
                0.0416732987152746,
                0.037258954503569246,
                0.04353772317711078,
                0.03931165069870719,
                0.10220904264518783,
            ],
            "spec": [
                0.04197036853196249,
                0.042690587767480526,
                0.05242743202671762,
                0.06460337120759087,
                0.11722904852802225,
            ],
            "spec-cluster": [
                0.22294582765365764,
                0.7739959862095483,
                0.9678272127016363,
                0.25622110009024557,
                0.2887440088213392,
            ],
        }
        for name, expected in scores.items():
            fit = FeatureRanker(name, method_params={"clusters": 3}).fit(x)
            np.testing.assert_allclose(fit.scores_, expected, rtol=1e-11)
            order = np.argsort(expected)
            if name == "spec-cluster":
                order = order[::-1]
            np.testing.assert_array_equal(order, fit.order_)

    def test_graphs_constants_and_validation(self):
        x = np.c_[np.arange(5), np.arange(5) ** 2, np.ones(5)]
        fit = lambda name, **params: FeatureRanker(name, method_params=params).fit(x)
        self.assertTrue(np.isnan(fit("spec").scores_[-1]))
        np.testing.assert_allclose(fit("spec-energy", affinity=np.eye(5)).scores_, 0.0)
        # All nonconstant directions have the same eigenvalue on an identity graph.
        scores = fit("spec-cluster", affinity=np.eye(5)).scores_
        permuted = FeatureRanker(
            "spec-cluster", method_params={"affinity": np.eye(5)}
        ).fit(x[::-1])
        np.testing.assert_allclose(scores, permuted.scores_, atol=1e-12)
        for params in ({"gamma": 0}, {"clusters": 1}, {"affinity": np.zeros((5, 5))}):
            with self.assertRaises(ValueError):
                fit("spec", **params)


class SeparationTests(unittest.TestCase):
    def test_scikit_feature_and_welch_parity(self):
        from scipy.stats import ttest_ind

        x = np.random.RandomState(97).normal(size=(36, 5))
        y = np.arange(36) % 2
        expected = [
            0.5643735276505818,
            0.7093509449791529,
            1.3364000573497872,
            0.19490329213936955,
            0.10092764493434359,
        ]
        np.testing.assert_allclose(
            FeatureRanker("t-score").fit(x, y).scores_, expected, atol=1e-12
        )
        y[0] = 1
        fit = FeatureRanker("t-score", method_params={"ddof": 1}).fit(x, y)
        np.testing.assert_allclose(
            fit.scores_, abs(ttest_ind(x[y == 0], x[y == 1], equal_var=False).statistic)
        )
        expected = [
            0.037157371089004076,
            0.05235988901420785,
            0.17971024654715406,
            0.028505363862885602,
            0.017188407877762435,
        ]
        np.testing.assert_allclose(
            FeatureRanker("fisher-score").fit(x, (np.arange(36) % 3)).scores_,
            expected,
            atol=1e-12,
        )

    def test_undefined_variance_and_targets(self):
        x = np.c_[np.zeros(6), np.repeat([0, 1], 3), np.ones(6)]
        y = np.repeat([0, 1], 3)
        for method in ("fisher-score", "t-score"):
            np.testing.assert_array_equal(
                FeatureRanker(method).fit(x, y).scores_, [0, np.inf, 0]
            )
        with self.assertRaises(ValueError):
            FeatureRanker("t-score").fit(x, np.arange(6) % 3)
        with self.assertRaises(ValueError):
            FeatureRanker("t-score", method_params={"ddof": 1}).fit(
                x, np.arange(6) == 0
            )
        with self.assertRaises(ValueError):
            FeatureRanker("t-score", method_params={"ddof": 2}).fit(x, y)
        result = FeatureRanker("t-score", target_strategy="auto").fit(
            x, np.arange(6) % 3
        )
        self.assertEqual(result.target_strategy_, "one-vs-rest")


class RobustTests(unittest.TestCase):
    def test_upstream_coefficients(self):
        x = np.random.RandomState(31).normal(size=(18, 4))
        cases = [
            (
                "classification",
                np.arange(18) % 3,
                [
                    0.07433853766196787,
                    0.44490875476453345,
                    0.15792590210237195,
                    0.05720403260091776,
                ],
            ),
            (
                "multilabel",
                (x[:, :2] > 0).astype(int),
                [
                    0.06864481679126405,
                    0.04427903760554591,
                    0.0046555607314027634,
                    0.002221810425076753,
                ],
            ),
            (
                "regression",
                x[:, 1] ** 2,
                [
                    0.0011939988261550317,
                    0.4063245907418482,
                    0.00037427486144211134,
                    0.0031284095184881,
                ],
            ),
            (
                "multioutput",
                x[:, :2] ** 2,
                [
                    0.05531366799957507,
                    0.7058512768060414,
                    0.08164728544169246,
                    0.0045613027816912375,
                ],
            ),
        ]
        for task, y, expected in cases:
            with self.subTest(task=task):
                fit = FeatureRanker("rfs", task=task, method_params={"gamma": 0.8}).fit(
                    x, y
                )
                np.testing.assert_allclose(fit.scores_, expected, rtol=1e-8, atol=1e-11)

    def test_validation(self):
        from reliefe.robust import robust_scores

        x = np.ones((4, 2))
        y = np.array([0, 0, 1, 1])
        for params in ({"gamma": 0}, {"tolerance": np.nan}, {"max_iter": 0}):
            with self.assertRaises(ValueError):
                robust_scores(x, y, "classification", **params)
        with self.assertRaises(ValueError):
            robust_scores(x, y * np.nan, "regression")
