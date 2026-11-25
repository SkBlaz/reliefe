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
