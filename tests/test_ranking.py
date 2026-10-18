import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from reliefe import ReliefE, rank_features


X = [[0., 0., 1.], [0., 1., 1.], [1., 0., 0.], [1., 1., 0.]]
Y = [0, 0, 1, 1]


def reference_relief(x, y, k, iterations):
    """Small, independent dense ReliefF reference; explicitly exclude self."""
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    ranges = np.ptp(x, axis=0)
    x = (x - x.min(axis=0)) / np.where(ranges == 0, 1., ranges)
    classes, counts = np.unique(y, return_counts=True)
    priors = dict(zip(classes, counts / len(y)))
    scores = np.zeros(x.shape[1])
    for sample in np.random.RandomState(42).permutation(len(y))[:iterations]:
        for label in classes:
            members = np.flatnonzero((y == label) & (np.arange(len(y)) != sample))
            distances = np.abs(x[members] - x[sample]).sum(axis=1)
            neighbours = members[np.argsort(distances)[:k]]
            weight = -1. if label == y[sample] else priors[label] / (1. - priors[y[sample]])
            scores += weight * np.abs(x[sample] - x[neighbours]).mean(axis=0)
    return scores / iterations


class RankingTests(unittest.TestCase):
    def test_named_ranking_ties_and_json(self):
        result = rank_features(X, Y, ["a", "b", "c"], top_k=2)
        self.assertEqual(result["scores"], [1., -1., 1.])
        self.assertEqual([item["index"] for item in result["ranking"]], [0, 2])
        self.assertEqual([item["name"] for item in result["ranking"]], ["a", "c"])
        self.assertEqual(result["num_iter"], 4)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertEqual(rank_features(X, ["a", "a", "b", "b"])["scores"], result["scores"])
        self.assertIsNone(rank_features(X, Y)["ranking"][0]["name"])

    def test_reference_multiclass_and_repeatability(self):
        rng = np.random.RandomState(17)
        x = rng.normal(size=(18, 7))
        y = np.repeat([0, 1, 2], [4, 6, 8])
        x[:, -1] = 1.
        for k in (1, 3):
            for iterations in (7, 18):
                with self.subTest(k=k, iterations=iterations):
                    result = rank_features(x, y, k=k, num_iter=iterations)
                    np.testing.assert_allclose(result["scores"], reference_relief(x, y, k, iterations), atol=1e-14)
                    self.assertEqual(result, rank_features(x, y, k=k, num_iter=iterations))

    def test_duplicate_rows_and_extreme_scales(self):
        # This data used to fail during unnecessary latent dimension estimation.
        duplicates = np.array([[0.] * 4] * 3 + [[1.] * 4])
        result = rank_features(duplicates, Y)
        self.assertTrue(np.isfinite(result["scores"]).all())
        np.testing.assert_allclose(rank_features(np.zeros((4, 4)), Y)["scores"], 0.)
        scaled = np.array(X) * np.array([1e308, 1e-308, 1.])
        scaled[:2, 0] = -1e308
        self.assertEqual(rank_features(scaled, Y)["scores"], [1., -1., 1.])

    def test_direct_fit_skips_unused_dimension_estimation(self):
        with patch.object(ReliefE, "determine_latent_dim", side_effect=AssertionError("unused")):
            model = ReliefE(num_iter=4, k=1, embedding_based_distances=False, sparsity_threshold=1.)
            model.fit(np.array(X), np.array(Y))
        np.testing.assert_allclose(model.feature_importances_, [1., -1., 1.])

    def test_native_dense_and_sparse_inputs_agree(self):
        for x in (np.array(X), sparse.csr_matrix(X), sparse.csc_matrix(X), sparse.coo_matrix(X)):
            with self.subTest(format=type(x).__name__):
                model = ReliefE(num_iter=4, k=1, embedding_based_distances=False,
                                sparsity_threshold=1.)
                model.fit(x, np.array(Y))
                np.testing.assert_allclose(model.feature_importances_, [1., -1., 1.])

    def test_invalid_data(self):
        invalid = [
            ([], Y), ([[], [], [], []], Y), ([[1.], [2.]], [0, 1]),
            ([[1.]] * 513, [0] * 256 + [1] * 257),
            (np.zeros((4, 1025)), Y), (np.zeros((512, 257)), [0] * 256 + [1] * 256),
            ([[float("nan")]] * 4, Y), ([[float("inf")]] * 4, Y),
            ([["1"]] * 4, Y), ([[True]] * 4, Y), ([[1j]] * 4, Y),
            ([[0.], [1., 2.], [2.], [3.]], Y),
            (X, [[0], [0], [1], [1]]), (X, [0, 1]),
            (X, [True, True, False, False]), (X, [0., 0., float("nan"), 1.]),
            (X, [0, 0, 0, 0]), (X, [0, 0, 0, 1]),
        ]
        for x, y in invalid:
            with self.subTest(shape=np.shape(x) if not isinstance(x, list) else len(x)):
                with self.assertRaises((ValueError, TypeError)):
                    rank_features(x, y)

    def test_invalid_parameters(self):
        for kwargs in [dict(k=0), dict(k=2), dict(k=True), dict(k=1.5),
                       dict(num_iter=0), dict(num_iter=513), dict(num_iter=True),
                       dict(top_k=0), dict(top_k=4), dict(top_k=True),
                       dict(feature_names=["a"]), dict(feature_names=["a", "a", "b"]),
                       dict(feature_names=["a", "b", 1]), dict(feature_names="abc")]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    rank_features(X, Y, **kwargs)

    def test_nonfinite_result_is_an_error(self):
        def invalid_fit(model, *args, **kwargs):
            model.feature_importances_ = np.full(3, np.nan)
        with patch.object(ReliefE, "fit", invalid_fit):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                rank_features(X, Y)


if __name__ == "__main__":
    unittest.main()
