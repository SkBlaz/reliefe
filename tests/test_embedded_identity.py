import unittest
from unittest.mock import patch

import numpy as np
from reliefe import ReliefE


class Projection:
    def __init__(self, column):
        self.column = column

    def fit(self, x):
        return self

    def transform(self, x):
        return x[:, [self.column]]


class EmbeddedIdentityTests(unittest.TestCase):
    def test_projection_preserves_original_feature_indices(self):
        for signal in range(3):
            x = np.zeros((4, 3))
            x[:, signal] = [1, 1, -1, -1]
            model = ReliefE(k=1, embedding_based_distances=True,
                            normalize_descriptive=False, sparsity_threshold=1)
            with patch.object(model, 'determine_latent_dim', return_value=1):
                model.fit(x, np.array([0, 0, 1, 1]), Projection(signal))
            np.testing.assert_array_equal(model.feature_importances_, 2 * np.eye(3)[signal])

    def test_refit_recomputes_effective_parameters(self):
        x = np.eye(8)
        y = np.repeat([0, 1], 4)
        model = ReliefE(k=3, num_iter=1.0)
        model.fit(x[::2], y[::2])
        self.assertEqual(model.k_, 1)
        model.fit(x, y)
        self.assertEqual(model.k_, 3)
        np.testing.assert_array_equal(model.num_iter_, [8])
        np.testing.assert_array_equal(model.feature_importances_, ReliefE(k=3).fit(x, y).feature_importances_)

    def test_sparse_signed_sampling_bounded_and_input_lists(self):
        from scipy import sparse
        x = np.array([[0., -10., 2., 0.], [0., -10., 2., 0.], [1., 10., 2., 0.],
                      [1., 10., 2., 0.], [2., 9., 2., 0.], [2., 9., 2., 0.]])
        y = [0, 0, 1, 1, 2, 2]
        native = ReliefE(k=1)
        native.fit(x.tolist(), y)
        np.testing.assert_allclose(native.feature_importances_, ReliefE(k=1).fit(x, y).feature_importances_)
        # Two groups out of three requested. COO input supports no row slicing.
        with patch.object(native, "determine_latent_dim", return_value=1):
            native.embedding_based_distances = True
            native.samples = 2
            projection = Projection(1)
            with patch.object(projection, "fit", wraps=projection.fit) as call:
                native.fit(sparse.coo_matrix(x), y, projection)
                sample = call.call_args[0][0]
                self.assertEqual(sample.shape, (2, 4))
                self.assertLess(sample.min(), -.5)
            native.samples = 0
            with self.assertRaises(ValueError):
                native.fit(x, y)
        with self.assertRaises(ValueError):
            ReliefE().fit(x, y[:-1])
        # Five duplicates and one distinct row leave only one positive radius.
        duplicate = np.vstack([np.zeros((5, 4)), np.ones((1, 4))])
        self.assertGreater(ReliefE().determine_latent_dim(duplicate), 0)

    def test_explicit_multilabel_on_single_positive_fold(self):
        x = np.eye(4)
        y = np.eye(2)[[0, 0, 1, 1]]
        model = ReliefE(k=1, mlc_distance="hamming").fit(x, y, task_type="multilabel")
        self.assertEqual(model.task_type, "multi-label classification")
        for task, target in [("other", y), ("multilabel", [0, 0, 1, 1]),
                             ("multilabel", 3*y), ("classification", np.ones_like(y))]:
            with self.assertRaises(ValueError):
                model.fit(x, target, task_type=task)
