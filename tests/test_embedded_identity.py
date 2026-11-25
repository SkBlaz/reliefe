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
