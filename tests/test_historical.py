import unittest

import numpy as np

from reliefe import FeatureRanker
from reliefe.historical import METHODS_1994


class HistoricalTests(unittest.TestCase):
    def test_complete_binary_and_original_class_prior(self):
        x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for key in METHODS_1994:
            model = FeatureRanker(key).fit(x, [0, 0, 1, 1])
            np.testing.assert_allclose(model.scores_, [.5 if key.endswith("-f") else 1, -1])

    def test_paper_missing_distance_on_single_attribute(self):
        # In class 0 the only observed category is 0, and in class 1 it is 1.
        # Expected within-class mismatch is zero under D and across-class mismatch is one.
        x = np.c_[[0, np.nan, 1, np.nan], [np.nan] * 4]
        y = [0, 0, 1, 1]
        for letter, expected in [("b", 0.), ("c", 0.), ("d", 1.), ("e", 1.), ("f", .5)]:
            model = FeatureRanker("relief-1994-" + letter).fit(x, y)
            np.testing.assert_allclose(model.scores_, [expected, 0])
            self.assertEqual(model.transform(x).shape, x.shape)
        with self.assertRaises(ValueError):
            FeatureRanker("relief-1994-a").fit(x, y)

    def test_multiclass_and_unseen_class_distribution(self):
        x = np.array([[0], [0], [1], [1], [2], [2]], dtype=float)
        y = [0, 0, 1, 1, 2, 2]
        for letter, expected in [("e", 1.), ("f", 2/3)]:
            model = FeatureRanker("relief-1994-" + letter).fit(x, y)
            np.testing.assert_allclose(model.scores_, [expected])
        x[-2:] = np.nan
        self.assertTrue(np.isfinite(model.fit(x, y).scores_).all())
        with self.assertRaises(ValueError):
            FeatureRanker("relief-1994-d").fit(x, y)
        with self.assertRaises(ValueError):
            FeatureRanker("relief-1994-e", method_params={"k": 2}).fit(x, y)
