import unittest

import numpy as np
from reliefe import FeatureRanker


class SurvivalTests(unittest.TestCase):
    def test_comparable_pairs_ties_and_direction(self):
        x = [[3, 1, 0], [2, 2, 0], [1, 3, 0], [0, 4, 0]]
        # First event precedes three rows, including the censored row tied on time.
        # The third row's event precedes the fourth row. Censored rows never anchor.
        y = [[1, 1], [0, 1], [1, 2], [0, 3]]
        model = FeatureRanker("survival-concordance").fit(x, y)
        np.testing.assert_allclose(model.concordance_, [1, 0, .5])
        np.testing.assert_allclose(model.scores_, [1, 1, 0])
        for invalid in ([[0, 2]] * 4, [[2, 0]] * 4, [1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                model.fit(x, invalid)
