import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from reliefe import FeatureRanker, compare_rankings, grouped_permutation_importance


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]] * 5)
        self.y = self.x[:, 0].astype(int)

    def test_fold_local_and_grouped(self):
        folds = list(StratifiedKFold(3).split(self.x, self.y))
        ranker = FeatureRanker("forest", top_k=1)
        configs = {"forest": ranker, "full": None}
        results = compare_rankings(self.x, self.y, configs, LogisticRegression(), folds)
        np.testing.assert_allclose(results["forest"]["selection_frequency"], [1, 0])
        self.assertEqual(results["forest"]["mean_jaccard"], 1)
        self.assertFalse(hasattr(ranker, "order_"))
        with self.assertRaises(ValueError):
            compare_rankings(self.x, self.y, configs, LogisticRegression(), folds[:1])
        with self.assertRaises(ValueError):
            compare_rankings(self.x, self.y, configs, LogisticRegression(), [([0, 1], [1, 2])] * 2)
        model = LogisticRegression().fit(self.x, self.y)
        for opts in (dict(n_repeats=0), dict(groups={"bad": [2]})):
            with self.assertRaises(ValueError):
                grouped_permutation_importance(model, self.x, self.y, **dict({"groups": {}}, **opts))
        changes = grouped_permutation_importance(model, self.x, self.y, {"signal": [0], "null": [1]})
        np.testing.assert_allclose(changes["null"], 0)
        self.assertGreater(changes["signal"].mean(), .2)

    def test_named_groups(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("optional named-table test requires pandas")
        x = pd.DataFrame(self.x, columns=["a", "b"])
        model = LogisticRegression().fit(x, self.y)
        actual = grouped_permutation_importance(model, x, self.y, {"joint": [0, 1]})
        self.assertGreater(actual["joint"].mean(), .2)
        np.testing.assert_array_equal(x.to_numpy(), self.x)

    def test_reliefseq_shadow_and_targets(self):
        params = {"k_min": 1, "k_max": 3}
        model = FeatureRanker("reliefseq", method_params=params).fit(self.x, self.y)
        expected = [FeatureRanker("relieff", method_params={"k": k}).fit(self.x, self.y).scores_
                    for k in range(1, 4)]
        np.testing.assert_allclose(model.scores_, np.max(expected, axis=0))
        shadow = FeatureRanker("shadow-forest", method_params={"n_repeats": 3}).fit(self.x, self.y)
        self.assertGreater(shadow.scores_[0], shadow.scores_[1])
        targets = np.c_[self.y, np.ones_like(self.y)]
        result = FeatureRanker("per-target", method_params={"base_method": "forest"}).fit(self.x, targets)
        np.testing.assert_allclose(result.per_target_ranks_, [[1, 2], [1.5, 1.5]])
        for m, opts in [("per-target", {"base_method": "per-target"}), ("reliefseq", {"k_max": 20}),
                        ("shadow-forest", {"n_repeats": 0})]:
            with self.assertRaises(ValueError):
                FeatureRanker(m, method_params=opts).fit(self.x, targets if m == "per-target" else self.y)

    def test_laplacian(self):
        x = np.array([[0., 1.], [0., 1.], [1., 1.], [1., 1.]])
        ranker = FeatureRanker("laplacian", method_params={"k": 1}).fit(x)
        self.assertEqual(ranker.scores_[0], 0.)
        self.assertTrue(np.isnan(ranker.scores_[1]))
        np.testing.assert_array_equal(ranker.order_, [0, 1])
        with self.assertRaises(ValueError):
            FeatureRanker("laplacian", method_params={"k": 5}).fit(x)
        with self.assertRaises(ValueError):
            FeatureRanker("laplacian", method_params={"k": 1, "heat_width": 1e-10}).fit(np.eye(4))
