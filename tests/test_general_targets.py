import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from examples.try_rankers import dataset, main
from reliefe import FeatureRanker, classification_strategies


class TargetTests(unittest.TestCase):
    def test_catalogue_and_example(self):
        with self.assertRaises(ValueError):
            classification_strategies("regression")
        for task in ("multiclass", "multilabel"):
            with redirect_stdout(StringIO()) as out:
                self.assertEqual(main(["--task", task, "--list"]), 0)
                self.assertIn("numba-irelief", out.getvalue())
                self.assertEqual(main(["--task", task, "f", "relief"]), 0)
        self.assertNotIn("rrelieff", classification_strategies("multilabel"))

    def test_binary_relevance_matches_individual_fits(self):
        x, y = dataset("multilabel")
        y = np.c_[y, np.ones(len(x))]
        for method in ("f", "relief", "cmim", "estimator"):
            tree = DecisionTreeClassifier(max_depth=2, random_state=12) if method == "estimator" else None
            model = FeatureRanker(method, target_strategy="auto", estimator=tree, top_k=2)
            result = clone(model).fit(x, sparse.csr_matrix(y))
            expected = [rankdata(FeatureRanker(method, estimator=tree).fit(x, col).ranks_) for col in y.T[:-1]]
            expected.append(np.full(x.shape[1], (x.shape[1] + 1) / 2))
            np.testing.assert_allclose(result.per_target_ranks_, expected)
            np.testing.assert_allclose(result.scores_, np.mean(expected, axis=0))
            self.assertEqual(result.target_strategy_, "binary-relevance")
            self.assertEqual(result.transform(x).shape, (len(x), 2))

    def test_multiclass_and_refit(self):
        x, y = dataset("multiclass")
        labels = np.array(["A", "B", "C"])[y]
        for method in ("relief", "relief-1994-c", "stir-relieff"):
            result = FeatureRanker(method, target_strategy="auto").fit(x, labels)
            expected = [rankdata(FeatureRanker(method).fit(x, y == c).ranks_) for c in range(3)]
            np.testing.assert_allclose(result.scores_, np.mean(expected, axis=0))
            np.testing.assert_array_equal(result.target_names_, ["A", "B", "C"])
            self.assertEqual(result.target_strategy_, "one-vs-rest")
            result.fit(x, y == 0)
            self.assertFalse(hasattr(result, "per_target_ranks_"))
            self.assertEqual(result.target_strategy_, "direct")
        direct = FeatureRanker("f", target_strategy="auto").fit(x, labels)
        np.testing.assert_allclose(direct.scores_, FeatureRanker("f").fit(x, labels).scores_)
        self.assertEqual(direct.target_strategy_, "direct")

    def test_joint_signal_from_label_combinations(self):
        # Neither label alone carries information about their parity.
        y = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (8, 1))
        x = np.c_[np.zeros(len(y)), y[:, 0] ^ y[:, 1]]
        separate = FeatureRanker("numba-info-gain", target_strategy="auto").fit(x, y)
        np.testing.assert_array_equal(separate.ranks_, [1, 1])
        joint = FeatureRanker("numba-info-gain", target_strategy="label-powerset").fit(x, y)
        np.testing.assert_array_equal(joint.ranks_, [2, 1])
        np.testing.assert_array_equal(joint.label_combinations_, y[:4])
        self.assertEqual(joint.target_strategy_, "label-powerset")
        np.testing.assert_array_equal(joint.fit(x, np.ones_like(y)).ranks_, [1, 1])
        for method, strategy in [("rrelieff", "label-powerset"), ("f", "unsupported")]:
            with self.assertRaises(ValueError):
                FeatureRanker(method, target_strategy=strategy).fit(x, y)

    def test_native_catalogue_multiclass_and_multilabel(self):
        for task in ("multiclass", "multilabel"):
            x, y = dataset(task)
            for method, strategy in classification_strategies(task).items():
                if not method.startswith(("numba-", "relief-1994-", "stir-")):
                    continue
                with self.subTest(task=task, method=method):
                    fitted = FeatureRanker(method, target_strategy="auto").fit(x, y)
                    np.testing.assert_array_equal(np.sort(fitted.order_), np.arange(x.shape[1]))
                    self.assertEqual(fitted.target_strategy_, strategy)

    def test_missing_optional_packages(self):
        import subprocess
        from unittest.mock import patch
        x, y = dataset("multiclass")
        missing = subprocess.CompletedProcess([], 42, stdout="", stderr="Install the reference package")
        with patch("shutil.which", return_value="runtime"), patch("subprocess.run", return_value=missing):
            for method in ("irelief", "core-relief"):
                with self.assertRaisesRegex(ImportError, "Install the reference package"):
                    FeatureRanker(method, target_strategy="auto").fit(x, y)
