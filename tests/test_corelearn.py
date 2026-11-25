from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from reliefe import FeatureRanker, list_methods
from reliefe.corelearn import CORE_METHODS


class CorelearnTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.RandomState(0).randint(0, 4, size=(40, 4))
        self.y = self.x[:, 2] >= 2

    def test_dependency_and_options(self):
        model = FeatureRanker("core-relieffequalk")
        with patch("shutil.which", return_value=None), self.assertRaises(ImportError):
            model.fit(self.x, self.y)
        with patch("shutil.which", return_value="Rscript"):
            for opts in ({"cost_matrix": [[0]]}, {"categorical_features": [4]}, {"data": 1}):
                with self.assertRaises(ValueError):
                    model.set_params(method_params=opts).fit(self.x, self.y)
            failed = subprocess.CompletedProcess([], 1, "", "test error")
            with patch("subprocess.run", return_value=failed), self.assertRaisesRegex(RuntimeError, "test error"):
                model.set_params(method_params={"kNearestEqual": 1}).fit(self.x, self.y)

    def test_bridge_order_and_binary_format(self):
        def score(args, **kwargs):
            folder = Path(args[3])
            features = np.fromfile(folder / "x.bin", dtype="<f8").reshape(self.x.shape)
            np.testing.assert_array_equal(features, self.x)
            np.testing.assert_array_equal(np.fromfile(folder / "cost.bin", dtype="<f8"), [0, 2, 1, 0])
            np.testing.assert_array_equal(np.fromfile(folder / "y.bin", dtype="<f8"), self.y)
            self.assertEqual(args[4:11], ["ReliefFexpRank", "classification", "40", "4", "42", "1", "kNearestExpRank"])
            np.array([0., .1, 1., .3], dtype="<f8").tofile(folder / "scores.bin")
            return subprocess.CompletedProcess(args, 0, "", "")
        options = dict(cost_matrix=[[0, 2], [1, 0]], categorical_features=[1], kNearestExpRank=3)
        with patch("shutil.which", return_value="Rscript"), patch("subprocess.run", side_effect=score):
            model = FeatureRanker("core-relieffexprank", method_params=options).fit(self.x, self.y)
        self.assertEqual(model.ranking()[0]["index"], 2)

    def test_all_corelearn_methods(self):
        r = shutil.which("Rscript")
        if r is None or subprocess.run([r, "--vanilla", "-e",
                "quit(status=as.integer(!requireNamespace('CORElearn',quietly=TRUE)))"]).returncode:
            self.skipTest("Optional R package CORElearn unavailable")
        for key in CORE_METHODS:
            task = list_methods()[key].tasks[0]
            with self.subTest(key=key):
                model = FeatureRanker(key, task=task)
                if task == "classification":
                    model.set_params(method_params={"cost_matrix": [[0, 1], [2, 0]],
                                                    "categorical_features": [0]})
                model.fit(self.x, self.y if task == "classification" else self.x[:, 2] + .1 * self.x[:, 1])
                self.assertEqual(model.scores_.shape, (4,))
                self.assertTrue(np.isfinite(model.scores_).all())
                if key in ("core-infgain", "core-rrelieffexprank"):
                    self.assertEqual(model.order_[0], 2)
