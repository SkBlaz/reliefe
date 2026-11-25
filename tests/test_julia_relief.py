import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from reliefe import FeatureRanker
from reliefe.julia_relief import JULIA_OPTIONS


class JuliaReliefTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.RandomState(4).uniform(size=(32, 5))
        self.y = np.repeat([0, 1], 16)
        self.x[:, 1] += self.y * 3

    def test_validation_and_exchange(self):
        with patch("shutil.which", return_value=None), self.assertRaises(ImportError):
            FeatureRanker("irelief").fit(self.x, self.y)
        with patch("shutil.which", return_value="julia"):
            for m, opts in [("irelief", {"bad": 1}), ("irelief", {"k_width": 0}),
                            ("boostedsurf", {"m": 33}), ("boostedsurf", {"m": 0})]:
                with self.assertRaises(ValueError):
                    FeatureRanker(m, method_params=opts).fit(self.x, self.y)
            def execute(args, **kwargs):
                path = Path(args[3])
                np.testing.assert_array_equal(np.fromfile(path / "labels", dtype="<i8"), self.y + 1)
                np.array([3, 1, 5, 2, 4], dtype="<f8").tofile(path / "scores")
                return subprocess.CompletedProcess(args, 0, "", "")
            with patch("subprocess.run", side_effect=execute):
                model = FeatureRanker("ec-relieff").fit(self.x, self.y)
            self.assertIsNone(model.scores_)
            self.assertEqual(model.order_[0], 1)
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "missing")):
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    FeatureRanker("irelief").fit(self.x, self.y)

    @unittest.skipUnless(os.environ.get("RELIEFE_TEST_JULIA") == "1", "optional Julia integration")
    def test_reference_implementations(self):
        for m in JULIA_OPTIONS:
            with self.subTest(method=m):
                model = FeatureRanker(m).fit(self.x, self.y)
                np.testing.assert_array_equal(np.sort(model.order_), np.arange(5))
                if model.scores_ is not None:
                    self.assertTrue(np.isfinite(model.scores_).all())
