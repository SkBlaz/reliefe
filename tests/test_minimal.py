import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.decomposition import TruncatedSVD

import reliefe
from reliefe.utils import load_arff


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class ReliefESmokeTests(unittest.TestCase):

    def test_operator_sgn(self):
        self.assertEqual(reliefe.sgn(-1), -1)
        self.assertEqual(reliefe.sgn(0), 0)
        self.assertEqual(reliefe.sgn(1), 1)

    def test_dense_classification_smoke(self):
        x = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0]])
        y = np.array([0, 0, 1, 1])
        model = reliefe.ReliefE(num_iter=4,
                                k=1,
                                embedding_based_distances=False,
                                verbose=False)
        model.fit(x, y)
        np.testing.assert_allclose(model.feature_importances_,
                                   np.array([1.0, -1.0, 1.0]))

    def test_custom_embedding_smoke(self):
        mat_obj = sio.loadmat(ROOT / "data" / "mcc" / "chess.mat")
        x = mat_obj["input_space"][:256]
        y = np.asarray(mat_obj["target_space"]).reshape(-1)[:256]
        model = reliefe.ReliefE(embedding_based_distances=True,
                                num_iter=32,
                                samples=128,
                                verbose=False)
        model.fit(x, y, embedding_method=TruncatedSVD(n_components=8))
        self.assertEqual(model.feature_importances_.shape[0], x.shape[1])
        self.assertTrue(np.isfinite(model.feature_importances_).all())

    def test_load_arff_multilabel_and_single_target(self):
        features = [0, 1, 2, 3, 4]
        targets = [5, 6]

        x_multi, y_multi, attr_ranges = load_arff(ROOT / "data" / "test.arff",
                                                  features, targets)
        self.assertEqual(x_multi.shape[0], y_multi.shape[0])
        self.assertEqual(y_multi.shape[1], 2)
        self.assertEqual(len(attr_ranges), len(features))

        x_single, y_single, _ = load_arff(ROOT / "data" / "test.arff",
                                          features, targets[0])
        self.assertEqual(x_single.shape[0], y_single.shape[0])
        self.assertGreaterEqual(y_single.shape[1], 2)

    def test_multilabel_embedding_smoke(self):
        x = np.array([[1.0, 0.0, 0.2, 0.0], [0.9, 0.1, 0.1, 0.0],
                      [0.0, 1.0, 0.0, 0.3], [0.1, 0.8, 0.0, 0.2],
                      [0.4, 0.4, 0.7, 0.1], [0.3, 0.5, 0.6, 0.2]])
        y = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1],
                      [1, 1, 0], [1, 1, 0]])

        model = reliefe.ReliefE(embedding_based_distances=True,
                                mlc_distance="cosine",
                                k=1,
                                num_iter=6,
                                samples=6,
                                verbose=False)
        model.fit(x, y, embedding_method="SVD")
        self.assertEqual(model.feature_importances_.shape[0], x.shape[1])
        self.assertTrue(np.isfinite(model.feature_importances_).all())

    def test_examples_run(self):
        for script_name in [
                "custom_embeddings.py", "example_synthetic.py",
                "minimal_reliefe.py", "minimal_reliefe2.py",
                "minimal_reliefe3.py"
        ]:
            with self.subTest(script=script_name):
                self.run_example(script_name)

    def run_example(self, script_name):
        env = dict(os.environ)
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not python_path else (
            f"{ROOT}{os.pathsep}{python_path}")
        completed = subprocess.run([sys.executable, str(EXAMPLES / script_name)],
                                   cwd=ROOT,
                                   env=env,
                                   capture_output=True,
                                   text=True,
                                   timeout=120)
        if completed.returncode != 0:
            self.fail(f"{script_name} failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
        self.assertTrue(completed.stdout.strip() or completed.stderr.strip())


if __name__ == "__main__":
    unittest.main()
