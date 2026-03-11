import os
import unittest
import ast
import io
import runpy
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import scipy.io as sio
from sklearn.decomposition import TruncatedSVD

import reliefe
from reliefe.utils import load_arff


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class ReliefESmokeTests(unittest.TestCase):

    @staticmethod
    def parse_list_after_prefix(output, prefix):
        for line in output.splitlines():
            if line.startswith(prefix):
                return ast.literal_eval(line.split(":", 1)[1].strip())
        raise AssertionError(
            f"Did not find line starting with {prefix!r} in output:\n{output}")

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
        x = mat_obj["input_space"][:128]
        y = np.asarray(mat_obj["target_space"]).reshape(-1)[:128]
        model = reliefe.ReliefE(embedding_based_distances=True,
                                num_iter=24,
                                samples=64,
                                verbose=False)
        model.fit(x, y, embedding_method=TruncatedSVD(n_components=8))
        self.assertEqual(model.feature_importances_.shape[0], x.shape[1])
        self.assertTrue(np.isfinite(model.feature_importances_).all())

    def test_default_embedding_method_is_usable(self):
        mat_obj = sio.loadmat(ROOT / "data" / "mcc" / "chess.mat")
        x = mat_obj["input_space"][:128]
        y = np.asarray(mat_obj["target_space"]).reshape(-1)[:128]
        model = reliefe.ReliefE(embedding_based_distances=True,
                                num_iter=24,
                                samples=64,
                                verbose=False)
        model.fit(x, y)
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
                      [0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0], [0.0, 0.1, 0.9, 0.0],
                      [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
        y = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1],
                      [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]])

        model = reliefe.ReliefE(embedding_based_distances=True,
                                mlc_distance="cosine",
                                k=1,
                                num_iter=8,
                                samples=8,
                                verbose=False)
        model.fit(x, y, embedding_method="SVD")
        self.assertEqual(model.feature_importances_.shape[0], x.shape[1])
        self.assertTrue(np.isfinite(model.feature_importances_).all())
        self.assertGreater(np.abs(model.feature_importances_).sum(), 0.0)

    def test_examples_run(self):
        for script_name in [
                "custom_embeddings.py", "example_synthetic.py",
                "minimal_reliefe.py", "minimal_reliefe2.py",
                "minimal_reliefe3.py"
        ]:
            with self.subTest(script=script_name):
                output = self.run_example(script_name)
                self.assert_example_output(script_name, output)

    def run_example(self, script_name):
        previous_cwd = Path.cwd()
        buffer = io.StringIO()
        try:
            os.chdir(ROOT)
            with redirect_stdout(buffer):
                runpy.run_path(str(EXAMPLES / script_name), run_name="__main__")
        finally:
            os.chdir(previous_cwd)
        output = buffer.getvalue()
        self.assertTrue(output.strip())
        return output

    def assert_example_output(self, script_name, output):
        if script_name in {"custom_embeddings.py", "minimal_reliefe2.py"}:
            top = self.parse_list_after_prefix(output, "Top features")
            self.assertEqual(len(top), 5)
            self.assertEqual(len(set(top)), 5)
            self.assertTrue(all(isinstance(idx, int) for idx in top))
        elif script_name == "example_synthetic.py":
            self.assertIn("Successfully retrieved top features.", output)
            top = self.parse_list_after_prefix(output, "Top three")
            self.assertIn(0, top)
            self.assertIn(1, top)
        elif script_name == "minimal_reliefe.py":
            weights = self.parse_list_after_prefix(output, "Feature importances")
            self.assertEqual(len(weights), 10)
            self.assertGreater(max(weights), min(weights))
        elif script_name == "minimal_reliefe3.py":
            embedded = self.parse_list_after_prefix(output,
                                                    "Embedded input distances")
            origin = self.parse_list_after_prefix(output,
                                                  "Origin input distances")
            self.assertGreater(sum(abs(x) for x in embedded), 0.0)
            self.assertGreater(sum(abs(x) for x in origin), 0.0)
            self.assertNotEqual(embedded, origin)


if __name__ == "__main__":
    unittest.main()
