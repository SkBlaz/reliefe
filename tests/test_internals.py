import io
import logging
import os
import runpy
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

import reliefe
from reliefe import reliefe_classes as rc
from reliefe import reliefe_evaluation as reval
from reliefe import utils as ru


ROOT = Path(__file__).resolve().parents[1]


class FakeReducer:
    def __init__(self, fail_on_sparse=False, fail_on_fit=False):
        self.fail_on_sparse = fail_on_sparse
        self.fail_on_fit = fail_on_fit
        self.fit_calls = 0

    def fit(self, x):
        self.fit_calls += 1
        if self.fail_on_fit:
            raise RuntimeError("fit failed")
        return self

    def transform(self, x):
        if self.fail_on_sparse and sparse.issparse(x):
            raise TypeError("sparse not supported")
        arr = x.toarray() if sparse.issparse(x) else np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] == 1:
            return np.hstack([arr, arr])
        return arr[:, :max(1, min(2, arr.shape[1]))]


class FakeUMAPModule:
    class UMAP(FakeReducer):
        def __init__(self, *args, **kwargs):
            super().__init__()


class ReliefeInternalTests(unittest.TestCase):

    def test_sparsify_matrix_kernel(self):
        matrix = np.array([[0.1, 10.0], [0.2, 0.3]], dtype=float)
        with patch.object(rc.np.random, "random", return_value=0.0):
            out = rc._sparsify_matrix_kernel(matrix.copy(), epsilon=1.0)
        self.assertEqual(out.shape, matrix.shape)
        self.assertEqual(out[0, 1], 10.0)
        self.assertNotEqual(out[0, 0], 0.1)

    def test_numba_distance_variants(self):
        u = np.array([0.0, 1.0])
        v = np.array([1.0, 1.0])
        self.assertAlmostEqual(rc._numba_distance(u, v, "euclidean"), 1.0)
        self.assertAlmostEqual(rc._numba_distance(u, v, "cityblock"), 1.0)
        self.assertAlmostEqual(rc._numba_distance(u, v, "cosine"), 1 - 1 / np.sqrt(2))
        with self.assertRaises(ValueError):
            rc._numba_distance(u, v, "bad")

    def test_get_sparse_row(self):
        mat = sparse.csr_matrix([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
        row = rc._get_sparse_row(1, mat.data, mat.indptr, mat.indices)
        np.testing.assert_allclose(row, np.array([3.0, 0.0, 4.0]))

    def test_classification_kernel_paths(self):
        x = sparse.csr_matrix([[0.0, 1.0], [0.0, 2.0], [1.0, 0.0], [2.0, 0.0]])
        y = sparse.csr_matrix(np.array([[1, 0], [1, 0], [0, 1], [0, 1]]))
        class_members, example_to_class, class_first_indices = rc.ReliefE.compute_members(y)
        priors = np.array([0.5, 0.5])
        weights, kvec = rc._compiled_classification_update_weights(
            np.array([0, 2]),
            np.array([1, 2]),
            x.data,
            x.indptr,
            x.indices,
            priors,
            1,
            "cityblock",
            class_members,
            example_to_class,
            class_first_indices,
            x.shape[0],
            x.shape[1],
            True,
            True,
        )
        self.assertEqual(weights.shape, (2, 2))
        self.assertEqual(kvec.shape[0], 4)

    def test_numba_distance_target_variants(self):
        a = np.array([1.0, 0.0, 1.0])
        b = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(rc._numba_distance_target(a, b, "f1"), 0.5)
        self.assertAlmostEqual(rc._numba_distance_target(a, b, "accuracy"), 2 / 3)
        self.assertEqual(rc._numba_distance_target(a, a, "subset"), 3)
        self.assertAlmostEqual(rc._numba_distance_target(a, b, "hamming"), 2 / 3)
        self.assertAlmostEqual(rc._numba_distance_target(a, b, "cosine"), 0.5)
        self.assertEqual(rc._numba_distance_target(np.zeros(2), np.array([1.0, 0.0]),
                                                  "cosine"), 1.0)
        h1 = np.array([1.0, 1.0])
        h2 = np.array([-1.0, -1.0])
        self.assertAlmostEqual(rc._numba_distance_target(h1, h2, "hyperbolic"),
                               np.arccosh(2.0))
        with self.assertRaises(ValueError):
            rc._numba_distance_target(a, b, "bad")

    def test_multilabel_kernel_paths(self):
        x = sparse.csr_matrix(np.array([
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ]))
        y = sparse.csr_matrix(np.array([
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 1],
        ]))
        weights = rc._compiled_multi_label_classification_update_weights(
            np.array([0, 1]),
            np.array([1, 2]),
            x.data,
            x.indptr,
            x.indices,
            y.data,
            y.indptr,
            y.indices,
            y.shape[1],
            1,
            "cityblock",
            "cosine",
            x.shape[0],
            x.shape[1],
            True,
            False,
        )
        self.assertEqual(weights.shape, (2, 2))

        zero_weights = rc._compiled_multi_label_classification_update_weights(
            np.array([0]),
            np.array([1]),
            x.data,
            x.indptr,
            x.indices,
            sparse.csr_matrix(np.ones((4, 2))).data,
            sparse.csr_matrix(np.ones((4, 2))).indptr,
            sparse.csr_matrix(np.ones((4, 2))).indices,
            2,
            1,
            "cityblock",
            "subset",
            x.shape[0],
            x.shape[1],
            False,
            True,
        )
        self.assertTrue((zero_weights == 0).all())

    def test_reliefe_helpers_and_parameter_checks(self):
        model = rc.ReliefE(num_iter=[0.5, 2], verbose=True)
        model._compute_iterations(10)
        self.assertEqual(model.num_iter[0], 5)

        self.assertEqual(model.determine_latent_dim(np.ones((3, 3))), 3)

        with patch.object(rc.np.linalg, "lstsq", side_effect=RuntimeError("boom")):
            latent = model.determine_latent_dim(np.ones((4, 4)))
        self.assertEqual(latent, 4)
        self.assertIsInstance(
            model.determine_latent_dim(np.eye(4), empirical_left_continuous=False), int)

        model.task_type = ru.TaskTypes.HMLC
        model.use_average_neighbour = True
        model.mlc_distance = "hamming"
        model.k = 10
        model.num_iter = [0, 10]
        model._check_parameters(4, np.array([2, 2]))
        self.assertFalse(model.use_average_neighbour)
        self.assertEqual(model.k, 1)

        bad = rc.ReliefE()
        bad.task_type = ru.TaskTypes.CLASSIFICATION
        bad.mlc_distance = "bad"
        with self.assertRaises(ValueError):
            bad._check_parameters(4, np.array([2, 2]))

        too_small = rc.ReliefE(k=1)
        too_small.task_type = ru.TaskTypes.CLASSIFICATION
        with self.assertRaises(ValueError):
            too_small._check_parameters(2, np.array([1, 1]))

        y = sparse.csr_matrix(np.array([[1, 0], [0, 1], [1, 0]]))
        members, mapping, first = rc.ReliefE.compute_members(y)
        np.testing.assert_array_equal(members, np.array([0, 2, 1]))
        np.testing.assert_array_equal(mapping, np.array([0, 1, 0]))
        np.testing.assert_array_equal(first, np.array([0, 2, 3]))

        self.assertEqual(rc.ReliefE._compute_task_type(y), ru.TaskTypes.CLASSIFICATION)
        self.assertEqual(rc.ReliefE._compute_task_type(sparse.csr_matrix(np.array([[1, 1], [0, 1]]))),
                         ru.TaskTypes.MLC)

        tx, ty = rc.ReliefE._transform_x_y([[1.0, 0.0], [0.0, 1.0]], [0, 1])
        self.assertEqual(tx.getformat(), "csr")
        self.assertEqual(ty.getformat(), "csr")

        silent = rc.ReliefE(verbose=False)
        silent.send_message("quiet")

    def test_resolve_embedding_method(self):
        with patch.object(rc, "umap", None):
            self.assertEqual(rc.ReliefE._resolve_embedding_method("auto"), "SVD")
        with patch.object(rc, "umap", FakeUMAPModule):
            self.assertEqual(rc.ReliefE._resolve_embedding_method("auto"), "UMAP")
        self.assertEqual(rc.ReliefE._resolve_embedding_method("SVD"), "SVD")

    def test_fit_branches_and_failures(self):
        x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 1, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir) / "k.npy"
            model = rc.ReliefE(num_iter=4, k=1, embedding_based_distances=False, samples=2)
            model.fit(x, y, store_neighborhoods=store)
            self.assertTrue(store.exists())

        dense_fallback = rc.ReliefE(num_iter=4, k=1, embedding_based_distances=True, samples=4)
        dense_fallback.fit(x, y, embedding_method=FakeReducer(fail_on_sparse=True))
        self.assertEqual(dense_fallback.feature_importances_.shape[0], 2)

        fallback = rc.ReliefE(num_iter=4, k=1, embedding_based_distances=True, samples=4)
        fallback.fit(x, y, embedding_method=FakeReducer(fail_on_fit=True))
        self.assertEqual(fallback.feature_importances_.shape[0], 2)

        with patch.object(rc, "umap", None):
            with self.assertRaises(ImportError):
                rc.ReliefE(num_iter=4, k=1, embedding_based_distances=True,
                           samples=4, LD_friendly=False).fit(
                    x, y, embedding_method="UMAP")

        ml_x = np.array([
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ])
        ml_y = np.array([
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 1],
        ])
        model = rc.ReliefE(num_iter=4, k=1, embedding_based_distances=True,
                           mlc_distance="cosine", samples=4)
        model.fit(ml_x, ml_y, embedding_method=FakeReducer())
        self.assertEqual(model.feature_importances_.shape[0], 2)

        hyper = rc.ReliefE(num_iter=4, k=1, embedding_based_distances=True,
                           mlc_distance="hyperbolic", samples=4)
        with patch.object(rc, "_compiled_multi_label_classification_update_weights",
                          return_value=np.ones((1, 2))):
            hyper.fit(ml_x, ml_y, embedding_method=FakeReducer())
        np.testing.assert_allclose(hyper.feature_importances_, np.array([0.25, 0.25]))

        with patch.object(rc.ReliefE, "_compute_task_type", return_value=ru.TaskTypes.REGRESSION):
            with self.assertRaises(ValueError):
                rc.ReliefE().fit(x, y)

    def test_fit_verbose_umap_and_custom_branches(self):
        x = np.ones((6, 4))
        y = np.array([0, 0, 1, 1, 0, 1])

        with patch.object(rc, "umap", FakeUMAPModule), \
             patch.object(rc.ReliefE, "determine_latent_dim", return_value=2), \
             patch.object(rc, "_compiled_classification_update_weights",
                          return_value=(np.ones((1, 4)), np.arange(8))), \
             patch.object(rc.logger, "info") as info:
            model = rc.ReliefE(num_iter=4,
                               k=1,
                               embedding_based_distances=True,
                               determine_k_automatically=True,
                               samples=4,
                               verbose=True,
                               sparsity_threshold=0.0,
                               LD_friendly=False)
            model.fit(x, y, embedding_method="UMAP")
            self.assertTrue(info.called)

        with patch.object(rc.ReliefE, "determine_latent_dim", return_value=10), \
             patch.object(rc.logger, "info") as info:
            model = rc.ReliefE(num_iter=4,
                               k=1,
                               embedding_based_distances=True,
                               samples=4,
                               verbose=True,
                               LD_friendly=False)
            model.fit(x, y, embedding_method=FakeReducer(fail_on_fit=True))
            self.assertTrue(info.called)

        with patch.object(rc.ReliefE, "determine_latent_dim", return_value=1), \
             patch.object(rc.logger, "info") as info:
            model = rc.ReliefE(num_iter=5,
                               k=1,
                               embedding_based_distances=True,
                               samples=5,
                               verbose=True,
                               LD_friendly=False)
            model.fit(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
                                [0.0, 1.0], [1.0, 2.0], [2.0, 1.0]]),
                      np.array([0, 0, 1, 1, 1, 1]),
                      embedding_method=FakeReducer(fail_on_sparse=True))
            self.assertTrue(info.called)

        with patch.object(rc.ReliefE, "determine_latent_dim", return_value=2):
            model = rc.ReliefE(num_iter=5,
                               k=1,
                               embedding_based_distances=True,
                               samples=5,
                               verbose=True,
                               LD_friendly=True)
            model.fit(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
                                [0.0, 1.0], [1.0, 2.0], [2.0, 1.0]]),
                      np.array([0, 0, 1, 1, 1, 1]),
                      embedding_method="SVD")

    def test_fit_mlc_verbose_and_umap_import_errors(self):
        x = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8],
                      [1.0, 0.0], [0.0, 1.0]])
        y = np.array([[1, 0], [1, 1], [0, 1], [0, 1], [1, 0], [1, 1]])

        with patch.object(rc.ReliefE, "determine_latent_dim", side_effect=[2, 1]), \
             patch.object(rc, "_compiled_multi_label_classification_update_weights",
                          return_value=np.ones((1, 2))), \
             patch.object(rc.logger, "info") as info:
            model = rc.ReliefE(num_iter=4,
                               k=1,
                               embedding_based_distances=True,
                               mlc_distance="cosine",
                               samples=4,
                               verbose=True,
                               LD_friendly=False)
            model.fit(x, y, embedding_method=FakeReducer())
            self.assertTrue(info.called)

        with patch.object(rc.ReliefE, "determine_latent_dim", side_effect=[2, 1]), \
             patch.object(rc, "_compiled_multi_label_classification_update_weights",
                          return_value=np.ones((1, 2))), \
             patch.object(rc.logger, "info") as info:
            model = rc.ReliefE(num_iter=4,
                               k=1,
                               embedding_based_distances=True,
                               mlc_distance="hyperbolic",
                               samples=4,
                               verbose=True,
                               LD_friendly=False)
            model.fit(x, y, embedding_method="SVD")
            self.assertTrue(info.called)

        with patch.object(rc, "umap", None):
            with self.assertRaises(ImportError):
                rc.ReliefE(num_iter=4, k=1, embedding_based_distances=False,
                           mlc_distance="cosine", samples=4,
                           LD_friendly=False).fit(x, y, embedding_method="UMAP")
            with self.assertRaises(ImportError):
                rc.ReliefE(num_iter=4, k=1, embedding_based_distances=False,
                           mlc_distance="hyperbolic", samples=4,
                           LD_friendly=False).fit(x, y, embedding_method="UMAP")

        with patch.object(rc, "umap", FakeUMAPModule), \
             patch.object(rc, "_compiled_multi_label_classification_update_weights",
                          return_value=np.ones((1, 2))):
            rc.ReliefE(num_iter=4, k=1, embedding_based_distances=False,
                       mlc_distance="cosine", samples=4,
                       LD_friendly=False).fit(x, y, embedding_method="UMAP")
            rc.ReliefE(num_iter=4, k=1, embedding_based_distances=False,
                       mlc_distance="hyperbolic", samples=4,
                       LD_friendly=False).fit(x, y, embedding_method="UMAP")

    def test_direct_kernel_edge_cases(self):
        x = sparse.csr_matrix(np.array([[np.nan, 1.0], [0.0, 1.0], [1.0, 0.0],
                                        [1.0, 1.0]]))
        y = sparse.csr_matrix(np.array([[1, 0], [1, 0], [0, 1], [0, 1]]))
        class_members, example_to_class, class_first_indices = rc.ReliefE.compute_members(y)
        weights, _ = rc._compiled_classification_update_weights(
            np.array([0]),
            np.array([1]),
            x.data,
            x.indptr,
            x.indices,
            np.array([0.5, 0.5]),
            1,
            "cityblock",
            class_members,
            example_to_class,
            class_first_indices,
            x.shape[0],
            x.shape[1],
            False,
            False,
        )
        self.assertEqual(weights.shape, (1, 2))

        self.assertEqual(rc._numba_distance_target(np.array([1, 0]),
                                                  np.array([0, 1]),
                                                  "subset"), 2)

        x2 = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2],
                                         [0.2, 0.8]]))
        y2 = sparse.csr_matrix(np.array([[1, 0], [0, 1], [1, 1], [1, 1]]))
        weights2 = rc._compiled_multi_label_classification_update_weights(
            np.array([0]),
            np.array([1]),
            x2.data,
            x2.indptr,
            x2.indices,
            y2.data,
            y2.indptr,
            y2.indices,
            y2.shape[1],
            1,
            "cityblock",
            "cosine",
            x2.shape[0],
            x2.shape[1],
            False,
            True,
        )
        self.assertEqual(weights2.shape, (1, 2))

        singleton_x = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        singleton_y = sparse.csr_matrix(np.array([[1, 0], [0, 1]]))
        singleton_members, singleton_mapping, singleton_first = rc.ReliefE.compute_members(
            singleton_y)
        singleton_weights, singleton_kvec = rc._compiled_classification_update_weights(
            np.array([0]),
            np.array([1]),
            singleton_x.data,
            singleton_x.indptr,
            singleton_x.indices,
            np.array([0.5, 0.5]),
            1,
            "cityblock",
            singleton_members,
            singleton_mapping,
            singleton_first,
            singleton_x.shape[0],
            singleton_x.shape[1],
            False,
            False,
        )
        self.assertEqual(singleton_weights.shape, (1, 2))
        self.assertEqual(singleton_kvec.shape, (2,))

        single_x = sparse.csr_matrix(np.array([[1.0, 0.0]]))
        single_y = sparse.csr_matrix(np.array([[1, 0]]))
        single_mlc_weights = rc._compiled_multi_label_classification_update_weights(
            np.array([0]),
            np.array([1]),
            single_x.data,
            single_x.indptr,
            single_x.indices,
            single_y.data,
            single_y.indptr,
            single_y.indices,
            single_y.shape[1],
            1,
            "cityblock",
            "cosine",
            single_x.shape[0],
            single_x.shape[1],
            False,
            False,
        )
        self.assertEqual(single_mlc_weights.shape, (1, 2))


class UtilsAndEvaluationTests(unittest.TestCase):

    def test_make_one_hot_encoder_and_measure_time(self):
        encoder = ru.make_one_hot_encoder(categories=[["a", "b"]], sparse_output=False)
        transformed = encoder.fit_transform([["a"], ["b"]])
        self.assertEqual(transformed.shape, (2, 2))

        calls = []

        def fake_encoder(*args, **kwargs):
            calls.append(kwargs)
            if "sparse_output" in kwargs:
                raise TypeError("old sklearn")
            return "fallback"

        with patch.object(ru, "OneHotEncoder", side_effect=fake_encoder):
            result = ru.make_one_hot_encoder([["a"]], False)
        self.assertEqual(result, "fallback")
        self.assertEqual(len(calls), 2)

        @ru.measure_time
        def add(a, b):
            return a + b

        result, elapsed = add(1, 2)
        self.assertEqual(result, 3)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_library_modules_do_not_configure_root_logging(self):
        self.assertEqual(rc.logger.name, "reliefe.reliefe_classes")
        self.assertEqual(ru.logger.name, "reliefe.utils")
        self.assertEqual(rc.logger.handlers, [])
        self.assertEqual(ru.logger.handlers, [])

    def test_metadata_and_imputation_helpers(self):
        with tempfile.NamedTemporaryFile("w+", delete=False) as handle:
            handle.write("toy;0:2;3:4\n")
            spec_path = handle.name
        self.assertEqual(ru.read_meta("toy", spec_path), ([0, 1, 2], [3, 4]))
        os.unlink(spec_path)

        with tempfile.NamedTemporaryFile("w+", delete=False) as handle:
            handle.write("toy;0:2;3\n")
            spec_path = handle.name
        self.assertEqual(ru.read_meta("toy", spec_path), ([0, 1, 2], [3]))
        os.unlink(spec_path)

        self.assertEqual(ru.aggregate_num([1, 2, 3]), 2)
        self.assertEqual(ru.aggregate_nom(["a", "a", "b"]), "a")

        numeric = [1.0, np.nan, 3.0]
        self.assertEqual(ru.impute_missing_values(numeric, [], True, "?"), 1)
        self.assertEqual(numeric[1], 2.0)

        nominal = ["x", "?", "x"]
        self.assertEqual(ru.impute_missing_values(nominal, ["x", "y"], False, "?"), 1)
        self.assertEqual(nominal[1], "x")

        all_missing_numeric = [np.nan]
        ru.impute_missing_values(all_missing_numeric, [], True, "?")
        self.assertEqual(all_missing_numeric[0], 0.0)

        all_missing_nominal = ["?"]
        ru.impute_missing_values(all_missing_nominal, ["a"], False, "?")
        self.assertEqual(all_missing_nominal[0], "a")

    def test_sparse_parsing_and_task_type(self):
        self.assertEqual(ru.basic_parse_sparse_line("{}", 0), {})
        self.assertEqual(ru.basic_parse_sparse_line("{0 1, 2 3}", 0), {0: "1", 2: "3"})
        with self.assertRaises(ValueError):
            ru.basic_parse_sparse_line("0 1")

        self.assertEqual(ru.get_task_type([["a", "b"]]), ru.TaskTypes.CLASSIFICATION)
        self.assertEqual(ru.get_task_type([[]]), ru.TaskTypes.REGRESSION)
        self.assertEqual(ru.get_task_type([["0", "1"], ["0", "1"]]), ru.TaskTypes.MLC)
        with self.assertRaises(ValueError):
            ru.get_task_type([["0", "1"], ["yes", "no"]])

    def test_arff_helpers_and_wrapper(self):
        x, y, ranges = ru.load_arff(ROOT / "data" / "test.arff", [0, 1, 2, 3, 4], [5, 6])
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertEqual(len(ranges), 5)

        with self.assertRaises(ValueError):
            ru.load_arff(ROOT / "data" / "test.arff", [0, 1], [1, 2])

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation demo\n@attribute a string\n@attribute c {x,y}\n@data\nfoo,x\n")
            bad_type = handle.name
        with self.assertRaises(ValueError):
            ru.load_arff(bad_type, [0], [1])
        os.unlink(bad_type)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation demo\n@attribute a numeric\n@attribute target numeric\n@data\n1,2\n")
            regression = handle.name
        with self.assertRaises(NotImplementedError):
            ru.load_arff(regression, [0], [1])
        os.unlink(regression)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation demo\n@attribute a numeric\n@attribute target {0,1}\n@data\n1,0\n")
            short = handle.name
        with self.assertRaises(ValueError):
            ru.load_arff(short, [0, 2], [1])
        os.unlink(short)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation sparse\n@attribute a {x,y}\n@attribute target {0,1}\n@data\n{}\n")
            sparse_bad = handle.name
        with self.assertRaises(ValueError):
            ru.load_arff(sparse_bad, [0], [1])
        os.unlink(sparse_bad)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation sparse\n@attribute a numeric\n@attribute b numeric\n@data\n{2 1}\n")
            sparse_one = handle.name
        self.assertEqual(ru.is_one_based_sparse_arff(sparse_one, 2), (True, 1))
        os.unlink(sparse_one)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation sparse\n@attribute a numeric\n@attribute b numeric\n@data\n\n{0 1}\n")
            sparse_zero = handle.name
        self.assertEqual(ru.is_one_based_sparse_arff(sparse_zero, 2), (True, 0))
        os.unlink(sparse_zero)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation sparse\n@attribute a numeric\n@attribute b numeric\n@attribute c numeric\n@data\n{1 1}\n")
            sparse_unknown = handle.name
        with self.assertRaises(ValueError):
            ru.is_one_based_sparse_arff(sparse_unknown, 3)
        os.unlink(sparse_unknown)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation dense\n@attribute a numeric\n@attribute b numeric\n@data\n1,2\n")
            dense = handle.name
        self.assertEqual(ru.is_one_based_sparse_arff(dense, 2), (False, 0))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            ru.show_arff_attributes(dense, 1)
        self.assertIn("Found 2 attributes", buffer.getvalue())
        os.unlink(dense)

        def ranker(xs, ys, scale):
            self.assertEqual(xs.shape[0], ys.shape[0])
            return np.arange(xs.shape[1]) * scale

        scores = ru.feature_ranking_wrapper(ROOT / "data" / "test.arff",
                                            [1, 2, 3, 4, 5], 6, ranker, 1)
        self.assertEqual(set(scores.keys()), {1, 2, 3, 4, 5})

        with patch.object(ru, "load_arff",
                          return_value=(np.ones((2, 3)), np.array([0, 1]),
                                        {0: (0, 1), 1: (1, 2), 2: (2, 3)})):
            scores = ru.feature_ranking_wrapper("dummy", None, 0,
                                                lambda xs, ys: np.array([1, 2, 3]))
        self.assertEqual(scores, {0: 1, 1: 2, 2: 3})

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation demo\n@attribute\n@data\n1\n")
            malformed = handle.name
        with patch.object(ru.re, "search", return_value=None):
            with self.assertRaises(ValueError):
                ru.load_arff(malformed, [0], [0])
        os.unlink(malformed)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation demo\n@attribute skip string\n@attribute a numeric\n@attribute target {0,1}\n@data\nfoo,1,0\n")
            unsupported = handle.name
        with self.assertRaises(NotImplementedError):
            ru.load_arff(unsupported, [1], [2])
        os.unlink(unsupported)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation sparse\n@attribute a numeric\n@attribute b {0,1}\n@attribute c {0,1}\n@data\n{0 1.5,2 1}\n")
            sparse_mlc = handle.name
        x_sparse, y_sparse, _ = ru.load_arff(sparse_mlc, [0], [1, 2])
        self.assertEqual(x_sparse.shape[0], y_sparse.shape[0])
        os.unlink(sparse_mlc)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation sparse\n@attribute a numeric\n@attribute b {1,2}\n@attribute c {0,1}\n@data\n{0 1.5,2 1}\n")
            sparse_nominal_missing = handle.name
        with self.assertRaises(ValueError):
            ru.load_arff(sparse_nominal_missing, [0, 1], [2])
        os.unlink(sparse_nominal_missing)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation demo\n@attribute a numeric\n@attribute target {0,1}\n@data\n?,0\n")
            missing_numeric = handle.name
        x_missing, _, _ = ru.load_arff(missing_numeric, [0], [1], impute_missing=False)
        self.assertTrue(np.isnan(x_missing[0, 0]))
        os.unlink(missing_numeric)

        with patch.object(ru, "get_task_type", return_value="wrong"):
            with self.assertRaises(NotImplementedError):
                ru.load_arff(ROOT / "data" / "test.arff", [0, 1, 2, 3, 4], [5, 6])

    def test_test_stuff_and_main(self):
        with patch.object(ru, "load_arff") as loader, \
             patch.object(ru.glob, "glob", return_value=[]):
            loader.return_value = (np.array([]), np.array([]), {})
            ru.test_stuff(test_first=True, test_second=True)
            self.assertEqual(loader.call_count, 2)

        with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as handle:
            handle.write("@relation demo\n@attribute a numeric\n@data\n1\n")
            demo = handle.name
        try:
            with patch("glob.glob", return_value=[demo]):
                runpy.run_path(str(ROOT / "reliefe" / "utils.py"), run_name="__main__")
        finally:
            os.unlink(demo)

    def test_evaluation_wrapper(self):
        x_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.9, 0.8]])
        x_test = np.array([[0.0, 0.1], [1.0, 0.9]])
        y_train = np.array([0, 1, 0, 1])
        y_test = np.array([0, 1])
        score = reval.evaluate_importances_with_logistic_regression(
            x_train, x_test, y_train, y_test, np.array([0.9, 0.1]), 1)
        self.assertGreaterEqual(score, 0.0)

        with patch.object(reval, "f1_score", side_effect=[ValueError("x"), 0.5]):
            score = reval.evaluate_importances_with_logistic_regression(
                x_train, x_test, y_train, y_test.reshape(-1, 1), np.array([0.9, 0.1]), 1)
        self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
