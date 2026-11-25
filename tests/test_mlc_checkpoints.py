"""Numerical checkpoint contracts; also run directly with Numba enabled."""
import unittest

import numpy as np
from scipy import sparse

from reliefe import ReliefE, reliefe_classes as rc


def kernel_scores(x, y, samples, checkpoints, *, average=False, adaptive=False, k=1,
                  distance='cityblock', target_distance='hamming', kernel=None):
    x = sparse.csr_matrix(np.asarray(x, dtype=float))
    y = sparse.csr_matrix(np.asarray(y, dtype=float))
    if kernel is None:
        kernel = rc._compiled_multi_label_classification_update_weights
    return kernel(np.asarray(samples, dtype=np.int64),
                  np.asarray(checkpoints, dtype=np.int64),
                  x.data, x.indptr, x.indices, y.data, y.indptr, y.indices,
                  y.shape[1], k, distance, target_distance, *x.shape,
                  adaptive, average)


class MLCCheckpointTests(unittest.TestCase):
    def test_leading_skipped_checkpoint(self):
        x = [[0, 0], [.1, 0], [1, 1], [1.1, 1]]
        y = [[1, 0], [1, 0], [1, 1], [0, 1]]
        # Samples 1 and 0 have identical-label neighbours: no update.
        # Samples 3 and 2 each add [.9, 1] under the existing update rule.
        expected = [[0, 0], [.9, 1], [.9, 1], [1.8, 2]]
        for average in (False, True):
            with self.subTest(average=average):
                actual = kernel_scores(x, y, [1, 3, 0, 2], [1, 2, 3, 4],
                                       average=average)
                np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_skipped_checkpoint_carries_nonzero_weights(self):
        x = [[0, 0], [.1, 0], [2, 2], [2.1, 2]]
        for skipped_target in ([1, 0], [0, 1]):
            # Target distances zero and one both skip, after a valid update.
            y = [[1, 0], [1, 1], [1, 0], skipped_target]
            for average in (False, True):
                with self.subTest(skipped_target=skipped_target, average=average):
                    actual = kernel_scores(x, y, [0, 2, 1], [1, 2, 3],
                                           average=average)
                    np.testing.assert_allclose(actual, [[.9, 1], [.9, 1], [1.8, 2]],
                                               atol=1e-14)
                    single = kernel_scores(x, y, [0, 2], [2], average=average)
                    np.testing.assert_array_equal(actual[1], single[0])

    def test_all_skipped_and_no_neighbours(self):
        cases = [([[0, 0]], [[1, 0]], [0, 0, 0], 1),
                 ([[0, 0], [.1, 0]], [[1, 0], [1, 1]], [0, 1, 0], 0),
                 ([[0, 0], [.1, 0]], [[1, 0], [1, 0]], [0, 1, 0], 1),
                 ([[0, 0], [.1, 0]], [[1, 0], [0, 1]], [0, 1, 0], 1)]
        for x, y, samples, k in cases:
            for average in (False, True):
                with self.subTest(rows=len(x), targets=y, k=k, average=average):
                    actual = kernel_scores(x, y, samples, [1, 2, 3], k=k,
                                           average=average)
                    np.testing.assert_array_equal(actual, np.zeros((3, 2)))

    def test_public_final_score_is_independent_of_extra_checkpoints(self):
        x = np.array([[0., 0.], [.1, 0.], [1., 1.], [1.1, 1.]])
        y = np.array([[1., 0.], [1., 0.], [1., 1.], [0., 1.]])
        for average in (False, True):
            for checkpoints in (2, [1, 2], [2, 3], [1, 2, 3, 4]):
                with self.subTest(average=average, checkpoints=checkpoints):
                    model = ReliefE(num_iter=checkpoints, k=1, samples=4,
                                    sparsity_threshold=1., normalize_descriptive=False,
                                    mlc_distance='hamming', use_average_neighbour=average)
                    model.fit(x, y)
                    expected = {1: [0, 0], 2: [.45, .5],
                                3: [.3, 1 / 3], 4: [.45, .5]}
                    for n, weights in zip(model.num_iter_, model.all_weights):
                        np.testing.assert_allclose(weights, expected[n], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
