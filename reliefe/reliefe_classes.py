"""
ReliefE algorithm : Skrlj and Petkovic, 2021
"""

from typing import List, Union
import numpy as np
from scipy import sparse
import umap.umap_ as umap
from sklearn.preprocessing import OneHotEncoder, normalize
from scipy.sparse import csr_matrix
from typing import Union, List
from numba import jit, prange
from sklearn import metrics
from .utils import TaskTypes, MLCDistances
import time
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


@jit(nopython=True)
def sgn(el):
    """
    The standard sgn function.
    :param el: a float.
    """

    if el < 0:
        return -1

    elif el == 0:
        return 0

    else:
        return 1


@jit(nopython=True, parallel=True)
def _sparsify_matrix_kernel(sparse_matrix_input, epsilon=0.001):
    """
    : param sparse_matrix_input: a matrix to be sparsified
    : param epsilon: approximation constant. See https://www.researchgate.net/publication/221462839_A_Fast_Random_Sampling_Algorithm_for_Sparsifying_Matrices
    """

    n = sparse_matrix_input.shape[1]
    for i in prange(n):
        for j in prange(n):
            matrix_el = sparse_matrix_input[i, j]
            sqrt_n = np.sqrt(n)
            if matrix_el > epsilon / sqrt_n:
                continue

            else:
                coin = np.random.random()
                probability = (np.sqrt(n) * np.abs(matrix_el)) / (epsilon)
                placeholder = 0
                if coin <= probability:
                    placeholder = sgn(matrix_el) * (epsilon) / sqrt_n
                sparse_matrix_input[i, j] = placeholder
    return sparse_matrix_input


@jit(nopython=True)
def _numba_distance(u, v, dist):
    """
    A method consisting of to-be compiled distance computations.
    : param u: vector 1
    : param v: vector 2
    : param dist: distance name
    """

    if dist == "euclidean":
        return np.linalg.norm(v - u)

    elif dist == "cityblock":
        return np.sum(np.abs(v - u))

    elif dist == "cosine":
        return 1.0 - np.dot(u, v) / ((np.dot(u, u)**.5) * (np.dot(v, v)**.5))

    else:
        raise ValueError()


@jit(nopython=True)
def _get_sparse_row(row_index, data, pointers, indices):
    """
    Unfolder csr indexing. See CSR format for more details.
    : param row_index: row indices.
    : param data: data
    : param pointers: pointers
    : param indices: indices
    """

    placeholder = np.zeros(np.max(indices) + 1, dtype=np.float64)
    crange_from = pointers[row_index]
    crange_to = pointers[row_index + 1]
    for j in range(crange_from, crange_to):
        placeholder[indices[j]] = data[j]
    return placeholder


@jit(nopython=True)
def _compiled_classification_update_weights(
        samples, num_iter, data, pointers, indices, priors, k,
        pairwise_distances, class_members, examples_to_class,
        class_first_indices, nrow_all, ncol_all, determine_k_automatically,
        use_average_neighbour):
    """
    A compiled kernel for the weight update step.
    :param samples: Number of instance samples
    :param num_iter: a list of numbers of iterations
    :param data: Input data set (vector of present floats)
    :param pointers: pointers to columnsets
    :param indices: columnsets
    :param priors: Prior probabilities of a given class
    :param k: num. of neighbors
    :param pairwise_distances: Distance to be computed
    :param class_members: [list of indices of examples that belong to class0, ... to class 1, ...] flattened
    :param examples_to_class: {example index: class index, ...}
    :param class_first_indices: list of first indices of the classes corresponding to the class_members.
    Contains an additional element: The last value is the length of the class_members.
    :param nrow_all: number of rows
    :param ncol_all: number of cols
    :param use_average_neighbour: see ReliefE constructor
    """

    number_of_columns = ncol_all
    ndim_via_pointer = np.max(indices.astype(np.int64)) + 1

    weights_final = np.zeros((len(num_iter), number_of_columns),
                             dtype=np.float64)
    weights = np.zeros(number_of_columns, dtype=np.float64)  # current weights
    num_iter_position = 0

    for i_sample, sample in enumerate(samples):
        considered_class = examples_to_class[sample]

        for c, number_of_samples in enumerate(class_first_indices[:-1]):
            first_index = class_first_indices[c]
            last_index = class_first_indices[c + 1]
            members = class_members[first_index:last_index]

            # extract distances
            distances = np.zeros(len(members), dtype=np.float64)
            ith_sample_row = _get_sparse_row(sample, data, pointers, indices)

            for i_neighbor, neighbour in enumerate(members):
                neigh_row = _get_sparse_row(neighbour, data, pointers, indices)

                internal_distance = _numba_distance(ith_sample_row, neigh_row,
                                                    pairwise_distances)
                distances[i_neighbor] = internal_distance
            top_neighbour_indices = np.argsort(distances)

            if determine_k_automatically:
                sorted_distances = distances[
                    top_neighbour_indices]  #np.sort(distances)
                diffs = np.diff(sorted_distances)
                fdiffs = diffs[1:-1]
                fdiffs2 = diffs[2:]
                ratios = np.divide(fdiffs2, fdiffs)

                if len(ratios) > 0:
                    k = np.argmax(ratios) + 1

            if considered_class == c:
                offset = 1  # ignore itself when computing the neighbours
                prior = -1.0  # 1 negative, so the weight gets lower for hits

            else:
                offset = 0
                prior = priors[c] / (1 - priors[considered_class])

            top_neighbors = top_neighbour_indices[offset:k + offset]
            top_neighbor_members = members[top_neighbors]
            nearest_neighbours = np.zeros(
                (len(top_neighbor_members), ndim_via_pointer))

            for enx, tnm in enumerate(top_neighbor_members):
                rx = _get_sparse_row(tnm, data, pointers, indices)
                assert rx.shape[0] == nearest_neighbours.shape[1]
                nearest_neighbours[enx] = rx

            # Update weights
            sample_row = _get_sparse_row(sample, data, pointers, indices)

            for j in range(number_of_columns):

                # Difference between near hits/misses for the j-th feature
                if use_average_neighbour:
                    diff = np.abs(sample_row[j] -
                                  np.mean(nearest_neighbours[:, j]))

                else:
                    diff = np.mean(
                        np.abs(sample_row[j] - nearest_neighbours[:, j]))

                if np.isnan(diff):
                    update = 0

                else:
                    update = prior * diff

                weights[j] += update

        if i_sample + 1 == num_iter[num_iter_position]:
            weights_final[num_iter_position, :] = weights
            num_iter_position = num_iter_position + 1

    return weights_final


@jit(nopython=True)
def _numba_distance_target(row1, row2, dist):
    """
    Compute distance amongst targets.
    """

    if dist == "f1":
        intersection = (row1 * row2.T).sum()
        first = len(row1.nonzero()[0])
        second = len(row2.nonzero()[0])
        s = first + second
        return 1 - 2 * intersection / s if s > 0 else 0.0

    elif dist == "accuracy":
        intersection = (row1 * row2.T).sum()
        union = len((row1 + row2).nonzero()[0])
        return 1.0 - intersection / union if union > 0 else 0.0

    elif dist == "subset":
        ns1 = np.sum(row1 == row2)
        if ns1 > 0:
            return ns1
        else:
            return len(row1)

    elif dist == "hamming":
        return np.abs(row1 - row2).sum() / len(row1)

    elif dist == "cosine":
        return 1 - np.dot(row1,
                          row2) / (np.linalg.norm(row1) * np.linalg.norm(row2))

    elif dist == "hyperbolic":
        return np.arccosh(np.sum(-(row1 * row2)))

    else:
        raise ValueError()


@jit(nopython=True)
def _compiled_multi_label_classification_update_weights(
        samples, num_iter, data, pointers, indices, data_y, pointers_y,
        indices_y, k, pairwise_distances, mlc_distance, nrow, ncol):
    """
    A compiled kernel for the weight update step.
    :param samples: Number of instance samples
    :param num_iter: list of numbers of iterations
    :param xs: Input data set
    :param y: target data
    :param k: num. of neighbors
    :param pairwise_distances: Distances
    :param mlc_distance: distance type for MLC
    :param nrow: number of rows
    :param ncol: number of columns
    """

    number_of_columns = ncol
    n_examples = nrow
    ndim_via_pointer = np.max(indices.astype(np.int64)) + 1

    weights_final = np.zeros((len(num_iter), number_of_columns),
                             dtype=np.float64)
    weights = np.zeros(number_of_columns, dtype=np.float64)  # current weights
    num_iter_position = 0

    for i_sample, sample in enumerate(samples):

        # extract distances
        distances = np.zeros(n_examples)
        ith_sample_row = _get_sparse_row(sample, data, pointers, indices)

        for i_neighbor in range(n_examples):
            neigh_row = _get_sparse_row(i_neighbor, data, pointers, indices)
            internal_distance = _numba_distance(ith_sample_row, neigh_row,
                                                pairwise_distances)
            distances[i_neighbor] = internal_distance

        top_neighbor_indices = np.argsort(distances)[1:k + 1]
        nearest_neighbours = np.zeros(
            (len(top_neighbor_indices), ndim_via_pointer))

        for enx, tnm in enumerate(top_neighbor_indices):
            rx = _get_sparse_row(tnm, data, pointers, indices)
            assert rx.shape[0] == nearest_neighbours.shape[1]
            nearest_neighbours[enx] = rx

        # Update weights
        target_diffs = np.zeros(k)

        y_sample_row = _get_sparse_row(sample, data_y, pointers_y, indices_y)

        for i in range(k):
            col_idx = top_neighbor_indices[i]
            ith_y_row = _get_sparse_row(col_idx, data_y, pointers_y, indices_y)
            tdist = _numba_distance_target(y_sample_row, ith_y_row,
                                           mlc_distance)
            target_diffs[i] = tdist

        t_diff = np.mean(target_diffs)

        if not (0 < t_diff < 1):
            # skipping those for which the update is ill defined
            continue

        sample_row = _get_sparse_row(sample, data, pointers, indices)
        for j in range(number_of_columns):
            # Difference between near hits/misses for the j-th feature
            descriptive_diffs = np.abs(sample_row[j] -
                                       nearest_neighbours[:, j])
            d_diff = np.mean(descriptive_diffs)
            t_d_diff = np.mean(target_diffs * descriptive_diffs)
            weights[j] += t_d_diff / t_diff - (d_diff - t_diff) / (1.0 -
                                                                   t_diff)

        if i_sample + 1 == num_iter[num_iter_position]:
            weights_final[num_iter_position, :] = weights
            num_iter_position += 1

    return weights_final


class ReliefE:
    def __init__(self,
                 num_iter: Union[float, int, List[Union[float, int]]] = 1.0,
                 k=8,
                 normalize_descriptive=True,
                 embedding_based_distances=False,
                 num_threads="all",
                 verbose=False,
                 mlc_distance="f1",
                 latent_dimension=128,
                 determine_k_automatically=False,
                 use_average_neighbour=False):
        """
        Initiate the Relief object. Some standard parameters can be assigned:
        :param num_iter:
        :param k:
        :param normalize_descriptive:
        :param embedding_based_distances:
        :param num_threads:
        :param verbose:
        :param mlc_distance:
        :param latent_dimension:
        :param determine_k_automatically:
        :param use_average_neighbour: Should a) compute the average neighbour, and b) perform weight updates with it?
        Note that standard Relief a) uses all neighbours, b) performs updates, and c) averages the updates
        update
        """

        self.num_iter = num_iter if isinstance(num_iter, list) else [num_iter]
        self.determine_k_automatically = determine_k_automatically
        self.latent_dimension = latent_dimension
        self.k = k
        self.normalize = normalize_descriptive
        self.embedding_based_distances = embedding_based_distances
        self.verbose = verbose
        self.mlc_distance = mlc_distance
        self.timed = {}
        self.sparsity_threshold = 0.2
        self.max_ram = 8
        self.samples = 3000
        self.use_average_neighbour = use_average_neighbour
        self.feature_importances_ = None
        self.task_type = "unknown"

    def _compute_iterations(self, n_examples):
        for i, value in enumerate(self.num_iter):
            if isinstance(value, float):
                self.num_iter[i] = round(value * n_examples)

    def determine_latent_dim(self,
                             xs,
                             true_dim=None,
                             empirical_left_continuous=True,
                             metric="euclidean"):
        """
        A method for computing the latent dimension,
        based on the paper: https://www.nature.com/articles/s41598-017-11873-y.pdf
        :param xs: data matrix X
        :param true_dim: the actual dimension or None
        :param empirical_left_continuous:
        """

        # data, pointers, indices = xs.data, xs.indptr, xs.indices

        if self.verbose:
            logging.info("Estimating latent dimension.")

        if xs.shape[1] <= 3:
            return xs.shape[1]

        # dists = metrics.pairwise_distances(xs, metric = _cyclic_euclidean_kernel)
        dists = metrics.pairwise.euclidean_distances(xs)
        dists = np.sort(dists, axis=1)
        i0 = (dists > 0).argmax(axis=1)
        dmat = []

        for j in range(len(i0)):
            cx = i0[j]
            ds = dists[j, cx]
            ds1 = dists[j, cx + 1]
            dmat.append([ds, ds1])

        dists = np.array(dmat)

        if self.verbose:
            self.send_message("Distances computed.")

        # compute r2 / r1
        mu = dists[:, 1] / dists[:, 0]
        mu = np.nan_to_num(mu)

        # Cumulative Femp
        f_empirical = [-1 for _ in range(xs.shape[0])]
        max_position = xs.shape[0]  # something out of range
        for order, position in enumerate(np.argsort(mu)):
            f_empirical[position] = order
            if empirical_left_continuous and order == xs.shape[0] - 1:
                max_position = position

        # handle log(0)
        if empirical_left_continuous:
            f_empirical = (1.0 + np.array(f_empirical)) / xs.shape[0]
            ok_indices = list(range(max_position)) + list(
                range(max_position + 1, xs.shape[0]))
            mu = mu[ok_indices]
            f_empirical = f_empirical[ok_indices]

        else:
            f_empirical = np.array(f_empirical) / xs.shape[0]

        # X - axis
        mu_first = np.log(mu)

        # Y - axis
        mu_second = np.log(1 - f_empirical)

        # some technicalities
        mu_first = mu_first.reshape(-1, 1)

        # fit
        try:

            latent_dim, _, _, _ = np.linalg.lstsq(mu_first,
                                                  mu_second,
                                                  rcond=None)
            latent_dim = int(round(-latent_dim[0]))

        except Exception as es:
            self.send_message(es)
            latent_dim = min(128, xs.shape[0])

        latent_dim = max(16, latent_dim)

        if self.verbose:
            logging.info(
                "Estimated latent dimension is: {}".format(latent_dim))

        return latent_dim

    def _check_parameters(self, n_examples, class_counts):
        message = "Number of {} ({}) was not between {} and {} ({}). Now, it is set to {}."
        min_iter = 128
        for i, iterations in enumerate(self.num_iter):
            self.num_iter[i] = max(iterations, min_iter)
            self.num_iter[i] = min(iterations, n_examples)
            if self.num_iter[i] != iterations:
                logging.info(
                    message.format("iterations", iterations, min_iter,
                                   "number samples", n_examples,
                                   self.num_iter[i]))
        self.num_iter.sort()
        self.num_iter = np.unique(np.array(self.num_iter, dtype=np.int64))
        self.num_iter = self.num_iter[self.num_iter > 0]
        logging.info("Number of iterations set to {}".format(self.num_iter))

        # k
        k = self.k
        min_k = 1
        min_class = int(min(class_counts))
        self.k = max(self.k, 1)
        self.k = min(min_class - 1, self.k)

        if self.k < 1:
            raise ValueError(
                "All classes should have size at least 2! (class sizes: {})".
                format(class_counts))

        if k != self.k:
            logging.info(
                message.format("neighbours", k, min_k, "smallest class size",
                               min_class, self.k))

        # mlc distance
        if self.mlc_distance not in MLCDistances.DISTANCES:
            raise ValueError(
                "MLC distance {} should be en element of {}".format(
                    self.mlc_distance, MLCDistances.DISTANCES))
        # average neighbour
        if self.task_type not in [
                TaskTypes.CLASSIFICATION, TaskTypes.REGRESSION
        ] and self.use_average_neighbour:
            self.use_average_neighbour = False
            logging.info(
                "Cannot use average neighbour for {}. Setting the parameter to False."
                .format(self.task_type))

    @staticmethod
    def compute_members(y):
        n_examples, n_classes = y.shape
        members = [[] for _ in range(n_classes)]
        example_to_class = [-1 for _ in y]
        for i_example in range(n_examples):
            for i_class in range(n_classes):
                if y[i_example, i_class] == 1:
                    members[i_class].append(i_example)
                    example_to_class[i_example] = i_class

        class_first_indices = [0]
        for i in range(n_classes):
            members[i] = np.array(members[i])
            class_first_indices.append(len(members[i]))

        class_first_indices = np.cumsum(
            class_first_indices)  # contains an additional element at the end
        return np.concatenate(members), np.array(
            example_to_class), class_first_indices

    @staticmethod
    def _compute_task_type(y):
        for t in y:
            if t.sum() != 1:
                return TaskTypes.MLC
        return TaskTypes.CLASSIFICATION

    @staticmethod
    def _transform_x_y(x, y):

        if isinstance(x, np.ndarray):
            x = sparse.csr_matrix(x)

        if x.getformat() != "csr":
            x = x.tocsr()

        if isinstance(y, list):
            y = np.array(y)

        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                # this means standard classification
                y = y.reshape(-1, 1)
                y = OneHotEncoder(categories="auto").fit_transform(y)

        if not isinstance(y, csr_matrix):
            y = csr_matrix(y)

        return x, y

    def send_message(self, message):
        if self.verbose:
            logging.info(message)

    def fit(self, x, y):
        """
        Key idea of ReliefE:
        embed the instance space. Compute mean embedding for each of the classes.
        compare, in the update step, to the joint label embedding instead of the single instance
        :param x: Feature space, array-like.
        :param y: Target space, a 0/1 array-like structure (1-hot encoded for classification).
        """

        if self.verbose:
            self.send_message("Dataset shape X: {} Y: {}".format(
                x.shape, y.shape))

        subsamples = min(self.samples, x.shape[0])

        if self.samples < x.shape[0]:
            self.send_message(
                "Subsampling to {} instances.".format(subsamples))

            unique_indice_maps = {}

            if y.ndim > 1:
                # MLC
                for j in range(y.shape[0]):
                    nzc = np.nonzero(y[j])[1]
                    nze = np.array2string(nzc)

                    if not nze in unique_indice_maps:
                        unique_indice_maps[nze] = [j]

                    else:
                        unique_indice_maps[nze].append(j)
            else:
                # CC
                unique_hashes = np.unique(y)
                for uh in unique_hashes:
                    unique_indice_maps[uh] = np.where(y == uh)[0].tolist()

            indices_sample = []
            counter = 0

            while counter < subsamples:
                for k, v in unique_indice_maps.items():
                    if len(v) > 0:
                        indices_sample.append(v.pop(0))
                        counter += 1
                    else:
                        continue

            indices_sample = np.array(indices_sample)
            x_sampled = x[indices_sample]
            y[indices_sample]

        else:
            x_sampled = x
            indices_sample = list(range(x.shape[0]))

        ts = time.time()

        # sparsify the input matrix some more
        sparsity_var = len(
            x_sampled.nonzero()[0]) / (x_sampled.shape[1] * x_sampled.shape[0])

        ts = time.time()

        if not isinstance(x_sampled, np.ndarray):
            x_sampled = x_sampled.todense()

        if sparsity_var > self.sparsity_threshold:

            if self.normalize:
                x_sampled = normalize(x_sampled)

            quad1 = x_sampled
            quad3 = x_sampled.T

            quad2 = np.zeros((quad1.shape[0], quad3.shape[1]))
            quad4 = np.zeros((quad3.shape[0], quad1.shape[1]))

            # (n+m)x(n+m) --> symmetry
            C = np.block([[quad2, quad1], [quad3, quad4]])
            var_sum = np.max(np.mean(C, axis=0))  # heuristic

            self.send_message("Initial sparsity: {}".format(sparsity_var))

            x_sampled = _sparsify_matrix_kernel(C, var_sum)
            x_sampled = x_sampled[0:quad1.shape[0], quad2.shape[1]:]
            x_sampled = sparse.csr_matrix(x_sampled)
            sparsity_var = len(x_sampled.nonzero()[0]) / (x_sampled.shape[1] *
                                                          x_sampled.shape[0])
            self.send_message("Final sparsity: {}".format(sparsity_var))

        ts2 = time.time()
        self.timed["sparsification"] = ts2 - ts

        # do some preprocessing
        x, y = ReliefE._transform_x_y(x, y)

        nrow_raw = x.shape[0]
        ncol_raw = x.shape[1]

        assert x.shape[0] == y.shape[0]
        n_examples = x.shape[0]

        self.task_type = ReliefE._compute_task_type(y)

        if self.task_type == TaskTypes.CLASSIFICATION:
            class_counts = np.array(np.sum(y, axis=0).tolist())[0]
            class_priors = np.array(class_counts) / n_examples
            class_members, examples_to_class, class_first_indices = ReliefE.compute_members(
                y)

        elif self.task_type == TaskTypes.MLC:
            class_counts = np.array(
                [n_examples])  # needed so that resetting parameters works
            class_priors, class_members, examples_to_class, class_first_indices = None, None, None, None

        else:
            raise ValueError("Unsupported task type.")

        # check the parameters
        self._compute_iterations(n_examples)
        self._check_parameters(n_examples, class_counts)

        self.send_message("Feature importance estimation start")

        ts2 = time.time()
        self.timed["initialization"] = ts2 - ts

        ts = time.time()
        if self.embedding_based_distances:

            # Project to low dim
            self.send_message("Estimating embedding from {}.".format(
                x_sampled.shape))
            latent_dim = min(self.determine_latent_dim(x_sampled), 8)
            reducer = umap.UMAP(n_components=latent_dim,
                                n_neighbors=self.k,
                                low_memory=True,
                                init="spectral")

            try:
                # very low-dim datasets can be problematic
                transf_um = reducer.fit(x_sampled)
                transf = transf_um.transform(x)
                x_embedded = sparse.csr_matrix(transf)

            except Exception as es:
                self.send_message(es)
                x_embedded = sparse.csr_matrix(x)

            if self.normalize:
                x_embedded = normalize(x_embedded)  # l2 norm

            self.send_message("Embedding obtained.")

        else:

            # generic Relief
            x_embedded = x

            if self.normalize:
                factors = (np.max(x_embedded, axis=0) -
                           np.min(x_embedded, axis=0)).todense()
                factors[np.abs(factors) ==
                        0] = 1.0  # make them effectively constant  < 10 ** -12
                factors = np.asarray(factors)[0]
                diags = sparse.diags(1 / factors)
                x_embedded = x_embedded.dot(diags)

        ts2 = time.time()
        self.timed["embedding"] = ts2 - ts

        ts = time.time()

        samples = np.random.RandomState(
            seed=42).permutation(n_examples)[:max(self.num_iter)]
        if self.embedding_based_distances:
            pairwise_distances = "cosine"
        else:
            # cdist(x_embedded[samples], x_embedded, metric="cityblock")
            pairwise_distances = "cityblock"

        self.send_message("Using {} distance for the Relief iteration.".format(
            pairwise_distances))

        # Run the Numba kernel
        data, pointers, indices = x_embedded.data, x_embedded.indptr, x_embedded.indices
        if self.task_type == TaskTypes.CLASSIFICATION:
            weights = _compiled_classification_update_weights(
                samples, self.num_iter, data, pointers, indices, class_priors,
                self.k, pairwise_distances, class_members, examples_to_class,
                class_first_indices, nrow_raw, ncol_raw,
                self.determine_k_automatically, self.use_average_neighbour)
        elif self.task_type == TaskTypes.MLC:

            latent_dim = self.determine_latent_dim(y)
            if self.mlc_distance == "cosine":

                if self.verbose:
                    self.send_message("Computing embedding of target space.")

                # compute embedding of target space.
                reducer = umap.UMAP(n_components=latent_dim,
                                    n_neighbors=self.k,
                                    low_memory=True,
                                    init="spectral")
                y = sparse.csr_matrix(
                    reducer.fit(y[indices_sample]).transform(y))

            elif self.mlc_distance == "hyperbolic":
                reducer = umap.UMAP(output_metric="hyperboloid",
                                    n_components=latent_dim,
                                    n_neighbors=self.k,
                                    low_memory=True,
                                    init="spectral")
                y = sparse.csr_matrix(
                    reducer.fit(y[indices_sample]).transform(y))

            data_y, pointers_y, indices_y = y.data, y.indptr, y.indices
            weights = _compiled_multi_label_classification_update_weights(
                samples, self.num_iter, data, pointers, indices, data_y,
                pointers_y, indices_y, self.k, pairwise_distances,
                self.mlc_distance, nrow_raw, ncol_raw)
        else:
            raise ValueError("Unsupported task type.")

        weights = np.nan_to_num(weights)
        for i, iterations in enumerate(self.num_iter):
            weights[i, :] = np.divide(weights[i, :], iterations)
        self.all_weights = weights

        weights = weights[len(self.num_iter) - 1]
        self.feature_importances_ = weights.reshape(-1)
        ts2 = time.time()
        self.timed["Feature weight updates"] = ts2 - ts

        self.send_message("Feature importance estimation ended.")
        return self
