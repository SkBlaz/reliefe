from numba import jit, prange
import numpy as np


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
