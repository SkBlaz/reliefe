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
                ## MLC
                for j in range(y.shape[0]):
                    nzc = np.nonzero(y[j])[1]
                    nze = np.array2string(nzc)

                    if not nze in unique_indice_maps:
                        unique_indice_maps[nze] = [j]

                    else:
                        unique_indice_maps[nze].append(j)
            else:
                ## CC
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

        ## sparsify the input matrix some more
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

            ## (n+m)x(n+m) --> symmetry
            C = np.block([[quad2, quad1], [quad3, quad4]])
            var_sum = np.max(np.mean(C, axis=0))  ## heuristic

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
                ## very low-dim datasets can be problematic
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
            pairwise_distances = "cityblock"  # cdist(x_embedded[samples], x_embedded, metric="cityblock")

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

                ## compute embedding of target space.
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
