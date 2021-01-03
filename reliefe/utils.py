# helper methods
from typing import List, Callable, Union
import scipy.stats
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import re
from scipy.sparse import csr_matrix
import logging
import warnings
import glob

warnings.filterwarnings(
    "ignore", category=RuntimeWarning)  # aggregate_nom call causes it

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

import time


def measure_time(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        return result, te - ts


class TaskTypes:
    CLASSIFICATION = "classification"
    MLC = "multi-label classification"
    HMLC = "hierarchical multi-label classification"
    REGRESSION = "regression"


class MLCDistances:
    HAMMING = "hamming"
    F1 = "f1"
    ACCURACY = "accuracy"
    SUBSET = "subset"

    DISTANCES = [HAMMING, F1, ACCURACY, SUBSET, "cosine", "hyperbolic"]


def read_meta(dframe, specs="dataset_specifications.txt"):
    with open(specs) as df:
        for line in df:
            df_name, features, targets = line.strip().split(";")
            if df_name in dframe:
                feats = features.split(":")
                features = list(range(int(feats[0]), (int(feats[1]) + 1)))
                tars = targets.split(":")
                if len(tars) == 1:
                    targets = [int(x) for x in tars]
                else:
                    targets = list(range(int(tars[0]), (int(tars[1]) + 1)))
                return features, targets


def aggregate_num(values):
    return np.mean(values)


def aggregate_nom(values):
    return scipy.stats.mode(values)[0][0]


def impute_missing_values(column, possible_values, is_numeric,
                          missing_char_nom):
    """
    Replaces missing_char_nom for nominal and np.NaNs for numeric attributes with the modes and averages.
    This is done in-place.
    :param column: list of values
    :param possible_values: empty for numeric (and ignored), and list of possible values otherwise
    :param is_numeric: is the attribute numeric
    :param missing_char_nom: character that denotes missing nominal value
    :return: number of missing values
    """
    def is_missing(value):
        if is_numeric:
            return np.isnan(value)
        else:
            return v == missing_char_nom

    missing_no_yes = [[], []]
    for i, v in enumerate(column):
        position = 1 if is_missing(v) else 0
        missing_no_yes[position].append(i)
    if missing_no_yes[1]:
        non_missing_values = [column[j] for j in missing_no_yes[0]]
        if is_numeric:
            if missing_no_yes[0]:
                imputed = aggregate_num(non_missing_values)
            else:
                imputed = 0.0  # does not matter what
        else:
            if missing_no_yes[0]:
                imputed = aggregate_nom(non_missing_values)
            else:
                imputed = possible_values[0]  # does not matter what
        for i in missing_no_yes[1]:
            column[i] = imputed
    return len(missing_no_yes[1])


def basic_parse_sparse_line(line, offset=0):
    if not (line[0] == '{' and line[-1] == '}'):
        raise ValueError("Sparse lines should start with { and end with }")
    line = line[1:-1].strip()
    if not line:
        return {}
    else:
        pairs = [[c.strip() for c in re.split('\\s+', pair)]
                 for pair in line.split(',')]
        return {int(position) - offset: value for position, value in pairs}


def is_one_based_sparse_arff(path_to_data, n_attributes):
    """
    Determines if the data is given in sparse format. If it is, determines whether the attribute indices
    are 0- or 1-based
    :param path_to_data:
    :param n_attributes:
    :return: (True/False, 0/1) The second component is not used if the arff is not sparse, i.e., if the first component
    (equals True).
    """
    data_tag = "@data"
    f = open(path_to_data)
    for line in f:
        if line.lower().startswith(data_tag):
            break
    is_sparse = None
    for line in f:
        if len(line) <= 1:
            continue
        line = line.strip()
        if is_sparse is None:
            is_sparse = line[0] == '{' and line[-1] == '}'
        if not is_sparse:
            return False, 0
        parsed = basic_parse_sparse_line(line)
        if 0 in parsed:
            return True, 0
        elif n_attributes in parsed:
            return True, 1
    f.close()
    raise ValueError("Neither 0 nor", n_attributes,
                     "(n_attributes) found. Minimal possible index unknown.")


def get_task_type(target_attributes_values: List[List[str]]):
    """
    Finds the task type from the possible attribute values.
    :param target_attributes_values:
    :return:
    """
    mlc_values = ["0", "1"]
    if len(target_attributes_values) == 1:
        if target_attributes_values[0]:
            return TaskTypes.CLASSIFICATION
        else:
            return TaskTypes.REGRESSION
    for attribute_values in target_attributes_values:
        if attribute_values != mlc_values:
            raise ValueError("Unrecognizable task type")
    return TaskTypes.MLC


def load_arff(path_to_data,
              descriptive_indices: List[int],
              target_indices: Union[int, List[int]],
              missing_character='?',
              impute_missing=True):
    """
    Loads arff in sparse or dense form. Nominal attributes (descriptive or target) are 1-hot encoded.

    Due to some inconsistencies in use of quotation marks, of data-creators, single and double quotation marks
    are simply removed prior to any processing.

    :param path_to_data: path to arff
    :param descriptive_indices: list of 0-based indices of descriptive attributes
    :param target_indices: analogue of descriptive indices. Can be a single number.
    :param missing_character: character that denotes missing value in the arff
    :param impute_missing: should missing values be imputed? Is so, simple per-column means/modes are computed.
    Otherwise, missing numeric values are converted to np.NaNs.
    :return: descriptive matrix (dense), target matrix (sparse), attribute_ranges
    The ranges are given as a dictionary {index of original descriptive attribute: (start, end), ...},
    where columns with indices in range(start, end) in descriptive matrix belong to the descriptive attribute.
    For example, if the attribute was originally numeric, then start = end - 1.
    """

    logging.info("Reading {}".format(path_to_data))
    numeric = "numeric"
    numeric_types = [numeric, "real", "integer"]  # define them in lower case
    nominal = "nominal"
    data_tag = "@data"
    attribute_tag = "@attribute"
    attribute_line = "{}\\s+(\\S+)\\s+(.+)".format(attribute_tag)
    if isinstance(target_indices, int):
        target_indices = [target_indices]
    assert isinstance(target_indices, list)
    intersection = set(target_indices) & set(descriptive_indices)
    if intersection:
        raise ValueError(
            "The intersection of descriptive and target indices must be empty, "
            "but is {}".format(intersection))
    # read meta data
    meta = []
    f = open(path_to_data)
    for line in f:
        line = re.sub("[\"']", "", line)
        if line.lower().startswith(data_tag):
            break
        elif line.lower().startswith(attribute_tag):
            match = re.search(attribute_line, line, flags=re.IGNORECASE)
            if match is None:
                raise ValueError("Could not parse the line {}".format(line))
            name = match.group(1)
            values = match.group(2)
            is_numeric = values.lower() in numeric_types
            is_nominal = values[0] == '{' and values[-1] == '}'
            is_kept = len(meta) in descriptive_indices or len(
                meta) in target_indices
            if not (is_numeric or is_nominal) and is_kept:
                raise ValueError(
                    "Exclude the attribute {}: "
                    "its type was not recognized as nominal or numeric. "
                    "If necessary, extend the numeric types {}.".format(
                        line, numeric_types))
            if is_numeric:
                meta.append((name, numeric, []))
            elif is_nominal:
                parsed_values = sorted(
                    [v.strip() for v in values[1:-1].split(',')])
                meta.append((name, nominal, parsed_values))
            else:
                raise NotImplementedError("The load_arff code needs updates.")
    maximal_index = max(max(descriptive_indices), max(target_indices))
    n_attributes = len(meta)
    if maximal_index >= len(meta):
        raise ValueError(
            "We must have every specified attribute index < {} (number of attributes), "
            "but one of your indices is {}.".format(n_attributes,
                                                    maximal_index))
    indices_both = [descriptive_indices, target_indices]
    are_numeric = [[meta[i][1] == numeric for i in indices]
                   for indices in indices_both]
    possible_values_projected = [[meta[i][-1] for i in indices]
                                 for indices in indices_both]
    # determine sparsity and task type
    is_sparse, sparse_offset = is_one_based_sparse_arff(
        path_to_data, n_attributes)
    task_type = get_task_type([meta[i][-1] for i in target_indices])
    if task_type not in [TaskTypes.CLASSIFICATION, TaskTypes.MLC]:
        raise NotImplementedError(
            "Processing of {} data not implemented.".format(task_type))
    # read data
    data = [[], []]  # [descriptive, target]
    for line in f:
        line = re.sub("[\"']", "", line).strip()
        if not line:
            continue
        if is_sparse:
            values_basic = basic_parse_sparse_line(line, sparse_offset)
        else:
            values_basic = [v.strip() for v in line.split(',')]
        for indices, numeric, data_part in zip(indices_both, are_numeric,
                                               data):
            line = []
            for i, is_numeric in zip(indices, numeric):
                if is_sparse:
                    if is_numeric:
                        v = values_basic[i] if i in values_basic else "0.0"
                    elif i in values_basic:
                        v = values_basic[i]
                    else:
                        # raise KeyError("Value for the non-numeric attribute {} missing.".format(meta[i]))
                        if "0" not in meta[i][-1]:
                            raise ValueError(
                                "Values of the sparse attribute {} must be given explicitly, "
                                "since 0 is not among its possible values {}.".
                                format(*meta[i][0::2]))
                        v = "0"
                else:
                    v = values_basic[i]
                if is_numeric:
                    if v != missing_character:
                        v = float(v)
                    else:
                        v = np.NaN
                line.append(v)
            data_part.append(line)
    f.close()
    logging.info("Read. Processing the data.")
    # transpose
    for i in range(len(data)):
        data[i] = [list(column) for column in zip(*data[i])]
    assert len(data[0]) == len(descriptive_indices)
    assert len(data[1]) == len(target_indices)
    if impute_missing:
        n_imputed = 0
        for columns, numeric, possible_values in zip(
                data, are_numeric, possible_values_projected):
            for column, is_numeric, values in zip(columns, numeric,
                                                  possible_values):
                n_imputed += impute_missing_values(column, values, is_numeric,
                                                   missing_character)
        logging.info("Values imputed: {}".format(n_imputed))
    # sparse for target
    if task_type == TaskTypes.CLASSIFICATION:
        possible_values = possible_values_projected[1][0]

        encoder = OneHotEncoder(categories=[possible_values],
                                sparse=True,
                                dtype=np.int8)
        target_data = csr_matrix(encoder.fit_transform([[v]
                                                        for v in data[1][0]]),
                                 dtype=np.int8)
    elif task_type == TaskTypes.MLC:
        target_data = [[
            int(v) if v != missing_character else np.NaN for v in column
        ] for column in zip(*data[1])]
        target_data = csr_matrix(target_data, dtype=np.int8)
    else:
        raise ValueError("Wrong task")
    # 1-hot for descriptive
    attribute_ranges = {
    }  # {index of descriptive: (start, end), ...}, end exclusive
    columns = data[0]
    descriptive_data = []
    for j, column in enumerate(columns):
        if not are_numeric[0][j]:
            possible_values = possible_values_projected[0][j]
            encoder = OneHotEncoder(categories=[possible_values], sparse=False)
            transformed = encoder.fit_transform([[v]
                                                 for v in column]).T.tolist()
            attribute_ranges[j] = (len(descriptive_data),
                                   len(descriptive_data) + len(transformed))
            descriptive_data += transformed
        else:
            attribute_ranges[j] = (len(descriptive_data),
                                   len(descriptive_data) + 1)
            descriptive_data.append(column)
    logging.info("Finished processing the data.")
    return np.array(descriptive_data).T, target_data, attribute_ranges


def show_arff_attributes(path_to_data, k=5):
    attributes = []
    with open(path_to_data) as f:
        for line in f:
            if line.lower().startswith("@attribute"):
                attributes.append(line.strip())
            elif line.lower().startswith("@data"):
                break
    print("Found {} attributes of {}. Showing first and last {}:".format(
        len(attributes), path_to_data, k))
    first = str(attributes[:k])[1:-1]
    last = str(attributes[-k:])[1:-1]
    print("attributes = [{}, ..., {}]".format(first, last))


def feature_ranking_wrapper(path_to_arff: str,
                            descriptive_indices: Union[None, List[int]],
                            target_index: Union[None, int],
                            feature_ranking: Callable, *args):
    """
    Takes care of the datasets that contain nominal attributes:
    - loads the data from arff file and 1-hot encode the nominal features
    - computed the relevance of the transformed features
    - compute the relevance of original features by summing up the relevance of the 1-hot encoded groups

    E.g., if the data has two features: x1 which is numeric and x2 which can take values A, B, and C, the pipeline
    is like this:
    - load data which has now 4 features: x1, x2-A, x2-B, x2-C (e.g., if originally, the first example is 2.3,B,
      the converted example is 2.3,0,1,0.)
    - compute relevance for the transformed features, e.g., [0.1, 0.2, 0.3, 0.05]
    - compute relevance for the original features: [0.1, 0.55], since 0.2 + 0.3 + 0.05 = 0.55.
    """

    xs, y, attribute_ranges = load_arff(path_to_arff, descriptive_indices,
                                        target_index)
    scores = feature_ranking(xs, y, *args)
    if descriptive_indices is None:
        descriptive_indices = list(range(len(scores)))
    assert isinstance(descriptive_indices, list)
    # sum up the one-hot encoded scores
    feature_importance = {i: 0 for i in descriptive_indices}
    for i in descriptive_indices:
        feature_importance[i] = sum(
            [scores[j] for j in range(*attribute_ranges[i - 1])])
    return feature_importance


def test_stuff(test_first=False, test_second=False):
    if test_first:
        ova = "../data/OVA.arff"
        features = list(range(10936))
        targets = list(range(10936, 10945))
        load_arff(ova, features, targets)
    if test_second:
        test_arff = "../data/test_sparse.arff"
        features = [0, 1, 2, 3, 4]
        targets = [5, 6]
        x, y, r = load_arff(test_arff, features, targets)
        print(x)
        print(y)
        print(r)


if __name__ == "__main__":
    for filex in glob.glob("../data/cc/*"):
        # features, targets = read_meta(filex, specs="dataset_specifications.txt")
        show_arff_attributes(filex, 3)
        print()
