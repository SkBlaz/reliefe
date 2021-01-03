import reliefe
import scipy.io as sio
from utils import load_arff


def test_experiment1(x, y):

    mat_obj = sio.loadmat("../preprocessed_sparse_matrices/ohscal.wc.mat")
    x = mat_obj['input_space']
    y = mat_obj['target_space']  ## this is not one hot for scc

    if y.shape[1] >= 2 and "scipy.sparse" in type(y).__module__:
        y = np.array(np.argmax(y, axis=1)).reshape(-1)

    else:
        y = np.array(y).reshape(-1)

    wrange = []
    for u in range(x.shape[0]):
        if 2**u <= x.shape[0]:
            wrange.append(int(2**u))
    wrange.append(int(x.shape[0]))

    relief_b3_instance = reliefe.ReliefE(embedding_based_distances=True,
                                         verbose=True,
                                         use_average_neighbour=True,
                                         determine_k_automatically=True,
                                         num_iter=wrange)

    relief_b3_instance.fit(x, y)
    relief_b3_instance.feature_importances_
    all_weights = relief_b3_instance.all_weights
    print(all_weights)
    for k, v in relief_b3_instance.timed.items():
        print(k + "\t" + str(v))


def test_experiment2():
    ova = "../data/test.arff"
    features = [0, 1, 2, 3, 4]
    targets = [5, 6]
    x_train, y_train, _ = load_arff(ova, features, targets)
    neighbours = 2

    print("ReliefF ..")
    relief_b3_instance = reliefe.ReliefE(embedding_based_distances=False,
                                         verbose=True,
                                         k=neighbours)
    relief_b3_instance.fit(x_train, y_train)
    importances2 = relief_b3_instance.feature_importances_
    print(importances2)

    print("ReliefE ..")
    relief_b3_instance = reliefe.ReliefE(embedding_based_distances=True,
                                         verbose=True,
                                         k=neighbours,
                                         latent_dimension=2)
    relief_b3_instance.fit(x_train, y_train)
    importances2 = relief_b3_instance.feature_importances_
    print(importances2)

    x_train, y_train, _ = reliefe.utils.load_arff(ova, features, targets[0])
    neighbours = 2

    print("ReliefF ..")
    relief_b3_instance = reliefe.ReliefE(embedding_based_distances=False,
                                         verbose=True,
                                         k=neighbours)
    relief_b3_instance.fit(x_train, y_train)
    importances2 = relief_b3_instance.feature_importances_
    print(importances2)

    print("ReliefE ..")
    relief_b3_instance = reliefe.ReliefE(embedding_based_distances=True,
                                         verbose=True,
                                         k=neighbours,
                                         latent_dimension=2)
    relief_b3_instance.fit(x_train, y_train)
    importances2 = relief_b3_instance.feature_importances_
    print(importances2)
