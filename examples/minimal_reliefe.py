import reliefe
import numpy as np
import scipy.io as sio

mat_obj = sio.loadmat("../data/mlc/Health1.mat")
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
print(relief_b3_instance.feature_importances_)
