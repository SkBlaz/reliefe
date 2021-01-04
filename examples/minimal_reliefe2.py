import scipy.io as sio
import numpy as np
import reliefe

mat_obj = sio.loadmat("../data/mcc/chess.mat")
x = mat_obj['input_space']
y = mat_obj['target_space']  ## this is not one hot for scc
y = np.array(y).reshape(-1)
wrange = []

for u in range(x.shape[0]):
    if 2**u <= x.shape[0]:
        wrange.append(int(2**u))
wrange.append(int(x.shape[0]))

# Fully fledged ReliefE (with all functionality)
reliefE_instance = reliefe.ReliefE(embedding_based_distances=True,
                                         verbose=True,
                                         use_average_neighbour=True,
                                         determine_k_automatically=True,
                                         num_iter=wrange)

reliefE_instance.fit(x, y)
print(reliefE_instance.feature_importances_)
