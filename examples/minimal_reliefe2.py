import scipy.io as sio
import numpy as np
import reliefe

mat_obj = sio.loadmat("../data/mcc/chess.mat")
x = mat_obj['input_space']
y = mat_obj['target_space']  ## this is not one hot for scc
y = np.array(y).reshape(-1)
wrange = []

# Fully fledged ReliefE (with all functionality)
reliefE_instance = reliefe.ReliefE(embedding_based_distances=True,
                                   verbose=True,
                                   use_average_neighbour=True,
                                   determine_k_automatically=True,
                                   num_iter=[100])

reliefE_instance.fit(x, y)
print(reliefE_instance.feature_importances_)
