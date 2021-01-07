import scipy.io as sio
import reliefe

mat_obj = sio.loadmat("../data/mlc/medical.mat")
x = mat_obj['input_space']
y = mat_obj['target_space']  ## this is not one hot for scc
wrange = []

# Fully fledged MLC - ReliefE (with all functionality)
reliefE_instance = reliefe.ReliefE(embedding_based_distances=False,
                                   verbose=True,
                                   use_average_neighbour=False,
                                   determine_k_automatically=False,
                                   mlc_distance="hamming")
reliefE_instance.fit(x, y)
print(reliefE_instance.feature_importances_)
