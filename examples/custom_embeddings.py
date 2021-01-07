import reliefe
import scipy.io as sio
from sklearn.decomposition import TruncatedSVD

mat_obj = sio.loadmat("../data/mcc/chess.mat")
x = mat_obj['input_space']
y = mat_obj['target_space']  ## this is not one hot for scc
y = y.reshape(-1)
print(x.shape, y.shape)

reliefe_instance = reliefe.ReliefE(embedding_based_distances=True,
                                   verbose=True)

# Simply provide a sklearn-like transform object
emb_custom = TruncatedSVD()  # Let's do SVD

# Provide it as the "embedding_method" parameter
reliefe_instance.fit(x, y, embedding_method=emb_custom)
assert len(reliefe_instance.feature_importances_) > 0
print(reliefe_instance.feature_importances_)
