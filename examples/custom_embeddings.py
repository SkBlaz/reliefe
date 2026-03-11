import scipy.io as sio
import numpy as np
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reliefe

mat_obj = sio.loadmat(ROOT / "data" / "mcc" / "chess.mat")
x = mat_obj['input_space'][:128]
y = np.asarray(mat_obj['target_space']).reshape(-1)[:128]  ## this is not one hot for scc
print(x.shape, y.shape)

reliefe_instance = reliefe.ReliefE(embedding_based_distances=True,
                                   verbose=False,
                                   num_iter=24,
                                   samples=64)

# Simply provide a sklearn-like transform object
emb_custom = TruncatedSVD()  # Let's do SVD

# Provide it as the "embedding_method" parameter
reliefe_instance.fit(x, y, embedding_method=emb_custom)
assert len(reliefe_instance.feature_importances_) > 0
top = np.argsort(reliefe_instance.feature_importances_)[-5:][::-1]
print("Top features:", top.tolist())
