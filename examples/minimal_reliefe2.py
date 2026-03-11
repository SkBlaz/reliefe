import scipy.io as sio
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reliefe

mat_obj = sio.loadmat(ROOT / "data" / "mcc" / "chess.mat")
x = mat_obj['input_space'][:128]
y = np.asarray(mat_obj['target_space']).reshape(-1)[:128]  ## this is not one hot for scc

# Small multiclass example with embedded input distances.
reliefE_instance = reliefe.ReliefE(embedding_based_distances=True,
                                   verbose=False,
                                   use_average_neighbour=True,
                                   determine_k_automatically=True,
                                   num_iter=[24],
                                   samples=64)

reliefE_instance.fit(x, y, embedding_method="SVD")
top = np.argsort(reliefE_instance.feature_importances_)[-5:][::-1]
print("Top features:", top.tolist())
