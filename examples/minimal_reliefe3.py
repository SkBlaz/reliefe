import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reliefe

x = np.array([[1.0, 0.0, 0.2, 0.0], [0.9, 0.1, 0.1, 0.0],
              [0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0], [0.0, 0.1, 0.9, 0.0],
              [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
y = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1],
              [1, 0, 1], [1, 1, 0], [0, 1, 1]])

# Multi-label example with embedded input distances and embedded target space.
reliefE_instance = reliefe.ReliefE(determine_k_automatically=False,
                                   embedding_based_distances=True,
                                   num_iter=8,
                                   k=1,
                                   mlc_distance="cosine",
                                   LD_friendly=False,
                                   samples=8,
                                   verbose=False)

reliefE_instance.fit(x, y, embedding_method="SVD")  # Compute rankings
print("Embedded input distances:",
      reliefE_instance.feature_importances_.round(4).tolist())

reliefE_instance = reliefe.ReliefE(determine_k_automatically=False,
                                   embedding_based_distances=False,
                                   num_iter=8,
                                   k=1,
                                   mlc_distance="cosine",
                                   samples=8,
                                   verbose=False)

reliefE_instance.fit(x, y, embedding_method="SVD")
print("Origin input distances:",
      reliefE_instance.feature_importances_.round(4).tolist())
