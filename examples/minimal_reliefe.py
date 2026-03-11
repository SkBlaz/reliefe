import scipy.io as sio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reliefe
from reliefe.utils import load_arff

features = [0, 1, 2, 3, 4]
targets = [5, 6]
x, y, _ = load_arff(ROOT / "data" / "test.arff", features, targets)

# Small multi-label example using the origin space distances.
reliefE_instance = reliefe.ReliefE(embedding_based_distances=False,
                                   verbose=False,
                                   use_average_neighbour=False,
                                   determine_k_automatically=False,
                                   mlc_distance="hamming",
                                   k=2,
                                   num_iter=8,
                                   samples=8)
reliefE_instance.fit(x, y)
print("Feature importances:", reliefE_instance.feature_importances_.round(4).tolist())
