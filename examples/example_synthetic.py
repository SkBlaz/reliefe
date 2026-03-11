## A simple example on the quadrants data set

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reliefe

def test_importances(importances, method="ReliefE"):
    sorted_importances = np.argsort(importances)[::-1]

    ## If the first two features are amongst the top 3, this is a relatively OK sign.
    if 0 in sorted_importances[0:3] and 1 in sorted_importances[0:3]:
        print("Successfully retrieved top features.")
        print("Top three:", sorted_importances[:3].tolist())
        return 1, method
    else:
        return 0, method

data = np.genfromtxt(ROOT / "examples" / "synthetic.txt",
                     delimiter=",",
                     names=True,
                     dtype=None,
                     encoding="utf-8")
x = np.column_stack([data[f"a{i}"] for i in range(1, 9)])
y = np.array([1 if value.strip('"') == "TRUE" else 0 for value in data["Class"]])

## ReliefE will revert to the core version in LD settings; the point of reducing the dimension makes no sense in LD; the core version is fast for this type of problems.
reliefE_instance = reliefe.ReliefE(embedding_based_distances=True,
                                   num_iter=64,
                                   verbose=False,
                                   use_average_neighbour=True,
                                   determine_k_automatically=True)

reliefE_instance.fit(x, y)
importances = reliefE_instance.feature_importances_
test_importances(importances, method="ReliefE")
