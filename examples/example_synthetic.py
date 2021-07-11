## A simple example on the quadrants data set

import tqdm
import pandas as pd ## pip install pandas
import numpy as np
import scipy.io as sio
import reliefe

def test_importances(importances, method = "ReliefE"):
    sorted_importances = np.argsort(importances)[::-1]

    ## If the first two features are amongst the top 3, this is a relatively OK sign.
    if 0 in sorted_importances[0:3] and 1 in sorted_importances[0:3]:
        print("Successfully retrieved top features.")
        return 1, method
    else:
        return 0, method

df = pd.read_csv("synthetic.txt",sep = ",")
print(df)


## Features a1 and a2 are here crucial for performance.
x = df[[f"a{x}" for x in range(1,9)]].values
y = df.Class.astype(int).values

## ReliefE will revert to the core version in LD settings; the point of reducing the dimension makes no sense in LD; the core version is fast for this type of problems.
reliefE_instance = reliefe.ReliefE(embedding_based_distances=True,
                                   num_iter = 0.05, ## Few-shot?
                                   verbose=True,
                                   use_average_neighbour=True,
                                   determine_k_automatically=True)

reliefE_instance.fit(x, y)
importances = reliefE_instance.feature_importances_
test_importances(importances, method = f"ReliefE")
