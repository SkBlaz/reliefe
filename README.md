# ReliefE: Feature Ranking via Manifold Embeddings
Welcome to the official implementation sites of ReliefE.
This repository hosts the ReliefE idea, suitable both for _multiclass_ and _multilabel_ classification problems.

![Scheme](images/scheme.png)
## Getting started
To install stable ReliefE, try:
```
pip install reliefe
```

To install the bleeding-edge repo version, you can also
```
pip install git+https://github.com/SkBlaz/reliefe
```

Required dependencies can be installed via
```
pip install -r requirements.txt
```

## Tests
An important check after each installation are the *tests*. To verify ReliefE works OK on your hardware, please go to "tests" folder, and input:
```
py.test (if using pytest-sugar)
python -m pytest * (otherwise)
```

## Toy examples
ReliefE was build in sklearn-like syntax (as closely as possible), and is hence relatively simple to use. To instantiate a ranking class, first

```
import reliefe
```
This call imports the reliefe namespace with all the required libraries.
To perform ranking, let's consider the following example:


```
import scipy.io as sio
import numpy as np

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

```
Returns:
```
[ 9.89309972e-003  1.20427974e-002  1.34322081e-002  9.30750360e-003
  4.11818629e-003  1.60451980e-002  3.65662586e-003  4.95127759e-003
 -1.21291185e+304 -1.21724878e+304 -5.00084339e+302 -1.19073976e+293
  3.64385300e+226 -3.60334541e+266 -3.85113882e+293  1.38678002e-003
 -1.21291185e+304 -1.21724878e+304 -1.12834143e+303 -2.91441558e+302
 -3.06704357e+247  1.81048784e+303  1.65435470e+303  1.65414412e+303
 -1.21291185e+304 -1.21724542e+304  0.00000000e+000 -2.91441558e+302
 -1.57470695e+298  1.31441274e+302 -2.39764884e+301 -1.83316404e+302
 -1.21290833e+304 -1.21724481e+304  0.00000000e+000 -2.91441558e+302
 -3.15257597e+298 -2.46423553e+301]
```