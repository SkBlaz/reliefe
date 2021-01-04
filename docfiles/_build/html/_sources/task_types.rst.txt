Multiclass and multilabel classification
===============

ReliefE was adapted for various classification tasks. The key difference between doing ranking in a multiclass or multilabel setting is the `shape` of the output matrix. Having considered the multiclass example before, let's inspect how does the code differ in a multilabel case:

.. code:: python3

	  
	import reliefe
	import scipy.io as sio
	import numpy as np

	# Load the data first
	mat_obj = sio.loadmat("data/mlc/Science1.mat")
	x = mat_obj['input_space'] ## scipy csr sparse matrix (or numpy dense)
	y = mat_obj['target_space']  ## scipy csr sparse matrix (or numpy dense)

	print(y.shape) # 40 possible labels
	
	reliefE_instance = reliefe.ReliefE() # Initialize default ReliefE
	reliefE_instance.fit(x, y) # Compute rankings
	print(reliefE_instance.feature_importances_) # rankings for features (same order as x)


**There is no difference**. ReliefE automatically recognizes that as the shape of `y` is > 1, it needs to perform **multilabel** ranking.
