Getting started
===============
The key idea behind ReliefE is simplicity. The purpose of this library is to offer off-the-shelf
functionality not supported elsewhere, with minimal user effort.
The data used in the example is accessible at: https://github.com/SkBlaz/reliefe/tree/master/data

The minimal example is given next.

.. code:: python3

	  
	import reliefe
	import scipy.io as sio
	import numpy as np

	# Load the data first
	mat_obj = sio.loadmat("data/mcc/chess.mat")
	x = mat_obj['input_space'] ## scipy csr sparse matrix (or numpy dense)
	y = mat_obj['target_space']  ## scipy csr sparse matrix (or numpy dense)

	# Fully fledged ReliefE (with all functionality)
	reliefE_instance = reliefe.ReliefE() # Initialize default ReliefE
	reliefE_instance.fit(x, y) # Compute rankings
	print(reliefE_instance.feature_importances_) # rankings for features (same order as x)


Returns the following ranking:

.. code:: python3

	  
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

