Using custom embeddings
===============
UMAP is only one of many possible embedding techniques that can be exploited by ReliefE. The following example shows how SVD-based representations can be used in the very same manner as e.g., the UMAP-based ones.

.. code:: python3

	  
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
    emb_custom = TruncatedSVD() # Let's do SVD

    # Provide it as the "embedding_method" parameter
    reliefe_instance.fit(x, y, embedding_method = emb_custom)
    print(reliefe_instance.feature_importances_)

 
Returns the following ranking:

.. code:: python3

	  
    [ 3.27528470e-04  8.12616155e-04  3.15436806e-04  0.00000000e+00
      0.00000000e+00  0.00000000e+00  2.39045842e-04 -5.65957574e-04
      1.93557981e-04 -4.57150539e-04  1.47203139e-04 -3.43643779e-04
      9.86991482e-05 -2.30556572e-04  4.88615029e-05 -1.14931228e-04
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00]
