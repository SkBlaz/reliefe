ReliefE Hyperparameters
===============

ReliefE has multiple hyperparameters, which determine its performance. Let's discuss the main ones next.

.. code:: python3

	  
	import reliefe
	
	reliefE_instance = reliefe.ReliefE(embedding_based_distances=True,
	    verbose=True,
	    use_average_neighbour=True,
	    determine_k_automatically=True,
	    num_iter=[128])
	

This code snipped initialized ReliefE with all the functionality described in the paper. The parameters are:

.. list-table:: Hyperparameter descriptions. Below you can find key hyperparameters.
   :widths: 20 60 20
   :header-rows: 1

   * - Hyperparameter
     - Description
     - Values
   * - embedding_based_distances
     - Whether to embed the input and output space if possible (ReliefE). Instances are compared via *cosine* distance, but the distance for targets needs to be specified (see below).
     - True/False
   * - use_average_neighbour
     - Whether to average the neighbors' embeddings during computation
     - True/False
   * - determine_k_automatically
     - Whether to determine the size of neighborhood
     - True/False
   * - num_iter
     - Number of iterations
     - Integer
   * - normalize_descriptive
     - Normalization of descriptive attributes?
     - True/False
   * - latent_dimension
     - Embedding dimension
     - Integer
   * - mlc_distance
     - Distance used for comparison in MLC setting
     - ["f1","cosine","hyperbolic","hamming","accuracy","subset"]
   * - sparsity_threshold
     - If number of non-zero elements is larger than this, sparsify.
     - float between 0 and 1
   * - samples
     - Number of samples if the number of instances is too large.
     - Integer
More detailed descriptions can be found in the method description pages in
:ref:`genindex`.
