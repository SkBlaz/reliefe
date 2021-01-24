Key idea behind ReliefE
===============

Feature ranking has been widely adopted in machine learning applications such as high-throughput biology and social sciences. The approaches of the popular Relief family of algorithms assign importances to features by iteratively accounting for nearest relevant and irrelevant instances. Despite high utility, these algorithms can be computationally expensive and not-well suited for high-dimensional sparse input spaces.

In contrast, the recent wave of embedding-based methods learn compact, low-dimensional representations, potentially facilitating the down-stream learning capabilities of conventional learners. This paper explores how the Relief branch of algorithms can be adapted to benefit from (Riemannian) manifold-based embeddings of instance and target spaces, where the embeddings dimensionality is  intrinsic to the dimensionality of the considered data set.

The developed ReliefE algorithm is shown to perform faster and to result in better feature rankings, when evaluated on 20 real-life data sets considering multiclass and multilabel classification tasks. The utility of ReliefE for high-dimensional data sets is ensured by its implementation utilizing sparse matrix algebraic operations.

* Embedding-based ranking by considering both instance and target embeddings
* Extension to multiclass and multilabel classification
* Efficient, Numba-based implementation
