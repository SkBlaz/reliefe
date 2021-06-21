# ReliefE: Feature Ranking via Manifold Embeddings

This repository hosts the implementation of the ReliefE idea, suitable both for _multiclass_ and _multilabel_ **feature ranking** problems.
The paper can be cited as:

```
@Article{Škrlj2021,
author={{\v{S}}krlj, Bla{\v{z}}
and D{\v{z}}eroski, Sa{\v{s}}o
and Lavra{\v{c}}, Nada
and Petkovi{\'{c}}, Matej},
title={ReliefE: feature ranking in high-dimensional spaces via manifold embeddings},
journal={Machine Learning},
year={2021},
month={Jun},
day={17},
abstract={Feature ranking has been widely adopted in machine learning applications such as high-throughput biology and social sciences. The approaches of the popular Relief family of algorithms assign importances to features by iteratively accounting for nearest relevant and irrelevant instances. Despite their high utility, these algorithms can be computationally expensive and not-well suited for high-dimensional sparse input spaces. In contrast, recent embedding-based methods learn compact, low-dimensional representations, potentially facilitating down-stream learning capabilities of conventional learners. This paper explores how the Relief branch of algorithms can be adapted to benefit from (Riemannian) manifold-based embeddings of instance and target spaces, where a given embedding's dimensionality is intrinsic to the dimensionality of the considered data set. The developed ReliefE algorithm is faster and can result in better feature rankings, as shown by our evaluation on 20 real-life data sets for multi-class and multi-label classification tasks. The utility of ReliefE for high-dimensional data sets is ensured by its implementation that utilizes sparse matrix algebraic operations. Finally, the relation of ReliefE to other ranking algorithms is studied via the Fuzzy Jaccard Index.},
issn={1573-0565},
doi={10.1007/s10994-021-05998-5},
url={https://doi.org/10.1007/s10994-021-05998-5}
}
```

![Scheme](images/scheme.png)

To get started with ReliefE, please consult the [documentation pages](https://skblaz.github.io/reliefe/)
