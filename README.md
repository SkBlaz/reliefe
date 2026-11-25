# ReliefE

Feature ranking for classification, multilabel prediction, regression, survival and unlabeled data.

![Feature ranking methods grouped by how they score a feature](images/feature-ranking-tree.svg)

## How to use

From the repository root, run every multiclass ranker on the same dataset. Use `--task multilabel` for binary label matrices.

```bash
uv run --extra rebate --extra all-relevant python examples/try_rankers.py --task multiclass --all
```

To rank columns in your own data and select the top five:

```python
from reliefe import FeatureRanker

ranker = FeatureRanker("relieff", top_k=5, target_strategy="auto").fit(X_train, y_train)
print(ranker.order_)  # column indices, best first
X_selected = ranker.transform(X_test)
```

## Methods

<details>
<summary>One feature at a time</summary>

- **Variance, correlation and rank tests** — `variance`, `numba-variance`, `pearson`, `numba-pearson`, `spearman`, `numba-spearman`, `kendall`, `kruskal`, `f`, `numba-f`, `chi2`, `numba-chi2`, `survival-concordance`
- **Information gain and split impurity** — `mi`, `numba-info-gain`, `numba-gain-ratio`, `numba-symmetrical-uncertainty`, `numba-gini`, `core-infgain`, `core-gainratio`, `core-mdl`, `core-gini`, `core-dkm`, `core-accuracy`, `core-mseofmean`, `core-myopicrelieff`, `core-dkmcost`, `core-gainratiocost`, `core-mdlsmp`, `core-uniformdkm`, `core-uniformgini`, `core-uniforminf`, `core-uniformaccuracy`, `core-equaldkm`, `core-equalgini`, `core-equalinf`
- **Distance between class distributions** — `core-impurityeuclid`, `core-impurityhellinger`, `core-equalhellinger`, `core-disthellinger`, `core-distauc`, `core-distangle`, `core-disteuclid`

</details>

<details>
<summary>Sample neighborhoods</summary>

- **Nearest hits and misses (Relief, ReliefF, RReliefF)** — `relief`, `relieff`, `rebate-relieff`, `core-relief`, `core-relieffequalk`, `relief-1994-a`, `relief-1994-b`, `relief-1994-c`, `relief-1994-d`, `relief-1994-e`, `relief-1994-f`, `core-relieffexprank`, `core-relieffdistance`, `core-relieffsqrdistance`, `core-relieffmerit`, `reliefmss`, `numba-reliefmss`, `mu-relief`, `numba-mu-relief`, `reliefseq`, `core-relieffbestk`, `core-rrelieffbestk`, `rrelieff`, `core-rrelieffequalk`, `core-rrelieffexprank`, `core-rrelieffdistance`, `core-rrelieffsqrdistance`, `core-rrelieffwithmse`, `core-relieffexpc`, `core-relieffavgc`, `core-relieffpe`, `core-relieffpa`, `core-relieffsmp`, `core-reliefkukar`
- **Global or local radius (SURF, SWRF, MultiSURF)** — `surf`, `numba-surf`, `surf-star`, `numba-surf-star`, `swrf`, `numba-swrf`, `swrf-star`, `numba-swrf-star`, `multisurf`, `numba-multisurf`, `multisurf-star`, `numba-multisurf-star`, `multiswrf`, `numba-multiswrf`, `multiswrf-star`, `numba-multiswrf-star`, `multiswrfdb`, `numba-multiswrfdb`, `multiswrfdb-star`, `numba-multiswrfdb-star`
- **Iterative weighting and feature pruning** — `irelief`, `numba-irelief`, `iterative-relief`, `numba-iterative-relief`, `boostedsurf`, `iter`, `turf`, `numba-turf`, `vls`, `numba-vls`, `ec-relieff`
- **Hit/miss variance (STIR)** — `stir-relieff`, `stir-surf`, `stir-multisurf`
- **Embedding distance (ReliefE)** — `reliefe`
- **Graph smoothness (Laplacian Score)** — `laplacian`

</details>

<details>
<summary>Conditional information</summary>

- **Relevance given prior selections (mRMR, JMI, CMIM, CIFE)** — `mrmr`, `jmi`, `cmim`, `cife`

</details>

<details>
<summary>Fitted models</summary>

- **Coefficients, tree importance and split error** — `linear`, `forest`, `estimator`, `core-mseofmodel`, `core-maeofmodel`
- **Elimination, permutation and shadow features** — `rfe`, `permutation`, `shadow-forest`, `boruta`

</details>

<details>
<summary>Ranks and baselines</summary>

- **Random order and per-target ranks** — `random`, `per-target`

</details>

## Citation

Škrlj et al. (2021). [ReliefE: feature ranking in high-dimensional spaces via manifold embeddings](https://doi.org/10.1007/s10994-021-05998-5). *Machine Learning*.

Original algorithms and implementations: [Relief](https://cdn.aaai.org/AAAI/1992/AAAI92-020.pdf), [Relief A–F](https://doi.org/10.1007/3-540-57868-4_57), [ReliefF and RReliefF](https://doi.org/10.1023/A:1025667309714), [ReliefSeq](https://pubmed.ncbi.nlm.nih.gov/24339943/), [ReBATE](https://github.com/UrbsLab/scikit-rebate), [CORElearn](https://cran.r-project.org/src/contrib/Archive/CORElearn/), [Relief.jl](https://github.com/jernejvivod/Relief.jl), [STIR](https://github.com/insilico/stir), [information criteria](https://jmlr.org/papers/v13/brown12a.html), [Laplacian Score](https://proceedings.neurips.cc/paper/2005/hash/b5b03f06271f8917685d14cea7c6c50a-Abstract.html), [BorutaPy](https://github.com/scikit-learn-contrib/boruta_py), [scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html), [SciPy](https://docs.scipy.org/doc/scipy/reference/stats.html).
