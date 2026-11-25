# ReliefE

ReliefE is a Numba-accelerated feature ranking library for multiclass and multi-label classification. It extends the Relief family with optional manifold-based embeddings for the input and target spaces.

## Install

```bash
uv sync
uv sync --group dev
```

This is a `uv`-first repository. Use the base sync for the library, and add the `dev` group when you want the test toolchain.

## Run

```bash
uv run python examples/minimal_reliefe2.py
uv run python examples/minimal_reliefe.py
uv run python examples/custom_embeddings.py
```

## Test

```bash
make test
```

## Layout

- `reliefe/`: core algorithm and utility code
- `examples/`: runnable smoke-sized examples
- `data/`: bundled datasets used by examples and tests
- `tests/`: standard-library smoke tests
- `docs/`: published documentation assets

## Notes

- `ReliefE.fit(..., embedding_method="auto")` prefers UMAP and falls back to SVD when UMAP is unavailable.
- The bundled examples are intentionally small enough to be used as smoke tests.

## Reference

Paper: Škrlj et al., "ReliefE: feature ranking in high-dimensional spaces via manifold embeddings", Machine Learning (2021).

Docs: https://skblaz.github.io/reliefe/

## Rank a small table

```python
from reliefe import rank_features

result = rank_features(
    x=[[0., 0., 1.], [0., 1., 1.], [1., 0., 0.], [1., 1., 0.]],
    y=["control", "control", "case", "case"],
    feature_names=["signal", "noise", "inverse_signal"],
    top_k=2,
)
print(result["ranking"])
# [{'index': 0, 'score': 1.0, 'name': 'signal'},
#  {'index': 2, 'score': 1.0, 'name': 'inverse_signal'}]
```

This convenience function and the MCP tool share the same implementation. They
use direct Relief distances with column range scaling, fixed sampling, and no
embedding randomness. Supply finite numeric features and integer or string class
labels. There must be at least two classes with at least two examples each;
`k` defaults to 1 and must be smaller than the smallest class size. `num_iter`
defaults to 128 and is capped at the number of rows. Ties retain column order.
Scores are signed relevance estimates, not probabilities or causal effects.
Fit feature selection on training folds only to avoid data leakage.

The tool accepts at most 512 rows, 1024 features, 131072 cells, and 512 iterations.
`top_k` limits the returned ranking; `scores` always contains every feature score
in original column order. The result also records dimensions, effective `k` and
iteration count, method, and normalization. Impute missing values and encode
categorical features first. For larger or sparse inputs, manifold embeddings,
or multi-label targets, use the existing `ReliefE` estimator directly.

## Use as an MCP tool

The optional MCP server requires Python 3.10+ and uses the official MCP Python
SDK's v1 line (`mcp<2`). The base library retains Python 3.8+ support and does not
require the SDK. From a checkout:

```bash
uv sync --extra mcp
uv run --extra mcp reliefe-mcp
# Alternatively: uv run --extra mcp python -m reliefe.mcp_server
```

For pip installations, use `pip install '.[mcp]'` from the checkout, then run
`reliefe-mcp`. The extra will also be available as `reliefe[mcp]` when this change
is released. The server uses stdio, so launch it from an MCP client; it waits for
protocol messages when started in a terminal.

Add this to your MCP client's server configuration, replacing the path with the
absolute checkout location (it works independently of the client's directory):

```json
{
  "mcpServers": {
    "reliefe": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/reliefe",
               "--extra", "mcp", "reliefe-mcp"]
    }
  }
}
```

The server exposes one tool, `rank_features`, with the arguments and result
shown above. Its schemas describe input bounds and structured results. Invalid
calls return MCP tool errors; the server remains available for the next call.
It computes locally on the supplied table without reading files, writing files,
or opening a network listener. Library logs use stderr; stdout is reserved for
MCP messages. The first compiled call can take longer while Numba initializes.

`make test` now checks 100% statement coverage, a real stdio MCP exchange, and
separate numerical tests with Numba JIT enabled. On Python 3.8/3.9 the optional
server tests are skipped; use Python 3.10+ for the full coverage gate.

### Changes to native ReliefE rankings

Embedded fits use the embedding to find neighbors and attribute their feature
differences to original input columns. Earlier versions could put a latent
coordinate's score under an unrelated input column. Multilabel checkpoints now
retain accumulated weights across skipped updates, and subset distance is zero
for equal label sets and one otherwise. The sparsifier operates on the sampled
rectangular CSR matrix. Use `random_state` to control sampling and built-in
embeddings. Inspect `effective_embedding_`, `k_` and `num_iter_` after fitting.

## Rank and select features in Python

`FeatureRanker` fits on a numeric table, returns a full ranking in input-feature
coordinates, and selects columns in their original order. It works as a step
in a scikit-learn pipeline. Use an explicit regression task for integer-valued
responses. Fit the entire pipeline inside each training fold.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from reliefe import FeatureRanker, list_methods

X, y = load_breast_cancer(return_X_y=True)
pipeline = make_pipeline(FeatureRanker("relieff", top_k=5), StandardScaler(),
                         LogisticRegression(max_iter=1000))
fold_scores = cross_val_score(pipeline, X, y, cv=3)
pipeline.fit(X, y)
print(pipeline[0].ranking()[:5])
print(list_methods()["rrelieff"])
```

| Method | Targets | Inputs | Meaning |
| --- | --- | --- | --- |
| `relief` | Binary | Dense | Original range-normalized, squared-difference update using every query |
| `relieff`, `reliefe` | Classification, plus multilabel for ReliefE | Dense, sparse | Signed feature weights, with direct or embedding-based neighborhoods |
| `rrelieff` | Regression | Dense | Rank-weighted conditional difference in feature changes |
| `variance`, `random` | Unsupervised | Dense, sparse | Population variance or seeded random baseline |
| `f`, `mi` | Classification, regression | Dense, sparse | F statistic or mutual information. For sparse MI sklearn expects discrete predictors |
| `chi2` | Classification | Nonnegative dense or sparse | Chi-square statistic |
| `mrmr` | Classification | Dense | Greedy relevance-minus-redundancy selection order with quantile discretization |
| `linear`, `forest` | Classification, regression | Dense, sparse | Absolute standardized coefficients averaged across classes, or impurity decrease |
| `rfe` | Classification, regression | Dense, sparse | Elimination rank from a cloned estimator |
| `permutation` | Classification, regression | Dense | Score decrease on an internal validation split. Inspect `validation_indices_`, `estimator_`, `scoring_` and `permutation_std_` |
| ReBATE methods below | Classification, regression | Dense | Upstream feature weights, including wrapper elimination tiers for TuRF |

Install the optional **ReBATE** integration on Python 3.9+ with
`pip install 'reliefe[rebate]'` (from a checkout, `uv sync --extra rebate`).
The catalog includes `rebate-relieff`, `surf`, `surf-star`, `multisurf`,
`multisurf-star`, `swrf`, `swrf-star`, `multiswrf`, `multiswrf-star`,
`multiswrfdb`, `multiswrfdb-star`, `mu-relief`, `turf`, `vls`, and `iter`.
Set backend options with `method_params`. For wrappers, `base_method` selects
a ReBATE core method and `base_params` configures it. The Python ranking
interface accepts finite encoded numeric features. ReBATE distance methods
can require quadratic memory in sample count. See the
[upstream implementations](https://github.com/UrbsLab/scikit-rebate).

`scores_` preserves native units in original column order. Ordinal methods
expose `scores_=None`. `ranks_` starts at one and permits ties. Equal values
retain column order at the top-k boundary. Undefined univariate statistics sort
last. Labeled tables retain their original names. Run
`uv run python examples/minimal_benchmark.py` for fold-local comparisons.

### CORElearn methods

An optional local R backend exposes all 37 classification and nine regression
attribute evaluators in CORElearn 1.56.0. This includes equal-k, exponential
rank, best-k, distance and squared-distance neighborhoods, merit and cost
sensitive ReliefF variants, myopic ReliefF, Kukar's Relief, RReliefF with MSE,
information gain, gain ratio, MDL, Gini, Hellinger and other split criteria.
Use `[m for m in list_methods() if m.startswith("core-")]` for exact identifiers.
Method names refer to implementations. Several backends implement the same
published algorithm.

Install R with `Rscript` on PATH and install CORElearn in that R environment.
The version used by CI is available in the [CRAN archive](https://cran.r-project.org/src/contrib/Archive/CORElearn/).
For example, in R:

```r
install.packages(c("cluster", "nnet", "plotrix", "rpart.plot"))
install.packages("https://cran.r-project.org/src/contrib/Archive/CORElearn/CORElearn_1.56.0.tar.gz",
                 repos = NULL, type = "source")
```

```python
ranker = FeatureRanker("core-relieffexprank", method_params={"kNearestExpRank": 5})
ranker.fit(X, y)
```

The bridge launches a local R process per fit and exchanges finite dense numeric
arrays in a temporary directory. Declare encoded categorical columns with
`categorical_features=[0, 3]`, supply a misclassification `cost_matrix` in
sorted class order, or pass named scalar CORElearn options via `method_params`.

### Comparing rankings and repeated measurements

* `reliefseq` takes the largest direct ReliefF score for each feature across
  `k_min ... k_max`. Inspect `best_k_` to see the winning neighborhood size.
* `laplacian` ranks unsupervised numeric features by their normalized graph
  energy on a symmetric heat-weighted k-neighbor graph. Lower scores rank first.
* `per-target` fits a base ranker to each response and averages tied ranks.
  Use `base_method` and `base_task` to configure multilabel or multi-output
  regression baselines. Constant targets contribute a neutral tied rank.
* `shadow-forest` estimates how often a variable beats the best independently
  shuffled shadow variable over repeated forest fits. Its scores are empirical
  frequencies. They carry no multiple-testing or false-discovery guarantee.

`compare_rankings(X, y, rankers, estimator, splits, scoring=...)` fits each
configuration separately on each training fold. It reports paired fold scores,
per-feature selection frequencies, and mean pairwise Jaccard overlap. Supply a
mapping from labels to rankers (use None for all columns) and precomputed splits
for group or time-series evaluation. Choose hyperparameters inside training folds.

Use `grouped_permutation_importance(fitted_estimator, X_test, y_test, groups)`
to shuffle correlated or one-hot columns together on held-out data. The returned
mapping contains one score decrease per repeat for each named group.

### The 1994 RELIEF series

`relief-1994-a` through `relief-1994-f` implement the six historical variants
on **discrete, numeric-coded attributes**. Use `method_params={"k": 3}` to
average over neighbors. A averages multiple hits and misses. B assigns uniform
mismatch probability to missing values. C uses B's search distance and omits
unobserved pairs from each attribute's hit/miss averages. D uses empirical
class-conditional mismatch probabilities. E extends D with pooled multiclass
misses. F draws misses per class and weights them by the unconditional class
priors in the 1994 equation. B–F accept NaN. Entirely missing columns score
zero; a class with no observed values uses the pooled training distribution.
The separate `reliefe` method implements the 2021 embedding algorithm.
[Kononenko, 1994](https://doi.org/10.1007/3-540-57868-4_57)

### Complementary Relief.jl methods

`reliefmss`, `boostedsurf`, `iterative-relief`, `irelief`, and `ec-relieff`
run the corresponding implementations in Relief.jl 0.2.0. Iterative Relief and
I-Relief have distinct neighborhood and update rules. Evaporative Cooling
returns an elimination rank. These methods accept dense numeric classification
tables and start a local Julia process for each fit. Select
`method_params={"f_type": "discrete"}` for discrete attributes.
Use Julia 1.10 with a Python environment that contains sklearn:

```bash
PYTHON="$PWD/.venv/bin/python" julia -e 'using Pkg; Pkg.add(PackageSpec(name="Relief", version="0.2.0")); using Relief'
```

Set `JULIA_PROJECT` when using a non-default Julia environment. See the
[Relief.jl reference implementations](https://github.com/jernejvivod/Relief.jl).
