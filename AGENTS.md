# Repository Guide

## Purpose

This repo packages `reliefe`, a feature-ranking library centered on the `ReliefE` implementation in `reliefe/reliefe_classes.py`. The codebase mixes library code, bundled datasets, runnable examples, and prebuilt docs.

## Layout

- `reliefe/`
  - `reliefe_classes.py`: main algorithm, Numba kernels, estimator API.
  - `utils.py`: ARFF loading, task-type helpers, utility enums.
  - `reliefe_evaluation.py`: evaluation helpers.
- `examples/`
  - Small runnable scripts intended to work from the repo root with `python examples/<name>.py`.
  - Keep them representative, but avoid benchmark-scale runtimes in “minimal” examples.
- `data/`
  - Bundled `.mat` and `.arff` files used by examples and tests.
- `tests/`
  - Smoke-style regression tests, runnable with `make test`.
- `docs/`, `docfiles/`
  - Documentation source and generated site artifacts.

## Working Rules

- Preserve the public import surface: `import reliefe` should continue to expose `ReliefE` and helpers from `__init__.py`.
- Prefer small, deterministic smoke tests over long benchmark-style tests.
- When changing examples, keep them runnable from the repository root and avoid hidden working-directory assumptions.
- Treat `docs/` as generated/static output unless the task explicitly asks for documentation-site edits.
- Prefer `uv` for local workflows. Reach for `uv sync`, `uv run ...`, and `make test` before ad hoc environment setup.
- Use `uv sync --group dev` when you need coverage or other test-only tooling.

## Commands

- Sync the project environment: `uv sync`
- Sync with test tooling: `uv sync --group dev`
- Run tests: `make test`
- Run a representative example: `uv run python examples/minimal_reliefe2.py`

## Library concepts and usage

Relief methods rank features by comparing nearby examples: differences between
classes increase a feature's score; differences within a class decrease it.
ReliefE can use a manifold embedding to choose neighbours. The implementation
also supports multi-label targets. Higher scores mean greater relevance, and
scores can be negative; they are neither probabilities nor causal effects.

A deterministic, small classification example (run from the repo root):

```python
import numpy as np
from reliefe import ReliefE

x = np.array([[0., 0., 1.], [0., 1., 1.],
              [1., 0., 0.], [1., 1., 0.]])
y = np.array([0, 0, 1, 1])
model = ReliefE(num_iter=4, k=1, embedding_based_distances=False,
                verbose=False)
model.fit(x, y, embedding_method="SVD")
scores = model.feature_importances_  # one score per input feature
ranking = np.argsort(-scores, kind="stable")  # descending, zero-based indices
```

- `x` is a numeric `(n_samples, n_features)` NumPy array or SciPy sparse
  matrix (converted internally to CSR). Lists are converted in some helpers;
  use arrays at `fit` because preprocessing accesses `.shape` and `.ndim`.
- For classification, `y` is a one-dimensional vector of class labels, or a
  one-hot target matrix. Each class needs at least two examples. Use finite,
  preprocessed features and at least two classes for meaningful rankings.
- Multi-label targets are binary matrices with one column per label. Task
  detection treats matrices whose rows all sum to one as classification.
  Regression is not implemented, despite the utility enum's regression value.
- `num_iter` accepts an integer, a fraction of the number of rows, or a list of
  checkpoints. The estimator clamps iterations to the number of examples and
  samples without replacement using a fixed permutation. `all_weights` contains
  the checkpoint scores; `feature_importances_` contains the last checkpoint.
- `k` is clamped against the smallest class size. `determine_k_automatically`
  enables a distance-gap heuristic; `use_average_neighbour` changes the update
  rule. Keep both explicit in comparisons.
- `normalize_descriptive=True` is the default. Direct distances scale columns
  by their range; embedded distances use row normalization. Impute missing
  values and encode categorical features before fitting; `load_arff` can help
  with ARFF files and returns encoded-column ranges.
- Enable `embedding_based_distances=True` explicitly to request embeddings.
  `fit(..., embedding_method="auto")` chooses UMAP when installed, otherwise
  SVD. `"SVD"` or a sklearn-style object with `fit` and `transform` can be
  supplied explicitly. `LD_friendly=True` may switch back to the original
  space for low-dimensional data. Input embedding failures can also fall back
  to original-space data; inspect behaviour before claiming an embedding ran.
- Embedding and sparsification paths use randomness without a public
  `random_state` parameter. The fixed iteration permutation alone does not
  make every fit reproducible. Recreate the estimator for independent fits:
  fitting mutates parameters such as `num_iter`, `k`, and the embedding flag.
- Feature selection must be fitted inside each training fold. Do not rank on
  the full dataset before a train/test split or cross-validation.

## Verification and regression policy

No regressions is a hard requirement. Run `uv sync --group dev` and `make test`
before proposing any code change. The current coverage configuration requires
100% statement coverage. Preserve that gate; do not lower the threshold, exclude
new library code to pass it, or suppress failing tests. Existing `pragma: no
cover` markers are not a pattern for new code. Coverage is necessary but does
not prove numerical correctness.

`make test` disables Numba JIT for fast branch coverage. Changes touching kernels
also require a separate run with JIT enabled and a numerical reference or
before/after comparison. Run a representative example for changes affecting
installation, imports, estimators, or examples. Keep unit tests small and
independent of network access. Check that the CI job fails when tests fail.

## Benchmark and PR policy

- Keep documentation, agent tooling, and algorithm changes in separate PRs.
  Open draft PRs with clear descriptions. Across all work for this repository,
  open at most two PRs per Europe/Ljubljana calendar day, including drafts and
  PRs that are subsequently closed. Count existing PRs before creating one;
  when the count cannot be verified, prepare locally and defer publication.
- Before an algorithm PR, state the hypothesis and inspect the relevant
  primary literature and existing implementation. Fix correctness before
  claiming speed or ranking-quality improvements. Never claim state of the art
  from smoke tests or one dataset.
- Benchmark the base commit and candidate on the same machine and inputs.
  Record commit IDs, exact command, dataset and split, seeds, parameters,
  dependency versions, hardware/thread settings, and raw results. Separate
  cold JIT compilation from warmed runtimes, repeat timings, and report
  variability and memory when relevant.
- Check score equivalence against a reference for optimizations that should
  preserve results. For changes to the ranking rule, compare downstream
  performance across multiple datasets/seeds with selection inside training
  folds, and report regressions as well as improvements. A meaningful result
  must exceed measurement noise and justify the added complexity.
- Daily code-quality sweeps should propose focused, evidence-backed refactors
  only when useful. Do not manufacture changes to fill the PR quota. Include
  the reproducer or benchmark and exact validation results in each draft PR.
- The requested improvement campaign lasts seven days, with algorithm scans
  every eight hours and a code-quality sweep once per local calendar day.
  Store its deadlines, run history, and benchmark artifacts locally; do not
  turn this finite campaign into a permanent recurring commitment.
- Work only inside `/home/blazs/reliefe` on this machine. Put temporary files,
  isolated checkouts, benchmark outputs, and campaign logs under `.agent-local/`
  (locally ignored). Do not edit other repositories or generated `docs/`.
