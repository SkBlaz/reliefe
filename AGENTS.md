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

## Commands

- Install runtime dependencies: `pip install -r requirements.txt`
- Install in editable mode: `pip install -e .`
- Run tests: `make test`
- Run a representative example: `python examples/minimal_reliefe2.py`
