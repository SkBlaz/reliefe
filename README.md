# ReliefE

ReliefE is a Numba-accelerated feature ranking library for multiclass and multi-label classification. It extends the Relief family with optional manifold-based embeddings for the input and target spaces.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

`umap-learn` is optional at runtime if you pass `embedding_method="SVD"` or another sklearn-like embedder.

## Run

```bash
python examples/minimal_reliefe2.py
python examples/minimal_reliefe.py
python examples/custom_embeddings.py
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

## Reference

Paper: Škrlj et al., "ReliefE: feature ranking in high-dimensional spaces via manifold embeddings", Machine Learning (2021).

Docs: https://skblaz.github.io/reliefe/
