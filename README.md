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
