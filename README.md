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
