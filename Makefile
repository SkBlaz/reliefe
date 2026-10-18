.PHONY: test

test:
	NUMBA_DISABLE_JIT=1 uv run --extra mcp --group dev coverage erase
	NUMBA_DISABLE_JIT=1 uv run --extra mcp --group dev coverage run -m unittest discover -s tests -v
	uv run --extra mcp --group dev coverage report -m
	NUMBA_NUM_THREADS=1 uv run --extra mcp --group dev python -m unittest discover -s tests -p test_ranking.py -v
