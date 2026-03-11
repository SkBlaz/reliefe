.PHONY: test

test:
	NUMBA_DISABLE_JIT=1 uv run --group dev coverage erase
	NUMBA_DISABLE_JIT=1 uv run --group dev coverage run -m unittest discover -s tests -v
	uv run --group dev coverage report -m
