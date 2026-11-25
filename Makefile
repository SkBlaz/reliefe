.PHONY: test

test:
	NUMBA_DISABLE_JIT=1 uv run --extra mcp --extra rebate --extra all-relevant --group dev coverage erase
	NUMBA_DISABLE_JIT=1 uv run --extra mcp --extra rebate --extra all-relevant --group dev coverage run -m unittest discover -s tests -v
	uv run --extra mcp --extra rebate --extra all-relevant --group dev coverage report -m
	NUMBA_DISABLE_JIT=0 NUMBA_NUM_THREADS=1 uv run --extra mcp --extra rebate --extra all-relevant --group dev python -m unittest discover -s tests -p test_ranking.py -v
	NUMBA_DISABLE_JIT=0 NUMBA_NUM_THREADS=1 uv run --extra mcp --extra rebate --extra all-relevant --group dev python -m unittest discover -s tests -p test_embedded_identity.py -v
	NUMBA_DISABLE_JIT=0 NUMBA_NUM_THREADS=1 uv run --extra mcp --extra rebate --extra all-relevant --group dev python -m unittest discover -s tests -p test_mlc_checkpoints.py -v
	NUMBA_DISABLE_JIT=0 NUMBA_NUM_THREADS=1 uv run --extra mcp --extra rebate --extra all-relevant --group dev python -m unittest discover -s tests -p test_native_kernels.py -v
	NUMBA_DISABLE_JIT=0 NUMBA_NUM_THREADS=1 uv run --extra mcp --extra rebate --extra all-relevant --group dev python -m unittest discover -s tests -p "test_*parity.py" -v
