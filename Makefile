.PHONY: test

test:
	PYTHONPATH=. python -m unittest discover -s tests -v
