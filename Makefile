.PHONY: test reproduce-baseline check
PYTHON ?= python3

test:
	$(PYTHON) -m pytest -q

reproduce-baseline:
	$(PYTHON) scripts/reproduce_baseline.py

check: test reproduce-baseline
