.PHONY: test reproduce-baseline export-replay check-replay check
PYTHON ?= python3

test:
	$(PYTHON) -m pytest -q

reproduce-baseline:
	$(PYTHON) scripts/reproduce_baseline.py

export-replay:
	$(PYTHON) scripts/export_historical_replay.py

check-replay:
	$(PYTHON) scripts/export_historical_replay.py --check

check: test reproduce-baseline check-replay
