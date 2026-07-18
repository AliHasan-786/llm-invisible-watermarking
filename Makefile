.PHONY: test reproduce-baseline check

test:
	python3 -m unittest discover -s tests -v

reproduce-baseline:
	python3 scripts/reproduce_baseline.py

check: test reproduce-baseline
