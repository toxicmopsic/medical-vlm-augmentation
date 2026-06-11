PYTHON ?= python

.PHONY: prepare train eval predict test lint

prepare:
	$(PYTHON) -m thesis_rrg.cli.prepare_data

train:
	$(PYTHON) -m thesis_rrg.cli.train

eval:
	$(PYTHON) -m thesis_rrg.cli.eval

predict:
	$(PYTHON) -m thesis_rrg.cli.predict

test:
	pytest -q

lint:
	ruff check src tests
