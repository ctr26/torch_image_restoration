.PHONY: all build install test lint typecheck prepush

all: build install test

build:
	uv build

install:
	uv sync

test:
	uv run pytest

test.notebooks:
	uv run pytest --nbmake **/*ipynb

lint:
	uv run ruff check .

typecheck:
	uv run mypy .

prepush: lint typecheck test
	@echo "✅ Ready to push"

get.fly.data:
	wget -nc https://download.fht.org/jug/n2v/BSD68_reproducibility.zip \
	https://download.fht.org/jug/n2v/RGB.zip \
	https://download.fht.org/jug/n2v/flywing-data.zip \
	https://download.fht.org/jug/n2v/SEM.zip

figures:
	python scripts/generate_figures.py
