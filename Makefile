.PHONY: venv install qdrant ingest ui test lint

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

qdrant:
	docker compose up -d qdrant

ingest:
	. .venv/bin/activate && python -m ragbook.cli ingest ./books --config ./config.yaml

ui:
	. .venv/bin/activate && python -m ragbook.cli ui --config ./config.yaml

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check src tests
