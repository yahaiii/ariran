.PHONY: dev-up dev-down install test lint format check

# ── Environment ─────────────────────────────────────────────────────────
install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/python -m spacy download en_core_web_trf

# ── Docker ──────────────────────────────────────────────────────────────
dev-up:
	docker compose up -d
	@echo "PostGIS ready at localhost:5432"
	@echo "pgAdmin ready at http://localhost:5050"

dev-down:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d

# ── Database ─────────────────────────────────────────────────────────────
init-db:
	.venv/bin/python scripts/init_db.py

load-boundaries:
	.venv/bin/python scripts/load_boundaries.py

# ── Connectors ───────────────────────────────────────────────────────────
run-nbs:
	.venv/bin/python -m connectors.nbs.connector

run-acled:
	.venv/bin/python -m connectors.acled.connector

# ── Code quality ─────────────────────────────────────────────────────────
lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .

type-check:
	.venv/bin/mypy connectors db enrichment pipeline

check: lint type-check

# ── Tests ────────────────────────────────────────────────────────────────
test:
	.venv/bin/pytest --cov --cov-report=term-missing -q

test-unit:
	.venv/bin/pytest tests/unit -q

test-integration:
	.venv/bin/pytest tests/integration -q
