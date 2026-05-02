.PHONY: dev-up dev-down install test lint format check
.PHONY: dev-up dev-down install test lint format check run-api

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
	@echo "API ready at http://localhost:8000 (if service built successfully)"

dev-down:
	docker compose down

dev-up-api:
	docker compose up -d api db pgadmin
	@echo "Full stack ready:"
	@echo "  - API:     http://localhost:8000"
	@echo "  - Docs:    http://localhost:8000/docs"
	@echo "  - pgAdmin: http://localhost:5050"
	@echo "  - DB:      localhost:5432"

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

run-api:
	.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

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

test-integration-live:
	INTEGRATION_TESTS=1 .venv/bin/pytest tests/integration -m live -q
