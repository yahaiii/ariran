# Nigeria Open Crime Database (ariran)

The largest open-source crime dataset in Nigeria — multi-source, spatially indexed, provenance-first.

## Architecture
```
Data Sources → Ingestion Bus → Enrichment (NLP + Geocoding) → PostGIS → API / Exports
```

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Docker + Docker Compose
- VSCode (recommended)

### 2. Clone and install
```bash
git clone https://github.com/your-org/ariran.git
cd ariran
cp .env.example .env          # fill in DB_PASSWORD, ACLED_API_KEY
make install                  # creates .venv and installs all deps
```

### 3. Start the database and stack
```bash
make dev-up                   # starts PostGIS + pgAdmin + API via Docker
```

**Or**, run the database and run the API locally:
```bash
docker compose up -d db pgadmin  # database and admin UI only
make init-db                  # verify connection and seed data
make run-api                  # start the FastAPI app on http://localhost:8000
```

### 4. Initialize the database
```bash
make init-db                  # verifies connection and seed data
```

### 5. Run connectors
```bash
make run-nbs                  # ingest NBS annual crime stats
make run-acled                # ingest ACLED conflict events (needs API key)
```

### 6. Run integration tests
```bash
make test-integration         # run integration tests (live tests skip by default)
INTEGRATION_TESTS=1 make test-integration-live
```

Live tests require real credentials and network access. The harness currently includes ACLED, RSS news, Nairaland, and Twitter when `TWITTER_API_KEY` is set.

To run live integration tests in GitHub Actions, open the `Integration Tests` workflow and trigger it with `run_live_connectors=true`.
Configure repository secrets: `ACLED_EMAIL`, `ACLED_PASSWORD`, `ACLED_API_KEY`, `TWITTER_API_KEY`.

### 7. Access the API
Once running, the API is available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### 8. Open in VSCode
```bash
code .
```
Install recommended extensions when prompted (`.vscode/extensions.json`).

## VS Code Server / Remote Setup

This repository now includes a Dev Container setup for VS Code Server usage.

### Option A: WSL Remote (fastest)
1. Open the project in WSL from VS Code.
2. Run `make install` to create `.venv` and install dependencies.
3. Run `make dev-up` and `make init-db`.

### Option B: Dev Container (isolated)
1. Open the project in VS Code.
2. Run `Dev Containers: Reopen in Container`.
3. Wait for `postCreateCommand` to finish creating `.venv` and installing dependencies.
4. Use VS Code Tasks for bootstrap:
	- `Install Dependencies`
	- `Start DB Stack`
	- `Initialize Database`

The Dev Container config is defined in `.devcontainer/devcontainer.json` and shares
the project `docker-compose.yml` database service.

## Project Structure
```
ariran/
├── connectors/          # One sub-package per data source
│   ├── base.py          # BaseConnector — run logging, staging insert, retry
│   ├── nbs/             # NBS annual statistics (Excel)
│   ├── acled/           # ACLED API (conflict events)
│   ├── news/            # RSS scrapers (Punch, Vanguard, Channels, Premium Times)
│   ├── social/          # Twitter/X stream, Telegram
│   └── crowdsource/     # Public tipline intake
├── enrichment/
│   ├── nlp/             # spaCy NER: crime type, location, date, actors
│   ├── geocoder/        # Nigerian place name → lat/lon (LGA-level gazeteer)
│   └── dedup/           # Duplicate detection and canonical_id assignment
├── db/
│   ├── connection.py    # SQLAlchemy engine + session context manager
│   ├── models/          # SQLAlchemy ORM models (optional, raw SQL preferred)
│   └── migrations/      # Alembic migration scripts
├── pipeline/
│   ├── promote.py       # Staging → canonical promotion logic (scaffold)
│   ├── dags/            # Airflow DAG definitions
│   └── operators/       # Custom Airflow operators
├── api/                 # FastAPI public REST API (Phase 2)
├── scripts/
│   ├── init_db.py       # DB health check + seed verification
│   └── load_boundaries.py  # Boundary loader entrypoint (scaffold)
├── tests/
│   ├── unit/            # Connector + enrichment unit tests
│   └── integration/     # Full pipeline end-to-end tests
├── config/
│   └── settings.py      # Pydantic settings (env-driven)
├── .vscode/             # VSCode workspace config, launch configs, extensions
├── docker-compose.yml   # PostGIS 15 + pgAdmin
├── Makefile             # Dev workflow shortcuts
└── nigeria_crime_db_schema.sql  # Complete PostGIS DDL
```

## Data Quality Model
Every record carries:
- `confidence_score` (0.0–1.0) — composite from source trust, NLP confidence, geo/date precision
- `verification_status` — unverified → corroborated → police_confirmed
- `geo_precision` — coordinates / settlement / lga / state
- `date_precision` — exact / day / week / month / year
- `canonical_id` — links duplicate records from different sources to one event

## Sources
| Code           | Source                        | Type         | Default Confidence |
| -------------- | ----------------------------- | ------------ | ------------------ |
| NBS_ANNUAL     | National Bureau of Statistics | Government   | 0.80               |
| ACLED_API      | ACLED Conflict Events         | NGO          | 0.78               |
| NGA_WATCH      | Nigeria Watch                 | NGO          | 0.75               |
| UCDP_GED       | Uppsala Conflict Data Program | Academic     | 0.80               |
| PUNCH_RSS      | Punch Newspapers              | News         | 0.55               |
| VANGUARD_RSS   | Vanguard                      | News         | 0.55               |
| CHANNELS_RSS   | Channels TV                   | News         | 0.60               |
| PREMIUM_TIMES  | Premium Times                 | News         | 0.65               |
| TWITTER_STREAM | X (Twitter)                   | Social       | 0.30               |
| CROWDSOURCE    | ariran Tipline                | Crowdsourced | 0.20               |

## Contributing
See `docs/CONTRIBUTING.md`. All contributions must include tests and pass `make check`.

## Licence
Open Data Commons Open Database Licence (ODbL) v1.0
