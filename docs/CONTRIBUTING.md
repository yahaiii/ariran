# Contributing

Thanks for contributing to ariran.

## Local setup
1. Copy `.env.example` to `.env` and set required values.
2. Run `make install`.
3. Run `make dev-up` then `make init-db`.

## Quality gates
1. Run `make check` for lint + type checks.
2. Run `make test` (or `make test-unit` / `make test-integration`).

## Pull request expectations
1. Keep changes scoped and documented.
2. Add or update tests for behavioral changes.
3. Ensure all checks pass before requesting review.

## Connector development

This project uses a connector pattern to ingest data from external sources into the pipeline. Follow these guidelines when adding or modifying connectors.

- Connector layout: create a folder under `connectors/<source>/` with a `connector.py` implementing a subclass of `connectors.base.BaseConnector`.
- Required methods: implement `fetch(self, mode: str = "incremental") -> Iterator[dict]` that yields staging records in the shape accepted by the pipeline (must include `source_record_id`, `raw_payload`, `raw_text`, and optional `source_url`).
- `run()` is provided by `BaseConnector` — call `connector.run(mode="incremental")` or `connector.run(mode="backfill")` from runners.

Design notes
- Modes: support two modes — `incremental` (fetch recent updates, typically one or a few pages) and `backfill` (full historic pagination). The `mode` parameter must be accepted by `fetch()` even if ignored.
- Idempotency: ensure each yielded record has a stable `source_record_id` to allow the pipeline to avoid duplicates.
- HTTP safety: use `self._get_with_retry(url, params=...)` from `BaseConnector` for HTTP calls — it applies retry/backoff and respects settings.
- Configuration: read API keys or URLs from `config.settings` where possible. Add new settings to `config/settings.py` with clear `env` names.

Testing
- Unit tests: add unit tests under `tests/unit/` named `test_connectors_<source>.py`. Tests should mock network calls (`monkeypatch` the connector's `_get_with_retry`) and verify `fetch(mode="incremental")` and `fetch(mode="backfill")` behaviors.
- Use the project's existing test utilities and `SimpleNamespace` objects to fake JSON responses.
- Run `make test-unit` locally to validate.

CLI & DAG integration
- Scaffolding: use `scripts/new_connector.py <name>` to create a connector scaffold and test skeleton.
- Backfill runner: the repository provides `scripts/backfill_runner.py` to run connectors with checkpointing and a `--mock` flag for local dry-runs.
- Airflow: the DAG `pipeline/dags/connector_scheduler.py` contains daily incremental and manual backfill DAGs. Add connector tasks there when adding new sources.

Best practices
- Keep external-specific parsing inside the connector module; map to the canonical staging format before yielding.
- Avoid committing secrets; use `.env` and `config.settings`.
- Document any rate-limiting, pagination or licensing caveats in the connector docstring.

If you want, I can scaffold a new connector now using the template and add a test example.
