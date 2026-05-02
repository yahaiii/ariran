Connector extension guide

This file explains how to add new connectors to the project and the recommended backfill/incremental pattern.

1) Implement a new connector by creating a module under `connectors/<source>/connector.py`.

2) Inherit from `BaseConnector` and set `source_code` to match `public.sources.source_code`.

3) Implement `fetch(self, mode: str = "incremental") -> Iterator[dict]`.
   - `mode == "incremental"` should yield only the most-recent records (e.g., latest page, last N items).
   - `mode == "backfill"` should attempt to yield all historic records from the source (or as many as reasonable).

4) Each yielded record must be a dict with at minimum:
   - `source_record_id`: str  # unique ID for the source's item
   - `source_url`: str | None
   - `raw_payload`: dict      # original item structure
   - `raw_text`: str          # plain text for NLP/enrichment

5) Avoid writing to the DB in `fetch()`; let the base runner handle staging insertion via `_insert_staging()`.

6) Idempotency: the base `_insert_staging()` will skip duplicate `source_record_id` for the same source.

7) Tests: Provide a unit test that exercises `fetch()` in both `incremental` and `backfill` modes by mocking HTTP/feed responses.

Example quick start
-------------------
- Copy `connectors/template.py` as a starting point.
- Implement `fetch()` and add a small unit test under `tests/unit/` that mocks network calls.

Backfill considerations
----------------------
- Not all sources support efficient backfills. For sources with paginated archive endpoints, implement an archive walk in `backfill` mode.
- Respect rate limits and use `BaseConnector._get_with_retry()` for HTTP calls when appropriate.
- For very large historic backfills, consider running the connector with a larger `pipeline_runs` batch or staging in chunks.
