"""
BaseConnector — all source connectors inherit from this.
Enforces: run logging, staging insert, error handling, retry logic.
"""
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from db.connection import get_session

log = structlog.get_logger()


class BaseConnector(ABC):
    """
    Every connector must implement:
      - source_code: str  matching public.sources.source_code
      - fetch() -> Iterator[dict]  yields raw record dicts
    """

    source_code: str = ""

    def __init__(self):
        self._source_id: int | None = None
        self._run_id: uuid.UUID | None = None
        self._counters = {"fetched": 0, "inserted": 0, "skipped": 0, "failed": 0}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, mode: str = "incremental") -> dict:
        """Execute a full ingestion run.

        mode: 'incremental' | 'backfill'

        Returns run summary counters.
        """
        log.info("connector_run_start", source=self.source_code, mode=mode)
        self._run_id = self._open_pipeline_run()

        try:
            # Pass mode to fetch so connectors can implement backfill vs incremental
            for raw_record in self.fetch(mode=mode):
                self._counters["fetched"] += 1
                try:
                    inserted = self._insert_staging(raw_record)
                    if inserted:
                        self._counters["inserted"] += 1
                    else:
                        self._counters["skipped"] += 1
                except Exception as e:
                    self._counters["failed"] += 1
                    log.warning("staging_insert_failed",
                                source=self.source_code, error=str(e))

        except Exception as e:
            self._close_pipeline_run(status="failed", error=str(e))
            raise

        self._close_pipeline_run(status="completed")
        log.info("connector_run_done", source=self.source_code, **self._counters)
        return self._counters

        # ------------------------------------------------------------------
        # Abstract interface
        # ------------------------------------------------------------------

        @abstractmethod
        def fetch(self, mode: str = "incremental") -> Iterator[dict[str, Any]]:
                """
                Yields one raw record dict per incident. Must include at minimum:
                    - source_record_id: str
                    - source_url: str (optional)
                    - raw_payload: dict  (the full original record)
                    - raw_text: str      (plain text for NLP)
                """
                ...

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_source_id(self) -> int:
        if self._source_id:
            return self._source_id
        with get_session() as s:
            from sqlalchemy import text
            row = s.execute(
                text("SELECT id FROM public.sources WHERE source_code = :code"),
                {"code": self.source_code}
            ).fetchone()
            if not row:
                raise ValueError(f"Source '{self.source_code}' not found in sources table. "
                                 "Run scripts/init_db.py first.")
            self._source_id = row[0]
        return self._source_id

    def _open_pipeline_run(self) -> uuid.UUID:
        run_id = uuid.uuid4()
        with get_session() as s:
            from sqlalchemy import text
            s.execute(text("""
                INSERT INTO public.pipeline_runs
                    (id, source_id, status, run_started_at, pipeline_version)
                VALUES (:id, :source_id, 'running', NOW(), :version)
            """), {
                "id": str(run_id),
                "source_id": self._get_source_id(),
                "version": "0.1.0",
            })
        return run_id

    def _close_pipeline_run(self, status: str, error: str | None = None):
        with get_session() as s:
            from sqlalchemy import text
            s.execute(text("""
                UPDATE public.pipeline_runs SET
                    status = :status,
                    run_finished_at = NOW(),
                    records_fetched = :fetched,
                    records_inserted = :inserted,
                    records_skipped = :skipped,
                    records_failed = :failed,
                    error_log = :error_log
                WHERE id = :run_id
            """), {
                "status": status,
                "fetched": self._counters["fetched"],
                "inserted": self._counters["inserted"],
                "skipped": self._counters["skipped"],
                "failed": self._counters["failed"],
                "error_log": json.dumps({"error": error}) if error else None,
                "run_id": str(self._run_id),
            })

    def _insert_staging(self, record: dict) -> bool:
        """
        Inserts one record into raw.incidents_staging.
        Returns False (skipped) if source_record_id already exists for this source.
        """
        source_id = self._get_source_id()

        with get_session() as s:
            from sqlalchemy import text

            # Idempotency check — skip if already ingested
            exists = s.execute(text("""
                SELECT 1 FROM raw.incidents_staging
                WHERE source_id = :sid AND source_record_id = :rid
                LIMIT 1
            """), {
                "sid": source_id,
                "rid": str(record.get("source_record_id", "")),
            }).fetchone()

            if exists:
                return False

            s.execute(text("""
                INSERT INTO raw.incidents_staging (
                    id, pipeline_run_id, source_id,
                    source_record_id, source_url,
                    raw_payload, raw_text,
                    processing_status, ingested_at
                ) VALUES (
                    :id, :run_id, :source_id,
                    :source_record_id, :source_url,
                    :raw_payload, :raw_text,
                    'pending', NOW()
                )
            """), {
                "id": str(uuid.uuid4()),
                "run_id": str(self._run_id),
                "source_id": source_id,
                "source_record_id": str(record.get("source_record_id", "")),
                "source_url": record.get("source_url"),
                "raw_payload": json.dumps(record.get("raw_payload", {})),
                "raw_text": record.get("raw_text", ""),
            })

        return True

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_with_retry(self, url: str, **kwargs) -> Any:
        """HTTP GET with automatic retry and backoff."""
        import requests
        headers = {"User-Agent": settings.user_agent}
        resp = requests.get(url, headers=headers,
                            timeout=settings.request_timeout, **kwargs)
        resp.raise_for_status()
        return resp
