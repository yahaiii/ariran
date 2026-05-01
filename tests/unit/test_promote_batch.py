from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pipeline.promote as promote


@dataclass
class _FetchAllResult:
    rows: list[tuple[Any, ...]]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class _DummyResult:
    def fetchall(self) -> list[tuple[Any, ...]]:
        return []


class _FakeSession:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.execute_calls: int = 0

    def execute(self, _sql: Any, _params: dict[str, Any] | None = None) -> Any:
        self.execute_calls += 1
        if self.execute_calls == 1:
            return _FetchAllResult(self._rows)
        return _DummyResult()


class _SessionContext:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def __enter__(self) -> _FakeSession:
        return self._session

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> bool:
        return False


def _build_row(raw_payload: dict[str, Any] | None = None) -> tuple[Any, ...]:
    payload = raw_payload or {"state": "Lagos", "admin2": "Ikeja", "year": 2024}
    return (
        "staging-id-1",  # id
        1,  # source_id
        None,  # pipeline_run_id
        "src-1",  # source_record_id
        "https://example.com",  # source_url
        payload,  # raw_payload
        "raw text",  # raw_text
        "robbery",  # nlp_crime_type
        None,  # nlp_location_raw
        None,  # nlp_date_raw
        [],  # nlp_actors
        0,  # nlp_fatalities
        0,  # nlp_injuries
        0.9,  # nlp_confidence
        None,  # geocoded_geom
        None,  # geocoded_admin_id
        None,  # geocode_method
    )


def test_promote_batch_dry_run_no_rows(monkeypatch: Any) -> None:
    session = _FakeSession(rows=[])
    monkeypatch.setattr(promote, "get_session", lambda: _SessionContext(session))

    counters = promote.promote_batch(dry_run=True)

    assert counters == {"promoted": 0, "skipped": 0, "failed": 0}


def test_promote_batch_dry_run_promotes_row(monkeypatch: Any) -> None:
    session = _FakeSession(rows=[_build_row()])
    monkeypatch.setattr(promote, "get_session", lambda: _SessionContext(session))
    monkeypatch.setattr(promote, "resolve_admin_boundary", lambda *_args, **_kwargs: (11, "lga"))
    monkeypatch.setattr(promote, "find_duplicate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(promote, "compute_confidence", lambda *_args, **_kwargs: 0.88)

    counters = promote.promote_batch(dry_run=True)

    assert counters == {"promoted": 1, "skipped": 0, "failed": 0}


def test_promote_batch_marks_failed_when_processing_errors(monkeypatch: Any) -> None:
    session = _FakeSession(rows=[_build_row()])
    monkeypatch.setattr(promote, "get_session", lambda: _SessionContext(session))
    monkeypatch.setattr(promote, "resolve_admin_boundary", lambda *_args, **_kwargs: (11, "lga"))
    monkeypatch.setattr(promote, "find_duplicate", lambda *_args, **_kwargs: None)

    def _boom(*_args: Any, **_kwargs: Any) -> float:
        raise ValueError("forced failure")

    monkeypatch.setattr(promote, "compute_confidence", _boom)

    counters = promote.promote_batch(dry_run=True)

    assert counters["promoted"] == 0
    assert counters["failed"] == 1
    assert session.execute_calls >= 2
