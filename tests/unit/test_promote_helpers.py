from __future__ import annotations

import uuid
from datetime import date

from pipeline.promote import compute_confidence, find_duplicate, resolve_admin_boundary


class _FetchOneResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _SessionForResolveLga:
    def execute(self, _sql, params):
        if params.get("lga") == "Ikeja":
            return _FetchOneResult((123,))
        return _FetchOneResult(None)


class _SessionForResolveStateFallback:
    def __init__(self):
        self.calls = 0

    def execute(self, _sql, _params):
        self.calls += 1
        if self.calls == 1:
            return _FetchOneResult(None)
        return _FetchOneResult((99,))


class _SessionForResolveNoMatch:
    def execute(self, _sql, _params):
        return _FetchOneResult(None)


class _SessionForFindDuplicate:
    def __init__(self, row):
        self._row = row

    def execute(self, _sql, _params):
        return _FetchOneResult(self._row)


class _SessionForConfidence:
    def __init__(self):
        self.calls = 0

    def execute(self, _sql, _params):
        self.calls += 1
        if self.calls == 1:
            return _ScalarResult(0.8)
        return _ScalarResult(0.67)


def test_resolve_admin_boundary_prefers_lga_match() -> None:
    session = _SessionForResolveLga()

    admin_id, precision = resolve_admin_boundary(session, "Lagos", "Ikeja")

    assert admin_id == 123
    assert precision == "lga"


def test_resolve_admin_boundary_falls_back_to_state() -> None:
    session = _SessionForResolveStateFallback()

    admin_id, precision = resolve_admin_boundary(session, "Lagos", "Unknown LGA")

    assert admin_id == 99
    assert precision == "state"


def test_resolve_admin_boundary_returns_unknown_when_no_match() -> None:
    session = _SessionForResolveNoMatch()

    admin_id, precision = resolve_admin_boundary(session, "Unknown State", "Unknown LGA")

    assert admin_id is None
    assert precision == "unknown"


def test_find_duplicate_uses_existing_canonical_id() -> None:
    canonical = uuid.uuid4()
    session = _SessionForFindDuplicate((canonical, uuid.uuid4()))

    result = find_duplicate(session, date(2024, 1, 1), 7, "robbery")

    assert result == canonical


def test_find_duplicate_falls_back_to_existing_id() -> None:
    existing_id = uuid.uuid4()
    session = _SessionForFindDuplicate((None, existing_id))

    result = find_duplicate(session, date(2024, 1, 1), 7, "robbery")

    assert result == existing_id


def test_compute_confidence_returns_sql_function_value() -> None:
    session = _SessionForConfidence()

    result = compute_confidence(
        session=session,
        source_id=1,
        nlp_confidence=0.75,
        geo_precision="state",
        date_precision="day",
        corroborating=1,
    )

    assert result == 0.67
