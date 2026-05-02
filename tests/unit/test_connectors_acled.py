from __future__ import annotations

from types import SimpleNamespace

from connectors.acled.connector import ACLEDConnector


def _fake_resp(data):
    return SimpleNamespace(json=lambda: {"data": data})


def test_acled_incremental(monkeypatch):
    connector = ACLEDConnector()

    sample = [{"event_id": "e1", "event_type": "Violence", "actor1": "A", "actor2": "B", "notes": "note"}]

    monkeypatch.setattr(ACLEDConnector, "_get_with_retry", lambda self, url, params=None: _fake_resp(sample))

    results = list(connector.fetch(mode="incremental"))
    assert len(results) == 1
    assert results[0]["source_record_id"] == "e1"


def test_acled_backfill(monkeypatch):
    connector = ACLEDConnector(page_size=2)

    page1 = [
        {"event_id": "e1", "event_type": "X"},
        {"event_id": "e2", "event_type": "Y"},
    ]
    page2 = [
        {"event_id": "e3", "event_type": "Z"},
    ]

    calls = {"n": 0}

    def fake_get(self, url, params=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_resp(page1)
        elif calls["n"] == 2:
            return _fake_resp(page2)
        return _fake_resp([])

    monkeypatch.setattr(ACLEDConnector, "_get_with_retry", fake_get)

    results = list(connector.fetch(mode="backfill"))
    assert len(results) == 3
    ids = [r["source_record_id"] for r in results]
    assert ids == ["e1", "e2", "e3"]
