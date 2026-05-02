from __future__ import annotations

from types import SimpleNamespace

from connectors.acled.connector import ACLEDConnector


def _fake_resp(data, status=200):
    return SimpleNamespace(json=lambda: {"status": status, "data": data})


def _fake_token_resp():
    return SimpleNamespace(json=lambda: {"access_token": "fake-token", "expires_in": 3600, "refresh_token": "fake-refresh", "token_type": "Bearer"})


def test_acled_oauth_incremental(monkeypatch):
    monkeypatch.setenv("ACLED_EMAIL", "test@example.com")
    monkeypatch.setenv("ACLED_PASSWORD", "pw")

    connector = ACLEDConnector(date_from="2020-01-01", date_to="2020-01-02")

    sample = [{"event_id_cnty": "e1", "event_type": "Violence"}]

    # Mock token request and authenticated GET
    monkeypatch.setattr(ACLEDConnector, "_post_with_retry", lambda self, url, data: _fake_token_resp())
    monkeypatch.setattr(ACLEDConnector, "_get_with_retry_auth", lambda self, url, params=None: _fake_resp(sample))

    results = list(connector.fetch(mode="incremental"))
    assert len(results) == 1
    assert results[0]["source_record_id"].endswith("e1")


def test_acled_oauth_backfill(monkeypatch):
    monkeypatch.setenv("ACLED_EMAIL", "test@example.com")
    monkeypatch.setenv("ACLED_PASSWORD", "pw")

    # use small page size for test
    from connectors.acled import connector as acled_mod
    monkeypatch.setattr(acled_mod, "ACLED_PAGE_SIZE", 2)

    page1 = [
        {"event_id_cnty": "e1", "event_type": "X"},
        {"event_id_cnty": "e2", "event_type": "Y"},
    ]
    page2 = [
        {"event_id_cnty": "e3", "event_type": "Z"},
    ]

    calls = {"n": 0}

    def fake_get(self, url, params=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_resp(page1)
        elif calls["n"] == 2:
            return _fake_resp(page2)
        return _fake_resp([])

    monkeypatch.setattr(ACLEDConnector, "_post_with_retry", lambda self, url, data: _fake_token_resp())
    monkeypatch.setattr(ACLEDConnector, "_get_with_retry_auth", fake_get)

    connector = ACLEDConnector(date_from="2020-01-01", date_to="2020-12-31")
    results = list(connector.fetch())
    assert len(results) == 3
    ids = [r["source_record_id"] for r in results]
    assert ids == ["ACLED_e1", "ACLED_e2", "ACLED_e3"]
