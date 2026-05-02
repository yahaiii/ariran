from __future__ import annotations

from types import SimpleNamespace
from connectors.news.connector import NewsRSSConnector


def _make_entry(id_, link, title, summary):
    e = SimpleNamespace()
    e.id = id_
    e.link = link
    e.title = title
    e.summary = summary
    e.keys = lambda: ["id", "link", "title", "summary"]
    return e


def test_news_fetch_incremental(monkeypatch):
    connector = NewsRSSConnector(feeds=["https://example.org/feed"])

    fake_feed = SimpleNamespace()
    fake_feed.entries = [
        _make_entry("1", "https://example.org/1", "One", "First item"),
        _make_entry("2", "https://example.org/2", "Two", "Second item"),
    ]

    monkeypatch.setattr("feedparser.parse", lambda url: fake_feed)

    results = list(connector.fetch(mode="incremental"))
    assert len(results) == 2
    assert results[0]["source_record_id"] == "1"
    assert "First item" in results[0]["raw_text"]


def test_news_fetch_backfill(monkeypatch):
    connector = NewsRSSConnector(feeds=["https://example.org/feed"])

    fake_feed = SimpleNamespace()
    fake_feed.entries = [
        _make_entry("a", "https://example.org/a", "A", "Alpha"),
    ]

    monkeypatch.setattr("feedparser.parse", lambda url: fake_feed)

    results = list(connector.fetch(mode="backfill"))
    assert len(results) == 1
    assert results[0]["source_record_id"] == "a"
