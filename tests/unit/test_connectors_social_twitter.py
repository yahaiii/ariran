from __future__ import annotations

from types import SimpleNamespace

from connectors.social.twitter_connector import TwitterConnector
from config import settings as settings_mod


def _fake_twitter_resp(tweets_data, has_more=False):
    """Fake Twitter API response."""
    meta = {"result_count": len(tweets_data)}
    if has_more:
        meta["next_token"] = "b26v89c19zqg8o3fpz9g8o3fpza"
    
    users = []
    for tweet in tweets_data:
        if tweet.get("author_id"):
            users.append({"id": tweet["author_id"], "username": f"user_{tweet['author_id']}"})
    
    return SimpleNamespace(json=lambda: {
        "data": tweets_data,
        "includes": {"users": users},
        "meta": meta,
    })


def test_twitter_incremental(monkeypatch):
    # Mock settings to have twitter_api_key
    mock_settings = SimpleNamespace(twitter_api_key="fake_bearer_token_123")
    monkeypatch.setattr(settings_mod, "settings", mock_settings)

    connector = TwitterConnector(keywords=["crime"])

    tweets = [
        {
            "id": "1234567890",
            "text": "Armed robbery in Lagos today",
            "author_id": "user123",
        },
        {
            "id": "1234567891",
            "text": "Kidnapping incident in Abuja",
            "author_id": "user124",
        },
    ]

    monkeypatch.setattr(
        TwitterConnector, "_get_with_retry",
        lambda self, url, params=None, headers=None: _fake_twitter_resp(tweets),
    )

    results = list(connector.fetch(mode="incremental"))
    assert len(results) == 2
    assert results[0]["source_record_id"] == "TWITTER_1234567890"
    assert "Armed robbery" in results[0]["raw_text"]


def test_twitter_backfill_pagination(monkeypatch):
    # Mock settings to have twitter_api_key
    mock_settings = SimpleNamespace(twitter_api_key="fake_bearer_token_123")
    monkeypatch.setattr(settings_mod, "settings", mock_settings)

    connector = TwitterConnector(keywords=["crime"])

    page1_tweets = [
        {"id": "1", "text": "Crime 1", "author_id": "u1"},
        {"id": "2", "text": "Crime 2", "author_id": "u2"},
    ]
    page2_tweets = [
        {"id": "3", "text": "Crime 3", "author_id": "u3"},
    ]

    calls = {"n": 0}

    def fake_get(self, url, params=None, headers=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_twitter_resp(page1_tweets, has_more=True)
        elif calls["n"] == 2:
            return _fake_twitter_resp(page2_tweets, has_more=False)
        return _fake_twitter_resp([], has_more=False)

    monkeypatch.setattr(TwitterConnector, "_get_with_retry", fake_get)

    results = list(connector.fetch(mode="backfill"))
    assert len(results) == 3
    ids = [r["source_record_id"] for r in results]
    assert ids == ["TWITTER_1", "TWITTER_2", "TWITTER_3"]
