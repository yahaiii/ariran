from __future__ import annotations

from typing import Iterator, Any

import structlog

from config.settings import settings
from connectors.base import BaseConnector

log = structlog.get_logger()

CRIME_KEYWORDS = [
    "robbery", "theft", "kidnapping", "assault", "murder",
    "armed robbery", "rape", "fraud", "cybercrime", "gang",
    "police", "security", "arrested", "shooting", "stabbing",
]


class TwitterConnector(BaseConnector):
    """Twitter/X connector for Nigeria crime signals.

    Requires TWITTER_API_KEY (Bearer token) in .env.
    Uses Twitter API v2 recent_search endpoint.
    Incremental: last 24h tweets. Backfill: 7-day rolling archive.
    """

    source_code = "TWITTER_CRIME"

    def __init__(self, keywords: list[str] | None = None):
        super().__init__()
        self.keywords = keywords or CRIME_KEYWORDS
        self.api_base = "https://api.twitter.com/2/tweets/search/recent"

    def fetch(self, mode: str = "incremental") -> Iterator[dict[str, Any]]:
        if not settings.twitter_api_key:
            raise OSError("TWITTER_API_KEY (Bearer token) must be set in .env")

        query = self._build_query(mode)

        max_results = 100
        next_token = None

        while True:
            params = {
                "query": query,
                "max_results": max_results,
                "tweet.fields": "created_at,public_metrics,author_id",
                "expansions": "author_id",
                "user.fields": "username,location",
            }
            if next_token:
                params["next_token"] = next_token

            headers = {"Authorization": f"Bearer {settings.twitter_api_key}"}
            resp = self._get_with_retry(self.api_base, params=params, headers=headers)
            data = resp.json()

            tweets = data.get("data", [])
            if not tweets:
                break

            includes = data.get("includes", {})
            users = {u["id"]: u for u in includes.get("users", [])}

            for tweet in tweets:
                yield self._tweet_to_record(tweet, users)

            next_token = data.get("meta", {}).get("next_token")
            if not next_token or len(tweets) < max_results:
                break

    def _build_query(self, mode: str) -> str:
        """Build Twitter search query with Nigeria geo + crime keywords."""
        keyword_clause = " OR ".join(self.keywords)
        geo_clause = "(Lagos OR Abuja OR Kano OR Nigeria OR NigeriaCrime OR #NigeriaCrime)"
        query = f"({keyword_clause}) AND {geo_clause} -is:retweet -is:reply lang:en"
        return query

    def _tweet_to_record(self, tweet: dict, users: dict) -> dict:
        tweet_id = tweet.get("id", "")
        author_id = tweet.get("author_id", "")
        author = users.get(author_id, {})
        username = author.get("username", "unknown")

        return {
            "source_record_id": f"TWITTER_{tweet_id}",
            "source_url": f"https://twitter.com/{username}/status/{tweet_id}",
            "raw_payload": tweet,
            "raw_text": tweet.get("text", ""),
        }


if __name__ == "__main__":
    c = TwitterConnector()
    print("Running TwitterConnector (dry run)...")
    summary = c.run(mode="incremental")
    print(summary)
