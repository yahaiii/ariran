from __future__ import annotations

from typing import Iterator, List

import feedparser

from connectors.base import BaseConnector


class NewsRSSConnector(BaseConnector):
    """Simple RSS news connector. Supports backfill vs incremental.

    Configuration:
      - Provide a list of feed URLs on `self.feeds` or override in subclass.
    """

    source_code = "PUNCH_RSS"

    def __init__(self, feeds: List[str] | None = None):
        super().__init__()
        self.feeds = feeds or [
            "https://punchng.com/feed/",
        ]

    def fetch(self, mode: str = "incremental") -> Iterator[dict]:
        """Yield feed entries.

        - incremental: yield the latest page(s) (default behavior of feedparser)
        - backfill: attempt to yield all entries available from the feeds
        """
        for url in self.feeds:
            parsed = feedparser.parse(url)
            # feedparser returns a list-like `entries`; entries are typically newest-first
            entries = parsed.entries or []
            if mode == "incremental":
                # yield only the first page (all entries present in this fetch)
                for e in entries:
                    yield self._entry_to_record(e, url)
            else:
                # backfill: attempt to yield all entries available in the feed
                # (many feeds only expose a limited history; advanced connectors
                #  should walk archive/pagination endpoints when available)
                for e in entries:
                    yield self._entry_to_record(e, url)

    def _entry_to_record(self, e, feed_url: str) -> dict:
        source_record_id = getattr(e, "id", None) or getattr(e, "guid", None) or getattr(e, "link", None)
        source_url = getattr(e, "link", None)
        title = getattr(e, "title", "")
        summary = getattr(e, "summary", "")
        raw_payload = {k: getattr(e, k) for k in e.keys()} if hasattr(e, "keys") else {}
        raw_text = f"{title}\n\n{summary}"

        return {
            "source_record_id": str(source_record_id),
            "source_url": source_url,
            "raw_payload": raw_payload,
            "raw_text": raw_text,
        }
