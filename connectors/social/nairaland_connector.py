from __future__ import annotations

from typing import Iterator, Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import structlog

from connectors.base import BaseConnector

log = structlog.get_logger()


class NairalandConnector(BaseConnector):
    """Nairaland crime section scraper.

    Scrapes threads from the Nairaland Crime section.
    Incremental: first page (recent threads). Backfill: paginate through all pages.
    URL: https://www.nairaland.com/crime
    """

    source_code = "NAIRALAND_CRIME"
    BASE_URL = "https://www.nairaland.com/crime"

    def __init__(self, max_backfill_pages: int = 50):
        super().__init__()
        self.max_backfill_pages = max_backfill_pages

    def fetch(self, mode: str = "incremental") -> Iterator[dict[str, Any]]:
        if mode == "incremental":
            pages = [1]  # Just first page (recent)
        else:
            # Backfill: paginate through multiple pages of historic threads
            pages = list(range(1, self.max_backfill_pages + 1))

        for page_num in pages:
            url = f"{self.BASE_URL}/{page_num}" if page_num > 1 else self.BASE_URL
            log.info("nairaland_fetch_page", page=page_num, url=url)

            try:
                resp = self._get_with_retry(url)
                resp.encoding = "utf-8"
                html = resp.text
                yield from self._parse_threads(html, page_num)
            except Exception as e:
                log.error("nairaland_page_failed", page=page_num, error=str(e))
                if mode == "incremental":
                    break  # Stop on first error in incremental

    def _parse_threads(self, html: str, page_num: int) -> Iterator[dict]:
        """Extract thread titles and links from Nairaland page."""
        soup = BeautifulSoup(html, "html.parser")

        # Nairaland thread structure: <tr> with thread info
        for tr in soup.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 2:
                continue

            # Thread link is typically in first cell
            link_elem = cells[0].find("a")
            if not link_elem:
                continue

            thread_url = link_elem.get("href", "")
            thread_title = link_elem.get_text(strip=True)

            if not thread_url or not thread_title:
                continue

            # Make URL absolute
            if not thread_url.startswith("http"):
                thread_url = urljoin(self.BASE_URL, thread_url)

            record_id = f"NAIRALAND_{page_num}_{hash(thread_url) % 10000}"

            yield {
                "source_record_id": record_id,
                "source_url": thread_url,
                "raw_payload": {
                    "title": thread_title,
                    "url": thread_url,
                    "page": page_num,
                },
                "raw_text": thread_title,
            }


if __name__ == "__main__":
    c = NairalandConnector()
    print("Running NairalandConnector (dry run)...")
    summary = c.run(mode="incremental")
    print(summary)
