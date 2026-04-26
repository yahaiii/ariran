"""
ACLED Connector — Armed Conflict Location & Event Data Project
API docs: https://developer.acleddata.com
Coverage: Nigeria conflict/political violence events, 1997–present
Geo precision: LGA level, lat/lon provided by ACLED

ACLED requires free registration for API key + email.
Register at: https://acleddata.com/register/
"""

import argparse
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import structlog

from config.settings import settings
from connectors.base import BaseConnector

log = structlog.get_logger()

ACLED_NIGERIA_ISO = "NIG"
ACLED_PAGE_SIZE = 500  # max allowed by ACLED API


class ACLEDConnector(BaseConnector):
    """
    Polls ACLED API for Nigeria events within a date window.
    Default: last 30 days. For initial load, pass date_from='1997-01-01'.
    """

    source_code = "ACLED_API"

    def __init__(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
    ):
        super().__init__()
        self.date_from = date_from or str(date.today() - timedelta(days=30))
        self.date_to = date_to or str(date.today())

    def fetch(self) -> Iterator[dict[str, Any]]:
        if not settings.acled_api_key or not settings.acled_email:
            raise OSError(
                "ACLED_API_KEY and ACLED_EMAIL must be set in .env. "
                "Register at https://acleddata.com/register/"
            )

        page = 1
        total_fetched = 0

        while True:
            params = {
                "key": settings.acled_api_key,
                "email": settings.acled_email,
                "iso": 566,  # Nigeria ISO 3166-1 numeric
                "event_date": f"{self.date_from}|{self.date_to}",
                "event_date_where": "BETWEEN",
                "limit": ACLED_PAGE_SIZE,
                "page": page,
                "fields": (
                    "event_id_cnty|event_date|event_type|sub_event_type|"
                    "actor1|assoc_actor_1|actor2|assoc_actor_2|"
                    "admin1|admin2|admin3|location|latitude|longitude|"
                    "geo_precision|fatalities|notes|source|source_scale"
                ),
            }

            log.info("acled_fetch_page", page=page, date_from=self.date_from, date_to=self.date_to)

            resp = self._get_with_retry(settings.acled_base_url, params=params)
            data = resp.json()

            if data.get("status") != 200:
                log.error("acled_api_error", response=data)
                break

            events = data.get("data", [])
            if not events:
                break

            for event in events:
                yield self._transform(event)
                total_fetched += 1

            # ACLED paginates; stop when we get fewer than a full page
            if len(events) < ACLED_PAGE_SIZE:
                break

            page += 1

        log.info(
            "acled_fetch_complete",
            total=total_fetched,
            date_from=self.date_from,
            date_to=self.date_to,
        )

    def _transform(self, event: dict) -> dict:
        """Map ACLED event dict to ariran staging record format."""
        event_id = event.get("event_id_cnty", "")

        actors = [
            a
            for a in [
                event.get("actor1"),
                event.get("actor2"),
                event.get("assoc_actor_1"),
                event.get("assoc_actor_2"),
            ]
            if a
        ]

        narrative = event.get("notes", "")
        location_str = ", ".join(
            filter(
                None,
                [
                    event.get("location"),
                    event.get("admin2"),  # LGA
                    event.get("admin1"),  # State
                    "Nigeria",
                ],
            )
        )

        raw_text = (
            f"{event.get('event_type', '')} in {location_str} "
            f"on {event.get('event_date', '')}. "
            f"Actors: {', '.join(actors)}. "
            f"Fatalities: {event.get('fatalities', 0)}. "
            f"{narrative}"
        ).strip()

        return {
            "source_record_id": f"ACLED_{event_id}",
            "source_url": (f"https://acleddata.com/data-export-tool/?event_id={event_id}"),
            "raw_payload": event,
            "raw_text": raw_text,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACLED connector")
    parser.add_argument(
        "--date-from",
        dest="date_from",
        default=None,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--date-to",
        dest="date_to",
        default=None,
        help="End date in YYYY-MM-DD format",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    connector = ACLEDConnector(date_from=args.date_from, date_to=args.date_to)
    summary = connector.run()
    print(f"ACLED connector run complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
