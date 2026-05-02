from __future__ import annotations

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
import time
import os

import requests
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from connectors.base import BaseConnector

log = structlog.get_logger()

ACLED_NIGERIA_ISO = "NIG"
ACLED_PAGE_SIZE = 500  # max allowed by ACLED API
ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"


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

    def fetch(self, mode: str = "backfill") -> Iterator[dict[str, Any]]:
        self._sync_env_credentials()

        use_oauth = self._has_real_oauth_credentials()
        use_key = self._has_real_api_key_credentials()
        if not (use_oauth or use_key):
            # Allow operation without credentials to support unit tests that
            # monkeypatch the connector HTTP helpers. Live runs require creds.
            log.warning("acled_no_credentials", msg="No ACLED credentials found; proceeding (tests/mocks expected)")
        page = 1
        total_fetched = 0

        # Build base params
        base_params = {
            "key": settings.acled_api_key,
            "email": settings.acled_email,
            "iso": 566,  # Nigeria ISO 3166-1 numeric
            "event_date": f"{self.date_from}|{self.date_to}",
            "event_date_where": "BETWEEN",
            "limit": ACLED_PAGE_SIZE,
            "fields": (
                "event_id_cnty|event_date|event_type|sub_event_type|"
                "actor1|assoc_actor_1|actor2|assoc_actor_2|"
                "admin1|admin2|admin3|location|latitude|longitude|"
                "geo_precision|fatalities|notes|source|source_scale"
            ),
        }

        # Incremental: only fetch first page
        if mode == "incremental":
            params = dict(base_params)
            params["page"] = page

            log.info("acled_fetch_page", page=page, date_from=self.date_from, date_to=self.date_to, mode=mode)

            if use_oauth:
                resp = self._get_with_retry_auth(settings.acled_base_url, params=params)
            else:
                resp = self._get_with_retry(settings.acled_base_url, params=params)
            data = resp.json()

            if data.get("status") != 200:
                log.error("acled_api_error", response=data)
                return

            events = data.get("data", [])
            for event in events:
                yield self._transform(event)
                total_fetched += 1

            log.info(
                "acled_fetch_complete",
                total=total_fetched,
                date_from=self.date_from,
                date_to=self.date_to,
                mode=mode,
            )
            return

        # Backfill: iterate pages until fewer than a full page returned
        while True:
            params = dict(base_params)
            params["page"] = page

            log.info("acled_fetch_page", page=page, date_from=self.date_from, date_to=self.date_to, mode=mode)

            if use_oauth:
                resp = self._get_with_retry_auth(settings.acled_base_url, params=params)
            else:
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
            mode=mode,
        )

    def _post_with_retry(self, url: str, data: dict[str, Any]) -> requests.Response:
        @retry(stop=stop_after_attempt(settings.request_retries), wait=wait_exponential(multiplier=1, min=1, max=10))
        def _do():
            headers = {"User-Agent": settings.user_agent}
            resp = requests.post(url, data=data, headers=headers, timeout=settings.request_timeout)
            resp.raise_for_status()
            return resp

        return _do()

    def _get_with_retry_auth(self, url: str, params: dict | None = None) -> requests.Response:
        token = self._get_token()

        @retry(stop=stop_after_attempt(settings.request_retries), wait=wait_exponential(multiplier=1, min=1, max=10))
        def _do():
            headers = {"User-Agent": settings.user_agent, "Authorization": f"Bearer {token}"}
            resp = requests.get(url, params=params, headers=headers, timeout=settings.request_timeout)
            resp.raise_for_status()
            return resp

        return _do()

    def _get_with_retry(self, url: str, params: dict | None = None) -> requests.Response:
        """Override base GET to inject Authorization when ACLED credentials are available.

        Tests monkeypatch `ACLEDConnector._get_with_retry`, so keep this method name
        to preserve test compatibility. If no ACLED password is configured, fall
        back to the base connector's GET helper.
        """
        if settings.acled_email and settings.acled_password:
            return self._get_with_retry_auth(url, params=params)
        # fall back to BaseConnector's implementation
        return super()._get_with_retry(url, params=params)

    def _get_token(self) -> str:
        if getattr(self, "_access_token", None) and getattr(self, "_token_expires_at", 0) > time.time():
            return self._access_token

        payload = {
            "grant_type": "password",
            "client_id": "acled",
            "username": settings.acled_email,
            "password": settings.acled_password,
            "scope": "authenticated",
        }

        resp = self._post_with_retry(ACLED_TOKEN_URL, data=payload)
        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise OSError(f"ACLED token request failed: {data}")

        expires_in = int(data.get("expires_in", 3600))
        self._access_token = token
        self._token_expires_at = time.time() + expires_in - 60
        self._refresh_token = data.get("refresh_token")
        return token

    def _sync_env_credentials(self) -> None:
        email = os.getenv("ACLED_EMAIL")
        password = os.getenv("ACLED_PASSWORD")
        api_key = os.getenv("ACLED_API_KEY")

        if email is not None:
            settings.acled_email = email
        if password is not None:
            settings.acled_password = password
        if api_key is not None:
            settings.acled_api_key = api_key

    @staticmethod
    def _is_placeholder_credential(value: str | None) -> bool:
        if not value:
            return True

        normalized = value.strip().lower()
        placeholder_prefixes = ("your_", "changeme", "replace_me", "placeholder")
        return normalized.startswith(placeholder_prefixes)

    def _has_real_oauth_credentials(self) -> bool:
        return not self._is_placeholder_credential(settings.acled_email) and not self._is_placeholder_credential(settings.acled_password)

    def _has_real_api_key_credentials(self) -> bool:
        return not self._is_placeholder_credential(settings.acled_api_key) and not self._is_placeholder_credential(settings.acled_email)

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
