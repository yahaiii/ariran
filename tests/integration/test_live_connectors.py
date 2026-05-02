from __future__ import annotations

from collections.abc import Iterable

import pytest

from connectors.acled.connector import ACLEDConnector
from connectors.news.connector import NewsRSSConnector
from connectors.nbs.connector import NBSConnector
from connectors.social.nairaland_connector import NairalandConnector
from connectors.social.twitter_connector import TwitterConnector
from config.settings import settings


def _assert_record_shape(records: Iterable[dict]) -> None:
    for record in records:
        assert record["source_record_id"]
        assert record["raw_text"] is not None
        assert "raw_payload" in record


@pytest.mark.integration
@pytest.mark.live
def test_live_acled_incremental_fetch() -> None:
    if not settings.acled_email or not settings.acled_password:
        pytest.skip("ACLED credentials are required for live ACLED tests")

    connector = ACLEDConnector(date_from="2024-04-01", date_to="2024-04-02")
    records = list(connector.fetch(mode="incremental"))

    assert records, "expected at least one live ACLED record"
    _assert_record_shape(records)
    assert all(record["source_record_id"].startswith("ACLED_") for record in records)


@pytest.mark.integration
@pytest.mark.live
def test_live_news_rss_incremental_fetch() -> None:
    connector = NewsRSSConnector()
    records = list(connector.fetch(mode="incremental"))

    assert records, "expected at least one live RSS record"
    _assert_record_shape(records)


@pytest.mark.integration
@pytest.mark.live
def test_live_nairaland_incremental_fetch() -> None:
    connector = NairalandConnector(max_backfill_pages=1)
    records = list(connector.fetch(mode="incremental"))

    if not records:
        pytest.skip("Nairaland returned no parsable live threads")
    _assert_record_shape(records)


@pytest.mark.integration
@pytest.mark.live
def test_live_twitter_incremental_fetch() -> None:
    if not settings.twitter_api_key:
        pytest.skip("TWITTER_API_KEY is required for live Twitter tests")

    connector = TwitterConnector(keywords=["crime"])
    records = list(connector.fetch(mode="incremental"))

    assert records, "expected at least one live Twitter record"
    _assert_record_shape(records)


@pytest.mark.integration
def test_nbs_harness_is_present() -> None:
    connector = NBSConnector()
    assert connector.source_code == "NBS_ANNUAL"