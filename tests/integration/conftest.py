from __future__ import annotations

import os

import pytest


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def live_integration_enabled() -> bool:
    return any(
        _is_truthy(os.getenv(flag))
        for flag in (
            "INTEGRATION_TESTS",
            "ARIRAN_LIVE_INTEGRATION",
            "ACLED_INTEGRATION",
            "LIVE_CONNECTOR_TESTS",
        )
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that exercise the pipeline against real services or a real database",
    )
    config.addinivalue_line(
        "markers",
        "live: tests that hit live external services and require opt-in environment variables",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "live" in item.keywords and not live_integration_enabled():
        pytest.skip("live integration tests are disabled; set INTEGRATION_TESTS=1 to enable them")
