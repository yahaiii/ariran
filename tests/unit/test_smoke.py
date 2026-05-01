"""Minimal unit smoke tests to ensure test discovery works in CI."""

from config.settings import settings


def test_settings_smoke() -> None:
    """Basic assertion that project settings load with expected defaults."""
    assert settings.staging_batch_size > 0
