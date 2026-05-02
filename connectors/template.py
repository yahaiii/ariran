from __future__ import annotations

from typing import Iterator

from connectors.base import BaseConnector


class TemplateConnector(BaseConnector):
    """Connector scaffold. Copy this file to create a new connector package.

    Usage:
        - Create `connectors/your_source/connector.py`
        - Set `source_code` to the value in `public.sources.source_code`
        - Implement `fetch(self, mode: str = "incremental")`
    """

    source_code = "YOUR_SOURCE_CODE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fetch(self, mode: str = "incremental") -> Iterator[dict]:
        # Example: yield a single dummy record
        yield {
            "source_record_id": "example-1",
            "source_url": None,
            "raw_payload": {"example": True},
            "raw_text": "Example record text",
        }
