from __future__ import annotations

from connectors.social.nairaland_connector import NairalandConnector


SAMPLE_HTML_PAGE1 = """
<html>
<body>
<table>
<tr>
<td><a href="/1/2345678/crime-incident-in-lagos">Crime incident in Lagos</a></td>
<td>User123</td>
</tr>
<tr>
<td><a href="/1/2345679/robbery-in-abuja">Robbery in Abuja</a></td>
<td>User456</td>
</tr>
</table>
</body>
</html>
"""

SAMPLE_HTML_PAGE2 = """
<html>
<body>
<table>
<tr>
<td><a href="/1/2345680/kidnapping-case">Kidnapping case reported</a></td>
<td>User789</td>
</tr>
</table>
</body>
</html>
"""


def test_nairaland_incremental(monkeypatch):
    connector = NairalandConnector()

    monkeypatch.setattr(
        NairalandConnector, "_get_with_retry",
        lambda self, url: type('obj', (object,), {
            'text': SAMPLE_HTML_PAGE1,
            'encoding': 'utf-8'
        })(),
    )

    results = list(connector.fetch(mode="incremental"))
    assert len(results) == 2
    assert any("Lagos" in r["raw_text"] for r in results)
    assert any("Abuja" in r["raw_text"] for r in results)


def test_nairaland_backfill_pagination(monkeypatch):
    connector = NairalandConnector(max_backfill_pages=2)

    call_count = {"n": 0}

    def fake_get(self, url):
        call_count["n"] += 1
        if call_count["n"] == 1:
            html = SAMPLE_HTML_PAGE1
        elif call_count["n"] == 2:
            html = SAMPLE_HTML_PAGE2
        else:
            html = "<html><body></body></html>"
        
        return type('obj', (object,), {
            'text': html,
            'encoding': 'utf-8'
        })()

    monkeypatch.setattr(NairalandConnector, "_get_with_retry", fake_get)

    results = list(connector.fetch(mode="backfill"))
    assert len(results) == 3
    assert call_count["n"] == 2
    assert any("Lagos" in r["raw_text"] for r in results)
    assert any("kidnapping" in r["raw_text"].lower() for r in results)
