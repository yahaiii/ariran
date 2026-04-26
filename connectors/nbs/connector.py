"""
NBS Connector — National Bureau of Statistics Annual Crime Statistics
Source: https://nigerianstat.gov.ng
Format: Excel (.xlsx), annual releases, state-level aggregates

NBS data is structured (not scraped), so no NLP needed.
Records are promoted directly with high confidence.
"""
import re
from pathlib import Path
from typing import Iterator, Any

import pandas as pd
import requests
import structlog

from connectors.base import BaseConnector

log = structlog.get_logger()

# Known NBS crime statistics Excel URLs (extend as new reports are released)
NBS_KNOWN_URLS = [
    {
        "url": "https://nigerianstat.gov.ng/resource/CRIME%20STATISTICS%202017.xlsx",
        "year": 2017,
        "doc_id": "NBS_CRIME_2017",
    },
    # Add new annual URLs here as NBS publishes them
]

# Column name normalisation map (NBS changes column names between years)
COLUMN_ALIASES = {
    "state": ["state", "states", "state name"],
    "total_offences": ["total", "total offences", "total cases", "grand total"],
    "violent_crime": ["violent crime", "offences against person", "against person"],
    "property_crime": ["property crime", "offences against property", "against property"],
    "year": ["year", "period"],
}


class NBSConnector(BaseConnector):
    """
    Ingests NBS annual crime statistics Excel files.
    Each row in the Excel = one state/year aggregate record.
    """

    source_code = "NBS_ANNUAL"

    def __init__(self, local_file: str | None = None, year: int | None = None):
        super().__init__()
        self.local_file = local_file  # optional: path to already-downloaded file
        self.year = year

    def fetch(self) -> Iterator[dict[str, Any]]:
        sources = []

        if self.local_file:
            sources.append({"path": self.local_file, "year": self.year, "doc_id": "LOCAL"})
        else:
            sources = self._download_known_files()

        for src in sources:
            log.info("nbs_processing_file", path=src["path"], year=src["year"])
            yield from self._parse_excel(src["path"], src["year"], src["doc_id"])

    # ------------------------------------------------------------------

    def _download_known_files(self) -> list[dict]:
        results = []
        for entry in NBS_KNOWN_URLS:
            try:
                resp = self._get_with_retry(entry["url"])
                path = Path(f"/tmp/nbs_{entry['year']}.xlsx")
                path.write_bytes(resp.content)
                results.append({"path": str(path), **entry})
                log.info("nbs_downloaded", year=entry["year"])
            except Exception as e:
                log.error("nbs_download_failed", url=entry["url"], error=str(e))
        return results

    def _parse_excel(self, path: str, year: int, doc_id: str) -> Iterator[dict]:
        try:
            # Try all sheets; NBS sometimes puts data in non-default sheets
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                df = xl.parse(sheet, header=None)
                df = self._find_and_set_header(df)
                if df is None:
                    continue
                df = self._normalise_columns(df)
                if "state" not in df.columns:
                    continue

                for _, row in df.iterrows():
                    state = str(row.get("state", "")).strip()
                    if not state or state.lower() in ("state", "total", "nigeria", ""):
                        continue

                    record_id = f"{doc_id}_{year}_{state.upper().replace(' ', '_')}"
                    raw_payload = row.to_dict()

                    yield {
                        "source_record_id": record_id,
                        "source_url": next(
                            (e["url"] for e in NBS_KNOWN_URLS if e["year"] == year), None
                        ),
                        "raw_payload": {
                            "year": year,
                            "sheet": sheet,
                            "doc_id": doc_id,
                            **{k: (v if pd.notna(v) else None)
                               for k, v in raw_payload.items()},
                        },
                        "raw_text": (
                            f"NBS {year} crime statistics for {state}. "
                            f"Total offences: {row.get('total_offences', 'unknown')}. "
                            f"Violent crime: {row.get('violent_crime', 'unknown')}. "
                            f"Property crime: {row.get('property_crime', 'unknown')}."
                        ),
                    }

        except Exception as e:
            log.error("nbs_parse_failed", path=path, error=str(e))

    def _find_and_set_header(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Scan rows to find the actual header row (NBS embeds headers mid-sheet)."""
        for i, row in df.iterrows():
            values = [str(v).lower().strip() for v in row.values if pd.notna(v)]
            if any("state" in v for v in values):
                df.columns = df.iloc[i]
                df = df.iloc[i + 1:].reset_index(drop=True)
                return df
        return None

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map variant NBS column names to canonical names."""
        rename = {}
        for canonical, aliases in COLUMN_ALIASES.items():
            for col in df.columns:
                if str(col).lower().strip() in aliases:
                    rename[col] = canonical
                    break
        return df.rename(columns=rename)
