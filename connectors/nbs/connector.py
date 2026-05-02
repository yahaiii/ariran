"""
NBS Connector — National Bureau of Statistics Annual Crime Statistics
Source: https://nigerianstat.gov.ng
Format: Excel (.xlsx), annual releases, state-level aggregates

Expansion strategy: one staging record per state per crime category.
e.g. Abia 2017 → 4 rows: property_crime, violent_crime, lawful_authority, other
This makes NBS data consistent with event-level sources for filtering and mapping.
"""
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import structlog

from connectors.base import BaseConnector

log = structlog.get_logger()


class KnownNBSFile(TypedDict):
    url: str
    year: int
    doc_id: str


class NBSSource(TypedDict):
    path: str
    year: int
    doc_id: str
    url: str | None


NBS_KNOWN_URLS: list[KnownNBSFile] = [
    {
        "url": "https://nigerianstat.gov.ng/resource/CRIME%20STATISTICS%202017.xlsx",
        "year": 2017,
        "doc_id": "NBS_CRIME_2017",
    },
]

# Map raw NBS column names → canonical crime category + ariran crime_type
CATEGORY_MAP = {
    "property_crime":                       ("property_crime",  "property_crime"),
    "offences against property":            ("property_crime",  "property_crime"),
    "offences against persons":             ("violent_crime",   "assault_and_violence"),
    "offences against person":              ("violent_crime",   "assault_and_violence"),
    "offences against lawful authority":    ("financial_crime", "offences_against_authority"),
    "offences agains lawful authority":     ("financial_crime", "offences_against_authority"),
    "other offences":                       ("other",           "other"),
    "miscellaneous":                        ("other",           "other"),
}

COLUMN_ALIASES = {
    "state":            ["state", "states", "state name"],
    "total_offences":   ["total", "total offences", "total cases", "grand total"],
    "year":             ["year", "period"],
}


class NBSConnector(BaseConnector):
    source_code = "NBS_ANNUAL"

    def __init__(self, local_file: str | None = None, year: int | None = None):
        super().__init__()
        self.local_file = local_file
        self.year = year

    def fetch(self) -> Iterator[dict[str, Any]]:
        sources: list[NBSSource] = []
        if self.local_file:
            if self.year is None:
                raise ValueError("year is required when local_file is provided")
            sources.append(
                {
                    "path": self.local_file,
                    "year": self.year,
                    "doc_id": "LOCAL",
                    "url": None,
                }
            )
        else:
            sources = self._download_known_files()

        for src in sources:
            log.info("nbs_processing_file", path=src["path"], year=src["year"])
            yield from self._parse_excel(src["path"], src["year"], src["doc_id"])

    def _download_known_files(self) -> list[NBSSource]:
        results: list[NBSSource] = []
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
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                df = xl.parse(sheet, header=None)
                df = self._find_and_set_header(df)
                if df is None:
                    continue
                df = self._normalise_columns(df)
                if "state" not in df.columns:
                    continue

                # Identify crime category columns in this sheet
                category_cols = self._detect_category_columns(df.columns)
                if not category_cols:
                    log.warning("nbs_no_category_cols", sheet=sheet)
                    continue

                for _, row in df.iterrows():
                    state = str(row.get("state", "")).strip()
                    if not state or state.lower() in ("state", "total", "nigeria", ""):
                        continue

                    total = row.get("total_offences")

                    # Expand: one record per crime category
                    for raw_col, (crime_category, crime_type) in category_cols.items():
                        count = row.get(raw_col)
                        if pd.isna(count) or count is None:
                            continue

                        try:
                            count = int(count)
                        except (ValueError, TypeError):
                            continue

                        if count <= 0:
                            continue

                        record_id = (
                            f"{doc_id}_{year}_{state.upper().replace(' ', '_')}"
                            f"_{crime_type.upper()}"
                        )

                        yield {
                            "source_record_id": record_id,
                            "source_url": next(
                                (e["url"] for e in NBS_KNOWN_URLS if e["year"] == year),
                                None
                            ),
                            "raw_payload": {
                                "year": year,
                                "sheet": sheet,
                                "doc_id": doc_id,
                                "state": state,
                                "crime_category": crime_category,
                                "crime_type": crime_type,
                                "offence_count": count,
                                "total_offences": int(total) if pd.notna(total) else None,
                            },
                            "raw_text": (
                                f"NBS {year} statistics for {state}: "
                                f"{count} {crime_type.replace('_', ' ')} offences recorded."
                            ),
                        }

        except Exception as e:
            log.error("nbs_parse_failed", path=path, error=str(e))

    def _detect_category_columns(self, columns) -> dict:
        """Map actual DataFrame column names to (crime_category, crime_type) tuples."""
        result = {}
        for col in columns:
            normalised = str(col).lower().strip()
            # Remove trailing punctuation/spaces
            normalised = re.sub(r'[^a-z\s]', '', normalised).strip()
            # produce common variants to match CATEGORY_MAP keys which use
            # a mix of spaces and underscores
            norm_space = normalised
            norm_underscore = normalised.replace(' ', '_')
            if norm_underscore in CATEGORY_MAP:
                result[col] = CATEGORY_MAP[norm_underscore]
            elif norm_space in CATEGORY_MAP:
                result[col] = CATEGORY_MAP[norm_space]
        return result

    def _find_and_set_header(self, df: pd.DataFrame) -> pd.DataFrame | None:
        for i, row in df.iterrows():
            values = [str(v).lower().strip() for v in row.values if pd.notna(v)]
            if any("state" in v for v in values):
                df.columns = df.iloc[i]
                df = df.iloc[i + 1:].reset_index(drop=True)
                return df
        return None

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for canonical, aliases in COLUMN_ALIASES.items():
            for col in df.columns:
                if str(col).lower().strip() in aliases:
                    rename[col] = canonical
                    break
        return df.rename(columns=rename)


if __name__ == "__main__":
    connector = NBSConnector()
    result = connector.run()
    print(f"NBS connector run complete: {result}")
