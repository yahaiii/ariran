from __future__ import annotations

import pandas as pd
from connectors.nbs.connector import NBSConnector


def test_find_and_set_header_and_normalise():
    connector = NBSConnector()

    # Simulate an Excel sheet where the header row appears at index 0
    df = pd.DataFrame(
        [
            ["Report", "State", "Total offences", "Property Crime"],
            ["", "Abia", 100, 40],
            ["", "Lagos", 200, 80],
        ]
    )

    df2 = connector._find_and_set_header(df)
    assert df2 is not None
    df3 = connector._normalise_columns(df2)
    assert "state" in df3.columns
    assert "total_offences" in df3.columns


def test_detect_category_columns():
    connector = NBSConnector()
    cols = ["state", "property crime", "offences against persons", "other offences"]
    mapping = connector._detect_category_columns(cols)
    # Expect at least the known categories to be detected
    assert any("property" in v[0] for v in mapping.values())
    assert any("violent_crime" in v[0] or "assault" in v[1] for v in mapping.values())
