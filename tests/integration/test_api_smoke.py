from __future__ import annotations

from contextlib import contextmanager
from datetime import date

from fastapi.testclient import TestClient

from api.app import app
import api.routes.incidents as incidents_module


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class _Session:
    def execute(self, sql, params):
        statement = str(sql)
        if "v_hotspots_lga" in statement:
            return _Result([])
        if "WHERE id = :id" in statement:
            return _Result(
                [
                    {
                        "id": "incident-1",
                        "canonical_id": None,
                        "crime_category": "robbery",
                        "crime_type": "armed robbery",
                        "incident_title": "Smoke test incident",
                        "event_date": date(2024, 1, 2),
                        "state_name": "Lagos",
                        "lga_name": "Ikeja",
                        "location_description": "Ikeja axis",
                        "geometry": None,
                        "fatalities": 0,
                        "injuries": 0,
                        "confidence_score": 0.5,
                        "primary_source": "NBS",
                    }
                ]
            )
        return _Result(
            [
                {
                    "id": "incident-1",
                    "canonical_id": None,
                    "crime_category": "robbery",
                    "crime_type": "armed robbery",
                    "incident_title": "Smoke test incident",
                    "event_date": date(2024, 1, 2),
                    "state_name": "Lagos",
                    "lga_name": "Ikeja",
                    "location_description": "Ikeja axis",
                    "geometry": None,
                    "fatalities": 0,
                    "injuries": 0,
                    "confidence_score": 0.5,
                    "primary_source": "NBS",
                }
            ]
        )


@contextmanager
def _session_context():
    yield _Session()


def test_api_smoke_routes_are_exposed() -> None:
    def override_session():
        with _session_context() as session:
            yield session

    app.dependency_overrides[incidents_module.get_db_session] = override_session

    try:
        client = TestClient(app)
        schema = client.get("/openapi.json")

        assert schema.status_code == 200
        paths = schema.json()["paths"]
        assert "/incidents/" in paths
        assert "/incidents/{incident_id}" in paths
        assert "/incidents/hotspots/lga" in paths
    finally:
        app.dependency_overrides.clear()
