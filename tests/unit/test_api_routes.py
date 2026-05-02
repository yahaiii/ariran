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
    def __init__(self):
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((str(sql), params))
        if "v_hotspots_lga" in str(sql):
            return _Result(
                [
                    {
                        "state_name": "Lagos",
                        "lga_name": "Ikeja",
                        "admin_boundary_id": 7,
                        "incident_count": 12,
                        "total_fatalities": 3,
                        "total_injuries": 1,
                        "avg_confidence": 0.82,
                        "crime_categories": ["robbery"],
                        "latest_incident_date": date(2024, 1, 3),
                    }
                ]
            )
        if "/{incident_id}" in str(sql):
            return _Result(
                [
                    {
                        "id": "incident-1",
                        "canonical_id": "canonical-1",
                        "crime_category": "robbery",
                        "crime_type": "armed robbery",
                        "incident_title": "Test incident",
                        "event_date": date(2024, 1, 2),
                        "state_name": "Lagos",
                        "lga_name": "Ikeja",
                        "location_description": "Ikeja axis",
                        "geometry": {"type": "Point", "coordinates": [3.3, 6.6]},
                        "fatalities": 1,
                        "injuries": 0,
                        "confidence_score": 0.9,
                        "primary_source": "NBS",
                    }
                ]
            )
        return _Result(
            [
                {
                    "id": "incident-1",
                    "canonical_id": "canonical-1",
                    "crime_category": "robbery",
                    "crime_type": "armed robbery",
                    "incident_title": "Test incident",
                    "event_date": date(2024, 1, 2),
                    "state_name": "Lagos",
                    "lga_name": "Ikeja",
                    "location_description": "Ikeja axis",
                    "geometry": {"type": "Point", "coordinates": [3.3, 6.6]},
                    "fatalities": 1,
                    "injuries": 0,
                    "confidence_score": 0.9,
                    "primary_source": "NBS",
                }
            ]
        )


@contextmanager
def _session_context():
    yield _Session()


def test_incidents_routes_return_view_backed_shapes() -> None:
    def override_session():
        with _session_context() as session:
            yield session

    app.dependency_overrides[incidents_module.get_db_session] = override_session

    try:
        client = TestClient(app)

        list_response = client.get("/incidents/")
        detail_response = client.get("/incidents/incident-1")
        hotspots_response = client.get("/incidents/hotspots/lga")

        assert list_response.status_code == 200
        assert list_response.json()["count"] == 1
        assert list_response.json()["items"][0]["id"] == "incident-1"
        assert list_response.json()["items"][0]["event_date"] == "2024-01-02"
        assert detail_response.status_code == 200
        assert detail_response.json()["primary_source"] == "NBS"
        assert hotspots_response.status_code == 200
        assert hotspots_response.json()["items"][0]["incident_count"] == 12
    finally:
        app.dependency_overrides.clear()
