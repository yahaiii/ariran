"""Integration tests for the promotion pipeline against a real database.

These tests require:
- PostgreSQL 15+ with PostGIS extension
- Database initialized with nigeria_crime_db_schema.sql
- Environment variables: DB_USER, DB_PASSWORD, DB_HOST, DB_NAME, DB_PORT

Run with: pytest tests/integration/ -v --tb=short
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from config.settings import settings
from db.connection import get_session
from pipeline.promote import (
    compute_confidence,
    find_duplicate,
    promote_batch,
    resolve_admin_boundary,
)


@pytest.fixture
def session_fixture():
    """Provide a database session for integration tests."""
    with get_session() as session:
        yield session


@pytest.fixture
def setup_test_data(session_fixture):
    """Set up test data in the database for integration tests."""
    session = session_fixture

    # Verify database connectivity
    result = session.execute(text("SELECT PostGIS_Version()")).scalar()
    assert result is not None, "PostGIS not available"

    # Verify schema exists
    result = session.execute(
        text(
            "SELECT EXISTS("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_schema='public' AND table_name='incidents'"
            ")"
        )
    ).scalar()
    assert result, "incidents table not found - schema not initialized"

    # Create test pipeline run
    run_id = uuid4()
    source_id = 1  # Assume NBS_ANNUAL source exists from seed data

    session.execute(
        text(
            "INSERT INTO public.pipeline_runs "
            "(id, source_id, status) VALUES (:run_id, :source_id, 'running')"
        ),
        {"run_id": run_id, "source_id": source_id},
    )

    # Create test staging records
    staging_records = [
        {
            "id": uuid4(),
            "run_id": run_id,
            "source_id": source_id,
            "source_record_id": "TEST_001",
            "raw_payload": json.dumps(
                {
                    "date": "2024-01-15",
                    "location": "Lagos, Ikeja",
                    "description": "Armed robbery at bank",
                    "fatalities": 2,
                }
            ),
            "raw_text": "Armed robbery at bank in Lagos",
            "nlp_crime_type": "armed_robbery",
            "nlp_location_raw": "Ikeja",
            "nlp_date_raw": "2024-01-15",
            "nlp_confidence": 0.85,
        },
        {
            "id": uuid4(),
            "run_id": run_id,
            "source_id": source_id,
            "source_record_id": "TEST_002",
            "raw_payload": json.dumps(
                {
                    "date": "2024-01-16",
                    "location": "Abuja",
                    "description": "Kidnapping incident",
                    "victims": 1,
                }
            ),
            "raw_text": "Kidnapping in Abuja",
            "nlp_crime_type": "kidnapping",
            "nlp_location_raw": "Abuja",
            "nlp_date_raw": "2024-01-16",
            "nlp_confidence": 0.75,
        },
    ]

    for rec in staging_records:
        session.execute(
            text(
                "INSERT INTO raw.incidents_staging "
                "(id, pipeline_run_id, source_id, source_record_id, "
                " raw_payload, raw_text, nlp_crime_type, nlp_location_raw, "
                " nlp_date_raw, nlp_confidence, processing_status) "
                "VALUES (:id, :run_id, :source_id, :source_record_id, "
                "        :payload, :text, :crime_type, :loc, :date, "
                "        :confidence, 'enriched')"
            ),
            {
                "id": rec["id"],
                "run_id": rec["run_id"],
                "source_id": rec["source_id"],
                "source_record_id": rec["source_record_id"],
                "payload": rec["raw_payload"],
                "text": rec["raw_text"],
                "crime_type": rec["nlp_crime_type"],
                "loc": rec["nlp_location_raw"],
                "date": rec["nlp_date_raw"],
                "confidence": rec["nlp_confidence"],
            },
        )

    session.commit()

    return {
        "run_id": run_id,
        "source_id": source_id,
        "staging_ids": [r["id"] for r in staging_records],
    }


class TestDatabaseConnectivity:
    """Verify database is available and configured."""

    def test_db_connection_from_settings(self):
        """Verify database URL is configured."""
        assert settings.db_url is not None
        assert "postgresql" in settings.db_url

    def test_postgis_available(self, session_fixture):
        """Verify PostGIS extension is installed."""
        result = session_fixture.execute(text("SELECT PostGIS_Version()")).scalar()
        assert result is not None
        assert "POSTGIS" in str(result).upper()

    def test_schema_initialized(self, session_fixture):
        """Verify core schema tables exist."""
        session = session_fixture
        tables = [
            "public.incidents",
            "public.sources",
            "public.admin_boundaries",
            "raw.incidents_staging",
            "audit.incident_changes",
        ]

        for table in tables:
            schema, name = table.split(".")
            result = session.execute(
                text(
                    "SELECT EXISTS("
                    "  SELECT 1 FROM information_schema.tables "
                    "  WHERE table_schema=:schema AND table_name=:name"
                    ")"
                ),
                {"schema": schema, "name": name},
            ).scalar()
            assert result, f"Table {table} not found"


class TestResolveAdminBoundary:
    """Test LGA/state boundary resolution."""

    def test_resolve_lga_boundary(self, session_fixture):
        """Verify resolution of LGA-level boundaries."""
        session = session_fixture

        # Insert test boundary
        session.execute(
            text(
                "INSERT INTO public.admin_boundaries "
                "(admin_level, name, state_name, lga_name, geom) "
                "VALUES (2, 'Ikeja', 'Lagos', 'Ikeja', "
                "        ST_GeomFromText("
                "          'POLYGON((3.42 6.59, 3.44 6.59, 3.44 6.61, "
                "                    3.42 6.61, 3.42 6.59))', 4326))"
            )
        )
        session.commit()

        # Test resolution
        result = resolve_admin_boundary(session, "Lagos", "Ikeja")
        assert result is not None
        boundary_id, precision = result
        assert boundary_id is not None
        assert precision == "lga"

    def test_resolve_state_fallback(self, session_fixture):
        """Verify fallback to state when LGA not found."""
        session = session_fixture

        # Insert state boundary
        session.execute(
            text(
                "INSERT INTO public.admin_boundaries "
                "(admin_level, name, state_name, geom) "
                "VALUES (1, 'Lagos', 'Lagos', "
                "        ST_GeomFromText("
                "          'POLYGON((2.0 6.0, 5.0 6.0, 5.0 7.0, "
                "                    2.0 7.0, 2.0 6.0))', 4326))"
            )
        )
        session.commit()

        # Test fallback resolution (unknown LGA)
        result = resolve_admin_boundary(session, "Lagos", "Unknown_LGA")
        assert result is not None
        boundary_id, precision = result
        assert boundary_id is not None
        assert precision == "state"

    def test_resolve_unknown_boundary(self, session_fixture):
        """Verify handling of unrecognized boundaries."""
        session = session_fixture

        result = resolve_admin_boundary(session, "Nonexistent", "Fake_LGA")
        assert result is not None
        boundary_id, precision = result
        assert boundary_id is None
        assert precision == "unknown"


class TestComputeConfidence:
    """Test confidence score computation."""

    def test_compute_confidence_high_precision(self, session_fixture):
        """Verify confidence boost with exact precision."""
        session = session_fixture

        score = compute_confidence(
            session,
            source_id=1,
            nlp_confidence=0.85,
            geo_precision="coordinates",
            date_precision="exact",
            corroborating=1,
        )
        assert score is not None
        assert 0.0 <= score <= 1.0
        # High precision should yield high score
        assert score > 0.7

    def test_compute_confidence_low_precision(self, session_fixture):
        """Verify confidence with imprecise geo/date."""
        session = session_fixture

        score = compute_confidence(
            session,
            source_id=1,
            nlp_confidence=0.5,
            geo_precision="state",
            date_precision="year",
            corroborating=0,
        )
        assert score is not None
        assert 0.0 <= score <= 1.0
        # Low precision should yield moderate/low score
        assert score < 0.7


class TestPromoteWorkflow:
    """Test end-to-end promotion workflow."""

    def test_promote_batch_success(self, session_fixture, setup_test_data):
        """Verify successful batch promotion."""
        session = session_fixture
        test_data = setup_test_data

        # Get initial count
        before = session.execute(
            text("SELECT COUNT(*) FROM public.incidents WHERE is_deleted = FALSE")
        ).scalar()

        # Run promotion (promote_batch gets its own session internally)
        stats = promote_batch(dry_run=False)

        # Refresh session to see committed changes
        session.execute(text("ROLLBACK"))  # Clear any pending transaction
        session.execute(text("BEGIN"))

        # Verify incidents created
        after = session.execute(
            text("SELECT COUNT(*) FROM public.incidents WHERE is_deleted = FALSE")
        ).scalar()

        assert after > before, "No incidents were promoted"
        assert stats["promoted"] > 0
        assert (
            stats["promoted"] + stats["skipped"] + stats["failed"]
        ) > 0

    def test_promote_batch_dry_run(self, session_fixture, setup_test_data):
        """Verify dry-run doesn't commit changes."""
        session = session_fixture
        test_data = setup_test_data

        # Get initial count
        before = session.execute(
            text("SELECT COUNT(*) FROM public.incidents WHERE is_deleted = FALSE")
        ).scalar()

        # Run dry-run promotion
        stats = promote_batch(dry_run=True)

        # Verify no changes committed
        after = session.execute(
            text("SELECT COUNT(*) FROM public.incidents WHERE is_deleted = FALSE")
        ).scalar()

        assert after == before, "Dry-run should not commit"
        assert isinstance(stats, dict)
        assert "promoted" in stats


class TestAuditTrail:
    """Test audit logging functionality."""

    def test_incident_audit_logged_on_insert(self, session_fixture, setup_test_data):
        """Verify insert is logged to audit table."""
        session = session_fixture
        test_data = setup_test_data

        # Create and promote an incident
        promote_batch(dry_run=False)

        # Refresh to see committed changes
        session.execute(text("ROLLBACK"))
        session.execute(text("BEGIN"))

        # Verify audit entry was created
        audit_count = session.execute(
            text(
                "SELECT COUNT(*) FROM audit.incident_changes "
                "WHERE change_type = 'insert'"
            )
        ).scalar()

        assert audit_count > 0, "Insert audit entry not found"


class TestBoundaryLoader:
    """Test boundary loading functionality."""

    @pytest.mark.skip(reason="Requires external boundary file")
    def test_load_geojson_boundaries(self, session_fixture):
        """Verify GeoJSON boundary loading (requires test file)."""
        # This test requires a test GeoJSON file
        # Skip for now as integration test
        pass


@pytest.fixture(scope="session", autouse=True)
def check_db_availability():
    """Skip all integration tests if database is unavailable."""
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"Database not available: {e}")
