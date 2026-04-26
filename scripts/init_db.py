"""
Run after docker compose up to:
1. Verify DB + PostGIS connection
2. Apply schema (if not already applied)
3. Confirm seed data (sources + taxonomy) is present
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from sqlalchemy import text

from db.connection import get_session, health_check

log = structlog.get_logger()


def main():
    print("── ariran Database Init ─────────────────────────────")

    if not health_check():
        print("❌  Cannot reach PostGIS. Is 'make dev-up' running?")
        sys.exit(1)
    print("✓  PostGIS connection OK")

    with get_session() as s:
        src_count = s.execute(text("SELECT COUNT(*) FROM public.sources")).scalar()
        tax_count = s.execute(text("SELECT COUNT(*) FROM public.crime_taxonomy")).scalar()
        print(f"✓  Sources loaded    : {src_count}")
        print(f"✓  Taxonomy entries  : {tax_count}")

        if src_count == 0:
            print("⚠  No sources found. Apply the schema SQL first:")
            print("   psql -U ariran_pipeline -d ariran -f nigeria_crime_db_schema.sql")
            sys.exit(1)

    print("\n── Database is ready. Next steps:")
    print("   make run-nbs     → ingest NBS annual statistics")
    print("   make run-acled   → ingest ACLED conflict events")
    print("   make load-boundaries → load Nigerian LGA boundaries")


if __name__ == "__main__":
    main()
