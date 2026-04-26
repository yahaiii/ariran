"""
Promote records from raw staging into canonical incidents.

Current status: scaffold entrypoint only.
"""

from sqlalchemy import text

from db.connection import get_session


def main() -> int:
    with get_session() as session:
        pending = session.execute(
            text("SELECT COUNT(*) FROM raw.incidents_staging WHERE processing_status IN ('pending', 'enriched')")
        ).scalar()

    print(f"Promotion scaffold ready. Pending/enriched staging rows: {pending}")
    print("Full promotion logic is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
