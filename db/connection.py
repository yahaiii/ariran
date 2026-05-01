"""
Database connection pool — single import point for all DB access.
Usage:
    from db.connection import get_engine, get_session
"""
from contextlib import contextmanager

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings

log = structlog.get_logger()

engine = create_engine(
    settings.db_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,          # validates connections before use
    connect_args={"options": "-c search_path=public,raw,audit"},
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_engine():
    return engine


@contextmanager
def get_session() -> Session:
    """Context manager — always commits or rolls back cleanly."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def health_check() -> bool:
    """Returns True if the DB is reachable and PostGIS is installed."""
    try:
        with get_session() as s:
            result = s.execute(text("SELECT PostGIS_Version()")).scalar()
            log.info("db_health_ok", postgis_version=result)
            return True
    except Exception as e:
        log.error("db_health_failed", error=str(e))
        return False
