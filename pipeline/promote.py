"""
Ariran Promotion Pipeline
Moves enriched records from raw.incidents_staging → public.incidents.

Flow per record:
  1. Pull pending/enriched staging records in batches
  2. Attempt geocoding (admin_boundary lookup)
  3. Compute confidence score
  4. Dedup check — assign canonical_id if match found
  5. Insert into public.incidents
  6. Mark staging record as promoted
"""
import uuid
from datetime import date
from typing import Optional

import structlog
from sqlalchemy import text

from db.connection import get_session
from config.settings import settings

log = structlog.get_logger()

BATCH_SIZE = settings.staging_batch_size


# ── Helpers ────────────────────────────────────────────────────────────────────

def resolve_admin_boundary(session, state: Optional[str], lga: Optional[str]) -> tuple:
    """
    Returns (admin_boundary_id, geo_precision) by matching state/LGA
    against public.admin_boundaries.
    Falls back gracefully: LGA → state → None.
    """
    if lga:
        row = session.execute(text("""
            SELECT id FROM public.admin_boundaries
            WHERE admin_level = 2
              AND LOWER(lga_name) = LOWER(:lga)
              AND (:state IS NULL OR LOWER(state_name) = LOWER(:state))
            LIMIT 1
        """), {"lga": lga, "state": state}).fetchone()
        if row:
            return row[0], "lga"

    if state:
        row = session.execute(text("""
            SELECT id FROM public.admin_boundaries
            WHERE admin_level = 1
              AND LOWER(state_name) = LOWER(:state)
            LIMIT 1
        """), {"state": state}).fetchone()
        if row:
            return row[0], "state"

    return None, "unknown"


def find_duplicate(session, event_date: Optional[date], admin_boundary_id: Optional[int],
                   crime_type: Optional[str]) -> Optional[uuid.UUID]:
    """
    Simple dedup: same date + same LGA + same crime_type = likely duplicate.
    Returns canonical_id of existing record if found, else None.
    """
    if not event_date or not admin_boundary_id or not crime_type:
        return None

    row = session.execute(text("""
        SELECT canonical_id, id
        FROM public.incidents
        WHERE event_date = :dt
          AND admin_boundary_id = :ab
          AND crime_type = LOWER(:ct)
          AND is_deleted = FALSE
          AND is_canonical = TRUE
        LIMIT 1
    """), {"dt": event_date, "ab": admin_boundary_id, "ct": crime_type}).fetchone()

    if row:
        return row[0] or row[1]   # use canonical_id if set, else the matching record's id
    return None


def compute_confidence(session, source_id: int, nlp_confidence: Optional[float],
                       geo_precision: str, date_precision: str,
                       corroborating: int) -> float:
    """Calls the SQL function defined in the schema."""
    src = session.execute(text(
        "SELECT default_confidence FROM public.sources WHERE id = :id"
    ), {"id": source_id}).scalar() or 0.5

    result = session.execute(text("""
        SELECT public.compute_confidence(
            :source_conf, :nlp_conf, :geo_prec, :date_prec, :corr
        )
    """), {
        "source_conf": float(src),
        "nlp_conf": float(nlp_confidence) if nlp_confidence else 0.5,
        "geo_prec": geo_precision,
        "date_prec": date_precision,
        "corr": corroborating,
    }).scalar()

    return float(result)


# ── Main promotion logic ────────────────────────────────────────────────────────

def promote_batch(dry_run: bool = False) -> dict:
    """
    Promotes one batch of staging records to public.incidents.
    Returns counters: promoted, skipped, failed.
    """
    counters = {"promoted": 0, "skipped": 0, "failed": 0}

    with get_session() as session:
        # Pull pending batch
        rows = session.execute(text("""
            SELECT
                s.id, s.source_id, s.pipeline_run_id,
                s.source_record_id, s.source_url,
                s.raw_payload, s.raw_text,
                s.nlp_crime_type, s.nlp_location_raw, s.nlp_date_raw,
                s.nlp_actors, s.nlp_fatalities, s.nlp_injuries,
                s.nlp_confidence, s.geocoded_geom,
                s.geocoded_admin_id, s.geocode_method
            FROM raw.incidents_staging s
            WHERE s.processing_status IN ('pending', 'enriched')
            ORDER BY s.ingested_at ASC
            LIMIT :batch
        """), {"batch": BATCH_SIZE}).fetchall()

        if not rows:
            log.info("promote_nothing_pending")
            return counters

        log.info("promote_batch_start", count=len(rows))

        for row in rows:
            staging_id = row[0]
            try:
                source_id       = row[1]
                run_id          = row[2]
                source_record_id= row[3]
                source_url      = row[4]
                raw_payload     = row[5] or {}
                nlp_crime_type  = row[7]
                nlp_location    = row[8]
                nlp_date_raw    = row[9]
                nlp_actors      = row[10] or []
                nlp_fatalities  = row[11]
                nlp_injuries    = row[12]
                nlp_confidence  = row[13]
                geocoded_geom   = row[14]
                geocoded_admin_id = row[15]

                # ── Resolve location ───────────────────────────────────────
                # Prefer geocoded result from enrichment; fall back to raw_payload fields
                state = (raw_payload.get("state")
                        or raw_payload.get("admin1")
                        or raw_payload.get("state_name"))
                lga   = raw_payload.get("admin2") or raw_payload.get("lga_name")

                


                if geocoded_admin_id:
                    admin_boundary_id = geocoded_admin_id
                    geo_precision = "lga"
                else:
                    admin_boundary_id, geo_precision = resolve_admin_boundary(
                        session, state, lga
                    )
                if raw_payload.get("doc_id", "").startswith("NBS"):
                    geo_precision = "state"
                

                # ── Resolve date ───────────────────────────────────────────
                event_date = None
                date_precision = "unknown"

                raw_date = (raw_payload.get("event_date")
                            or raw_payload.get("date")
                            or nlp_date_raw)
                if raw_date:
                    try:
                        from dateutil import parser as dateparser
                        event_date = dateparser.parse(str(raw_date)).date()
                        date_precision = "day"
                    except Exception:
                        pass

                # NBS records have year only
                if not event_date and raw_payload.get("year"):
                    try:
                        event_date = date(int(raw_payload["year"]), 1, 1)
                        date_precision = "year"
                    except Exception:
                        pass

                # ── Crime classification ───────────────────────────────────
                crime_type = (
                    nlp_crime_type
                    or raw_payload.get("crime_type")
                    or raw_payload.get("event_type")
                    or "unknown"
                ).lower().replace(" ", "_")

                crime_category = (
                    raw_payload.get("crime_category")
                    or raw_payload.get("event_type")
                    or "unknown"
                )
                # ── Dedup ─────────────────────────────────────────────────
                existing_canonical = find_duplicate(
                    session, event_date, admin_boundary_id, crime_type
                )
                is_canonical = existing_canonical is None
                canonical_id = existing_canonical or uuid.uuid4()
                corroborating = 0 if is_canonical else 1

                # ── Confidence ────────────────────────────────────────────
                confidence = compute_confidence(
                    session, source_id, nlp_confidence,
                    geo_precision, date_precision, corroborating
                )

                # ── Build incident ─────────────────────────────────────────
                incident_id = uuid.uuid4()
                narrative = (
                    raw_payload.get("notes")
                    or raw_payload.get("narrative")
                    or row[6]   # raw_text
                    or ""
                )[:2000]

                if not dry_run:
                    session.execute(text("""
                        INSERT INTO public.incidents (
                            id, canonical_id, is_canonical,
                            crime_category, crime_type,
                            event_date, date_precision,
                            state_name, lga_name,
                            admin_boundary_id,
                            geo_precision,
                            fatalities, injuries,
                            perpetrator_groups,
                            narrative,
                            primary_source_id,
                            source_record_ids, source_urls, staging_ids,
                            corroborating_sources,
                            confidence_score, verification_status,
                            created_by_run
                        ) VALUES (
                            :id, :canonical_id, :is_canonical,
                            :crime_category, :crime_type,
                            :event_date, :date_precision,
                            :state_name, :lga_name,
                            :admin_boundary_id,
                            :geo_precision,
                            :fatalities, :injuries,
                            :perp_groups,
                            :narrative,
                            :source_id,
                            ARRAY[:source_record_id], ARRAY[:source_url], ARRAY[:staging_id]::uuid[],
                            :corroborating,
                            :confidence, 'unverified',
                            :run_id
                        )
                    """), {
                        "id": str(incident_id),
                        "canonical_id": str(canonical_id),
                        "is_canonical": is_canonical,
                        "crime_category": crime_category,
                        "crime_type": crime_type,
                        "event_date": event_date,
                        "date_precision": date_precision,
                        "state_name": state,
                        "lga_name": lga,
                        "admin_boundary_id": admin_boundary_id,
                        "geo_precision": geo_precision,
                        "fatalities": nlp_fatalities or raw_payload.get("fatalities"),
                        "injuries": nlp_injuries or raw_payload.get("injuries"),
                        "perp_groups": nlp_actors or [],
                        "narrative": narrative,
                        "source_id": source_id,
                        "source_record_id": str(source_record_id or ""),
                        "source_url": source_url or "",
                        "staging_id": str(staging_id),
                        "corroborating": corroborating,
                        "confidence": confidence,
                        "run_id": str(run_id) if run_id else None,
                    })

                    # Mark staging record as promoted
                    session.execute(text("""
                        UPDATE raw.incidents_staging
                        SET processing_status = 'promoted',
                            promoted_incident_id = :inc_id
                        WHERE id = :staging_id
                    """), {
                        "inc_id": str(incident_id),
                        "staging_id": str(staging_id),
                    })

                counters["promoted"] += 1
                log.info("promoted", incident_id=str(incident_id),
                         crime_type=crime_type, confidence=confidence,
                         geo_precision=geo_precision, date_precision=date_precision)

            except Exception as e:
                counters["failed"] += 1
                log.error("promote_failed", staging_id=str(staging_id), error=str(e))
                session.execute(text("""
                    UPDATE raw.incidents_staging
                    SET processing_status = 'rejected',
                        rejection_reason = :reason
                    WHERE id = :id
                """), {"reason": str(e)[:500], "id": str(staging_id)})

    log.info("promote_batch_done", **counters)
    return counters


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    result = promote_batch(dry_run=dry_run)
    print(result)