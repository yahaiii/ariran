from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from db.connection import get_session

router = APIRouter()


class IncidentOut(BaseModel):
    id: str
    canonical_id: Optional[str]
    crime_category: Optional[str]
    crime_type: Optional[str]
    incident_title: Optional[str]
    event_date: Optional[date]
    state_name: Optional[str]
    lga_name: Optional[str]
    location_description: Optional[str]
    geometry: Optional[dict]
    fatalities: Optional[int]
    injuries: Optional[int]
    confidence_score: Optional[float]
    primary_source: Optional[str]


class IncidentListResponse(BaseModel):
    items: list[IncidentOut]
    limit: int
    offset: int
    count: int


class HotspotOut(BaseModel):
    state_name: Optional[str]
    lga_name: Optional[str]
    admin_boundary_id: Optional[int]
    incident_count: int
    total_fatalities: Optional[int]
    total_injuries: Optional[int]
    avg_confidence: Optional[float]
    crime_categories: Optional[list[str]]
    latest_incident_date: Optional[date]


def get_db_session() -> Session:
    with get_session() as session:
        yield session


def _incident_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id")),
        "canonical_id": str(row.get("canonical_id")) if row.get("canonical_id") is not None else None,
        "crime_category": row.get("crime_category"),
        "crime_type": row.get("crime_type"),
        "incident_title": row.get("incident_title"),
        "event_date": row.get("event_date"),
        "state_name": row.get("state_name"),
        "lga_name": row.get("lga_name"),
        "location_description": row.get("location_description"),
        "geometry": row.get("geometry"),
        "fatalities": row.get("fatalities"),
        "injuries": row.get("injuries"),
        "confidence_score": float(row.get("confidence_score")) if row.get("confidence_score") is not None else None,
        "primary_source": row.get("primary_source"),
    }


@router.get("/", response_model=IncidentListResponse)
def list_incidents(
    session: Annotated[Session, Depends(get_db_session)],
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    state: Optional[str] = Query(None),
    lga: Optional[str] = Query(None),
    crime_type: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
):
    """List incidents using the `public.v_incidents_clean` view.

    Supports simple filtering by state, lga, crime_type and date range.
    """
    sql = "SELECT * FROM public.v_incidents_clean WHERE 1=1"
    params: dict[str, Any] = {}
    if state:
        sql += " AND LOWER(state_name) = LOWER(:state)"
        params["state"] = state
    if lga:
        sql += " AND LOWER(lga_name) = LOWER(:lga)"
        params["lga"] = lga
    if crime_type:
        sql += " AND LOWER(crime_type) = LOWER(:crime_type)"
        params["crime_type"] = crime_type
    if start_date:
        sql += " AND event_date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        sql += " AND event_date <= :end_date"
        params["end_date"] = end_date

    sql += " ORDER BY event_date DESC LIMIT :limit OFFSET :offset"
    params.update({"limit": limit, "offset": offset})

    rows = session.execute(text(sql), params).mappings().all()

    return {
        "items": [_incident_payload(dict(row)) for row in rows],
        "limit": limit,
        "offset": offset,
        "count": len(rows),
    }


@router.get("/{incident_id}", response_model=IncidentOut)
def get_incident(incident_id: str, session: Annotated[Session, Depends(get_db_session)]):
    row = session.execute(
        text("SELECT * FROM public.v_incidents_clean WHERE id = :id"), {"id": incident_id}
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Incident not found")
    return _incident_payload(dict(row))


class HotspotListResponse(BaseModel):
    items: list[HotspotOut]
    limit: int
    offset: int


@router.get("/hotspots/lga", response_model=HotspotListResponse)
def list_hotspots_lga(
    session: Annotated[Session, Depends(get_db_session)],
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    rows = session.execute(
        text(
            "SELECT * FROM public.v_hotspots_lga ORDER BY incident_count DESC, latest_incident_date DESC LIMIT :limit OFFSET :offset"
        ),
        {"limit": limit, "offset": offset},
    ).mappings().all()

    return {
        "items": [
            {
                "state_name": row.get("state_name"),
                "lga_name": row.get("lga_name"),
                "admin_boundary_id": row.get("admin_boundary_id"),
                "incident_count": int(row.get("incident_count") or 0),
                "total_fatalities": row.get("total_fatalities"),
                "total_injuries": row.get("total_injuries"),
                "avg_confidence": float(row.get("avg_confidence"))
                if row.get("avg_confidence") is not None
                else None,
                "crime_categories": row.get("crime_categories"),
                "latest_incident_date": row.get("latest_incident_date"),
            }
            for row in rows
        ],
        "limit": limit,
        "offset": offset,
    }
