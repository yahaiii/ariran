from __future__ import annotations

from datetime import datetime
from uuid import UUID
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from db.connection import get_session

router = APIRouter()


class PipelineRunOut(BaseModel):
    id: Optional[UUID]
    source_id: Optional[int]
    source_code: Optional[str]
    mode: Optional[str]
    status: Optional[str]
    run_started_at: Optional[datetime]
    run_finished_at: Optional[datetime]
    records_fetched: Optional[int]
    error_message: Optional[str]


class PipelineRunList(BaseModel):
    items: list[PipelineRunOut]
    limit: int
    offset: int


class PipelineSummaryOut(BaseModel):
    total_runs: int
    running_runs: int
    completed_runs: int
    failed_runs: int
    partial_runs: int
    latest_run_started_at: Optional[datetime]


class PipelineMapPointOut(BaseModel):
    id: str
    source_code: Optional[str]
    ingested_at: Optional[datetime]
    lat: float
    lon: float
    title: Optional[str]


def get_db_session() -> Session:
    with get_session() as session:
        yield session


@router.get("/runs", response_model=PipelineRunList)
def list_pipeline_runs(
    session: Annotated[Session, Depends(get_db_session)],
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    source: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    """List recent pipeline runs. Queries the `pipeline_runs` table if present."""
    sql = """
        SELECT
            pr.id,
            pr.source_id,
            s.source_code,
            COALESCE(pr.run_parameters->>'mode', 'incremental') AS mode,
            pr.status,
            pr.run_started_at,
            pr.run_finished_at,
            pr.records_fetched,
            COALESCE(pr.error_log->>'error', NULL) AS error_message
        FROM public.pipeline_runs pr
        JOIN public.sources s ON s.id = pr.source_id
        WHERE 1=1
    """
    params: dict[str, Any] = {}
    if source:
        sql += " AND s.source_code = :source"
        params["source"] = source
    if status:
        sql += " AND status = :status"
        params["status"] = status

    sql += " ORDER BY pr.run_started_at DESC LIMIT :limit OFFSET :offset"
    params.update({"limit": limit, "offset": offset})

    try:
        rows = session.execute(text(sql), params).mappings().all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    items = []
    for row in rows:
        items.append(
            PipelineRunOut(
                id=row.get("id"),
                source_id=row.get("source_id"),
                source_code=row.get("source_code"),
                mode=row.get("mode"),
                status=row.get("status"),
                run_started_at=row.get("run_started_at"),
                run_finished_at=row.get("run_finished_at"),
                records_fetched=row.get("records_fetched"),
                error_message=row.get("error_message"),
            )
        )

    return {"items": items, "limit": limit, "offset": offset}


@router.get("/runs/{source}", response_model=PipelineRunList)
def list_pipeline_runs_by_source(
    source: str,
    session: Annotated[Session, Depends(get_db_session)],
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return list_pipeline_runs(session=session, limit=limit, offset=offset, source=source)


@router.get("/summary", response_model=PipelineSummaryOut)
def pipeline_summary(session: Annotated[Session, Depends(get_db_session)]):
    sql = """
        SELECT
            COUNT(*) AS total_runs,
            COUNT(*) FILTER (WHERE status = 'running') AS running_runs,
            COUNT(*) FILTER (WHERE status = 'completed') AS completed_runs,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed_runs,
            COUNT(*) FILTER (WHERE status = 'partial') AS partial_runs,
            MAX(run_started_at) AS latest_run_started_at
        FROM public.pipeline_runs
    """
    try:
        row = session.execute(text(sql)).mappings().first() or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return PipelineSummaryOut(
        total_runs=int(row.get("total_runs") or 0),
        running_runs=int(row.get("running_runs") or 0),
        completed_runs=int(row.get("completed_runs") or 0),
        failed_runs=int(row.get("failed_runs") or 0),
        partial_runs=int(row.get("partial_runs") or 0),
        latest_run_started_at=row.get("latest_run_started_at"),
    )


@router.get("/map-points", response_model=list[PipelineMapPointOut])
def pipeline_map_points(
    session: Annotated[Session, Depends(get_db_session)],
    limit: int = Query(1000, ge=10, le=5000),
):
    """Return map-ready points from fetched staging rows.

    Priority for coordinates:
    1) geocoded_geom (if available)
    2) raw_payload latitude/longitude style keys
    """
    sql = """
        SELECT
            rs.id,
            rs.ingested_at,
            rs.raw_payload,
            rs.raw_text,
            s.source_code,
            CASE WHEN rs.geocoded_geom IS NOT NULL THEN ST_Y(rs.geocoded_geom) END AS geocoded_lat,
            CASE WHEN rs.geocoded_geom IS NOT NULL THEN ST_X(rs.geocoded_geom) END AS geocoded_lon
        FROM raw.incidents_staging rs
        JOIN public.sources s ON s.id = rs.source_id
        ORDER BY rs.ingested_at DESC
        LIMIT :limit
    """

    rows = session.execute(text(sql), {"limit": limit}).mappings().all()
    points: list[PipelineMapPointOut] = []

    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    centroid_cache: dict[tuple[Optional[str], Optional[str]], tuple[Optional[float], Optional[float]]] = {}

    def _lookup_admin_centroid(lga_name: Optional[str], state_name: Optional[str]) -> tuple[Optional[float], Optional[float]]:
        key = (
            (lga_name or "").strip().lower() or None,
            (state_name or "").strip().lower() or None,
        )
        if key in centroid_cache:
            return centroid_cache[key]

        result_lat: Optional[float] = None
        result_lon: Optional[float] = None

        if key[0] and key[1]:
            lga_row = session.execute(
                text(
                    """
                    SELECT
                        ST_Y(COALESCE(centroid, ST_PointOnSurface(geom))) AS lat,
                        ST_X(COALESCE(centroid, ST_PointOnSurface(geom))) AS lon
                    FROM public.admin_boundaries
                    WHERE admin_level = 2
                      AND LOWER(name) = :lga_name
                      AND LOWER(state_name) = :state_name
                    LIMIT 1
                    """
                ),
                {"lga_name": key[0], "state_name": key[1]},
            ).mappings().first()
            if lga_row:
                result_lat = _to_float(lga_row.get("lat"))
                result_lon = _to_float(lga_row.get("lon"))

        if (result_lat is None or result_lon is None) and key[1]:
            state_row = session.execute(
                text(
                    """
                    SELECT
                        ST_Y(COALESCE(centroid, ST_PointOnSurface(geom))) AS lat,
                        ST_X(COALESCE(centroid, ST_PointOnSurface(geom))) AS lon
                    FROM public.admin_boundaries
                    WHERE admin_level = 1
                      AND LOWER(name) = :state_name
                    LIMIT 1
                    """
                ),
                {"state_name": key[1]},
            ).mappings().first()
            if state_row:
                result_lat = _to_float(state_row.get("lat"))
                result_lon = _to_float(state_row.get("lon"))

        centroid_cache[key] = (result_lat, result_lon)
        return centroid_cache[key]

    for row in rows:
        payload = row.get("raw_payload") or {}
        geocoded_lat = _to_float(row.get("geocoded_lat"))
        geocoded_lon = _to_float(row.get("geocoded_lon"))

        lat = geocoded_lat
        lon = geocoded_lon

        if lat is None or lon is None:
            lat = _to_float(payload.get("latitude") or payload.get("lat") or payload.get("y"))
            lon = _to_float(payload.get("longitude") or payload.get("lng") or payload.get("lon") or payload.get("x"))

        if lat is None or lon is None:
            payload_lga = payload.get("admin2") or payload.get("lga")
            payload_state = payload.get("admin1") or payload.get("state")
            lat, lon = _lookup_admin_centroid(
                str(payload_lga) if payload_lga is not None else None,
                str(payload_state) if payload_state is not None else None,
            )

        if lat is None or lon is None:
            continue

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        title = payload.get("event_type") or payload.get("crime_type") or (row.get("raw_text") or "")[:140]

        points.append(
            PipelineMapPointOut(
                id=str(row.get("id")),
                source_code=row.get("source_code"),
                ingested_at=row.get("ingested_at"),
                lat=lat,
                lon=lon,
                title=title,
            )
        )

    return points

@router.get("/ui", response_class=HTMLResponse)
def pipeline_ui():
    """Single-file analytics dashboard for pipeline and incident views."""
    html = """
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width,initial-scale=1" />
                <title>Ariran Operations Dashboard</title>
                <link rel="preconnect" href="https://fonts.googleapis.com">
                <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3"></script>
                <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
                <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
                <style>
                    :root {
                        --bg: #f8f5ef;
                        --ink: #16222b;
                        --muted: #5b6973;
                        --panel: #fffdf8;
                        --line: #d9d0c4;
                        --accent: #0f766e;
                        --accent-soft: #d8f2ef;
                        --warn: #b45309;
                        --danger: #b91c1c;
                        --ok: #166534;
                        --mono: "IBM Plex Mono", monospace;
                        --sans: "Space Grotesk", system-ui, sans-serif;
                    }
                    * { box-sizing: border-box; }
                    body {
                        margin: 0;
                        color: var(--ink);
                        background:
                            radial-gradient(1200px 600px at 10% -20%, #e6f6f4 0%, transparent 60%),
                            radial-gradient(900px 500px at 100% 0%, #f6ead8 0%, transparent 65%),
                            var(--bg);
                        font-family: var(--sans);
                    }
                    .shell { max-width: 1280px; margin: 0 auto; padding: 24px 20px 28px; }
                    .hero {
                        display: grid;
                        grid-template-columns: 1.2fr .8fr;
                        gap: 16px;
                        margin-bottom: 16px;
                    }
                    .hero-card {
                        border: 1px solid var(--line);
                        border-radius: 18px;
                        background: linear-gradient(180deg, #fffefb 0%, #fffaf0 100%);
                        padding: 16px 18px;
                        box-shadow: 0 10px 24px rgba(22, 34, 43, 0.08);
                    }
                    .headline { margin: 0 0 6px; font-size: 1.7rem; letter-spacing: -.02em; }
                    .sub { margin: 0; color: var(--muted); }
                    .stamp {
                        font-family: var(--mono);
                        font-size: .8rem;
                        color: #114b46;
                        display: inline-block;
                        padding: 6px 10px;
                        border-radius: 999px;
                        background: var(--accent-soft);
                    }
                    .cards { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 12px; margin-bottom: 14px; }
                    .card {
                        border: 1px solid var(--line);
                        border-radius: 14px;
                        padding: 12px;
                        background: var(--panel);
                    }
                    .card .label {
                        color: var(--muted);
                        font-size: .73rem;
                        text-transform: uppercase;
                        letter-spacing: .08em;
                    }
                    .card .value { font-size: 1.6rem; margin-top: 5px; font-weight: 700; }
                    .controls {
                        display: grid;
                        grid-template-columns: repeat(4, minmax(120px, 1fr)) auto auto;
                        gap: 10px;
                        align-items: end;
                        margin-bottom: 14px;
                    }
                    .controls label { font-size: .82rem; color: var(--muted); display: flex; flex-direction: column; gap: 5px; }
                    input, select {
                        border: 1px solid var(--line);
                        background: #fff;
                        color: var(--ink);
                        border-radius: 10px;
                        padding: 10px 11px;
                        font: inherit;
                    }
                    button {
                        border: 1px solid #0c5f58;
                        background: #0f766e;
                        color: #fff;
                        border-radius: 10px;
                        padding: 10px 14px;
                        font-family: var(--sans);
                        font-weight: 600;
                        cursor: pointer;
                    }
                    button.ghost {
                        background: #fff;
                        color: #0f766e;
                        border-color: #79b8b1;
                    }
                    .grid {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 12px;
                        margin-bottom: 12px;
                    }
                    .panel {
                        border: 1px solid var(--line);
                        border-radius: 14px;
                        padding: 12px;
                        background: var(--panel);
                        min-height: 320px;
                    }
                    .panel h2 {
                        margin: 0 0 8px;
                        font-size: 1rem;
                        letter-spacing: .01em;
                    }
                    #incidentMap {
                        width: 100%;
                        height: 380px;
                        border: 1px solid #e8ddcf;
                        border-radius: 12px;
                        overflow: hidden;
                    }
                    .map-empty {
                        margin-top: 8px;
                        color: var(--muted);
                        font-size: .86rem;
                    }
                    .map-meta {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 8px;
                        color: var(--muted);
                        font-size: .82rem;
                    }
                    .map-meta .mono { font-size: .78rem; }
                    .table-wrap {
                        border: 1px solid var(--line);
                        border-radius: 14px;
                        overflow: hidden;
                        background: #fff;
                    }
                    table { border-collapse: collapse; width: 100%; }
                    th, td {
                        border-bottom: 1px solid #ece3d8;
                        padding: 10px 12px;
                        text-align: left;
                        vertical-align: top;
                        font-size: .9rem;
                    }
                    th {
                        background: #faf2e5;
                        font-size: .76rem;
                        color: #4b5b64;
                        text-transform: uppercase;
                        letter-spacing: .08em;
                    }
                    .mono { font-family: var(--mono); font-size: .8rem; }
                    .badge {
                        display: inline-block;
                        padding: 3px 8px;
                        border-radius: 999px;
                        font-size: .74rem;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: .06em;
                    }
                    .badge.completed { color: var(--ok); background: #dcfce7; }
                    .badge.running { color: #1d4ed8; background: #dbeafe; }
                    .badge.failed { color: var(--danger); background: #fee2e2; }
                    .badge.partial { color: var(--warn); background: #ffedd5; }
                    .badge.unknown { color: #374151; background: #e5e7eb; }
                    .pagination {
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        gap: 10px;
                        padding: 10px 12px;
                        background: #fff7ea;
                    }
                    .hotspots {
                        list-style: none;
                        margin: 0;
                        padding: 0;
                        display: grid;
                        gap: 8px;
                    }
                    .hotspots li {
                        border: 1px solid #e8ddcf;
                        border-radius: 10px;
                        padding: 10px;
                        background: #fffcf6;
                        display: flex;
                        justify-content: space-between;
                        gap: 8px;
                    }
                    .hotspots small { color: var(--muted); }
                    .empty { color: var(--muted); font-size: .9rem; }
                    @media (max-width: 980px) {
                        .hero { grid-template-columns: 1fr; }
                        .cards { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
                        .controls { grid-template-columns: 1fr 1fr; }
                        .grid { grid-template-columns: 1fr; }
                    }
                </style>
            </head>
            <body>
                <div class="shell">
                    <section class="hero">
                        <div class="hero-card">
                            <h1 class="headline">Ariran Operations Dashboard</h1>
                            <p class="sub">Live visibility into ingestion health, throughput, and emerging incident hotspots.</p>
                        </div>
                        <div class="hero-card">
                            <span class="stamp" id="lastUpdated">Last updated: --</span>
                        </div>
                    </section>

                    <section class="cards">
                        <div class="card"><div class="label">Total Runs</div><div class="value" id="totalRuns">0</div></div>
                        <div class="card"><div class="label">Completed</div><div class="value" id="completedRuns">0</div></div>
                        <div class="card"><div class="label">Running</div><div class="value" id="runningRuns">0</div></div>
                        <div class="card"><div class="label">Failed</div><div class="value" id="failedRuns">0</div></div>
                        <div class="card"><div class="label">Partial</div><div class="value" id="partialRuns">0</div></div>
                    </section>

                    <section class="controls">
                        <label>Source
                            <input id="source" placeholder="ACLED_API, NBS_ANNUAL" />
                        </label>
                        <label>Status
                            <select id="status">
                                <option value="">All</option>
                                <option value="running">running</option>
                                <option value="completed">completed</option>
                                <option value="failed">failed</option>
                                <option value="partial">partial</option>
                            </select>
                        </label>
                        <label>Rows
                            <input id="limit" type="number" min="5" max="100" value="12" />
                        </label>
                        <label>Hotspots
                            <input id="hotspotLimit" type="number" min="3" max="30" value="8" />
                        </label>
                        <button id="refresh">Refresh</button>
                        <button id="reset" class="ghost">Reset Filters</button>
                    </section>

                    <section class="grid">
                        <div class="panel">
                            <h2>Run Status Distribution</h2>
                            <canvas id="statusChart"></canvas>
                        </div>
                        <div class="panel">
                            <h2>Records Fetched by Source</h2>
                            <canvas id="sourceChart"></canvas>
                        </div>
                    </section>

                    <section class="grid">
                        <div class="panel">
                            <h2>Recent Run Throughput</h2>
                            <canvas id="throughputChart"></canvas>
                        </div>
                        <div class="panel">
                            <h2>Current LGA Hotspots</h2>
                            <ul id="hotspots" class="hotspots"></ul>
                            <p id="hotspotsEmpty" class="empty" style="display:none">No hotspot rows available yet.</p>
                        </div>
                    </section>

                    <section class="panel" style="margin-bottom: 12px; min-height: 450px;">
                        <h2>Fetched Crime Datapoints (Basemap)</h2>
                        <div class="map-meta">
                            <span>Plotted from fetched staging rows with lat/lon or geocoded geometry</span>
                            <span id="mapCount" class="mono">0 points</span>
                        </div>
                        <div id="incidentMap"></div>
                        <p id="mapEmptyHint" class="map-empty" style="display:none;">
                            No coordinate-bearing records yet. Ingest ACLED rows with latitude/longitude or run geocoding enrichment to populate map points.
                        </p>
                    </section>

                    <section class="table-wrap">
                        <table id="runs">
                            <thead>
                                <tr>
                                    <th>Run ID</th>
                                    <th>Source</th>
                                    <th>Mode</th>
                                    <th>Status</th>
                                    <th>Started</th>
                                    <th>Finished</th>
                                    <th>Records</th>
                                    <th>Error</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                        <div class="pagination">
                            <div>
                                <button id="prevPage" class="ghost">Previous</button>
                                <button id="nextPage" class="ghost">Next</button>
                            </div>
                            <span id="pageInfo" class="mono"></span>
                        </div>
                    </section>
                </div>

                <script>
                    let offset = 0;
                    let statusChart;
                    let sourceChart;
                    let throughputChart;
                    let incidentMap;
                    let incidentLayer;

                    async function fetchRuns({ source, status, limit, offset }) {
                        const params = new URLSearchParams();
                        if (source) params.set('source', source);
                        if (status) params.set('status', status);
                        params.set('limit', String(limit));
                        params.set('offset', String(offset));
                        const res = await fetch('/pipeline/runs?' + params.toString());
                        if (!res.ok) {
                            alert('Failed to load runs: ' + res.status);
                            return { items: [] };
                        }
                        return await res.json();
                    }

                    async function fetchSummary() {
                        const res = await fetch('/pipeline/summary');
                        if (!res.ok) return null;
                        return await res.json();
                    }

                    async function fetchHotspots(limit) {
                        const params = new URLSearchParams({ limit: String(limit), offset: '0' });
                        const res = await fetch('/incidents/hotspots/lga?' + params.toString());
                        if (!res.ok) return { items: [] };
                        return await res.json();
                    }

                    async function fetchIncidents(limit) {
                        const params = new URLSearchParams({ limit: String(limit) });
                        const res = await fetch('/pipeline/map-points?' + params.toString());
                        if (!res.ok) return [];
                        return await res.json();
                    }

                    function formatDate(value) {
                        if (!value) return '--';
                        const d = new Date(value);
                        if (Number.isNaN(d.getTime())) return String(value);
                        return d.toLocaleString();
                    }

                    function renderSummary(summary) {
                        if (!summary) return;
                        document.getElementById('totalRuns').textContent = summary.total_runs || 0;
                        document.getElementById('completedRuns').textContent = summary.completed_runs || 0;
                        document.getElementById('runningRuns').textContent = summary.running_runs || 0;
                        document.getElementById('failedRuns').textContent = summary.failed_runs || 0;
                        document.getElementById('partialRuns').textContent = summary.partial_runs || 0;
                        document.getElementById('lastUpdated').textContent = 'Last updated: ' + new Date().toLocaleString();
                    }

                    function renderRows(items) {
                        const tbody = document.querySelector('#runs tbody');
                        tbody.innerHTML = '';
                        if (!items.length) {
                            const tr = document.createElement('tr');
                            tr.innerHTML = '<td colspan="8" class="empty">No runs matched the current filters.</td>';
                            tbody.appendChild(tr);
                            return;
                        }
                        items.forEach(it => {
                            const status = String(it.status || 'unknown').toLowerCase();
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td class="mono">${it.id || ''}</td>
                                <td>${it.source_code || ''}</td>
                                <td>${it.mode || ''}</td>
                                <td><span class="badge ${status}">${status}</span></td>
                                <td>${formatDate(it.run_started_at)}</td>
                                <td>${formatDate(it.run_finished_at)}</td>
                                <td>${it.records_fetched || 0}</td>
                                <td>${it.error_message || ''}</td>
                            `;
                            tbody.appendChild(tr);
                        });
                    }

                    function renderHotspots(items) {
                        const root = document.getElementById('hotspots');
                        const empty = document.getElementById('hotspotsEmpty');
                        root.innerHTML = '';
                        if (!items.length) {
                            empty.style.display = 'block';
                            return;
                        }
                        empty.style.display = 'none';
                        items.forEach((row) => {
                            const li = document.createElement('li');
                            const location = [row.lga_name, row.state_name].filter(Boolean).join(', ');
                            const incidents = row.incident_count || 0;
                            li.innerHTML = `
                                <div>
                                    <strong>${location || 'Unknown location'}</strong><br>
                                    <small>Latest: ${row.latest_incident_date || '--'}</small>
                                </div>
                                <div class="mono">${incidents} incidents</div>
                            `;
                            root.appendChild(li);
                        });
                    }

                    function renderCharts(items) {
                        const statusCounts = {};
                        const sourceTotals = {};
                        const throughputLabels = [];
                        const throughputValues = [];

                        items.slice().reverse().forEach((it) => {
                            const status = String(it.status || 'unknown').toLowerCase();
                            statusCounts[status] = (statusCounts[status] || 0) + 1;

                            const source = it.source_code || 'unknown';
                            sourceTotals[source] = (sourceTotals[source] || 0) + (it.records_fetched || 0);

                            throughputLabels.push((it.run_started_at || '').slice(0, 16).replace('T', ' ') || 'run');
                            throughputValues.push(it.records_fetched || 0);
                        });

                        const statusCtx = document.getElementById('statusChart');
                        const sourceCtx = document.getElementById('sourceChart');
                        const throughputCtx = document.getElementById('throughputChart');

                        if (statusChart) statusChart.destroy();
                        if (sourceChart) sourceChart.destroy();
                        if (throughputChart) throughputChart.destroy();

                        statusChart = new Chart(statusCtx, {
                            type: 'doughnut',
                            data: {
                                labels: Object.keys(statusCounts),
                                datasets: [{
                                    data: Object.values(statusCounts),
                                    backgroundColor: ['#166534', '#1d4ed8', '#b91c1c', '#b45309', '#6b7280'],
                                }]
                            },
                            options: { responsive: true, maintainAspectRatio: false }
                        });

                        sourceChart = new Chart(sourceCtx, {
                            type: 'bar',
                            data: {
                                labels: Object.keys(sourceTotals),
                                datasets: [{
                                    label: 'Records fetched',
                                    data: Object.values(sourceTotals),
                                    backgroundColor: '#0f766e',
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: { legend: { display: false } },
                                scales: { y: { beginAtZero: true } }
                            }
                        });

                        throughputChart = new Chart(throughputCtx, {
                            type: 'line',
                            data: {
                                labels: throughputLabels,
                                datasets: [{
                                    label: 'Records per run',
                                    data: throughputValues,
                                    borderColor: '#b45309',
                                    backgroundColor: 'rgba(180, 83, 9, .18)',
                                    fill: true,
                                    tension: 0.25,
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: { y: { beginAtZero: true } }
                            }
                        });
                    }

                    function renderIncidentMap(points) {
                        document.getElementById('mapCount').textContent = `${points.length} points`;
                        const hint = document.getElementById('mapEmptyHint');

                        if (!incidentMap) {
                            incidentMap = L.map('incidentMap', { zoomControl: true }).setView([9.082, 8.6753], 6);
                            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                                maxZoom: 18,
                                attribution: '&copy; OpenStreetMap contributors',
                            }).addTo(incidentMap);
                            incidentLayer = L.layerGroup().addTo(incidentMap);
                        }

                        incidentLayer.clearLayers();

                        if (!points.length) {
                            hint.style.display = 'block';
                            incidentMap.setView([9.082, 8.6753], 6);
                            return;
                        }

                        hint.style.display = 'none';

                        const bounds = [];
                        points.forEach((p) => {
                            const marker = L.circleMarker([p.lat, p.lon], {
                                radius: 5,
                                color: '#b45309',
                                fillColor: '#f59e0b',
                                fillOpacity: 0.72,
                                weight: 1,
                            });
                            marker.bindPopup(`
                                <strong>${p.title || 'Fetched incident'}</strong><br>
                                Source: ${p.source_code || 'unknown'}<br>
                                Ingested: ${p.ingested_at || '--'}
                            `);
                            marker.addTo(incidentLayer);
                            bounds.push([p.lat, p.lon]);
                        });

                        const fitBounds = L.latLngBounds(bounds);
                        incidentMap.fitBounds(fitBounds.pad(0.2));
                    }

                    function updatePageInfo(count, limit, offset) {
                        const start = count === 0 ? 0 : offset + 1;
                        const end = offset + count;
                        document.getElementById('pageInfo').textContent = `Showing ${start}-${end} | offset ${offset}`;
                    }

                    async function loadRuns() {
                        const source = document.getElementById('source').value.trim();
                        const status = document.getElementById('status').value.trim();
                        const limit = parseInt(document.getElementById('limit').value || '10', 10);
                        const hotspotLimit = parseInt(document.getElementById('hotspotLimit').value || '8', 10);

                        const [data, summary, hotspots, points] = await Promise.all([
                            fetchRuns({ source, status, limit, offset }),
                            fetchSummary(),
                            fetchHotspots(hotspotLimit),
                            fetchIncidents(500),
                        ]);

                        const items = data.items || [];
                        renderSummary(summary);
                        renderRows(items);
                        renderCharts(items);
                        renderHotspots(hotspots.items || []);
                        renderIncidentMap(points || []);
                        updatePageInfo(items.length, limit, offset);
                        document.getElementById('prevPage').disabled = offset === 0;
                        document.getElementById('nextPage').disabled = items.length < limit;
                    }

                    document.getElementById('refresh').addEventListener('click', async () => {
                        offset = 0;
                        await loadRuns();
                    });

                    document.getElementById('prevPage').addEventListener('click', async () => {
                        const limit = parseInt(document.getElementById('limit').value || '10', 10);
                        offset = Math.max(0, offset - limit);
                        await loadRuns();
                    });

                    document.getElementById('nextPage').addEventListener('click', async () => {
                        const limit = parseInt(document.getElementById('limit').value || '10', 10);
                        offset += limit;
                        await loadRuns();
                    });

                    document.getElementById('reset').addEventListener('click', async () => {
                        document.getElementById('source').value = '';
                        document.getElementById('status').value = '';
                        document.getElementById('limit').value = '12';
                        document.getElementById('hotspotLimit').value = '8';
                        offset = 0;
                        await loadRuns();
                    });

                    // initial load
                    (async () => { await loadRuns(); })();
                </script>
            </body>
        </html>
        """
    return HTMLResponse(content=html)
