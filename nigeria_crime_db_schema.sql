-- =============================================================================
-- NIGERIA OPEN CRIME DATABASE (ariran)
-- PostGIS/PostgreSQL 15 Schema — v1.0
-- Designed for multi-source fusion with full provenance, confidence scoring,
-- deduplication, spatial precision flags, and immutable audit history.
-- =============================================================================

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;          -- fuzzy text matching for dedup
CREATE EXTENSION IF NOT EXISTS btree_gin;         -- composite GIN indexes


-- =============================================================================
-- SCHEMAS
-- raw      → immutable staging; records land here and are never modified
-- public   → normalized, enriched, production records
-- audit    → full change history for canonical records
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS audit;


-- =============================================================================
-- SCHEMA: public
-- SECTION 1 — LOOKUP / TAXONOMY TABLES
-- =============================================================================

-- 1.1  Crime type taxonomy (hierarchical)
-- e.g. category: 'violent_crime' → type: 'armed_robbery' → subtype: 'bank_robbery'
CREATE TABLE public.crime_taxonomy (
    id                  SERIAL PRIMARY KEY,
    category            TEXT NOT NULL,                   -- top-level: violent, property, cyber, etc.
    crime_type          TEXT NOT NULL,
    crime_subtype       TEXT,
    icc_code            TEXT,                            -- optional mapping to ICC/UNODC codes
    description         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (category, crime_type, crime_subtype)
);

COMMENT ON TABLE public.crime_taxonomy IS
    'Hierarchical crime type classification. All incidents reference this table.';


-- 1.2  Data sources registry
CREATE TABLE public.sources (
    id                  SERIAL PRIMARY KEY,
    source_code         TEXT NOT NULL UNIQUE,            -- e.g. 'NBS_ANNUAL', 'ACLED_API', 'PUNCH_RSS'
    source_name         TEXT NOT NULL,
    source_type         TEXT NOT NULL CHECK (source_type IN (
                            'government_statistics',
                            'ngo_database',
                            'news_media',
                            'social_media',
                            'crowdsourced',
                            'police_record',
                            'court_record',
                            'academic'
                        )),
    base_url            TEXT,
    api_endpoint        TEXT,
    update_frequency    TEXT,                            -- 'realtime', 'daily', 'weekly', 'annual'
    default_confidence  NUMERIC(3,2) NOT NULL CHECK (default_confidence BETWEEN 0 AND 1),
    reliability_notes   TEXT,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    first_ingested_at   TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE public.sources IS
    'Registry of all data sources feeding the pipeline. Every raw and canonical record '
    'must reference a source_id from this table.';

COMMENT ON COLUMN public.sources.default_confidence IS
    'Baseline confidence (0.0–1.0) assigned to records from this source before '
    'corroboration adjustments. Police records ~0.8, news ~0.6, social media ~0.35.';


-- 1.3  Administrative boundaries (geometry)
-- Populated from OSGOF/NBS official boundary shapefiles
CREATE TABLE public.admin_boundaries (
    id                  SERIAL PRIMARY KEY,
    admin_level         SMALLINT NOT NULL CHECK (admin_level IN (0,1,2,3)),
                        -- 0=country, 1=state/FCT, 2=LGA, 3=ward
    name                TEXT NOT NULL,
    name_alt            TEXT[],                          -- alternate spellings / Hausa / Yoruba / Igbo names
    state_name          TEXT,
    lga_name            TEXT,
    pcode               TEXT UNIQUE,                     -- NBS/OSGOF place code
    geom                GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,
    centroid            GEOMETRY(POINT, 4326),           -- pre-computed centroid for fast fallback geocoding
    population_2006     INTEGER,
    population_est      INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_admin_boundaries_geom   ON public.admin_boundaries USING GIST (geom);
CREATE INDEX idx_admin_boundaries_level  ON public.admin_boundaries (admin_level);
CREATE INDEX idx_admin_boundaries_state  ON public.admin_boundaries (state_name);
CREATE INDEX idx_admin_boundaries_name   ON public.admin_boundaries USING GIN (name gin_trgm_ops);

COMMENT ON TABLE public.admin_boundaries IS
    'Authoritative Nigerian administrative boundaries loaded from OSGOF shapefiles. '
    'Used for geocoding and spatial joins on all incident records.';


-- =============================================================================
-- SECTION 2 — PIPELINE RUNS (Ingestion audit trail)
-- =============================================================================

CREATE TABLE public.pipeline_runs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id           INTEGER NOT NULL REFERENCES public.sources(id),
    run_started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_finished_at     TIMESTAMPTZ,
    status              TEXT NOT NULL DEFAULT 'running' CHECK (status IN (
                            'running', 'completed', 'failed', 'partial'
                        )),
    records_fetched     INTEGER DEFAULT 0,
    records_inserted    INTEGER DEFAULT 0,
    records_skipped     INTEGER DEFAULT 0,
    records_failed      INTEGER DEFAULT 0,
    error_log           JSONB,
    pipeline_version    TEXT,                            -- git commit SHA of the ETL code
    run_parameters      JSONB                            -- any runtime config (date range, filters, etc.)
);

COMMENT ON TABLE public.pipeline_runs IS
    'One row per ETL pipeline execution. Every raw record references the run that '
    'produced it, creating a full audit trail from source to storage.';


-- =============================================================================
-- SECTION 3 — RAW STAGING SCHEMA (Immutable)
-- =============================================================================

-- 3.1  Raw incidents staging table
-- Records land here exactly as received — never modified after insert.
-- NLP and geocoding results populate the enriched columns but the
-- original raw payload is always preserved in raw_payload JSONB.

CREATE TABLE raw.incidents_staging (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id     UUID NOT NULL REFERENCES public.pipeline_runs(id),
    source_id           INTEGER NOT NULL REFERENCES public.sources(id),
    source_record_id    TEXT,                            -- original ID from the upstream source
    source_url          TEXT,                            -- direct URL to the source record
    raw_payload         JSONB NOT NULL,                  -- full original record, verbatim
    raw_text            TEXT,                            -- extracted plain text (for NLP input)

    -- NLP-extracted fields (populated by enrichment pipeline, never by human)
    nlp_crime_type      TEXT,
    nlp_location_raw    TEXT,                            -- location string as extracted by NLP
    nlp_date_raw        TEXT,                            -- date string as extracted by NLP
    nlp_actors          TEXT[],
    nlp_fatalities      INTEGER,
    nlp_injuries        INTEGER,
    nlp_confidence      NUMERIC(3,2),                   -- NLP model confidence on extraction
    nlp_model_version   TEXT,

    -- Geocoding results
    geocoded_geom       GEOMETRY(POINT, 4326),
    geocoded_admin_id   INTEGER REFERENCES public.admin_boundaries(id),
    geocode_method      TEXT,                           -- 'exact_match', 'fuzzy_match', 'lga_centroid', 'state_centroid', 'failed'

    -- Processing state
    processing_status   TEXT NOT NULL DEFAULT 'pending' CHECK (processing_status IN (
                            'pending', 'enriched', 'promoted', 'rejected', 'duplicate'
                        )),
    rejection_reason    TEXT,
    promoted_incident_id UUID,                          -- FK set when promoted to public.incidents
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Partial index: only pending/enriched records scanned during promotion
CREATE INDEX idx_staging_status      ON raw.incidents_staging (processing_status)
    WHERE processing_status IN ('pending', 'enriched');
CREATE INDEX idx_staging_source      ON raw.incidents_staging (source_id, ingested_at DESC);
CREATE INDEX idx_staging_geom        ON raw.incidents_staging USING GIST (geocoded_geom)
    WHERE geocoded_geom IS NOT NULL;
CREATE INDEX idx_staging_source_rec  ON raw.incidents_staging (source_id, source_record_id);

COMMENT ON TABLE raw.incidents_staging IS
    'Immutable raw landing zone. Records are inserted once and never updated. '
    'All enrichment results (NLP, geocoding) are appended in dedicated columns. '
    'Promoted records are linked to public.incidents via promoted_incident_id.';


-- =============================================================================
-- SECTION 4 — CANONICAL INCIDENTS (Production)
-- =============================================================================

-- 4.1  Main incidents table — partitioned by year for query performance
CREATE TABLE public.incidents (
    id                      UUID NOT NULL DEFAULT uuid_generate_v4(),
    canonical_id            UUID,                        -- groups duplicate records from different sources
    is_canonical            BOOLEAN NOT NULL DEFAULT TRUE,

    -- Event classification
    taxonomy_id             INTEGER REFERENCES public.crime_taxonomy(id),
    crime_category          TEXT NOT NULL,
    crime_type              TEXT NOT NULL,
    crime_subtype           TEXT,
    incident_title          TEXT,                        -- short human-readable summary

    -- Temporal fields
    event_date              DATE,
    event_date_end          DATE,                        -- for incidents spanning multiple days
    event_time              TIME,
    date_precision          TEXT NOT NULL DEFAULT 'day' CHECK (date_precision IN (
                                'exact', 'day', 'week', 'month', 'year', 'unknown'
                            )),
    report_date             DATE,                        -- date the incident was reported/published

    -- Spatial fields
    geom                    GEOMETRY(POINT, 4326),
    admin_boundary_id       INTEGER REFERENCES public.admin_boundaries(id),
    state_name              TEXT,
    lga_name                TEXT,
    ward_name               TEXT,
    location_description    TEXT,                        -- free-text e.g. "Beside Lugbe market"
    geo_precision           TEXT NOT NULL DEFAULT 'lga' CHECK (geo_precision IN (
                                'coordinates', 'settlement', 'lga', 'state', 'unknown'
                            )),

    -- Incident details
    fatalities              INTEGER CHECK (fatalities >= 0),
    injuries                INTEGER CHECK (injuries >= 0),
    arrests                 INTEGER CHECK (arrests >= 0),
    property_lost_ngn       BIGINT,                      -- estimated value in Naira
    perpetrator_groups      TEXT[],
    victim_groups           TEXT[],
    weapons_used            TEXT[],
    narrative               TEXT,                        -- cleaned summary narrative

    -- Provenance & trust
    primary_source_id       INTEGER NOT NULL REFERENCES public.sources(id),
    source_record_ids       TEXT[],                      -- all source record IDs that contributed
    source_urls             TEXT[],
    staging_ids             UUID[],                      -- raw.incidents_staging IDs this was built from
    corroborating_sources   INTEGER DEFAULT 0,           -- count of independent sources confirming
    confidence_score        NUMERIC(3,2) NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    verification_status     TEXT NOT NULL DEFAULT 'unverified' CHECK (verification_status IN (
                                'unverified',
                                'corroborated',          -- confirmed by 2+ independent sources
                                'police_confirmed',
                                'court_confirmed',
                                'rejected'
                            )),
    verification_notes      TEXT,

    -- Soft delete / lifecycle
    is_deleted              BOOLEAN NOT NULL DEFAULT FALSE,
    deletion_reason         TEXT,
    deleted_at              TIMESTAMPTZ,
    deleted_by              TEXT,

    -- Metadata
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_run          UUID REFERENCES public.pipeline_runs(id)

) PARTITION BY RANGE (event_date);

-- Partitions — extend annually
CREATE TABLE public.incidents_pre2000  PARTITION OF public.incidents
    FOR VALUES FROM (UNBOUNDED) TO ('2000-01-01');
CREATE TABLE public.incidents_2000_2009 PARTITION OF public.incidents
    FOR VALUES FROM ('2000-01-01') TO ('2010-01-01');
CREATE TABLE public.incidents_2010_2014 PARTITION OF public.incidents
    FOR VALUES FROM ('2010-01-01') TO ('2015-01-01');
CREATE TABLE public.incidents_2015_2019 PARTITION OF public.incidents
    FOR VALUES FROM ('2015-01-01') TO ('2020-01-01');
CREATE TABLE public.incidents_2020_2024 PARTITION OF public.incidents
    FOR VALUES FROM ('2020-01-01') TO ('2025-01-01');
CREATE TABLE public.incidents_2025_plus PARTITION OF public.incidents
    FOR VALUES FROM ('2025-01-01') TO (UNBOUNDED);

-- Indexes (applied across all partitions)
CREATE INDEX idx_incidents_geom         ON public.incidents USING GIST (geom)
    WHERE geom IS NOT NULL;
CREATE INDEX idx_incidents_date         ON public.incidents USING BRIN (event_date);
CREATE INDEX idx_incidents_canonical    ON public.incidents (canonical_id)
    WHERE canonical_id IS NOT NULL;
CREATE INDEX idx_incidents_state        ON public.incidents (state_name, event_date DESC);
CREATE INDEX idx_incidents_lga          ON public.incidents (lga_name, event_date DESC);
CREATE INDEX idx_incidents_type         ON public.incidents (crime_category, crime_type);
CREATE INDEX idx_incidents_confidence   ON public.incidents (confidence_score, verification_status);
CREATE INDEX idx_incidents_active       ON public.incidents (event_date DESC)
    WHERE is_deleted = FALSE;

COMMENT ON TABLE public.incidents IS
    'Canonical crime incident records. Partitioned by event_date for performance. '
    'Records from multiple sources representing the same event share a canonical_id; '
    'the authoritative record has is_canonical = TRUE. Records are never hard-deleted.';

COMMENT ON COLUMN public.incidents.confidence_score IS
    'Composite trust score (0.0–1.0). Starts at source default_confidence, '
    'increases with corroborating_sources, decreases for imprecise geo/date.';

COMMENT ON COLUMN public.incidents.geo_precision IS
    'Spatial resolution of the location. coordinates = exact GPS; '
    'settlement = named place; lga = LGA centroid used; state = state centroid used.';


-- 4.2  Trigger: auto-update updated_at on any row change
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_incidents_updated_at
    BEFORE UPDATE ON public.incidents
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();


-- =============================================================================
-- SECTION 5 — DEDUPLICATION SUPPORT
-- =============================================================================

-- 5.1  Deduplication candidates log
-- Written by the dedup pipeline before any merge decision is made.
CREATE TABLE public.dedup_candidates (
    id                  SERIAL PRIMARY KEY,
    incident_id_a       UUID NOT NULL,
    incident_id_b       UUID NOT NULL,
    similarity_score    NUMERIC(4,3) NOT NULL,           -- 0.0–1.0 composite similarity
    match_method        TEXT NOT NULL,                   -- 'spatial_temporal', 'text_hash', 'nlp_embedding'
    spatial_distance_m  NUMERIC,                         -- metres between the two points
    date_diff_days      INTEGER,
    text_similarity     NUMERIC(4,3),
    resolution          TEXT CHECK (resolution IN (
                            'pending', 'merged', 'distinct', 'manual_review'
                        )) DEFAULT 'pending',
    resolved_at         TIMESTAMPTZ,
    resolved_by         TEXT,
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (incident_id_a, incident_id_b)
);

CREATE INDEX idx_dedup_pending ON public.dedup_candidates (resolution)
    WHERE resolution = 'pending';

COMMENT ON TABLE public.dedup_candidates IS
    'Pairs of incidents flagged as potential duplicates by the deduplication pipeline. '
    'Resolution determines whether they are merged under one canonical_id or kept distinct.';


-- =============================================================================
-- SECTION 6 — AUDIT / CHANGE HISTORY
-- =============================================================================

-- 6.1  Incident audit log — append-only change history
CREATE TABLE audit.incident_changes (
    id                  BIGSERIAL PRIMARY KEY,
    incident_id         UUID NOT NULL,
    changed_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    changed_by          TEXT NOT NULL,                   -- pipeline run ID or user identifier
    change_type         TEXT NOT NULL CHECK (change_type IN (
                            'insert', 'update', 'soft_delete', 'restore',
                            'canonical_merge', 'verification_change'
                        )),
    field_name          TEXT,                            -- NULL for insert/delete
    old_value           TEXT,
    new_value           TEXT,
    change_reason       TEXT,
    pipeline_run_id     UUID REFERENCES public.pipeline_runs(id)
);

CREATE INDEX idx_audit_incident   ON audit.incident_changes (incident_id, changed_at DESC);
CREATE INDEX idx_audit_changed_at ON audit.incident_changes USING BRIN (changed_at);

COMMENT ON TABLE audit.incident_changes IS
    'Append-only log of every change to canonical incident records. '
    'Never updated or deleted. Provides full reconstruction of any record at any point in time.';


-- 6.2  Trigger: auto-log updates to incidents
CREATE OR REPLACE FUNCTION audit.log_incident_change()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit.incident_changes
            (incident_id, changed_by, change_type, change_reason)
        VALUES (NEW.id, COALESCE(NEW.created_by_run::TEXT, 'pipeline'), 'insert', 'Initial ingestion');

    ELSIF TG_OP = 'UPDATE' THEN
        -- Log verification status changes
        IF OLD.verification_status IS DISTINCT FROM NEW.verification_status THEN
            INSERT INTO audit.incident_changes
                (incident_id, changed_by, change_type, field_name, old_value, new_value)
            VALUES (NEW.id, 'system', 'verification_change',
                    'verification_status', OLD.verification_status, NEW.verification_status);
        END IF;

        -- Log soft deletes
        IF OLD.is_deleted = FALSE AND NEW.is_deleted = TRUE THEN
            INSERT INTO audit.incident_changes
                (incident_id, changed_by, change_type, change_reason)
            VALUES (NEW.id, COALESCE(NEW.deleted_by, 'system'),
                    'soft_delete', NEW.deletion_reason);
        END IF;

        -- Log confidence score changes
        IF OLD.confidence_score IS DISTINCT FROM NEW.confidence_score THEN
            INSERT INTO audit.incident_changes
                (incident_id, changed_by, change_type, field_name, old_value, new_value)
            VALUES (NEW.id, 'system', 'update',
                    'confidence_score',
                    OLD.confidence_score::TEXT, NEW.confidence_score::TEXT);
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_incidents_audit
    AFTER INSERT OR UPDATE ON public.incidents
    FOR EACH ROW EXECUTE FUNCTION audit.log_incident_change();


-- =============================================================================
-- SECTION 7 — ACTORS & RELATIONSHIPS
-- =============================================================================

-- 7.1  Actor registry (perpetrators, groups, organisations)
CREATE TABLE public.actors (
    id                  SERIAL PRIMARY KEY,
    actor_name          TEXT NOT NULL,
    actor_type          TEXT CHECK (actor_type IN (
                            'armed_group', 'criminal_gang', 'individual',
                            'security_force', 'political_group', 'unknown'
                        )),
    aliases             TEXT[],
    region_of_operation TEXT[],
    is_active           BOOLEAN DEFAULT TRUE,
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (actor_name)
);

CREATE INDEX idx_actors_name ON public.actors USING GIN (actor_name gin_trgm_ops);
CREATE INDEX idx_actors_aliases ON public.actors USING GIN (aliases);


-- 7.2  Incident-actor junction
CREATE TABLE public.incident_actors (
    incident_id         UUID NOT NULL,
    actor_id            INTEGER NOT NULL REFERENCES public.actors(id),
    role                TEXT CHECK (role IN ('perpetrator', 'victim', 'suspect', 'witness', 'respondent')),
    confidence          NUMERIC(3,2),
    PRIMARY KEY (incident_id, actor_id, role)
);

CREATE INDEX idx_incident_actors_actor ON public.incident_actors (actor_id);


-- =============================================================================
-- SECTION 8 — MEDIA / EVIDENCE ATTACHMENTS
-- =============================================================================

CREATE TABLE public.incident_media (
    id                  SERIAL PRIMARY KEY,
    incident_id         UUID NOT NULL,
    media_type          TEXT CHECK (media_type IN ('article', 'image', 'video', 'document', 'social_post')),
    url                 TEXT NOT NULL,
    title               TEXT,
    published_at        TIMESTAMPTZ,
    source_id           INTEGER REFERENCES public.sources(id),
    is_archived         BOOLEAN DEFAULT FALSE,           -- Wayback Machine / local archive copy exists
    archive_url         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_media_incident ON public.incident_media (incident_id);


-- =============================================================================
-- SECTION 9 — VIEWS FOR COMMON ACCESS PATTERNS
-- =============================================================================

-- 9.1  Clean public-facing view (active canonical records only)
CREATE VIEW public.v_incidents_clean AS
SELECT
    i.id,
    i.canonical_id,
    i.crime_category,
    i.crime_type,
    i.crime_subtype,
    i.incident_title,
    i.event_date,
    i.date_precision,
    i.state_name,
    i.lga_name,
    i.location_description,
    i.geo_precision,
    ST_AsGeoJSON(i.geom)::JSONB          AS geometry,
    i.fatalities,
    i.injuries,
    i.arrests,
    i.perpetrator_groups,
    i.victim_groups,
    i.weapons_used,
    i.narrative,
    i.confidence_score,
    i.verification_status,
    i.corroborating_sources,
    s.source_name                        AS primary_source,
    s.source_type                        AS primary_source_type,
    i.source_urls,
    i.created_at
FROM public.incidents i
JOIN public.sources s ON s.id = i.primary_source_id
WHERE i.is_deleted = FALSE
  AND i.is_canonical = TRUE;

COMMENT ON VIEW public.v_incidents_clean IS
    'Public-facing view: canonical, non-deleted incidents only. '
    'Suitable for API responses and researcher exports.';


-- 9.2  Hotspot aggregation view (LGA-level, rolling 12 months)
CREATE VIEW public.v_hotspots_lga AS
SELECT
    state_name,
    lga_name,
    admin_boundary_id,
    COUNT(*)                            AS incident_count,
    SUM(fatalities)                     AS total_fatalities,
    SUM(injuries)                       AS total_injuries,
    AVG(confidence_score)               AS avg_confidence,
    array_agg(DISTINCT crime_category)  AS crime_categories,
    MAX(event_date)                     AS latest_incident_date
FROM public.incidents
WHERE is_deleted = FALSE
  AND is_canonical = TRUE
  AND event_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY state_name, lga_name, admin_boundary_id;

COMMENT ON VIEW public.v_hotspots_lga IS
    'Rolling 12-month LGA-level crime aggregation for hotspot mapping and dashboard use.';


-- 9.3  Source coverage summary
CREATE VIEW public.v_source_coverage AS
SELECT
    s.source_code,
    s.source_name,
    s.source_type,
    s.default_confidence,
    COUNT(i.id)                         AS total_incidents,
    MIN(i.event_date)                   AS earliest_record,
    MAX(i.event_date)                   AS latest_record,
    AVG(i.confidence_score)             AS avg_confidence,
    COUNT(i.id) FILTER (
        WHERE i.verification_status = 'corroborated'
    )                                   AS corroborated_count,
    MAX(pr.run_started_at)              AS last_pipeline_run
FROM public.sources s
LEFT JOIN public.incidents i ON i.primary_source_id = s.id AND i.is_deleted = FALSE
LEFT JOIN public.pipeline_runs pr ON pr.source_id = s.id AND pr.status = 'completed'
GROUP BY s.id, s.source_code, s.source_name, s.source_type, s.default_confidence;

COMMENT ON VIEW public.v_source_coverage IS
    'Per-source record counts, date coverage, and last pipeline run — '
    'useful for monitoring ingestion health and source completeness.';


-- =============================================================================
-- SECTION 10 — CONFIDENCE SCORE FUNCTION
-- =============================================================================

-- Computes a composite confidence score when promoting a staging record.
-- Inputs: base source confidence, NLP confidence, geo_precision, date_precision,
--         and number of corroborating sources.
CREATE OR REPLACE FUNCTION public.compute_confidence(
    p_source_confidence     NUMERIC,
    p_nlp_confidence        NUMERIC,
    p_geo_precision         TEXT,
    p_date_precision        TEXT,
    p_corroborating_sources INTEGER
) RETURNS NUMERIC LANGUAGE plpgsql AS $$
DECLARE
    v_score     NUMERIC := 0;
    v_geo_w     NUMERIC;
    v_date_w    NUMERIC;
    v_corr_w    NUMERIC;
BEGIN
    -- Geo precision weight
    v_geo_w := CASE p_geo_precision
        WHEN 'coordinates' THEN 1.0
        WHEN 'settlement'  THEN 0.85
        WHEN 'lga'         THEN 0.65
        WHEN 'state'       THEN 0.40
        ELSE 0.10
    END;

    -- Date precision weight
    v_date_w := CASE p_date_precision
        WHEN 'exact'  THEN 1.0
        WHEN 'day'    THEN 0.90
        WHEN 'week'   THEN 0.70
        WHEN 'month'  THEN 0.50
        WHEN 'year'   THEN 0.25
        ELSE 0.10
    END;

    -- Corroboration bonus (logarithmic, caps at 0.20)
    v_corr_w := LEAST(0.20, LN(GREATEST(p_corroborating_sources + 1, 1)) * 0.10);

    -- Weighted composite
    v_score := (
        COALESCE(p_source_confidence, 0.5) * 0.35 +
        COALESCE(p_nlp_confidence,    0.5) * 0.25 +
        v_geo_w                           * 0.20 +
        v_date_w                          * 0.15 +
        v_corr_w                          * 0.05
    );

    RETURN ROUND(LEAST(GREATEST(v_score, 0.00), 1.00), 2);
END;
$$;

COMMENT ON FUNCTION public.compute_confidence IS
    'Returns a composite confidence score (0.0–1.0) from source trust, NLP extraction '
    'confidence, spatial precision, temporal precision, and corroboration count. '
    'Call this during the staging-to-canonical promotion step.';


-- =============================================================================
-- SECTION 11 — SEED DATA: SOURCE REGISTRY
-- =============================================================================

INSERT INTO public.sources
    (source_code, source_name, source_type, base_url, update_frequency, default_confidence, reliability_notes)
VALUES
    ('NBS_ANNUAL',    'National Bureau of Statistics — Crime Statistics',
     'government_statistics', 'https://nigerianstat.gov.ng', 'annual', 0.80,
     'Official police-reported aggregates. State-level only. Undercounting known.'),

    ('ACLED_API',     'Armed Conflict Location & Event Data Project',
     'ngo_database',  'https://acleddata.com', 'weekly', 0.78,
     'Best coverage for conflict/political violence. LGA-level geo available.'),

    ('NGA_WATCH',     'Nigeria Watch — Lethal Violence Database',
     'ngo_database',  'https://nigeriawatch.org', 'monthly', 0.75,
     'Covers 2006–present. GIS-tagged. Best historical violent crime dataset.'),

    ('UCDP_GED',      'UCDP Georeferenced Event Dataset',
     'academic',      'https://ucdp.uu.se', 'annual', 0.80,
     'Village-level geo precision. 1990–present. Conflict events focus.'),

    ('HDX_NIGERIA',   'Humanitarian Data Exchange — Nigeria',
     'ngo_database',  'https://data.humdata.org/group/nga', 'weekly', 0.70,
     'Aggregator; sources vary. Verify individual dataset provenance.'),

    ('PUNCH_RSS',     'Punch Newspapers',
     'news_media',    'https://punchng.com', 'realtime', 0.55,
     'High-volume. Requires NLP extraction. Sensationalism bias possible.'),

    ('VANGUARD_RSS',  'Vanguard Newspapers',
     'news_media',    'https://vanguardngr.com', 'realtime', 0.55,
     'Strong South-South/Southeast coverage.'),

    ('CHANNELS_RSS',  'Channels Television',
     'news_media',    'https://channelstv.com', 'realtime', 0.60,
     'Broadcast outlet; generally factual. Good Abuja/Lagos coverage.'),

    ('PREMIUM_TIMES', 'Premium Times Nigeria',
     'news_media',    'https://premiumtimesng.com', 'realtime', 0.65,
     'Strong investigative track record. Higher reliability than tabloids.'),

    ('TWITTER_STREAM','X (Twitter) — Nigeria Crime Stream',
     'social_media',  'https://api.twitter.com', 'realtime', 0.30,
     'Low individual confidence. Useful for early detection and corroboration signal.'),

    ('TELEGRAM_CHAN', 'Telegram — Nigeria Security Channels',
     'social_media',  NULL, 'realtime', 0.25,
     'Very low confidence. Monitor for signal only; always require corroboration.'),

    ('CROWDSOURCE',   'ariran Public Tipline',
     'crowdsourced',  NULL, 'realtime', 0.20,
     'Human-submitted reports via web form or USSD. Require review before promotion.');


-- =============================================================================
-- SECTION 12 — SEED DATA: CRIME TAXONOMY
-- =============================================================================

INSERT INTO public.crime_taxonomy (category, crime_type, crime_subtype) VALUES
    -- Violent crime
    ('violent_crime',   'homicide',          'murder'),
    ('violent_crime',   'homicide',          'manslaughter'),
    ('violent_crime',   'armed_robbery',     'bank_robbery'),
    ('violent_crime',   'armed_robbery',     'highway_robbery'),
    ('violent_crime',   'armed_robbery',     'one_chance'),
    ('violent_crime',   'kidnapping',        'ransom_kidnapping'),
    ('violent_crime',   'kidnapping',        'ritual_kidnapping'),
    ('violent_crime',   'kidnapping',        'child_abduction'),
    ('violent_crime',   'sexual_violence',   'rape'),
    ('violent_crime',   'sexual_violence',   'defilement'),
    ('violent_crime',   'assault',           'grievous_bodily_harm'),
    ('violent_crime',   'cult_violence',     NULL),
    ('violent_crime',   'mob_justice',       NULL),
    -- Terrorism / insurgency
    ('terrorism',       'bombing',           'suicide_bombing'),
    ('terrorism',       'bombing',           'ied'),
    ('terrorism',       'mass_attack',       'boko_haram'),
    ('terrorism',       'mass_attack',       'iswap'),
    ('terrorism',       'banditry',          NULL),
    ('terrorism',       'farmer_herder',     NULL),
    -- Property crime
    ('property_crime',  'burglary',          'residential'),
    ('property_crime',  'burglary',          'commercial'),
    ('property_crime',  'theft',             'vehicle_theft'),
    ('property_crime',  'theft',             'phone_snatching'),
    ('property_crime',  'arson',             NULL),
    ('property_crime',  'vandalism',         NULL),
    -- Financial crime
    ('financial_crime', 'fraud',             'advance_fee_fraud'),
    ('financial_crime', 'fraud',             'banking_fraud'),
    ('financial_crime', 'cybercrime',        'identity_theft'),
    ('financial_crime', 'cybercrime',        'romance_scam'),
    ('financial_crime', 'money_laundering',  NULL),
    ('financial_crime', 'corruption',        'bribery'),
    ('financial_crime', 'corruption',        'embezzlement'),
    -- Drug-related
    ('drug_crime',      'trafficking',       NULL),
    ('drug_crime',      'possession',        NULL),
    ('drug_crime',      'production',        NULL),
    -- Political violence
    ('political_violence', 'electoral',      'ballot_snatching'),
    ('political_violence', 'electoral',      'assassination'),
    ('political_violence', 'communal',       'ethnic_conflict'),
    ('political_violence', 'communal',       'land_dispute'),
    -- Other
    ('other',           'road_crash',        NULL),
    ('other',           'unknown',           NULL);


-- =============================================================================
-- GRANTS (adjust roles to your deployment)
-- =============================================================================

-- Read-only role for API layer and researchers
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ariran_reader') THEN
        CREATE ROLE ariran_reader;
    END IF;
END $$;

GRANT USAGE ON SCHEMA public TO ariran_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ariran_reader;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO ariran_reader;

-- Pipeline write role
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ariran_pipeline') THEN
        CREATE ROLE ariran_pipeline;
    END IF;
END $$;

GRANT USAGE ON SCHEMA public, raw, audit TO ariran_pipeline;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO ariran_pipeline;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA raw TO ariran_pipeline;
GRANT INSERT ON ALL TABLES IN SCHEMA audit TO ariran_pipeline;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO ariran_pipeline;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA raw TO ariran_pipeline;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA audit TO ariran_pipeline;


-- =============================================================================
-- END OF SCHEMA
-- Total objects: 3 schemas · 14 tables · 6 views/functions · 2 roles
-- Estimated initial size: ~50MB empty; scales to multi-GB with full ingestion
-- =============================================================================
