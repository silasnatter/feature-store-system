-- A feature view groups features that share an entity and are computed together.
CREATE TABLE IF NOT EXISTS feature_store.feature_views (
    view_id      INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    view_name    TEXT NOT NULL UNIQUE,
    entity_type  TEXT NOT NULL,            -- 'user', 'product'
    description  TEXT,
    owner        TEXT,
    ttl          INTERVAL,                 -- how long a value stays valid; NULL = forever
    source_sql   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feature_store.feature_definitions (
    feature_id       INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    view_id          INT NOT NULL REFERENCES feature_store.feature_views (view_id),
    feature_name     TEXT NOT NULL UNIQUE,
    description      TEXT,
    dtype            TEXT NOT NULL DEFAULT 'float',
    status           TEXT NOT NULL DEFAULT 'active',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Monitoring thresholds
    expected_min     DOUBLE PRECISION,
    expected_max     DOUBLE PRECISION,
    null_threshold   DOUBLE PRECISION NOT NULL DEFAULT 0.01,
    freshness_hours  INT NOT NULL DEFAULT 24,

    CONSTRAINT valid_dtype CHECK (dtype IN ('float', 'int', 'bool')),
    CONSTRAINT valid_status CHECK (status IN ('active', 'deprecated', 'testing'))
);

-- event_timestamp: the moment the value became true (what point-in-time joins use).
-- created_at: when the row was written (differs from event_timestamp on backfills).
-- The primary key index also serves "latest value at or before t" lookups.
CREATE TABLE IF NOT EXISTS feature_store.feature_values (
    feature_id       INT NOT NULL REFERENCES feature_store.feature_definitions (feature_id),
    entity_id        BIGINT NOT NULL,
    event_timestamp  TIMESTAMPTZ NOT NULL,
    value            DOUBLE PRECISION,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (feature_id, entity_id, event_timestamp)
);

-- For reading one whole snapshot of a feature ("all values as of this moment"),
-- which validation, monitoring and materialisation do.
CREATE INDEX IF NOT EXISTS idx_feature_values_feature_time
    ON feature_store.feature_values (feature_id, event_timestamp);
