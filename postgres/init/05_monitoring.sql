-- The outcome of every monitoring check. One row per thing checked, metric and
-- snapshot: checking the same snapshot again replaces the earlier outcome.
CREATE TABLE IF NOT EXISTS feature_store.monitor_logs (
    kind         TEXT NOT NULL,              -- 'feature' or 'model'
    name         TEXT NOT NULL,              -- the feature's or the model's name
    metric       TEXT NOT NULL,              -- 'null_share', 'psi', 'roc_auc', ...
    snapshot_at  TIMESTAMPTZ NOT NULL,       -- the moment the checked values describe
    value        DOUBLE PRECISION,
    status       TEXT NOT NULL,
    message      TEXT NOT NULL,
    checked_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (kind, name, metric, snapshot_at),
    CONSTRAINT valid_kind CHECK (kind IN ('feature', 'model')),
    CONSTRAINT valid_status CHECK (status IN ('ok', 'warning', 'alert'))
);
