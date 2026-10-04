-- One row per prediction served by the API, kept for monitoring.
-- features: the exact values the model saw; features_as_of: their event timestamp.
CREATE TABLE IF NOT EXISTS feature_store.predictions (
    prediction_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    entity_id       BIGINT NOT NULL,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    probability     DOUBLE PRECISION NOT NULL,
    features        JSONB NOT NULL,
    features_as_of  TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_model_time
    ON feature_store.predictions (model_name, predicted_at);
