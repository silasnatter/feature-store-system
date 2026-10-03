from datetime import UTC, date, datetime, timedelta

import psycopg

from feature_store.registry import FeatureRegistry, FeatureStatus

# The view's SQL returns one wide row per entity. to_jsonb turns it into one
# row per (entity, feature) so it fits feature_values.
# The WHERE on the update leaves unchanged rows alone, so a re-run writes nothing.
_UPSERT = """
WITH computed AS (
    {source_sql}
)
INSERT INTO feature_store.feature_values (feature_id, entity_id, event_timestamp, value)
SELECT
    fd.feature_id,
    computed.entity_id,
    %(as_of)s,
    (to_jsonb(computed) ->> fd.feature_name)::double precision
FROM computed
CROSS JOIN feature_store.feature_definitions fd
JOIN feature_store.feature_views fv USING (view_id)
WHERE fv.view_name = %(view_name)s
  AND fd.status <> 'deprecated'
ON CONFLICT (feature_id, entity_id, event_timestamp) DO UPDATE
    SET value = EXCLUDED.value, created_at = now()
    WHERE feature_values.value IS DISTINCT FROM EXCLUDED.value
"""


def compute_features(conn: psycopg.Connection, view_name: str, as_of: datetime) -> int:
    """Compute one feature view as of a moment and store the values.

    The values get `as_of` as their event timestamp. Returns the number of rows
    inserted or changed, which is 0 when the same moment is computed again.
    Does not commit.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")

    registry = FeatureRegistry(conn)
    view = registry.get_view(view_name)
    if not view.source_sql:
        raise ValueError(f"Feature view '{view_name}' has no source SQL")

    expected = {
        feature.feature_name
        for feature in registry.list_features(status=None, view_name=view_name)
        if feature.status != FeatureStatus.DEPRECATED
    }
    params = {"as_of": as_of, "view_name": view_name}
    probe = conn.execute(f"SELECT * FROM ({view.source_sql}) AS computed LIMIT 0", params)
    columns = {column.name for column in probe.description}
    missing = (expected | {"entity_id"}) - columns
    if missing:
        raise ValueError(f"SQL of view '{view_name}' does not return: {sorted(missing)}")

    return conn.execute(_UPSERT.format(source_sql=view.source_sql), params).rowcount


def backfill(conn: psycopg.Connection, view_name: str, start: date, end: date) -> int:
    """Compute the view at midnight UTC of every day from start to end, inclusive.

    Returns the number of rows inserted or changed. Does not commit.
    """
    written = 0
    day = start
    while day <= end:
        as_of = datetime(day.year, day.month, day.day, tzinfo=UTC)
        written += compute_features(conn, view_name, as_of)
        day += timedelta(days=1)
    return written
