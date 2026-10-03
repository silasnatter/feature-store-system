from collections.abc import Sequence
from datetime import datetime

import psycopg

from feature_store.registry import NotRegisteredError

# Reads as: "for every input row and every requested feature, find the newest
# stored value that the row was allowed to see".
_POINT_IN_TIME_JOIN = """
SELECT input.position, f.feature_name, v.value

-- The two Python lists become a table with one line per input row.
-- WITH ORDINALITY numbers the lines 1, 2, 3, ... so the input order survives.
FROM unnest(%(ids)s::bigint[], %(timestamps)s::timestamptz[])
     WITH ORDINALITY AS input (entity_id, ts, position)

-- Pair every input row with every requested feature.
CROSS JOIN (
    SELECT fd.feature_id, fd.feature_name, fv.ttl
    FROM feature_store.feature_definitions fd
    JOIN feature_store.feature_views fv USING (view_id)
    WHERE fd.feature_name = ANY(%(names)s)
) AS f

-- LATERAL runs this subquery once per (input row, feature) pair, and lets it
-- use that pair's entity_id, ts, feature_id and ttl.
LEFT JOIN LATERAL (
    SELECT fval.value
    FROM feature_store.feature_values fval
    WHERE fval.feature_id = f.feature_id
      AND fval.entity_id = input.entity_id
      -- Never a value from after the row's timestamp: that would be leakage
      AND fval.event_timestamp <= input.ts
      -- Not older than the view's TTL; a view without a TTL never expires
      AND (f.ttl IS NULL OR fval.event_timestamp >= input.ts - f.ttl)
    -- Of the values that remain, keep only the newest. A stored NULL counts:
    -- it is not skipped in favour of an older value.
    ORDER BY fval.event_timestamp DESC
    LIMIT 1
-- LEFT JOIN ... ON true keeps the pair even when nothing was found;
-- v.value is then NULL.
) AS v ON true
"""


def get_historical_features(
    conn: psycopg.Connection,
    entity_rows: Sequence[tuple[int, datetime]],
    feature_names: Sequence[str],
) -> list[dict]:
    """Point-in-time join: the feature values each row could have known at its timestamp.

    `entity_rows` is a list of (entity_id, event_timestamp). The result has one
    dict per input row, in the same order, with the keys `entity_id`,
    `event_timestamp` and one key per feature name.

    For each row and feature, the value is the stored one with the latest
    event_timestamp that is
      - at or before the row's timestamp (never after: that would be leakage), and
      - not older than the TTL of the feature's view (if the view has one).
    If there is no such value, the feature is None.

    Raises NotRegisteredError if a feature name is not in the registry.
    """
    # Step 1: check that every requested feature exists. Without this, a typo
    # in a feature name would silently produce no column instead of an error.
    registered = conn.execute(
        """
        SELECT feature_name
        FROM feature_store.feature_definitions
        WHERE feature_name = ANY(%s)
        """,
        (list(feature_names),),
    ).fetchall()
    missing = set(feature_names) - {row["feature_name"] for row in registered}
    if missing:
        raise NotRegisteredError(f"Features are not registered: {sorted(missing)}")

    # Step 2: nothing to look up.
    if not entity_rows:
        return []

    # Step 3: run the join. It returns one line per (input row, feature) pair:
    # the position of the input row, the feature name and the value found.
    pairs = conn.execute(
        _POINT_IN_TIME_JOIN,
        {
            "ids": [entity_id for entity_id, _ in entity_rows],
            "timestamps": [ts for _, ts in entity_rows],
            "names": list(feature_names),
        },
    ).fetchall()

    # Step 4: reshape from "one line per pair" to "one dict per input row".
    # Start with a dict per input row, already in input order, with every
    # feature set to None so the keys follow the order of `feature_names`, ...
    result = [
        {"entity_id": entity_id, "event_timestamp": ts, **dict.fromkeys(feature_names)}
        for entity_id, ts in entity_rows
    ]
    # ... then drop each value into the dict of its row. Positions from
    # Postgres start at 1, list indexes at 0.
    for pair in pairs:
        result[pair["position"] - 1][pair["feature_name"]] = pair["value"]
    return result
