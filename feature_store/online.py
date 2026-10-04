"""The online store: the latest feature values per entity, in Redis.

Layout in Redis: one hash per entity and feature view.

    key     user_purchase_stats:7
    fields  order_count_30d       -> "1.0"
            view_count_7d         -> "0.0"
            _event_timestamp      -> "2026-09-28T00:00:00+00:00"

A feature whose value is null has no field in the hash.

One extra key per view records the moment its values were last copied as of:

    key     _materialized_until:user_purchase_stats  -> "2026-09-28T00:00:00+00:00"
"""

from collections.abc import Sequence
from datetime import datetime
from typing import NamedTuple

import psycopg
import redis

from feature_store.config import Settings, get_settings
from feature_store.registry import FeatureRegistry

TIMESTAMP_FIELD = "_event_timestamp"


class OnlineRow(NamedTuple):
    """Everything the online store holds for one entity in one view."""

    entity_id: int
    event_timestamp: datetime
    values: dict[str, float | None]


def connect_redis(settings: Settings | None = None) -> redis.Redis:
    settings = settings or get_settings()
    # decode_responses: get str back from Redis instead of bytes
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def online_key(view_name: str, entity_id: int) -> str:
    return f"{view_name}:{entity_id}"


def watermark_key(view_name: str) -> str:
    return f"_materialized_until:{view_name}"


def write_online(client: redis.Redis, view_name: str, rows: Sequence[OnlineRow]) -> None:
    """Store each row as a hash, replacing whatever the entity had before.

    Per row: the key is `online_key(view_name, row.entity_id)`. The hash gets one
    field per feature whose value is not None, plus TIMESTAMP_FIELD holding
    `row.event_timestamp.isoformat()`. Fields from an earlier write that are
    no longer present must be gone afterwards.
    """

    pipe = client.pipeline()

    for row in rows:
        key = online_key(view_name=view_name, entity_id=row.entity_id)

        values_dict = {}
        for name, value in row.values.items():
            if value is not None:
                values_dict[name] = value

        values_dict[TIMESTAMP_FIELD] = row.event_timestamp.isoformat()

        pipe.delete(key)
        pipe.hset(key, mapping=values_dict)

    pipe.execute()


def read_online(
    client: redis.Redis, view_name: str, entity_id: int, feature_names: Sequence[str]
) -> dict[str, float | None]:
    """Read the given features of one entity.

    Returns one entry per feature name, as a float, or None if the entity or
    the field does not exist.
    """

    key = online_key(view_name=view_name, entity_id=entity_id)
    values = client.hmget(key, feature_names)

    result = zip(feature_names, values, strict=True)
    final_result = {}

    for name, value in result:
        value_parsed = None
        if value is not None:
            value_parsed = float(value)
        final_result[name] = value_parsed

    return final_result


def read_entity(client: redis.Redis, view_name: str, entity_id: int) -> OnlineRow | None:
    """Everything stored for one entity in one view, or None if there is nothing."""
    stored = client.hgetall(online_key(view_name, entity_id))
    if not stored:
        return None
    event_timestamp = datetime.fromisoformat(stored.pop(TIMESTAMP_FIELD))
    return OnlineRow(entity_id, event_timestamp, {k: float(v) for k, v in stored.items()})


def latest_values(conn: psycopg.Connection, view_name: str, as_of: datetime) -> list[OnlineRow]:
    """The newest value of every feature in the view, per entity, as of a moment.

    Values after `as_of` and values older than the view's TTL are ignored, so
    an entity whose values have all expired is not returned.
    """
    FeatureRegistry(conn).get_view(view_name)  # raises if the view is unknown
    found = conn.execute(
        """
        SELECT DISTINCT ON (fval.entity_id, fd.feature_name)
            fval.entity_id, fd.feature_name, fval.event_timestamp, fval.value
        FROM feature_store.feature_values fval
        JOIN feature_store.feature_definitions fd USING (feature_id)
        JOIN feature_store.feature_views fv USING (view_id)
        WHERE fv.view_name = %(view_name)s
          AND fd.status <> 'deprecated'
          AND fval.event_timestamp <= %(as_of)s
          AND (fv.ttl IS NULL OR fval.event_timestamp >= %(as_of)s - fv.ttl)
        ORDER BY fval.entity_id, fd.feature_name, fval.event_timestamp DESC
        """,
        {"view_name": view_name, "as_of": as_of},
    ).fetchall()

    rows: dict[int, OnlineRow] = {}
    for value in found:
        row = rows.get(value["entity_id"])
        if row is None:
            row = OnlineRow(value["entity_id"], value["event_timestamp"], {})
        elif value["event_timestamp"] > row.event_timestamp:
            row = row._replace(event_timestamp=value["event_timestamp"])
        row.values[value["feature_name"]] = value["value"]
        rows[row.entity_id] = row
    return list(rows.values())


def materialized_until(client: redis.Redis, view_name: str) -> datetime | None:
    """The moment the view's online values were last copied as of, or None if never."""
    stored = client.get(watermark_key(view_name))
    return datetime.fromisoformat(stored) if stored else None


def materialize(
    conn: psycopg.Connection,
    client: redis.Redis,
    view_name: str,
    as_of: datetime,
    force: bool = False,
) -> int | None:
    """Copy the latest values of a view from Postgres to Redis.

    Entities that have no valid value any more are removed from Redis.
    Returns the number of entities written.

    Redis only moves forward in time: if it already holds values as of a later
    moment, nothing is written and None is returned, unless `force` is set.
    This keeps a re-run or backfill of an old date from replacing newer values.
    """
    current_until = materialized_until(client, view_name)
    if current_until is not None and as_of < current_until and not force:
        return None

    rows = latest_values(conn, view_name, as_of)
    write_online(client, view_name, rows)

    current = {online_key(view_name, row.entity_id) for row in rows}
    stale = [key for key in client.scan_iter(match=f"{view_name}:*") if key not in current]
    if stale:
        client.delete(*stale)
    client.set(watermark_key(view_name), as_of.isoformat())
    return len(rows)
