from datetime import UTC, datetime, timedelta

import pytest

from feature_store.online import (
    TIMESTAMP_FIELD,
    OnlineRow,
    latest_values,
    materialize,
    read_online,
    write_online,
)
from feature_store.registry import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureView,
    NotRegisteredError,
)

VIEW = "stats"


def day(n: int, hour: int = 0) -> datetime:
    return datetime(2026, 3, n, hour, tzinfo=UTC)


# --- write_online / read_online: Redis only ---------------------------------


def test_write_creates_one_hash_per_entity(redis_client):
    write_online(
        redis_client,
        VIEW,
        [
            OnlineRow(1, day(2), {"spend": 20.0, "clicks": 5.0}),
            OnlineRow(2, day(2), {"spend": 70.0, "clicks": 0.0}),
        ],
    )

    assert sorted(redis_client.keys("*")) == ["stats:1", "stats:2"]
    assert redis_client.type("stats:1") == "hash"
    assert float(redis_client.hget("stats:1", "spend")) == 20.0
    assert float(redis_client.hget("stats:2", "clicks")) == 0.0


def test_write_stores_the_event_timestamp(redis_client):
    write_online(redis_client, VIEW, [OnlineRow(1, day(2), {"spend": 20.0})])

    assert redis_client.hget("stats:1", TIMESTAMP_FIELD) == "2026-03-02T00:00:00+00:00"


def test_write_does_not_store_null_values(redis_client):
    write_online(redis_client, VIEW, [OnlineRow(1, day(2), {"spend": None, "clicks": 5.0})])

    assert not redis_client.hexists("stats:1", "spend")
    assert redis_client.hexists("stats:1", "clicks")


def test_write_replaces_the_previous_hash(redis_client):
    write_online(redis_client, VIEW, [OnlineRow(1, day(1), {"spend": 10.0, "clicks": 5.0})])

    write_online(redis_client, VIEW, [OnlineRow(1, day(2), {"spend": None, "clicks": 6.0})])

    # spend became null, so its old value must not linger
    assert redis_client.hgetall("stats:1") == {
        "clicks": "6.0",
        TIMESTAMP_FIELD: "2026-03-02T00:00:00+00:00",
    }


def test_write_with_no_rows_does_nothing(redis_client):
    write_online(redis_client, VIEW, [])

    assert redis_client.keys("*") == []


def test_read_returns_floats_for_the_requested_features(redis_client):
    write_online(redis_client, VIEW, [OnlineRow(1, day(2), {"spend": 20.5, "clicks": 5.0})])

    assert read_online(redis_client, VIEW, 1, ["spend", "clicks"]) == {"spend": 20.5, "clicks": 5.0}
    assert read_online(redis_client, VIEW, 1, ["clicks"]) == {"clicks": 5.0}


def test_read_gives_none_for_missing_fields_and_entities(redis_client):
    write_online(redis_client, VIEW, [OnlineRow(1, day(2), {"spend": None, "clicks": 5.0})])

    assert read_online(redis_client, VIEW, 1, ["spend", "clicks"]) == {"spend": None, "clicks": 5.0}
    assert read_online(redis_client, VIEW, 99, ["spend", "clicks"]) == {
        "spend": None,
        "clicks": None,
    }


# --- latest_values / materialize: Postgres to Redis --------------------------


@pytest.fixture
def conn(conn):
    """The view `stats` with two features and a TTL of 2 days."""
    registry = FeatureRegistry(conn)
    registry.register_view(FeatureView(VIEW, "user", ttl=timedelta(days=2)))
    registry.register_feature(FeatureDefinition("spend", VIEW))
    registry.register_feature(FeatureDefinition("clicks", VIEW))
    return conn


def store(conn, feature_name, entity_id, event_timestamp, value):
    conn.execute(
        """
        INSERT INTO feature_store.feature_values (feature_id, entity_id, event_timestamp, value)
        SELECT feature_id, %s, %s, %s
        FROM feature_store.feature_definitions
        WHERE feature_name = %s
        """,
        (entity_id, event_timestamp, value, feature_name),
    )


def test_latest_values_picks_the_newest_snapshot_per_entity(conn):
    for n, spend in [(1, 10.0), (2, 20.0), (5, 999.0)]:
        store(conn, "spend", 1, day(n), spend)
        store(conn, "clicks", 1, day(n), None)
    store(conn, "spend", 2, day(1), 70.0)

    rows = latest_values(conn, VIEW, as_of=day(3))

    assert sorted(rows) == [
        OnlineRow(1, day(2), {"clicks": None, "spend": 20.0}),  # day 5 is after as_of
        OnlineRow(2, day(1), {"spend": 70.0}),
    ]


def test_latest_values_leaves_out_expired_entities(conn):
    store(conn, "spend", 1, day(1), 10.0)

    assert latest_values(conn, VIEW, as_of=day(3)) != []
    assert latest_values(conn, VIEW, as_of=day(3, hour=1)) == []


def test_latest_values_for_unknown_view_raises(conn):
    with pytest.raises(NotRegisteredError):
        latest_values(conn, "no_such_view", as_of=day(3))


def test_materialize_copies_latest_values_to_redis(conn, redis_client):
    store(conn, "spend", 1, day(1), 10.0)
    store(conn, "spend", 1, day(2), 20.0)
    store(conn, "clicks", 1, day(2), 5.0)

    written = materialize(conn, redis_client, VIEW, as_of=day(2))

    assert written == 1
    assert read_online(redis_client, VIEW, 1, ["spend", "clicks"]) == {"spend": 20.0, "clicks": 5.0}


def test_materialize_removes_entities_without_valid_values(conn, redis_client):
    store(conn, "spend", 1, day(1), 10.0)
    store(conn, "spend", 2, day(4), 70.0)
    materialize(conn, redis_client, VIEW, as_of=day(2))
    assert redis_client.keys("*") == ["stats:1"]

    materialize(conn, redis_client, VIEW, as_of=day(4))  # entity 1 has expired by now

    assert redis_client.keys("*") == ["stats:2"]
