from datetime import UTC, datetime, timedelta

import pytest

from feature_store.historical import get_historical_features
from feature_store.registry import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureView,
    NotRegisteredError,
)


def day(n: int, hour: int = 0) -> datetime:
    return datetime(2026, 3, n, hour, tzinfo=UTC)


@pytest.fixture
def conn(conn):
    """Two views: `spend` and `clicks` expire after 2 days, `segment` never expires."""
    registry = FeatureRegistry(conn)
    registry.register_view(FeatureView("with_ttl", "user", ttl=timedelta(days=2)))
    registry.register_view(FeatureView("without_ttl", "user"))
    registry.register_feature(FeatureDefinition("spend", "with_ttl"))
    registry.register_feature(FeatureDefinition("clicks", "with_ttl"))
    registry.register_feature(FeatureDefinition("segment", "without_ttl"))
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


def test_returns_the_latest_value_at_or_before_the_timestamp(conn):
    store(conn, "spend", 1, day(1), 10.0)
    store(conn, "spend", 1, day(2), 20.0)

    rows = get_historical_features(conn, [(1, day(2, hour=12))], ["spend"])

    assert rows == [{"entity_id": 1, "event_timestamp": day(2, hour=12), "spend": 20.0}]


def test_value_stamped_exactly_at_the_timestamp_is_used(conn):
    store(conn, "spend", 1, day(2), 20.0)

    rows = get_historical_features(conn, [(1, day(2))], ["spend"])

    assert rows[0]["spend"] == 20.0


def test_never_returns_a_value_from_the_future(conn):
    """The leakage test: a later value must not reach an earlier row."""
    store(conn, "spend", 1, day(1), 10.0)
    store(conn, "spend", 1, day(3), 999.0)

    rows = get_historical_features(conn, [(1, day(2))], ["spend"])

    assert rows[0]["spend"] == 10.0


def test_only_future_values_gives_none(conn):
    store(conn, "spend", 1, day(5), 999.0)

    rows = get_historical_features(conn, [(1, day(2))], ["spend"])

    assert rows[0]["spend"] is None


def test_value_older_than_the_ttl_gives_none(conn):
    store(conn, "spend", 1, day(1), 10.0)

    fresh = get_historical_features(conn, [(1, day(3))], ["spend"])
    expired = get_historical_features(conn, [(1, day(3, hour=1))], ["spend"])

    assert fresh[0]["spend"] == 10.0  # exactly 2 days old: still valid
    assert expired[0]["spend"] is None


def test_view_without_ttl_never_expires(conn):
    store(conn, "segment", 1, day(1), 3.0)

    rows = get_historical_features(conn, [(1, day(30))], ["segment"])

    assert rows[0]["segment"] == 3.0


def test_stored_null_is_returned_instead_of_an_older_value(conn):
    """A null is a real observation ("no orders"), not a gap to fill from the past."""
    store(conn, "spend", 1, day(1), 10.0)
    store(conn, "spend", 1, day(2), None)

    rows = get_historical_features(conn, [(1, day(2))], ["spend"])

    assert rows[0]["spend"] is None


def test_each_feature_is_looked_up_independently(conn):
    store(conn, "spend", 1, day(2), 20.0)
    store(conn, "clicks", 1, day(1), 5.0)
    store(conn, "segment", 1, day(1), 3.0)

    rows = get_historical_features(conn, [(1, day(2))], ["spend", "clicks", "segment"])

    assert rows == [
        {"entity_id": 1, "event_timestamp": day(2), "spend": 20.0, "clicks": 5.0, "segment": 3.0}
    ]


def test_same_entity_at_different_timestamps_and_input_order_is_kept(conn):
    store(conn, "spend", 1, day(1), 10.0)
    store(conn, "spend", 1, day(2), 20.0)
    store(conn, "spend", 2, day(1), 70.0)

    rows = get_historical_features(
        conn,
        [(1, day(2)), (2, day(1)), (1, day(1)), (3, day(1))],
        ["spend"],
    )

    assert [(row["entity_id"], row["event_timestamp"], row["spend"]) for row in rows] == [
        (1, day(2), 20.0),
        (2, day(1), 70.0),
        (1, day(1), 10.0),
        (3, day(1), None),  # entity without any values still gets a row
    ]


def test_duplicate_input_rows_are_kept(conn):
    store(conn, "spend", 1, day(1), 10.0)

    rows = get_historical_features(conn, [(1, day(1)), (1, day(1))], ["spend"])

    assert [row["spend"] for row in rows] == [10.0, 10.0]


def test_no_input_rows_gives_no_output(conn):
    assert get_historical_features(conn, [], ["spend"]) == []


def test_unknown_feature_raises(conn):
    with pytest.raises(NotRegisteredError, match="no_such_feature"):
        get_historical_features(conn, [(1, day(1))], ["spend", "no_such_feature"])
