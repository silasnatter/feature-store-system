from datetime import UTC, datetime

import pytest

from feature_store.registry import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureStatus,
    FeatureView,
    NotRegisteredError,
)
from feature_store.validation import validate_snapshot

VIEW = "stats"
AS_OF = datetime(2026, 3, 2, tzinfo=UTC)


@pytest.fixture
def conn(conn):
    """`spend`: 0 to 100, at most 25% nulls. `clicks`: no limits, no nulls allowed."""
    registry = FeatureRegistry(conn)
    registry.register_view(FeatureView(VIEW, "user"))
    registry.register_feature(
        FeatureDefinition("spend", VIEW, expected_min=0, expected_max=100, null_threshold=0.25)
    )
    registry.register_feature(FeatureDefinition("clicks", VIEW, null_threshold=0))
    return conn


def store(conn, feature_name, values, event_timestamp=AS_OF):
    """Store one value per entity, entity ids counting from 1."""
    for entity_id, value in enumerate(values, start=1):
        conn.execute(
            """
            INSERT INTO feature_store.feature_values
                (feature_id, entity_id, event_timestamp, value)
            SELECT feature_id, %s, %s, %s
            FROM feature_store.feature_definitions
            WHERE feature_name = %s
            """,
            (entity_id, event_timestamp, value, feature_name),
        )


def test_snapshot_within_all_thresholds_has_no_problems(conn):
    store(conn, "spend", [0.0, 50.0, 100.0, None])  # exactly 25% null: still allowed
    store(conn, "clicks", [1.0, 2.0, 3.0, 4.0])

    assert validate_snapshot(conn, VIEW, AS_OF) == []


def test_feature_without_values_for_the_moment_is_a_problem(conn):
    store(conn, "spend", [10.0])
    store(conn, "clicks", [1.0], event_timestamp=datetime(2026, 3, 1, tzinfo=UTC))

    assert validate_snapshot(conn, VIEW, AS_OF) == [
        "clicks: no values for 2026-03-02T00:00:00+00:00"
    ]


def test_too_many_nulls_is_a_problem(conn):
    store(conn, "spend", [10.0, None, None, 20.0])
    store(conn, "clicks", [1.0, 2.0, 3.0, 4.0])

    assert validate_snapshot(conn, VIEW, AS_OF) == [
        "spend: 50.0% of values are null, allowed 25.0%"
    ]


def test_values_outside_the_expected_range_are_a_problem(conn):
    store(conn, "spend", [-5.0, 50.0, 250.0])
    store(conn, "clicks", [-1.0, 1_000_000.0, 3.0])  # no limits defined for clicks

    assert validate_snapshot(conn, VIEW, AS_OF) == [
        "spend: minimum -5 is below the expected 0",
        "spend: maximum 250 is above the expected 100",
    ]


def test_only_nulls_is_reported_as_nulls_not_as_range(conn):
    store(conn, "spend", [None, None])
    store(conn, "clicks", [1.0, 2.0])

    assert validate_snapshot(conn, VIEW, AS_OF) == [
        "spend: 100.0% of values are null, allowed 25.0%"
    ]


def test_deprecated_features_are_not_checked(conn):
    FeatureRegistry(conn).register_feature(
        FeatureDefinition("old", VIEW, status=FeatureStatus.DEPRECATED)
    )
    store(conn, "spend", [10.0])
    store(conn, "clicks", [1.0])

    assert validate_snapshot(conn, VIEW, AS_OF) == []


def test_unknown_view_raises(conn):
    with pytest.raises(NotRegisteredError):
        validate_snapshot(conn, "no_such_view", AS_OF)
