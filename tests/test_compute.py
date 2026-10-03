from datetime import UTC, date, datetime, timedelta

import pytest

from feature_store import definitions
from feature_store.compute import backfill, compute_features
from feature_store.datagen import GeneratorConfig, generate, load
from feature_store.registry import FeatureDefinition, FeatureRegistry

VIEW = "user_purchase_stats"
AS_OF = datetime(2026, 2, 15, tzinfo=UTC)
SMALL = GeneratorConfig(seed=7, start=date(2026, 1, 1), days=60, n_users=150, n_products=30)


@pytest.fixture
def conn(conn):
    """Raw tables filled with a small dataset, definitions registered."""
    load(conn, generate(SMALL))
    definitions.apply(FeatureRegistry(conn))
    return conn


def stored_values(conn):
    return conn.execute(
        """
        SELECT fd.feature_name, fv.entity_id, fv.event_timestamp, fv.value
        FROM feature_store.feature_values fv
        JOIN feature_store.feature_definitions fd USING (feature_id)
        ORDER BY 1, 2, 3
        """
    ).fetchall()


def user_values(conn, user_id, as_of=AS_OF):
    rows = conn.execute(
        """
        SELECT fd.feature_name, fv.value
        FROM feature_store.feature_values fv
        JOIN feature_store.feature_definitions fd USING (feature_id)
        WHERE fv.entity_id = %s AND fv.event_timestamp = %s
        """,
        (user_id, as_of),
    ).fetchall()
    return {row["feature_name"]: row["value"] for row in rows}


def add_order(conn, user_id, order_ts, total):
    conn.execute(
        """
        INSERT INTO raw.orders (order_id, user_id, order_ts, total_amount)
        SELECT max(order_id) + 1, %s, %s, %s FROM raw.orders
        """,
        (user_id, order_ts, total),
    )


def test_compute_writes_one_value_per_user_and_feature(conn):
    users = conn.execute(
        "SELECT count(*) AS n FROM raw.users WHERE signup_ts < %s", (AS_OF,)
    ).fetchone()["n"]

    written = compute_features(conn, VIEW, AS_OF)

    assert written == users * len(definitions.FEATURES)
    assert {row["event_timestamp"] for row in stored_values(conn)} == {AS_OF}


def test_compute_matches_the_raw_data(conn):
    user_id = conn.execute(
        "SELECT user_id FROM raw.orders WHERE order_ts < %s ORDER BY order_ts DESC LIMIT 1",
        (AS_OF,),
    ).fetchone()["user_id"]
    expected = conn.execute(
        """
        SELECT count(*) AS orders, avg(total_amount)::double precision AS avg_value
        FROM raw.orders
        WHERE user_id = %(user)s
          AND order_ts < %(as_of)s
          AND order_ts >= %(as_of)s - INTERVAL '30 days'
        """,
        {"user": user_id, "as_of": AS_OF},
    ).fetchone()

    compute_features(conn, VIEW, AS_OF)

    values = user_values(conn, user_id)
    assert values["order_count_30d"] == expected["orders"]
    assert values["avg_order_value_30d"] == pytest.approx(expected["avg_value"])
    assert 0 <= values["days_since_last_order"] < 30


def test_user_without_orders_gets_zero_counts_and_null_stats(conn):
    user_id = conn.execute(
        """
        SELECT user_id FROM raw.users u
        WHERE signup_ts < %s
          AND NOT EXISTS (SELECT 1 FROM raw.orders o WHERE o.user_id = u.user_id)
        LIMIT 1
        """,
        (AS_OF,),
    ).fetchone()["user_id"]

    compute_features(conn, VIEW, AS_OF)

    values = user_values(conn, user_id)
    assert values["order_count_30d"] == 0
    assert values["avg_order_value_30d"] is None
    assert values["days_since_last_order"] is None


def test_computing_the_same_moment_twice_changes_nothing(conn):
    compute_features(conn, VIEW, AS_OF)
    before = stored_values(conn)

    written = compute_features(conn, VIEW, AS_OF)

    assert written == 0
    assert stored_values(conn) == before


def test_events_at_or_after_as_of_do_not_affect_values(conn):
    compute_features(conn, VIEW, AS_OF)
    user_id = conn.execute("SELECT min(user_id) AS id FROM raw.users").fetchone()["id"]
    add_order(conn, user_id, AS_OF, 500)
    add_order(conn, user_id, AS_OF + timedelta(hours=1), 500)

    assert compute_features(conn, VIEW, AS_OF) == 0


def test_recompute_picks_up_late_arriving_events(conn):
    compute_features(conn, VIEW, AS_OF)
    user_id = conn.execute("SELECT min(user_id) AS id FROM raw.users").fetchone()["id"]
    before = user_values(conn, user_id)
    add_order(conn, user_id, AS_OF - timedelta(hours=1), 500)

    written = compute_features(conn, VIEW, AS_OF)

    assert written > 0
    assert user_values(conn, user_id)["order_count_30d"] == before["order_count_30d"] + 1


def test_registered_feature_missing_from_sql_is_an_error(conn):
    FeatureRegistry(conn).register_feature(FeatureDefinition("not_in_sql", VIEW))

    with pytest.raises(ValueError, match="not_in_sql"):
        compute_features(conn, VIEW, AS_OF)


def test_naive_timestamp_is_rejected(conn):
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_features(conn, VIEW, datetime(2026, 2, 15))


def test_backfill_writes_one_snapshot_per_day(conn):
    backfill(conn, VIEW, date(2026, 2, 10), date(2026, 2, 12))

    assert {row["event_timestamp"] for row in stored_values(conn)} == {
        datetime(2026, 2, 10, tzinfo=UTC),
        datetime(2026, 2, 11, tzinfo=UTC),
        datetime(2026, 2, 12, tzinfo=UTC),
    }


def test_backfill_twice_changes_nothing(conn):
    backfill(conn, VIEW, date(2026, 2, 10), date(2026, 2, 12))
    before = stored_values(conn)

    assert backfill(conn, VIEW, date(2026, 2, 10), date(2026, 2, 12)) == 0
    assert stored_values(conn) == before
