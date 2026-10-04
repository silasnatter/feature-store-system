from datetime import UTC, date, datetime, timedelta

import pytest

from feature_store import definitions
from feature_store.compute import compute_features
from feature_store.datagen import GeneratorConfig, generate, load
from feature_store.registry import FeatureRegistry
from feature_store.training import build_training_set, monthly_snapshots

VIEW = "user_purchase_stats"
FEATURES = ["order_count_30d", "avg_order_value_30d", "days_since_last_order", "view_count_7d"]
SNAPSHOT = datetime(2026, 1, 20, tzinfo=UTC)
# 60 days of events from 1 January: labels for 20 January are final, for 15 February not
SMALL = GeneratorConfig(seed=7, start=date(2026, 1, 1), days=60, n_users=150, n_products=30)


@pytest.fixture
def conn(conn):
    load(conn, generate(SMALL))
    definitions.apply(FeatureRegistry(conn))
    compute_features(conn, VIEW, SNAPSHOT)
    return conn


def add_order(conn, user_id, order_ts):
    conn.execute(
        """
        INSERT INTO raw.orders (order_id, user_id, order_ts, total_amount)
        SELECT max(order_id) + 1, %s, %s, 50 FROM raw.orders
        """,
        (user_id, order_ts),
    )


def test_monthly_snapshots():
    assert monthly_snapshots(date(2026, 11, 15), date(2027, 2, 1)) == [
        datetime(2026, 12, 1, tzinfo=UTC),
        datetime(2027, 1, 1, tzinfo=UTC),
        datetime(2027, 2, 1, tzinfo=UTC),
    ]
    assert monthly_snapshots(date(2026, 2, 1), date(2026, 2, 1)) == [
        datetime(2026, 2, 1, tzinfo=UTC)
    ]


def test_one_row_per_user_with_features_and_label(conn):
    users = conn.execute(
        "SELECT count(*) AS n FROM raw.users WHERE signup_ts < %s", (SNAPSHOT,)
    ).fetchone()["n"]

    frame = build_training_set(conn, [SNAPSHOT], FEATURES)

    assert list(frame.columns) == ["entity_id", "event_timestamp", *FEATURES, "label"]
    assert len(frame) == users
    assert frame["entity_id"].is_unique
    assert set(frame["label"]) == {0, 1}
    assert frame["order_count_30d"].notna().all()


def test_label_matches_orders_after_the_snapshot(conn):
    buyers = conn.execute(
        """
        SELECT DISTINCT o.user_id
        FROM raw.orders o
        JOIN raw.users u USING (user_id)
        WHERE u.signup_ts < %(t)s
          AND o.order_ts >= %(t)s
          AND o.order_ts < %(t)s + INTERVAL '30 days'
        """,
        {"t": SNAPSHOT},
    ).fetchall()

    frame = build_training_set(conn, [SNAPSHOT], FEATURES)

    assert set(frame.loc[frame["label"] == 1, "entity_id"]) == {row["user_id"] for row in buyers}


def test_an_order_is_never_both_feature_and_label(conn):
    """An order exactly at the snapshot counts for the label, not for the features."""
    before = build_training_set(conn, [SNAPSHOT], FEATURES).set_index("entity_id")
    user_id = int(before.index[before["label"] == 0][0])
    add_order(conn, user_id, SNAPSHOT)
    compute_features(conn, VIEW, SNAPSHOT)

    after = build_training_set(conn, [SNAPSHOT], FEATURES).set_index("entity_id")

    assert after.loc[user_id, "label"] == 1
    assert after.loc[user_id, "order_count_30d"] == before.loc[user_id, "order_count_30d"]


def test_snapshot_with_unfinished_label_window_is_rejected(conn):
    with pytest.raises(ValueError, match="not final"):
        build_training_set(conn, [SNAPSHOT + timedelta(days=26)], FEATURES)
