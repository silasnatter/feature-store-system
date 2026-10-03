from collections import defaultdict
from datetime import UTC, date, datetime, timedelta

from feature_store.datagen import GeneratorConfig, generate, load

SMALL = GeneratorConfig(seed=7, start=date(2026, 1, 1), days=60, n_users=150, n_products=30)


def test_same_seed_gives_same_data():
    assert generate(SMALL) == generate(SMALL)


def test_different_seed_gives_different_data():
    other = GeneratorConfig(seed=8, start=SMALL.start, days=SMALL.days, n_users=150, n_products=30)

    assert generate(SMALL).orders != generate(other).orders


def test_events_fall_inside_the_window_and_after_signup():
    data = generate(SMALL)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(days=SMALL.days)
    signup = dict(data.users)

    assert data.product_views and data.orders
    for _, user_id, _, ts in data.product_views:
        assert window_start <= ts < window_end
        assert ts >= signup[user_id]
    for _, user_id, ts, _ in data.orders:
        # An order is placed up to 15 minutes after the last view of the day
        assert window_start <= ts < window_end + timedelta(minutes=15)
        assert ts >= signup[user_id]


def test_order_totals_match_their_items():
    data = generate(SMALL)
    item_totals = defaultdict(int)
    for order_id, _, quantity, unit_price in data.order_items:
        item_totals[order_id] += quantity * unit_price

    assert {order_id: total for order_id, _, _, total in data.orders} == item_totals


def test_load_replaces_raw_tables(conn):
    data = generate(SMALL)

    load(conn, data)
    counts = load(conn, data)  # second load must not duplicate rows

    assert counts["users"] == SMALL.n_users
    for table, expected in counts.items():
        row = conn.execute(f"SELECT count(*) AS n FROM raw.{table}").fetchone()
        assert row["n"] == expected
    total = conn.execute("SELECT sum(total_amount) AS total FROM raw.orders").fetchone()["total"]
    assert total == sum(order[3] for order in data.orders)
