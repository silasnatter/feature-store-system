"""Seeded synthetic e-commerce events for the raw schema.

Each user has a browsing rate, a conversion probability and a churn date, so
recent activity carries real signal about future purchases. The same config
always produces the same dataset.

    python -m feature_store.datagen --seed 42 --days 270
"""

import argparse
import math
import random
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import accumulate

import psycopg

from feature_store.db import connect

CATEGORIES = ["electronics", "home", "fashion", "sports", "books", "toys", "beauty", "garden"]
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    start: date = date(2026, 1, 1)
    days: int = 270
    n_users: int = 2000
    n_products: int = 200


@dataclass
class Dataset:
    """Rows per raw table, in the column order of the table."""

    users: list[tuple] = field(default_factory=list)  # user_id, signup_ts
    products: list[tuple] = field(default_factory=list)  # product_id, category, price
    product_views: list[tuple] = field(default_factory=list)  # view_id, user_id, product_id, ts
    orders: list[tuple] = field(default_factory=list)  # order_id, user_id, order_ts, total
    order_items: list[tuple] = field(default_factory=list)  # order_id, product_id, qty, price


def generate(config: GeneratorConfig | None = None) -> Dataset:
    config = config or GeneratorConfig()
    rng = random.Random(config.seed)
    start_ts = datetime(config.start.year, config.start.month, config.start.day, tzinfo=UTC)
    data = Dataset()

    prices = {}
    for product_id in range(1, config.n_products + 1):
        price = Decimal(round(rng.lognormvariate(3.3, 0.8) * 100)) / 100 + Decimal("0.99")
        prices[product_id] = price
        data.products.append((product_id, rng.choice(CATEGORIES), price))
    # Zipf-like popularity: a few products get most of the views
    product_ids = list(prices)
    rng.shuffle(product_ids)
    popularity = list(accumulate(1 / rank for rank in range(1, config.n_products + 1)))

    # (user_id, first active day, churn day, views per day, conversion probability)
    profiles = []
    for user_id in range(1, config.n_users + 1):
        if rng.random() < 0.6:
            signup_offset = -rng.uniform(0, 180)  # already a customer before the window
        else:
            signup_offset = rng.uniform(0, config.days)
        data.users.append((user_id, start_ts + timedelta(days=signup_offset)))
        profiles.append(
            (
                user_id,
                math.floor(signup_offset) + 1,
                signup_offset + rng.expovariate(1 / 150),
                rng.lognormvariate(-0.5, 0.8),
                rng.betavariate(2, 18),
            )
        )

    views = []  # (ts, user_id, product_id)
    orders = []  # (ts, user_id, {product_id: quantity})
    for day in range(config.days):
        day_start = start_ts + timedelta(days=day)
        for user_id, first_day, churn_day, view_rate, conversion in profiles:
            if day < first_day or day >= churn_day:
                continue
            cart: dict[int, int] = {}
            last_ts = day_start
            for _ in range(_poisson(rng, view_rate)):
                ts = day_start + timedelta(seconds=rng.randrange(SECONDS_PER_DAY))
                product_id = rng.choices(product_ids, cum_weights=popularity)[0]
                views.append((ts, user_id, product_id))
                if rng.random() < conversion:
                    cart[product_id] = cart.get(product_id, 0) + 1
                    last_ts = max(last_ts, ts)
            if cart:
                order_ts = last_ts + timedelta(seconds=rng.randrange(60, 900))
                orders.append((order_ts, user_id, cart))

    views.sort()
    for view_id, (ts, user_id, product_id) in enumerate(views, start=1):
        data.product_views.append((view_id, user_id, product_id, ts))

    orders.sort(key=lambda order: order[:2])
    for order_id, (ts, user_id, cart) in enumerate(orders, start=1):
        total = sum(prices[product_id] * quantity for product_id, quantity in cart.items())
        data.orders.append((order_id, user_id, ts, total))
        for product_id, quantity in cart.items():
            data.order_items.append((order_id, product_id, quantity, prices[product_id]))

    return data


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's algorithm; fine for the small rates used here
    threshold = math.exp(-lam)
    count, product = 0, rng.random()
    while product > threshold:
        count += 1
        product *= rng.random()
    return count


# Parents before children, so foreign keys hold while loading
_TABLES = ["users", "products", "product_views", "orders", "order_items"]


def load(conn: psycopg.Connection, data: Dataset) -> dict[str, int]:
    """Replace the contents of the raw tables with `data`. Does not commit."""
    conn.execute("TRUNCATE " + ", ".join(f"raw.{table}" for table in _TABLES))
    counts = {}
    with conn.cursor() as cur:
        for table in _TABLES:
            rows = getattr(data, table)
            with cur.copy(f"COPY raw.{table} FROM STDIN") as copy:
                for row in rows:
                    copy.write_row(row)
            counts[table] = len(rows)
    return counts


def main() -> None:
    defaults = GeneratorConfig()
    parser = argparse.ArgumentParser(description="Generate synthetic events into the raw schema")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--start", type=date.fromisoformat, default=defaults.start)
    parser.add_argument("--days", type=int, default=defaults.days)
    parser.add_argument("--users", type=int, default=defaults.n_users)
    parser.add_argument("--products", type=int, default=defaults.n_products)
    args = parser.parse_args()

    data = generate(
        GeneratorConfig(
            seed=args.seed,
            start=args.start,
            days=args.days,
            n_users=args.users,
            n_products=args.products,
        )
    )
    with connect() as conn:
        counts = load(conn, data)
    for table, count in counts.items():
        print(f"raw.{table}: {count:,} rows")


if __name__ == "__main__":
    main()
