"""Build training sets: labels joined with point-in-time correct features."""

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg

from feature_store.historical import get_historical_features

LABEL_SQL_DIR = Path(__file__).parent / "label_sql"

# How far each label looks into the future. A snapshot can only be used once
# this much time has passed, otherwise its labels are not final yet.
LABEL_HORIZONS = {"purchase_next_30d": timedelta(days=30)}


def monthly_snapshots(start: date, end: date) -> list[datetime]:
    """Midnight UTC of the first day of every month from start to end, inclusive."""
    snapshots = []
    year, month = start.year, start.month
    if start.day > 1:
        year, month = (year, month + 1) if month < 12 else (year + 1, 1)
    while date(year, month, 1) <= end:
        snapshots.append(datetime(year, month, 1, tzinfo=UTC))
        year, month = (year, month + 1) if month < 12 else (year + 1, 1)
    return snapshots


def build_training_set(
    conn: psycopg.Connection,
    snapshots: Sequence[datetime],
    feature_names: Sequence[str],
    label_name: str = "purchase_next_30d",
) -> pd.DataFrame:
    """One row per entity and snapshot: entity_id, event_timestamp, the features, label.

    Features are what was known at the snapshot (point-in-time join); the label
    is what happened after it. Raises ValueError for a snapshot whose label
    window reaches past the end of the raw data.
    """
    label_sql = (LABEL_SQL_DIR / f"{label_name}.sql").read_text()
    horizon = LABEL_HORIZONS[label_name]
    data_end = conn.execute("SELECT max(order_ts) AS ts FROM raw.orders").fetchone()["ts"]

    entity_rows = []
    labels = []
    for as_of in snapshots:
        if data_end is None or as_of + horizon > data_end:
            raise ValueError(
                f"Labels for {as_of.date()} are not final: they need data up to "
                f"{(as_of + horizon).date()}, but the raw data ends at {data_end}"
            )
        for row in conn.execute(label_sql, {"as_of": as_of}).fetchall():
            entity_rows.append((row["entity_id"], as_of))
            labels.append(row["label"])

    frame = pd.DataFrame(
        get_historical_features(conn, entity_rows, feature_names),
        columns=["entity_id", "event_timestamp", *feature_names],
    )
    frame["label"] = labels
    return frame
