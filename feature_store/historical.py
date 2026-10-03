from collections.abc import Sequence
from datetime import datetime

import psycopg


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
    raise NotImplementedError
