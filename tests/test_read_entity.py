from datetime import UTC, datetime

from feature_store.online import OnlineRow, read_entity, write_online

TIMESTAMP = datetime(2026, 3, 2, tzinfo=UTC)


def test_read_entity_returns_what_was_written_minus_nulls(redis_client):
    write_online(redis_client, "stats", [OnlineRow(1, TIMESTAMP, {"spend": 20.5, "clicks": None})])

    assert read_entity(redis_client, "stats", 1) == OnlineRow(1, TIMESTAMP, {"spend": 20.5})


def test_read_entity_unknown_entity_is_none(redis_client):
    assert read_entity(redis_client, "stats", 99) is None
