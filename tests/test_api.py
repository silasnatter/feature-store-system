from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from feature_store.api import app, get_conn, get_redis
from feature_store.monitoring import log_checks
from feature_store.online import materialize
from feature_store.registry import FeatureDefinition, FeatureRegistry, FeatureView
from feature_store.validation import Check

VIEW = "stats"


def day(n: int) -> datetime:
    return datetime(2026, 3, n, tzinfo=UTC)


@pytest.fixture
def api(conn, redis_client):
    """The app wired to the test database and test Redis, with a little data.

    User 1 has spend 10 on day 1 and spend 20, clicks null on day 2; Redis holds day 2.
    """
    registry = FeatureRegistry(conn)
    registry.register_view(FeatureView(VIEW, "user", ttl=timedelta(days=2)))
    registry.register_feature(FeatureDefinition("spend", VIEW))
    registry.register_feature(FeatureDefinition("clicks", VIEW))
    for feature_name, timestamp, value in [
        ("spend", day(1), 10.0),
        ("spend", day(2), 20.0),
        ("clicks", day(2), None),
    ]:
        conn.execute(
            """
            INSERT INTO feature_store.feature_values (feature_id, entity_id, event_timestamp, value)
            SELECT feature_id, 1, %s, %s
            FROM feature_store.feature_definitions
            WHERE feature_name = %s
            """,
            (timestamp, value, feature_name),
        )
    materialize(conn, redis_client, VIEW, as_of=day(2))

    app.dependency_overrides[get_conn] = lambda: conn
    app.dependency_overrides[get_redis] = lambda: redis_client
    yield TestClient(app)  # not used as a context manager, so the real lifespan never runs
    app.dependency_overrides.clear()


def test_health(api):
    response = api.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_features(api):
    response = api.get("/features")

    assert response.status_code == 200
    features = response.json()["features"]
    assert [feature["feature_name"] for feature in features] == ["clicks", "spend"]
    assert features[0]["view_name"] == VIEW


def test_online_returns_everything_stored_for_the_entity(api):
    response = api.get(f"/features/online/{VIEW}/1")

    assert response.status_code == 200
    body = response.json()
    assert datetime.fromisoformat(body.pop("event_timestamp")) == day(2)
    assert body == {
        "view": VIEW,
        "entity_id": 1,
        "features": {"spend": 20.0},  # clicks is null, so it is not in Redis
    }


def test_online_with_feature_parameters_returns_exactly_those(api):
    response = api.get(f"/features/online/{VIEW}/1", params={"feature": ["clicks", "spend"]})

    assert response.json()["features"] == {"clicks": None, "spend": 20.0}


def test_online_unknown_entity_is_404(api):
    assert api.get(f"/features/online/{VIEW}/99").status_code == 404
    assert api.get("/features/online/no_such_view/1").status_code == 404


def test_historical_returns_point_in_time_values(api):
    noon = timedelta(hours=12)
    response = api.post(
        "/features/historical",
        json={
            "entity_rows": [
                {"entity_id": 1, "event_timestamp": "2026-03-01T12:00:00Z"},
                {"entity_id": 1, "event_timestamp": "2026-03-02T12:00:00Z"},
            ],
            "features": ["spend", "clicks"],
        },
    )

    assert response.status_code == 200
    rows = response.json()["rows"]
    for row in rows:
        row["event_timestamp"] = datetime.fromisoformat(row["event_timestamp"])
    assert rows == [
        {"entity_id": 1, "event_timestamp": day(1) + noon, "spend": 10.0, "clicks": None},
        {"entity_id": 1, "event_timestamp": day(2) + noon, "spend": 20.0, "clicks": None},
    ]


def test_historical_unknown_feature_is_404(api):
    response = api.post(
        "/features/historical",
        json={
            "entity_rows": [{"entity_id": 1, "event_timestamp": "2026-03-02T00:00:00Z"}],
            "features": ["no_such_feature"],
        },
    )

    assert response.status_code == 404
    assert "no_such_feature" in response.json()["detail"]


def test_historical_rejects_timestamps_without_timezone(api):
    response = api.post(
        "/features/historical",
        json={
            "entity_rows": [{"entity_id": 1, "event_timestamp": "2026-03-02T00:00:00"}],
            "features": ["spend"],
        },
    )

    assert response.status_code == 422


def test_monitor_status_combines_logged_checks_with_freshness(api, conn):
    """The fixture stored values for day 1 and day 2; both features are fresh for 24 hours."""
    log_checks(conn, "feature", day(2), [Check("spend", "psi", 0.4, "alert", "drifted")])
    log_checks(conn, "model", day(2), [Check("purchase_model", "roc_auc", 0.93, "ok", "good")])

    response = api.get("/monitor/status", params={"as_of": "2026-03-02T06:00:00Z"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "alert"
    spend = {check["metric"]: check for check in body["features"]["spend"]["checks"]}
    assert spend["psi"]["status"] == "alert"
    assert spend["psi"]["message"] == "drifted"
    assert spend["freshness_hours"]["value"] == pytest.approx(6)
    assert spend["freshness_hours"]["status"] == "ok"
    assert body["features"]["clicks"]["status"] == "ok"
    assert body["models"]["purchase_model"]["status"] == "ok"


def test_monitor_status_defaults_to_now(api):
    """Asked without a moment, it measures freshness against the clock: March is long ago."""
    body = api.get("/monitor/status").json()

    assert body["status"] == "alert"
    (freshness,) = body["features"]["spend"]["checks"]
    assert freshness["metric"] == "freshness_hours"
    assert freshness["value"] > 24


def test_monitor_status_rejects_a_moment_without_timezone(api):
    assert api.get("/monitor/status", params={"as_of": "2026-03-02T06:00:00"}).status_code == 422
