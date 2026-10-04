from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from feature_store import api
from feature_store.api import app, get_conn, get_model, get_redis
from feature_store.modeling import (
    MODEL_FEATURES,
    MODEL_VIEW,
    LoadedModel,
    build_pipeline,
    predict_purchase,
)
from feature_store.online import OnlineRow, write_online

AS_OF = datetime(2026, 9, 28, tzinfo=UTC)
ACTIVE = {
    "order_count_30d": 3.0,
    "avg_order_value_30d": 40.0,
    "days_since_last_order": 2.0,
    "view_count_7d": 9.0,
}
INACTIVE = {
    "order_count_30d": 0.0,
    "avg_order_value_30d": None,
    "days_since_last_order": None,
    "view_count_7d": 0.0,
}


@pytest.fixture
def model():
    """A small model fitted in memory: active users buy, inactive ones do not."""
    frame = pd.DataFrame([ACTIVE, INACTIVE] * 20, columns=MODEL_FEATURES)
    labels = [1, 0] * 19 + [0, 1]
    return LoadedModel(version="7", pipeline=build_pipeline().fit(frame, labels))


@pytest.fixture
def client(conn, redis_client):
    """The app on the test database and test Redis; user 1 is active, user 2 is not."""
    write_online(
        redis_client,
        MODEL_VIEW,
        [OnlineRow(1, AS_OF, ACTIVE), OnlineRow(2, AS_OF, INACTIVE)],
    )
    app.dependency_overrides[get_conn] = lambda: conn
    app.dependency_overrides[get_redis] = lambda: redis_client
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def predict_api(client, model):
    app.dependency_overrides[get_model] = lambda: model
    return client


def logged_predictions(conn):
    return conn.execute("SELECT * FROM feature_store.predictions ORDER BY prediction_id").fetchall()


def test_predict_returns_the_model_probability_for_the_online_features(predict_api, model):
    response = predict_api.post("/predict", json={"user_id": 1})

    assert response.status_code == 200
    body = response.json()
    assert datetime.fromisoformat(body.pop("features_as_of")) == AS_OF
    assert body == {
        "user_id": 1,
        "purchase_probability": pytest.approx(predict_purchase(model.pipeline, ACTIVE)),
        "model": "purchase_model",
        "model_version": "7",
        "features": ACTIVE,
    }


def test_predict_passes_missing_features_to_the_model_as_null(predict_api, model):
    response = predict_api.post("/predict", json={"user_id": 2})

    body = response.json()
    assert body["features"] == INACTIVE
    assert body["purchase_probability"] == pytest.approx(predict_purchase(model.pipeline, INACTIVE))
    active = predict_api.post("/predict", json={"user_id": 1}).json()
    assert body["purchase_probability"] < active["purchase_probability"]


def test_predict_logs_the_prediction(predict_api, conn):
    probability = predict_api.post("/predict", json={"user_id": 2}).json()["purchase_probability"]

    (logged,) = logged_predictions(conn)
    assert logged["model_name"] == "purchase_model"
    assert logged["model_version"] == "7"
    assert logged["entity_id"] == 2
    assert logged["probability"] == pytest.approx(probability)
    assert logged["features"] == INACTIVE
    assert logged["features_as_of"] == AS_OF


def test_predict_for_unknown_user_is_404_and_logs_nothing(predict_api, conn):
    response = predict_api.post("/predict", json={"user_id": 99})

    assert response.status_code == 404
    assert logged_predictions(conn) == []


def test_predict_without_a_model_is_503(client, monkeypatch):
    def no_champion(tracking_uri):
        raise MlflowException("Registered model alias champion not found")

    monkeypatch.setattr(api, "load_champion", no_champion)
    monkeypatch.setattr(app.state, "model", None, raising=False)

    response = client.post("/predict", json={"user_id": 1})

    assert response.status_code == 503
    assert "No model available" in response.json()["detail"]


def test_the_model_is_loaded_once_and_then_reused(client, model, monkeypatch):
    loads = []

    def load(tracking_uri):
        loads.append(tracking_uri)
        return model

    monkeypatch.setattr(api, "load_champion", load)
    monkeypatch.setattr(app.state, "model", None, raising=False)

    client.post("/predict", json={"user_id": 1})
    client.post("/predict", json={"user_id": 2})

    assert len(loads) == 1
