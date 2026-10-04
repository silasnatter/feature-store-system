"""HTTP API of the feature store.

uvicorn feature_store.api:app --reload
"""

import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

import psycopg
import redis
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from mlflow.exceptions import MlflowException
from psycopg.types.json import Jsonb
from pydantic import AwareDatetime, BaseModel, Field

from feature_store.config import get_settings
from feature_store.db import create_pool
from feature_store.historical import get_historical_features
from feature_store.modeling import (
    MODEL_FEATURES,
    MODEL_NAME,
    MODEL_VIEW,
    LoadedModel,
    champion_info,
    fail_fast,
    load_champion,
    predict_purchase,
)
from feature_store.monitoring import current_status
from feature_store.online import connect_redis, read_entity
from feature_store.registry import FeatureRegistry, NotRegisteredError

# An API request must not hang for minutes when MLflow is down
fail_fast()

# How often to ask MLflow whether the champion has changed
MODEL_REFRESH_SECONDS = 60


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pool and one Redis client for the lifetime of the process
    app.state.pool = create_pool()
    app.state.redis = connect_redis()
    yield
    app.state.pool.close()
    app.state.redis.close()


app = FastAPI(title="Feature Store API", lifespan=lifespan)


def get_conn(request: Request) -> Iterator[psycopg.Connection]:
    """A pooled Postgres connection for one request."""
    with request.app.state.pool.connection() as conn:
        yield conn


def get_redis(request: Request) -> redis.Redis:
    return request.app.state.redis


def get_model(request: Request) -> LoadedModel:
    """The champion model, kept in memory.

    It is fetched from MLflow on first use; the feature endpoints work without
    a model, so it is not loaded at startup. After that, at most once every
    MODEL_REFRESH_SECONDS, a cheap lookup checks whether the champion has
    changed, and only then is the new one loaded. If MLflow cannot be reached,
    the model already in memory keeps serving.
    """
    state = request.app.state
    model = getattr(state, "model", None)
    checked_at = getattr(state, "model_checked_at", None)
    now = time.monotonic()
    if model is not None and checked_at is not None and now - checked_at < MODEL_REFRESH_SECONDS:
        return model

    tracking_uri = get_settings().mlflow_tracking_uri
    try:
        if model is None or champion_info(tracking_uri).version != model.version:
            model = load_champion(tracking_uri)
    except MlflowException as exc:
        if model is None:
            raise HTTPException(status_code=503, detail=f"No model available: {exc}") from exc
    state.model = model
    state.model_checked_at = now
    return model


Conn = Annotated[psycopg.Connection, Depends(get_conn)]
Redis = Annotated[redis.Redis, Depends(get_redis)]
Model = Annotated[LoadedModel, Depends(get_model)]


class EntityRow(BaseModel):
    entity_id: int
    event_timestamp: AwareDatetime


class HistoricalRequest(BaseModel):
    entity_rows: list[EntityRow] = Field(max_length=10_000)
    features: list[str] = Field(min_length=1)


class PredictRequest(BaseModel):
    user_id: int


@app.get("/health")
def health(conn: Conn, client: Redis):
    """Checks that both stores answer."""
    try:
        conn.execute("SELECT 1")
        client.ping()
    except (psycopg.Error, redis.RedisError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


@app.get("/features")
def list_features(conn: Conn):
    """The active features in the registry."""
    return {"features": FeatureRegistry(conn).list_features()}


@app.get("/features/online/{view_name}/{entity_id}")
def online_features(
    view_name: str,
    entity_id: int,
    client: Redis,
    feature: Annotated[list[str] | None, Query()] = None,
):
    """The latest values of one entity, served from Redis only.

    Without `feature` parameters, returns every stored feature. With them,
    returns exactly those; a feature without a stored value is null.
    """
    row = read_entity(client, view_name, entity_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No online values for {view_name}:{entity_id}")
    values = row.values if feature is None else {name: row.values.get(name) for name in feature}
    return {
        "view": view_name,
        "entity_id": entity_id,
        "event_timestamp": row.event_timestamp,
        "features": values,
    }


@app.post("/features/historical")
def historical_features(body: HistoricalRequest, conn: Conn):
    """Point-in-time correct values for training rows, served from Postgres."""
    entity_rows: list[tuple[int, datetime]] = [
        (row.entity_id, row.event_timestamp) for row in body.entity_rows
    ]
    try:
        rows = get_historical_features(conn, entity_rows, body.features)
    except NotRegisteredError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"rows": rows}


@app.post("/predict")
def predict(body: PredictRequest, conn: Conn, client: Redis, model: Model):
    """The probability that a user orders in the next 30 days.

    Features come from Redis, the model from MLflow. Every prediction is
    logged to Postgres with the feature values the model saw.
    """
    row = read_entity(client, MODEL_VIEW, body.user_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No online features for user {body.user_id}")
    # A feature without a stored value is None; the model fills the gap itself
    features = {name: row.values.get(name) for name in MODEL_FEATURES}
    probability = predict_purchase(model.pipeline, features)

    conn.execute(
        """
        INSERT INTO feature_store.predictions
            (model_name, model_version, entity_id, probability, features, features_as_of)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            MODEL_NAME,
            model.version,
            body.user_id,
            probability,
            Jsonb(features),
            row.event_timestamp,
        ),
    )
    return {
        "user_id": body.user_id,
        "purchase_probability": probability,
        "model": MODEL_NAME,
        "model_version": model.version,
        "features": features,
        "features_as_of": row.event_timestamp,
    }


@app.get("/monitor/status")
def monitor_status(conn: Conn, as_of: AwareDatetime | None = None):
    """The health of every feature and of the model, by default as of now.

    Quality, drift and model performance come from the latest recorded checks;
    freshness is measured on the spot. The overall status is the worst one.
    """
    return current_status(conn, as_of or datetime.now(UTC))
