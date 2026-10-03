"""HTTP API of the feature store.

uvicorn feature_store.api:app --reload
"""

from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import psycopg
import redis
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import AwareDatetime, BaseModel, Field

from feature_store.db import create_pool
from feature_store.historical import get_historical_features
from feature_store.online import connect_redis, read_entity
from feature_store.registry import FeatureRegistry, NotRegisteredError


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


Conn = Annotated[psycopg.Connection, Depends(get_conn)]
Redis = Annotated[redis.Redis, Depends(get_redis)]


class EntityRow(BaseModel):
    entity_id: int
    event_timestamp: AwareDatetime


class HistoricalRequest(BaseModel):
    entity_rows: list[EntityRow] = Field(max_length=10_000)
    features: list[str] = Field(min_length=1)


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
