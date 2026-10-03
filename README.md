# Feature store from scratch

A small feature store built to understand the moving parts: an offline store in PostgreSQL, an online store in Redis, point-in-time correct training sets, and a serving API.

Status: raw event data, the feature registry, feature computation and backfill exist. The point-in-time join, the Redis online store and the HTTP API work. Model training, orchestration and monitoring do not exist yet.

## Setup

Requires Docker, [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it if missing).

```bash
cp .env.example .env
docker compose up -d                      # Postgres on 5432, Redis on 6379
uv sync
uv run python -m feature_store.datagen    # fill the raw tables with synthetic events
uv run pytest
```

The SQL files in `postgres/init/` run once, when the Postgres volume is first created. To re-run them, reset the volume with `docker compose down -v`.

The tests create and drop their own `feature_store_test` database, so they never touch the data in `feature_store`.

## Data

`feature_store.datagen` generates seeded synthetic e-commerce events (users, products, product views, orders) into the `raw` schema. The same seed and options always produce the same rows, and each run replaces the previous contents. The defaults give 2,000 users and 270 days of events starting 2026-01-01; see `--help` for options.

## Computing features

```bash
uv run python -m feature_store apply      # write the feature definitions to the registry
uv run python -m feature_store backfill --view user_purchase_stats --start 2026-01-02 --end 2026-09-28
uv run python -m feature_store compute --view user_purchase_stats --as-of 2026-09-29
```

A feature view is computed as of a moment: its SQL reads only raw events from before that moment, and the values are stored with that moment as their event timestamp. `backfill` does this for midnight UTC of every day in the range. Computing the same moment again writes nothing unless the raw data changed.

## Serving features

```bash
uv run python -m feature_store materialize --view user_purchase_stats --as-of 2026-09-28
uv run uvicorn feature_store.api:app --reload     # http://localhost:8000/docs
```

`materialize` copies each entity's latest valid values from Postgres into Redis, one hash per entity and view (`user_purchase_stats:7`). Without `--as-of` it uses the current time; the generated data ends on 2026-09-28, so with the view's 2-day TTL a later moment finds nothing to copy.

| Endpoint | Reads from | Purpose |
|---|---|---|
| `GET /features/online/{view}/{entity_id}` | Redis | Latest values for one entity, for predictions |
| `POST /features/historical` | Postgres | Point-in-time correct values for training rows |
| `GET /features` | Postgres | The active features in the registry |
| `GET /health` | both | Checks that both stores answer |

```bash
curl localhost:8000/features/online/user_purchase_stats/9
curl -X POST localhost:8000/features/historical -H 'content-type: application/json' \
  -d '{"entity_rows": [{"entity_id": 7, "event_timestamp": "2026-06-01T12:00:00Z"}], "features": ["order_count_30d"]}'
```

## Layout

```
feature_store/
  config.py          Settings from the environment or .env
  db.py              Postgres connection and pool
  registry.py        Feature views and feature definitions
  datagen.py         Synthetic event generator
  definitions.py     The feature views and features of this project
  feature_sql/       One SQL file per feature view
  compute.py         Compute a view as of a moment; backfill a date range
  historical.py      Point-in-time join for training sets
  online.py          Redis online store and materialisation
  api.py             FastAPI app
  __main__.py        Command line
postgres/init/       Schema, applied on first container start
tests/               pytest suite
```
