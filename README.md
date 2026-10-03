# Feature store from scratch

A small feature store built to understand the moving parts: an offline store in PostgreSQL, an online store in Redis, point-in-time correct training sets, and a serving API.

Status: raw event data, schema and feature registry exist. Feature computation, the point-in-time join, the online store and the API do not exist yet.

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

## Layout

```
src/feature_store/
  config.py          Settings from the environment or .env
  db.py              Postgres connection and pool
  registry.py        Feature views and feature definitions
  datagen.py         Synthetic event generator
postgres/init/       Schema, applied on first container start
tests/               pytest suite
```
