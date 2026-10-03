# Feature store from scratch

A small feature store built to understand the moving parts: an offline store in PostgreSQL, an online store in Redis, point-in-time correct training sets, and a serving API.

Status: project skeleton only. Nothing below the setup section exists yet.

## Setup

Requires Docker, [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it if missing).

```bash
cp .env.example .env
docker compose up -d      # Postgres on 5432, Redis on 6379
uv sync
uv run pytest
```

The SQL files in `postgres/init/` run once, when the Postgres volume is first created. To re-run them, reset the volume with `docker compose down -v`.

## Layout

```
src/feature_store/   Python package
postgres/init/       Schema, applied on first container start
tests/               pytest suite
```
