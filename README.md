# Feature store from scratch

A small feature store built to understand the moving parts: an offline store in PostgreSQL, an online store in Redis, point-in-time correct training sets, a model registry in MLflow, a serving API, and a daily pipeline in Airflow.

Status: raw event data, the feature registry, feature computation and backfill exist. The point-in-time join, the Redis online store, model training with MLflow, the HTTP API including predictions, and the daily Airflow pipeline work. Monitoring and automatic retraining do not exist yet.

## Setup

Requires Docker, [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it if missing).

```bash
cp .env.example .env
docker compose up -d                      # Postgres 5432, Redis 6379, MLflow 5001, Airflow 8080
uv sync
uv run python -m feature_store.datagen    # fill the raw tables with synthetic events
uv run pytest
```

The scripts in `postgres/init/` run once, when the Postgres volume is first created. To re-run them, reset the volume with `docker compose down -v`.

The first `docker compose up` builds the Airflow image, which takes a few minutes.

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

Redis only moves forward in time: materialising as of an earlier moment than the one Redis already holds is skipped, so re-running an old date cannot replace newer values. `--force` overrides this.

| Endpoint | Reads from | Purpose |
|---|---|---|
| `GET /features/online/{view}/{entity_id}` | Redis | Latest values for one entity, for predictions |
| `POST /features/historical` | Postgres | Point-in-time correct values for training rows |
| `POST /predict` | Redis, MLflow | Probability that a user orders in the next 30 days |
| `GET /features` | Postgres | The active features in the registry |
| `GET /health` | both | Checks that both stores answer |

```bash
curl localhost:8000/features/online/user_purchase_stats/9
curl -X POST localhost:8000/features/historical -H 'content-type: application/json' \
  -d '{"entity_rows": [{"entity_id": 7, "event_timestamp": "2026-06-01T12:00:00Z"}], "features": ["order_count_30d"]}'
```

## Training and predicting

The model answers one question: will this user place an order in the next 30 days? Features are what was known at a snapshot date; the label is what happened in the 30 days after it.

```bash
uv run python -m feature_store train --start 2026-02-01 --end 2026-08-01 --test-from 2026-07-01
```

This builds a training set from the first of each month in the range (labels joined with point-in-time features), trains a logistic regression on the snapshots before `--test-from`, measures it on the rest, and stores the run and the model in MLflow (http://localhost:5001). The new model version gets the `champion` alias. Snapshots whose 30-day label window reaches past the end of the raw data are rejected.

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"user_id": 1727}'
```

`/predict` reads the user's features from Redis, applies the champion model and logs the prediction, with the feature values the model saw, to `feature_store.predictions`. The model is fetched from MLflow on the first request and kept in memory; restart the API to pick up a new champion. Without a reachable MLflow or a champion it answers 503, and the feature endpoints keep working.

The scripts in `ml/` are the exploration behind the model: `explore.py` looks at the training set, `train.py` compares two baselines, a logistic regression and gradient boosting on `data/training_set.csv` (written by `python -m feature_store training-set`).

## Daily pipeline

Airflow (http://localhost:8080, no login) runs the DAG `daily_features` once per day:

```
compute  ->  validate  ->  materialize
```

- `compute` stores the feature values as of midnight UTC of the run's date.
- `validate` checks them against the thresholds in the feature definitions (values exist, share of nulls, expected range) and fails the run if they break one, so `materialize` never copies bad values to Redis.
- `materialize` copies the latest values to Redis.

Each task calls the same command line you can run by hand, for example `python -m feature_store validate --view user_purchase_stats --as-of 2026-09-28`. The commands are safe to repeat, so a failed run can simply be run again.

The DAG is paused when first created; unpause it in the UI. It then catches up on every day from its start date (2026-09-01) to its end date (2026-09-28, where the generated data ends), one day at a time. Earlier days can be run from the UI as a backfill.

The Airflow container holds the project in its own virtualenv, separate from Airflow's packages. The code in `feature_store/` is mounted from the repo, so code changes take effect without a rebuild; after changing dependencies, run `docker compose build airflow`.

If your Postgres volume was created before Airflow was added, create its database once: `docker compose exec postgres psql -U postgres -c "CREATE DATABASE airflow"`.

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
  label_sql/         One SQL file per label
  training.py        Training sets: labels joined with point-in-time features
  modeling.py        The purchase model: train, store in MLflow, load, predict
  online.py          Redis online store and materialisation
  validation.py      Checks on a snapshot before it goes online
  api.py             FastAPI app
  __main__.py        Command line
ml/                  Exploration scripts
orchestration/       Airflow image and DAGs
postgres/init/       Schema and databases, applied on first container start
tests/               pytest suite
```
