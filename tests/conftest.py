import os
from pathlib import Path

import psycopg
import pytest
import redis
from psycopg.rows import dict_row

from feature_store.config import Settings

INIT_SQL_DIR = Path(__file__).parent.parent / "postgres" / "init"
TEST_DB = "feature_store_test"
TEST_REDIS_DB = 15  # the app uses database 0


@pytest.fixture(scope="session")
def test_dsn():
    """A throwaway database with the schema from postgres/init applied."""
    settings = Settings()
    try:
        admin = psycopg.connect(settings.postgres_dsn, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"Postgres is not reachable (run `docker compose up -d`): {exc}")

    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
        dsn = settings.model_copy(update={"postgres_db": TEST_DB}).postgres_dsn
        with psycopg.connect(dsn) as conn:
            for sql_file in sorted(INIT_SQL_DIR.glob("*.sql")):
                conn.execute(sql_file.read_text())
        yield dsn
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")


@pytest.fixture
def conn(test_dsn):
    """A connection whose changes are rolled back after the test."""
    with psycopg.connect(test_dsn, row_factory=dict_row) as connection:
        yield connection
        connection.rollback()


@pytest.fixture
def redis_client():
    """An empty Redis database, separate from the one the app uses."""
    settings = Settings()
    client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=TEST_REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=3,
    )
    try:
        client.flushdb()
    except redis.ConnectionError as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"Redis is not reachable (run `docker compose up -d`): {exc}")
    yield client
    client.flushdb()
    client.close()
