import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from feature_store.config import Settings, get_settings


def connect(settings: Settings | None = None) -> psycopg.Connection:
    """Open a single connection, for scripts and jobs."""
    settings = settings or get_settings()
    return psycopg.connect(settings.postgres_dsn, row_factory=dict_row)


def create_pool(settings: Settings | None = None, max_size: int = 10) -> ConnectionPool:
    """Open a connection pool, for long-running services.

    `with pool.connection() as conn:` commits on success and rolls back on error.
    """
    settings = settings or get_settings()
    return ConnectionPool(
        settings.postgres_dsn,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
        open=True,
    )
