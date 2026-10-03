from feature_store.config import Settings


def test_defaults_build_local_urls():
    settings = Settings(_env_file=None)

    assert settings.postgres_dsn == "postgresql://postgres:postgres@localhost:5432/feature_store"
    assert settings.redis_url == "redis://localhost:6379/0"


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "db")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("REDIS_HOST", "cache")

    settings = Settings(_env_file=None)

    assert settings.postgres_dsn == "postgresql://postgres:postgres@db:6543/feature_store"
    assert settings.redis_url == "redis://cache:6379/0"
