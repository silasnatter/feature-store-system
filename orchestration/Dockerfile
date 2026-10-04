# Airflow, plus this project's command line for the DAG tasks to call.
# Built from the repo root by `docker compose build airflow`.
FROM apache/airflow:3.3.1

# The project gets its own virtualenv, so its dependencies never mix with
# Airflow's. uv ships with the Airflow image and fetches Python 3.12 for it.
USER root
RUN mkdir /opt/feature_store && chown airflow:0 /opt/feature_store
USER airflow
WORKDIR /opt/feature_store
ENV UV_LINK_MODE=copy

# Dependencies first: this layer is rebuilt only when they change
COPY --chown=airflow:0 pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=airflow:0 feature_store ./feature_store
RUN uv sync --frozen --no-dev

WORKDIR /opt/airflow
