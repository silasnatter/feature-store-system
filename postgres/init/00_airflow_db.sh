#!/bin/sh
# Airflow keeps its own bookkeeping (DAG runs, task states) in a separate database.
# A shell script, not .sql: the tests apply every .sql file in this folder to
# their own throwaway database, and that should not get an Airflow database.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "CREATE DATABASE airflow"
