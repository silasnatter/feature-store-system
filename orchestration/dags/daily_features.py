"""Daily feature pipeline: compute the day's feature values, check them, copy them to Redis.

Every task calls the project's command line, which lives in its own virtualenv
inside the Airflow image (see orchestration/Dockerfile). The commands are safe
to run again for the same date, so a failed or repeated run does no harm.
"""

from datetime import UTC, datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

VIEW = "user_purchase_stats"
CLI = "/opt/feature_store/.venv/bin/python -m feature_store"

# The date a run stands for, as YYYY-MM-DD; the commands read it as midnight UTC.
# A scheduled run has a logical date. A run triggered by hand may have none,
# and then uses the date it was triggered on.
AS_OF = "{{ (dag_run.logical_date or dag_run.run_after) | ds }}"

with DAG(
    dag_id="daily_features",
    description="Compute, check and publish the daily feature values",
    # One run per day, at midnight UTC, for the events before that midnight
    schedule="@daily",
    start_date=datetime(2026, 9, 1, tzinfo=UTC),
    # The generated raw data ends on this day. Remove once new events arrive daily.
    end_date=datetime(2026, 9, 28, tzinfo=UTC),
    # Also run for every day since start_date that has not been run yet
    catchup=True,
    # One day at a time, oldest first, so Redis always moves forward
    max_active_runs=1,
    default_args={
        "cwd": "/opt/feature_store",
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["feature-store"],
):
    compute = BashOperator(
        task_id="compute",
        bash_command=f"{CLI} compute --view {VIEW} --as-of {AS_OF}",
    )

    # Exits with an error if the values break the thresholds in the feature
    # definitions. The task then fails and materialize never runs, so bad
    # values do not reach Redis. Trying again would give the same result.
    validate = BashOperator(
        task_id="validate",
        bash_command=f"{CLI} validate --view {VIEW} --as-of {AS_OF}",
        retries=0,
    )

    materialize = BashOperator(
        task_id="materialize",
        bash_command=f"{CLI} materialize --view {VIEW} --as-of {AS_OF}",
    )

    compute >> validate >> materialize
