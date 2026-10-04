"""Weekly check whether the model needs retraining, and retraining if it does.

    check --> retrain

`check` looks at two things as of the run's date: have the model's input
features drifted away from the data it was trained on, and does it score
clearly worse on recent data than when it was trained. If neither is the
case, the run stops there.

`retrain` trains a new model and compares it with the current champion on the
same recent data. The new one takes over only if it scores higher.
"""

from datetime import UTC, datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

CLI = "/opt/feature_store/.venv/bin/python -m feature_store"
AS_OF = "{{ (dag_run.logical_date or dag_run.run_after) | ds }}"

with DAG(
    dag_id="retrain_model",
    description="Retrain the purchase model when its inputs drift or it gets worse",
    # Every Sunday at midnight UTC
    schedule="@weekly",
    start_date=datetime(2026, 9, 1, tzinfo=UTC),
    # The generated raw data ends on this day. Remove once new events arrive daily.
    end_date=datetime(2026, 9, 28, tzinfo=UTC),
    catchup=True,
    # One week at a time, oldest first: each run compares against the champion
    # the run before it may have promoted
    max_active_runs=1,
    default_args={
        "cwd": "/opt/feature_store",
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["feature-store"],
):
    # The command exits with status 99 when no retraining is needed. Airflow
    # then marks this task as skipped, and with it everything downstream.
    check = BashOperator(
        task_id="check",
        bash_command=f"{CLI} retrain-needed --as-of {AS_OF}",
        skip_on_exit_code=99,
    )

    retrain = BashOperator(
        task_id="retrain",
        bash_command=f"{CLI} retrain --as-of {AS_OF}",
    )

    check >> retrain
