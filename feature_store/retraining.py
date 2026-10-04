"""Keeping the model current: watch its inputs, decide whether to retrain, retrain.

These are the jobs that need both the feature store and the model registry.
"""

from datetime import date, datetime

import psycopg
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score

from feature_store.modeling import (
    MODEL_FEATURES,
    MODEL_NAME,
    LoadedModel,
    TrainingResult,
    champion_info,
    champion_or_none,
    train_and_register,
)
from feature_store.monitoring import drift_checks, log_checks
from feature_store.training import LABEL_HORIZONS, build_training_set, monthly_snapshots
from feature_store.validation import ALERT, OK, Check, quality_checks

LABEL = "purchase_next_30d"

# How far the champion's score on recent data may fall below its score at
# training time before that counts as a reason to retrain.
MAX_AUC_DROP = 0.03


def monitor_features(
    conn: psycopg.Connection, view_name: str, as_of: datetime, tracking_uri: str
) -> tuple[list[Check], str | None]:
    """Run the quality and drift checks for one snapshot and record them.

    Drift is measured against the champion's training data. If that is not
    available, only quality is checked, and the second return value says why.
    Does not commit.
    """
    checks = quality_checks(conn, view_name, as_of)
    note = None
    try:
        reference = champion_info(tracking_uri).reference
    except MlflowException as exc:
        reference, note = None, f"drift not checked, the champion could not be looked up: {exc}"
    else:
        if reference is None:
            note = "drift not checked: the champion was trained before drift references existed"
    if reference is not None:
        checks += drift_checks(conn, as_of, reference)
    log_checks(conn, "feature", as_of, checks)
    return checks, note


def performance_check(conn: psycopg.Connection, champion: LoadedModel, as_of: datetime) -> Check:
    """Score the champion on the newest snapshot whose outcome is known at `as_of`.

    The label looks 30 days ahead, so that snapshot lies 30 days before
    `as_of`. The score is compared with the one the champion got on its own
    test rows when it was trained.
    """
    snapshot = as_of - LABEL_HORIZONS[LABEL]
    frame = build_training_set(conn, [snapshot], MODEL_FEATURES, LABEL)
    scores = champion.pipeline.predict_proba(frame[MODEL_FEATURES].astype("float64"))[:, 1]
    roc_auc = roc_auc_score(frame["label"], scores)

    message = f"ROC AUC {roc_auc:.3f} on the snapshot of {snapshot.date()}"
    if champion.test_roc_auc is None:
        return Check(MODEL_NAME, "roc_auc", roc_auc, OK, message)
    dropped = champion.test_roc_auc - roc_auc > MAX_AUC_DROP
    message += (
        f", {champion.test_roc_auc:.3f} when it was trained "
        f"(alert when more than {MAX_AUC_DROP} lower)"
    )
    return Check(MODEL_NAME, "roc_auc", roc_auc, ALERT if dropped else OK, message)


def retrain_reasons(conn: psycopg.Connection, as_of: datetime, tracking_uri: str) -> list[str]:
    """Why the model should be retrained as of `as_of`; an empty list means it should not.

    Reasons are: no champion, a champion without a drift reference, a feature
    that has drifted significantly from the champion's training data, or a
    champion that scores clearly worse on recent data than when it was trained.
    Records the checks it makes. Does not commit.
    """
    champion = champion_or_none(tracking_uri)
    if champion is None:
        return ["there is no champion model yet"]

    reasons = []
    if champion.reference is None:
        reasons.append("the champion has no drift reference to monitor its inputs against")
    else:
        drift = drift_checks(conn, as_of, champion.reference)
        log_checks(conn, "feature", as_of, drift)
        reasons += [
            f"{check.name} has drifted: {check.message}" for check in drift if check.status == ALERT
        ]

    performance = performance_check(conn, champion, as_of)
    log_checks(conn, "model", as_of, [performance])
    if performance.status == ALERT:
        reasons.append(f"the model has got worse: {performance.message}")
    return reasons


def retrain(
    conn: psycopg.Connection, as_of: datetime, start: date, tracking_uri: str
) -> TrainingResult:
    """Train a new model on what is known at `as_of`; promote it if it beats the champion.

    The newest snapshot with a known outcome (30 days before `as_of`) is held
    back as the test set. Training uses the first of each month from `start`,
    up to the last one whose own 30-day outcome window has closed before the
    test snapshot, so nothing the new model learns from overlaps with the test.
    """
    horizon = LABEL_HORIZONS[LABEL]
    test_snapshot = as_of - horizon
    train_snapshots = monthly_snapshots(start, (test_snapshot - horizon).date())
    if not train_snapshots:
        raise ValueError(f"No training snapshots between {start} and {test_snapshot - horizon}")
    frame = build_training_set(conn, [*train_snapshots, test_snapshot], MODEL_FEATURES, LABEL)
    return train_and_register(frame, test_snapshot, tracking_uri, promote_if_better=True)
