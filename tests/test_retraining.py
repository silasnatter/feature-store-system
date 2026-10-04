from datetime import UTC, date, datetime

import pytest

from feature_store import definitions, retraining
from feature_store.compute import compute_features
from feature_store.datagen import GeneratorConfig, generate, load
from feature_store.modeling import MODEL_FEATURES, LoadedModel, build_pipeline, load_champion
from feature_store.monitoring import reference_distribution
from feature_store.registry import FeatureRegistry
from feature_store.retraining import monitor_features, performance_check, retrain, retrain_reasons
from feature_store.training import build_training_set

VIEW = "user_purchase_stats"
START = date(2026, 2, 1)

# 150 days of events from 1 January. As of 25 May, the newest snapshot with a
# known 30-day outcome is 25 April; training can use 1 February and 1 March,
# whose outcome windows close before 25 April.
SMALL = GeneratorConfig(seed=7, start=date(2026, 1, 1), days=150, n_users=150, n_products=30)
AS_OF = datetime(2026, 5, 25, tzinfo=UTC)
TEST_SNAPSHOT = datetime(2026, 4, 25, tzinfo=UTC)
TRAIN_SNAPSHOTS = [datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)]


@pytest.fixture
def conn(conn):
    """Raw events, plus the feature snapshots that training and monitoring will read."""
    load(conn, generate(SMALL))
    definitions.apply(FeatureRegistry(conn))
    for moment in [*TRAIN_SNAPSHOTS, TEST_SNAPSHOT, AS_OF]:
        compute_features(conn, VIEW, moment)
    return conn


def users_before(conn, moment):
    return conn.execute(
        "SELECT count(*) AS n FROM raw.users WHERE signup_ts < %s", (moment,)
    ).fetchone()["n"]


def snapshot_values(conn, feature_name, moment):
    rows = conn.execute(
        """
        SELECT fval.value
        FROM feature_store.feature_values fval
        JOIN feature_store.feature_definitions fd USING (feature_id)
        WHERE fd.feature_name = %s AND fval.event_timestamp = %s
        """,
        (feature_name, moment),
    ).fetchall()
    return [row["value"] for row in rows]


def healthy_champion(conn, **changes) -> LoadedModel:
    """A model fitted on the test snapshot whose drift reference is today's data itself."""
    frame = build_training_set(conn, [TEST_SNAPSHOT], MODEL_FEATURES)
    pipeline = build_pipeline().fit(frame[MODEL_FEATURES].astype("float64"), frame["label"])
    reference = {
        name: reference_distribution(snapshot_values(conn, name, AS_OF)) for name in MODEL_FEATURES
    }
    fields = {"version": "1", "pipeline": pipeline, "test_roc_auc": None, "reference": reference}
    return LoadedModel(**(fields | changes))


def use_champion(monkeypatch, champion):
    monkeypatch.setattr(retraining, "champion_or_none", lambda tracking_uri: champion)


def logged_checks(conn):
    rows = conn.execute(
        "SELECT kind, name, metric, snapshot_at, status FROM feature_store.monitor_logs"
    ).fetchall()
    return {(row["kind"], row["name"], row["metric"]): row for row in rows}


# --- retrain ------------------------------------------------------------------


def test_retrain_trains_on_closed_months_and_tests_on_the_newest_known_snapshot(conn, tracking_uri):
    result = retrain(conn, AS_OF, START, tracking_uri)

    assert result.train_rows == sum(users_before(conn, moment) for moment in TRAIN_SNAPSHOTS)
    assert result.test_rows == users_before(conn, TEST_SNAPSHOT)
    assert result.promoted  # there was no champion to beat
    assert result.champion_roc_auc is None
    assert load_champion(tracking_uri).version == result.version


def test_retrain_on_the_same_data_does_not_replace_the_champion(conn, tracking_uri):
    first = retrain(conn, AS_OF, START, tracking_uri)

    second = retrain(conn, AS_OF, START, tracking_uri)

    assert not second.promoted
    assert second.champion_roc_auc == pytest.approx(second.metrics["roc_auc"])
    assert load_champion(tracking_uri).version == first.version


def test_retrain_without_any_closed_training_month_is_rejected(conn, tracking_uri):
    too_early = datetime(2026, 3, 15, tzinfo=UTC)  # test snapshot 13 Feb, training before 14 Jan

    with pytest.raises(ValueError, match="No training snapshots"):
        retrain(conn, too_early, START, tracking_uri)


# --- retrain_reasons ----------------------------------------------------------


def test_no_champion_is_a_reason_to_retrain(conn, tracking_uri):
    assert retrain_reasons(conn, AS_OF, tracking_uri) == ["there is no champion model yet"]


def test_healthy_champion_gives_no_reason_and_the_checks_are_recorded(conn, monkeypatch):
    use_champion(monkeypatch, healthy_champion(conn))

    assert retrain_reasons(conn, AS_OF, "unused") == []

    logged = logged_checks(conn)
    assert {key for key in logged if key[0] == "feature"} == {
        ("feature", name, "psi") for name in MODEL_FEATURES
    }
    model_check = logged[("model", "purchase_model", "roc_auc")]
    assert model_check["snapshot_at"] == AS_OF
    assert model_check["status"] == "ok"


def test_drifted_feature_is_a_reason_to_retrain(conn, monkeypatch):
    use_champion(monkeypatch, healthy_champion(conn))
    conn.execute(
        """
        UPDATE feature_store.feature_values fval
        SET value = value + 1000
        FROM feature_store.feature_definitions fd
        WHERE fd.feature_id = fval.feature_id
          AND fd.feature_name = 'view_count_7d'
          AND fval.event_timestamp = %s
        """,
        (AS_OF,),
    )

    reasons = retrain_reasons(conn, AS_OF, "unused")

    assert len(reasons) == 1
    assert reasons[0].startswith("view_count_7d has drifted: PSI ")
    assert logged_checks(conn)[("feature", "view_count_7d", "psi")]["status"] == "alert"


def test_champion_without_a_drift_reference_is_a_reason_to_retrain(conn, monkeypatch):
    use_champion(monkeypatch, healthy_champion(conn, reference=None))

    reasons = retrain_reasons(conn, AS_OF, "unused")

    assert reasons == ["the champion has no drift reference to monitor its inputs against"]
    assert ("model", "purchase_model", "roc_auc") in logged_checks(conn)  # still checked


def test_champion_that_got_clearly_worse_is_a_reason_to_retrain(conn, monkeypatch):
    # It claims a perfect score at training; on the recent snapshot it cannot match that
    use_champion(monkeypatch, healthy_champion(conn, test_roc_auc=1.2))

    reasons = retrain_reasons(conn, AS_OF, "unused")

    assert len(reasons) == 1
    assert reasons[0].startswith("the model has got worse: ROC AUC ")


def test_small_drop_in_score_is_tolerated(conn):
    champion = healthy_champion(conn)
    measured = performance_check(conn, champion, AS_OF)
    slightly_higher_at_training = healthy_champion(conn, test_roc_auc=measured.value + 0.02)

    check = performance_check(conn, slightly_higher_at_training, AS_OF)

    assert check.status == "ok"
    assert 0.5 < check.value <= 1
    assert "on the snapshot of 2026-04-25" in check.message


# --- monitor_features ---------------------------------------------------------


def test_monitor_records_quality_and_drift_against_the_champion(conn, tracking_uri):
    retrain(conn, AS_OF, START, tracking_uri)

    checks, note = monitor_features(conn, VIEW, AS_OF, tracking_uri)

    assert note is None
    assert {check.name for check in checks if check.metric == "psi"} == set(MODEL_FEATURES)
    assert {check.metric for check in checks} >= {"row_count", "null_share", "psi"}
    logged = logged_checks(conn)
    assert len(logged) == len(checks)
    assert all(row["snapshot_at"] == AS_OF for row in logged.values())


def test_monitor_without_a_champion_records_quality_only(conn, tracking_uri):
    checks, note = monitor_features(conn, VIEW, AS_OF, tracking_uri)

    assert "drift not checked" in note
    assert checks
    assert all(check.metric != "psi" for check in checks)
    assert len(logged_checks(conn)) == len(checks)
