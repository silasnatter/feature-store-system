from datetime import UTC, datetime, timedelta

import pytest

from feature_store.monitoring import (
    current_status,
    drift_checks,
    freshness_checks,
    log_checks,
    population_stability_index,
    reference_distribution,
)
from feature_store.registry import FeatureDefinition, FeatureRegistry, FeatureView
from feature_store.validation import ALERT, OK, WARNING, Check

VIEW = "stats"


def day(n: int, hour: int = 0) -> datetime:
    return datetime(2026, 3, n, hour, tzinfo=UTC)


# --- the drift measure: no database needed -----------------------------------


def test_reference_bins_hold_equal_shares_of_evenly_spread_values():
    reference = reference_distribution([float(n) for n in range(1, 101)])

    assert len(reference["edges"]) == 9
    assert reference["shares"] == pytest.approx([0.1] * 10 + [0.0])  # last entry: missing values


def test_reference_merges_repeated_edges_and_counts_missing_values():
    values = [0.0] * 6 + [1.0, 2.0] + [None, float("nan")]

    reference = reference_distribution(values)

    assert reference["edges"] == sorted(set(reference["edges"]))
    assert reference["edges"][0] == 0.0
    assert reference["shares"][0] == pytest.approx(0.6)  # the zeros
    assert reference["shares"][-1] == pytest.approx(0.2)  # None and NaN
    assert sum(reference["shares"]) == pytest.approx(1.0)


def test_psi_is_zero_for_the_same_distribution():
    values = [float(n % 17) for n in range(500)]

    assert population_stability_index(reference_distribution(values), values) == pytest.approx(0)


def test_psi_grows_with_the_size_of_the_shift():
    values = [float(n) for n in range(1000)]
    reference = reference_distribution(values)

    small = population_stability_index(reference, [value + 30 for value in values])
    large = population_stability_index(reference, [value + 400 for value in values])

    assert 0 < small < 0.1 < 0.25 < large


def test_psi_sees_a_change_in_the_share_of_missing_values():
    values = [float(n) for n in range(100)]
    reference = reference_distribution(values)

    half_missing = values[:50] + [None] * 50

    assert population_stability_index(reference, half_missing) > 0.25


def test_psi_stays_finite_when_a_bin_becomes_empty():
    reference = reference_distribution([float(n) for n in range(100)])

    psi = population_stability_index(reference, [5.0] * 100)

    assert 0.25 < psi < float("inf")


# --- checks against the database ---------------------------------------------


@pytest.fixture
def conn(conn):
    """The view `stats` with `spend` (fresh for 24 hours) and `clicks` (48 hours)."""
    registry = FeatureRegistry(conn)
    registry.register_view(FeatureView(VIEW, "user"))
    registry.register_feature(FeatureDefinition("spend", VIEW, freshness_hours=24))
    registry.register_feature(FeatureDefinition("clicks", VIEW, freshness_hours=48))
    return conn


def store(conn, feature_name, values, event_timestamp):
    """Store one value per entity, entity ids counting from 1."""
    for entity_id, value in enumerate(values, start=1):
        conn.execute(
            """
            INSERT INTO feature_store.feature_values
                (feature_id, entity_id, event_timestamp, value)
            SELECT feature_id, %s, %s, %s
            FROM feature_store.feature_definitions
            WHERE feature_name = %s
            """,
            (entity_id, event_timestamp, value, feature_name),
        )


def test_drift_checks_grade_each_feature_against_its_reference(conn):
    training_values = [float(n) for n in range(200)]
    reference = {
        "spend": reference_distribution(training_values),
        "clicks": reference_distribution(training_values),
    }
    store(conn, "spend", training_values, day(2))  # unchanged
    store(conn, "clicks", [value + 150 for value in training_values], day(2))  # shifted

    checks = {check.name: check for check in drift_checks(conn, day(2), reference)}

    assert checks["spend"].metric == "psi"
    assert checks["spend"].status == OK
    assert checks["spend"].value == pytest.approx(0)
    assert checks["clicks"].status == ALERT
    assert checks["clicks"].value > 0.25


def test_drift_check_without_values_for_the_moment_is_a_warning(conn):
    reference = {"spend": reference_distribution([1.0, 2.0, 3.0])}
    store(conn, "spend", [1.0, 2.0, 3.0], day(1))

    (check,) = drift_checks(conn, day(2), reference)

    assert (check.status, check.value) == (WARNING, None)
    assert "no values for 2026-03-02" in check.message


def test_freshness_measures_the_age_of_the_newest_values(conn):
    store(conn, "spend", [1.0], day(1))
    store(conn, "spend", [1.0], day(2))
    store(conn, "spend", [1.0], day(9))  # after the moment asked about: ignored
    store(conn, "clicks", [1.0], day(1))

    checks = {check.name: check for check in freshness_checks(conn, day(3, hour=6))}

    assert checks["spend"].value == pytest.approx(30)  # day 2 to day 3, 06:00
    assert checks["spend"].status == ALERT  # allowed: 24 hours
    assert checks["clicks"].value == pytest.approx(54)
    assert checks["clicks"].status == ALERT  # allowed: 48 hours
    fresh = {check.name: check.status for check in freshness_checks(conn, day(2, hour=12))}
    assert fresh == {"spend": OK, "clicks": OK}


def test_feature_that_never_had_values_is_not_fresh(conn):
    store(conn, "spend", [1.0], day(2))

    checks = {check.name: check for check in freshness_checks(conn, day(2))}

    assert checks["clicks"].status == ALERT
    assert checks["clicks"].message == "no values at all"


def test_logging_the_same_snapshot_again_replaces_the_outcome(conn):
    log_checks(conn, "feature", day(2), [Check("spend", "psi", 0.05, OK, "fine")])

    log_checks(conn, "feature", day(2), [Check("spend", "psi", 0.4, ALERT, "drifted")])

    rows = conn.execute("SELECT value, status, message FROM feature_store.monitor_logs").fetchall()
    assert rows == [{"value": 0.4, "status": ALERT, "message": "drifted"}]


def test_status_reports_the_latest_outcome_of_each_check_and_the_worst_overall(conn):
    store(conn, "spend", [1.0], day(3))
    store(conn, "clicks", [1.0], day(3))
    log_checks(conn, "feature", day(2), [Check("spend", "psi", 0.4, ALERT, "drifted")])
    log_checks(conn, "feature", day(3), [Check("spend", "psi", 0.02, OK, "stable")])
    log_checks(conn, "feature", day(3), [Check("clicks", "psi", 0.15, WARNING, "moving")])
    log_checks(conn, "model", day(3), [Check("purchase_model", "roc_auc", 0.93, OK, "good")])

    status = current_status(conn, as_of=day(3, hour=6))

    assert status["status"] == WARNING
    assert status["as_of"] == day(3, hour=6)
    spend = {check["metric"]: check for check in status["features"]["spend"]["checks"]}
    assert spend["psi"]["status"] == OK  # day 3 replaced the alert of day 2
    assert spend["psi"]["snapshot_at"] == day(3)
    assert spend["freshness_hours"]["value"] == pytest.approx(6)
    assert status["features"]["spend"]["status"] == OK
    assert status["features"]["clicks"]["status"] == WARNING
    assert status["models"]["purchase_model"]["status"] == OK


def test_status_as_of_an_earlier_moment_ignores_later_outcomes(conn):
    store(conn, "spend", [1.0], day(2))
    store(conn, "clicks", [1.0], day(2))
    log_checks(conn, "feature", day(2), [Check("spend", "psi", 0.4, ALERT, "drifted")])
    log_checks(conn, "feature", day(3), [Check("spend", "psi", 0.02, OK, "stable")])

    status = current_status(conn, as_of=day(2, hour=6))

    assert status["status"] == ALERT
    assert status["features"]["spend"]["status"] == ALERT


def test_status_turns_to_alert_when_the_data_goes_stale(conn):
    store(conn, "spend", [1.0], day(2))
    store(conn, "clicks", [1.0], day(2))

    assert current_status(conn, as_of=day(2) + timedelta(hours=1))["status"] == OK
    assert current_status(conn, as_of=day(2) + timedelta(days=5))["status"] == ALERT
