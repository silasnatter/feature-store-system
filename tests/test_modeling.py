from datetime import UTC, datetime

import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from feature_store.modeling import (
    CHAMPION,
    MODEL_FEATURES,
    MODEL_NAME,
    build_pipeline,
    champion_info,
    load_champion,
    predict_purchase,
    train_and_register,
)

ACTIVE = {
    "order_count_30d": 3.0,
    "avg_order_value_30d": 40.0,
    "days_since_last_order": 2.0,
    "view_count_7d": 9.0,
}
INACTIVE = {
    "order_count_30d": 0.0,
    "avg_order_value_30d": None,
    "days_since_last_order": None,
    "view_count_7d": 0.0,
}


def training_frame(misleading_training_labels: bool = False) -> pd.DataFrame:
    """Two snapshots of 40 users: active users buy, inactive ones do not.

    May is the training month, June the test month. With
    `misleading_training_labels`, May says the opposite, so a model trained on
    it does badly on June.
    """
    rows = []
    for month in (5, 6):
        for user_id in range(40):
            features = ACTIVE if user_id % 2 == 0 else INACTIVE
            # one user per group goes against the pattern, so the fit is not perfect
            buys = (user_id % 2 == 0) != (user_id in (0, 1))
            if misleading_training_labels and month == 5:
                buys = not buys
            rows.append(
                {
                    "entity_id": user_id,
                    "event_timestamp": datetime(2026, month, 1, tzinfo=UTC),
                    **features,
                    "label": int(buys),
                }
            )
    return pd.DataFrame(rows)


JUNE = datetime(2026, 6, 1, tzinfo=UTC)


def test_pipeline_handles_missing_values():
    frame = training_frame()
    pipeline = build_pipeline().fit(frame[MODEL_FEATURES], frame["label"])

    active = predict_purchase(pipeline, ACTIVE)
    inactive = predict_purchase(pipeline, INACTIVE)

    assert 0 <= inactive < active <= 1


def test_train_and_register_stores_the_model_as_champion(tracking_uri):
    result = train_and_register(training_frame(), datetime(2026, 6, 1, tzinfo=UTC), tracking_uri)

    assert (result.train_rows, result.test_rows) == (40, 40)
    assert result.metrics["roc_auc"] > 0.8
    champion = MlflowClient(tracking_uri).get_model_version_by_alias(MODEL_NAME, CHAMPION)
    assert str(champion.version) == result.version


def test_loaded_champion_predicts_like_the_trained_model(tracking_uri):
    frame = training_frame()
    test_from = datetime(2026, 6, 1, tzinfo=UTC)
    result = train_and_register(frame, test_from, tracking_uri)
    train = frame[frame["event_timestamp"] < test_from]
    expected = build_pipeline().fit(train[MODEL_FEATURES], train["label"])

    model = load_champion(tracking_uri)

    assert model.version == result.version
    for features in (ACTIVE, INACTIVE):
        assert predict_purchase(model.pipeline, features) == pytest.approx(
            predict_purchase(expected, features)
        )


def test_retraining_moves_the_champion_to_the_new_version(tracking_uri):
    first = train_and_register(training_frame(), datetime(2026, 6, 1, tzinfo=UTC), tracking_uri)

    second = train_and_register(training_frame(), datetime(2026, 6, 1, tzinfo=UTC), tracking_uri)

    assert second.version != first.version
    assert load_champion(tracking_uri).version == second.version


def test_split_that_leaves_one_side_empty_is_rejected(tracking_uri):
    with pytest.raises(ValueError, match="leaves"):
        train_and_register(training_frame(), datetime(2026, 9, 1, tzinfo=UTC), tracking_uri)


def test_load_champion_without_a_registered_model_raises(tracking_uri):
    with pytest.raises(MlflowException):
        load_champion(tracking_uri)


def test_training_stores_a_drift_reference_and_the_test_score_with_the_model(tracking_uri):
    result = train_and_register(training_frame(), JUNE, tracking_uri)

    model = load_champion(tracking_uri)

    assert model.test_roc_auc == pytest.approx(result.metrics["roc_auc"])
    assert set(model.reference) == set(MODEL_FEATURES)
    # Half of the training users are inactive and have no order value
    assert model.reference["avg_order_value_30d"]["shares"][-1] == pytest.approx(0.5)


def test_champion_info_gives_the_same_facts_without_loading_the_model(tracking_uri):
    result = train_and_register(training_frame(), JUNE, tracking_uri)

    info = champion_info(tracking_uri)

    assert info.version == result.version
    assert info.test_roc_auc == pytest.approx(result.metrics["roc_auc"])
    assert set(info.reference) == set(MODEL_FEATURES)


def test_first_model_is_promoted_even_when_it_has_to_be_better(tracking_uri):
    result = train_and_register(training_frame(), JUNE, tracking_uri, promote_if_better=True)

    assert result.promoted
    assert result.champion_roc_auc is None
    assert load_champion(tracking_uri).version == result.version


def test_model_that_is_not_better_does_not_replace_the_champion(tracking_uri):
    first = train_and_register(training_frame(), JUNE, tracking_uri)

    # Same data, so the same model: equal, not better
    second = train_and_register(training_frame(), JUNE, tracking_uri, promote_if_better=True)

    assert not second.promoted
    assert second.version != first.version  # it is still stored as a version
    assert second.champion_roc_auc == pytest.approx(second.metrics["roc_auc"])
    assert load_champion(tracking_uri).version == first.version


def test_better_model_replaces_the_champion(tracking_uri):
    train_and_register(training_frame(misleading_training_labels=True), JUNE, tracking_uri)

    better = train_and_register(training_frame(), JUNE, tracking_uri, promote_if_better=True)

    assert better.promoted
    assert better.metrics["roc_auc"] > better.champion_roc_auc
    assert load_champion(tracking_uri).version == better.version
