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


def training_frame() -> pd.DataFrame:
    """Two snapshots of 40 users: active users buy, inactive ones do not."""
    rows = []
    for month in (5, 6):
        for user_id in range(40):
            features = ACTIVE if user_id % 2 == 0 else INACTIVE
            rows.append(
                {
                    "entity_id": user_id,
                    "event_timestamp": datetime(2026, month, 1, tzinfo=UTC),
                    **features,
                    # one user per group goes against the pattern, so the fit is not perfect
                    "label": int((user_id % 2 == 0) != (user_id in (0, 1))),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def tracking_uri(tmp_path, monkeypatch):
    """A private MLflow store in a temporary directory, no server needed."""
    monkeypatch.chdir(tmp_path)  # MLflow writes model files below the working directory
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


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
