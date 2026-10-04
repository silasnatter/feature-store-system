"""The purchase model: its features, how it is trained, stored in MLflow and loaded.

The model answers: will this user place an order in the next 30 days?
"""

from dataclasses import dataclass
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "purchase_model"
# The alias that marks the version the API serves
CHAMPION = "champion"

# The model reads these features, in this order, from this view. Training gets
# them from the point-in-time join, serving from Redis.
MODEL_VIEW = "user_purchase_stats"
MODEL_FEATURES = [
    "order_count_30d",
    "avg_order_value_30d",
    "days_since_last_order",
    "view_count_7d",
]


@dataclass(frozen=True)
class LoadedModel:
    version: str
    pipeline: Pipeline


@dataclass(frozen=True)
class TrainingResult:
    version: str
    train_rows: int
    test_rows: int
    metrics: dict[str, float]


def build_pipeline() -> Pipeline:
    """Logistic regression that accepts raw feature values, missing ones included."""
    return make_pipeline(
        # Fill gaps with the training median and add a "was missing" column per feature
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(),
    )


def predict_purchase(pipeline: Pipeline, features: dict[str, float | None]) -> float:
    """The probability that a user with these feature values buys in the next 30 days."""
    row = pd.DataFrame([features], columns=MODEL_FEATURES, dtype="float64")
    return float(pipeline.predict_proba(row)[0, 1])


def train_and_register(
    frame: pd.DataFrame, test_from: datetime, tracking_uri: str
) -> TrainingResult:
    """Train on the snapshots before `test_from`, measure on the rest, store in MLflow.

    `frame` is a training set from `build_training_set`. The run, its metrics
    and the model go to MLflow; the new model version becomes the champion.
    """
    is_test = frame["event_timestamp"] >= test_from
    train, test = frame[~is_test], frame[is_test]
    if train.empty or test.empty:
        raise ValueError(
            f"Splitting at {test_from} leaves {len(train)} train, {len(test)} test rows"
        )

    x_train = train[MODEL_FEATURES].astype("float64")
    x_test = test[MODEL_FEATURES].astype("float64")
    pipeline = build_pipeline().fit(x_train, train["label"])

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = pipeline.predict(x_test)
    metrics = {
        "roc_auc": roc_auc_score(test["label"], probabilities),
        "accuracy": accuracy_score(test["label"], predictions),
        "precision": precision_score(test["label"], predictions, zero_division=0),
        "recall": recall_score(test["label"], predictions),
        "test_base_rate": test["label"].mean(),
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "features": ",".join(MODEL_FEATURES),
                "feature_view": MODEL_VIEW,
                "train_snapshots": f"{train['event_timestamp'].min()} to "
                f"{train['event_timestamp'].max()}",
                "test_snapshots": f"{test['event_timestamp'].min()} to "
                f"{test['event_timestamp'].max()}",
                "train_rows": len(train),
                "test_rows": len(test),
            }
        )
        mlflow.log_metrics(metrics)
        logged = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            # Records the expected input columns and the output type with the model
            signature=infer_signature(x_train, pipeline.predict_proba(x_train)[:, 1]),
            registered_model_name=MODEL_NAME,
            # The model file format only loads types it is told to trust; the
            # pipeline stores numpy column types, which are plain data.
            skops_trusted_types=["numpy.dtype"],
        )
    version = str(logged.registered_model_version)
    MlflowClient(tracking_uri).set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    return TrainingResult(version, len(train), len(test), metrics)


def load_champion(tracking_uri: str) -> LoadedModel:
    """Load the model version that carries the champion alias.

    Raises MlflowException if MLflow cannot be reached or no champion exists.
    """
    mlflow.set_tracking_uri(tracking_uri)
    version = MlflowClient(tracking_uri).get_model_version_by_alias(MODEL_NAME, CHAMPION).version
    pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    return LoadedModel(str(version), pipeline)
