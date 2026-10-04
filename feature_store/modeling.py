"""The purchase model: its features, how it is trained, stored in MLflow and loaded.

The model answers: will this user place an order in the next 30 days?
"""

import os
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version as installed_version

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_store.monitoring import reference_distribution

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


# Stored with every training run: how each feature was distributed in the
# training data, which is what drift is later measured against.
DRIFT_REFERENCE_FILE = "drift_reference.json"


@dataclass(frozen=True)
class ChampionInfo:
    """What MLflow records about the champion, without the model itself."""

    version: str
    test_roc_auc: float | None  # its score on its own test rows when it was trained
    reference: dict[str, dict] | None  # None for models trained before references existed


@dataclass(frozen=True)
class LoadedModel:
    version: str
    pipeline: Pipeline
    test_roc_auc: float | None = None
    reference: dict[str, dict] | None = None


@dataclass(frozen=True)
class TrainingResult:
    version: str
    train_rows: int
    test_rows: int
    metrics: dict[str, float]
    promoted: bool  # whether the new version became the champion
    champion_roc_auc: float | None  # the previous champion on the same test rows, if compared


def fail_fast() -> None:
    """Make MLflow give up quickly when its server cannot be reached.

    By default it retries for about four minutes, too long for an API request
    or a monitoring check. Settings already in the environment win.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")


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
    frame: pd.DataFrame,
    test_from: datetime,
    tracking_uri: str,
    promote_if_better: bool = False,
) -> TrainingResult:
    """Train on the snapshots before `test_from`, measure on the rest, store in MLflow.

    `frame` is a training set from `build_training_set`. The run, its metrics,
    the drift reference and the model go to MLflow as a new model version.

    By default the new version becomes the champion. With `promote_if_better`
    it has to earn that: the current champion is scored on the same test rows,
    and the new version takes over only if its ROC AUC is higher (or if there
    is no champion yet).
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

    champion = champion_or_none(tracking_uri) if promote_if_better else None
    champion_roc_auc = None
    if champion is not None:
        champion_roc_auc = roc_auc_score(
            test["label"], champion.pipeline.predict_proba(x_test)[:, 1]
        )
        metrics["champion_roc_auc"] = champion_roc_auc
    promoted = champion_roc_auc is None or metrics["roc_auc"] > champion_roc_auc

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
        mlflow.set_tag("promoted", str(promoted).lower())
        reference = {
            name: reference_distribution(x_train[name].tolist()) for name in MODEL_FEATURES
        }
        mlflow.log_dict(reference, DRIFT_REFERENCE_FILE)
        logged = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            # Records the expected input columns and the output type with the model
            signature=infer_signature(x_train, pipeline.predict_proba(x_train)[:, 1]),
            registered_model_name=MODEL_NAME,
            # The model file format only loads types it is told to trust; the
            # pipeline stores numpy column types, which are plain data.
            skops_trusted_types=["numpy.dtype"],
            # What is needed to load and run the model. Listing it also skips
            # MLflow's own detection, which takes several seconds per model.
            pip_requirements=[
                f"{package}=={installed_version(package)}"
                for package in ("mlflow", "scikit-learn", "skops", "pandas")
            ],
        )
    version = str(logged.registered_model_version)
    if promoted:
        MlflowClient(tracking_uri).set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    return TrainingResult(version, len(train), len(test), metrics, promoted, champion_roc_auc)


def champion_info(tracking_uri: str) -> ChampionInfo:
    """Look up the champion's version, recorded test score and drift reference.

    Raises MlflowException if MLflow cannot be reached or no champion exists.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    run = client.get_run(version.run_id)
    reference = None
    if any(file.path == DRIFT_REFERENCE_FILE for file in client.list_artifacts(version.run_id)):
        reference = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/{DRIFT_REFERENCE_FILE}")
    return ChampionInfo(str(version.version), run.data.metrics.get("roc_auc"), reference)


def load_champion(tracking_uri: str) -> LoadedModel:
    """Load the model version that carries the champion alias.

    Raises MlflowException if MLflow cannot be reached or no champion exists.
    """
    info = champion_info(tracking_uri)
    pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{info.version}")
    return LoadedModel(info.version, pipeline, info.test_roc_auc, info.reference)


def champion_or_none(tracking_uri: str) -> LoadedModel | None:
    try:
        return load_champion(tracking_uri)
    except MlflowException as exc:
        # Either no model is registered yet, or no version carries the alias
        if exc.error_code in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            return None
        raise
